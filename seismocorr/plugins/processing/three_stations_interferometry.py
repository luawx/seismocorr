# seismocorr/plugins/three_stations_interferometry.py
"""
Three-Station Interferometry (三台/三站干涉) - Minimal Orchestrator

1) 输入：直接输入 traces (N,T) + (i,j) / pairs + k_list（None 默认全部k）
2) 输出：多对 NCF（每对输出多条 ncf_ijk），输出结构可直接接入 stacking.py
3) 本模块只调用外部互相关函数，不调用叠加函数，不做预处理

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np

Array = np.ndarray
LagsAndCCF = Tuple[np.ndarray, np.ndarray]
XCorrFunc = Callable[..., LagsAndCCF]  # 外部互相关函数：xcorr(x,y,**kw)->(lags,ccf)


class PairNCFResult(TypedDict):
    lags2: np.ndarray
    ccfs: List[np.ndarray]
    ks: List[int]


@dataclass
class ThreeStationConfig:
    """
    mode:
      - "correlation": 二次干涉固定用互相关
      - "convolution": 二次干涉固定用卷积
      - "auto": 线性阵列自动分段：
          k 在 i/j 中间 -> convolution
          否则 -> correlation
    """
    mode: str = "auto"                      # "correlation" | "convolution" | "auto"
    second_stage_nfft: Optional[int] = None
    max_lag2: Optional[float] = None


class ThreeStationInterferometry:
    def __init__(
        self,
        sampling_rate: float,
        xcorr_func: XCorrFunc,
        cfg: Optional[ThreeStationConfig] = None,
    ):
        self.sr = float(sampling_rate)
        self.xcorr = xcorr_func
        self.cfg = cfg or ThreeStationConfig()

        if self.cfg.mode not in ("correlation", "convolution", "auto"):
            raise ValueError('ThreeStationConfig.mode must be "correlation", "convolution", or "auto"')

    def compute_pair(
        self,
        traces: np.ndarray,
        i: int,
        j: int,
        k_list: Optional[Sequence[int]] = None,
        **xcorr_kwargs,
    ) -> PairNCFResult:
        if traces.ndim != 2:
            raise ValueError("traces must be a 2D array with shape (N, T)")
        n_stations = traces.shape[0]
        if not (0 <= i < n_stations and 0 <= j < n_stations):
            raise IndexError("i/j out of range")
        if i == j:
            raise ValueError("i and j must be different")

        # 默认 k：全部（排除 i/j）
        if k_list is None:
            ks = [k for k in range(n_stations) if k not in (i, j)]
        else:
            ks = [int(k) for k in k_list if 0 <= int(k) < n_stations and int(k) not in (i, j)]

        if not ks:
            return {"lags2": np.array([]), "ccfs": [], "ks": []}

        xi = traces[i]
        xj = traces[j]

        # 用第一个 k 定标：固定二次输出长度（保证可直接 stacking）
        k0 = ks[0]
        _, ccf_ik0 = self.xcorr(xi, traces[k0], **xcorr_kwargs)
        _, ccf_jk0 = self.xcorr(xj, traces[k0], **xcorr_kwargs)

        if ccf_ik0.size == 0 or ccf_jk0.size == 0:
            return {"lags2": np.array([]), "ccfs": [], "ks": []}

        base_len = min(ccf_ik0.shape[-1], ccf_jk0.shape[-1])
        nfft2 = self._choose_nfft2(base_len)

        lags2_full = self._lags_for_len(nfft2)
        if self.cfg.max_lag2 is not None:
            crop_start, crop_end = self._crop_indices(nfft2, float(self.cfg.max_lag2))
            lags2 = lags2_full[crop_start:crop_end]
        else:
            crop_start, crop_end = 0, nfft2
            lags2 = lags2_full

        ccfs: List[np.ndarray] = []
        ks_used: List[int] = []

        # 线性阵列分段依据（索引顺序即空间顺序）
        lo, hi = (i, j) if i < j else (j, i)

        for k in ks:
            xk = traces[k]

            _, ccf_ik = self.xcorr(xi, xk, **xcorr_kwargs)
            _, ccf_jk = self.xcorr(xj, xk, **xcorr_kwargs)
            if ccf_ik.size == 0 or ccf_jk.size == 0:
                continue

            m = min(ccf_ik.shape[-1], ccf_jk.shape[-1], base_len)
            ccf_ik = ccf_ik[:m]
            ccf_jk = ccf_jk[:m]

            # ========= 自动分段：为每个 k 选择二次干涉模式 =========
            mode_k = self._mode_for_k(i=i, j=j, k=k, lo=lo, hi=hi)

            ncf_full = self._second_stage(ccf_ik, ccf_jk, nfft2, mode_k=mode_k)
            ncf = ncf_full[crop_start:crop_end]

            ccfs.append(ncf)
            ks_used.append(int(k))

        return {"lags2": lags2, "ccfs": ccfs, "ks": ks_used}

    def compute_many(
        self,
        traces: np.ndarray,
        pairs: Sequence[Tuple[int, int]],
        k_list: Optional[Sequence[int]] = None,
        **xcorr_kwargs,
    ) -> Dict[str, PairNCFResult]:
        results: Dict[str, PairNCFResult] = {}
        for (i, j) in pairs:
            results[f"{i}--{j}"] = self.compute_pair(
                traces=traces, i=i, j=j, k_list=k_list, **xcorr_kwargs
            )
        return results

    def _mode_for_k(self, *, i: int, j: int, k: int, lo: int, hi: int) -> str:
        """
        线性DAS阵列自动分段：
          - k 在 (lo, hi) 开区间内 => convolution
          - 否则 => correlation

        若 cfg.mode 不是 auto，则直接返回 cfg.mode。
        """
        if self.cfg.mode != "auto":
            return self.cfg.mode

        # k 是否在 i/j 中间（按索引顺序）
        between = (lo < k < hi)
        return "convolution" if between else "correlation"

    def _choose_nfft2(self, base_len: int) -> int:
        if self.cfg.second_stage_nfft is None:
            return int(2 ** np.ceil(np.log2(max(2, base_len))))
        nfft = int(self.cfg.second_stage_nfft)
        if nfft < base_len:
            raise ValueError(f"second_stage_nfft ({nfft}) must be >= base_len ({base_len}).")
        return nfft

    def _second_stage(self, ccf_ik: Array, ccf_jk: Array, nfft2: int, *, mode_k: str) -> Array:
        Fik = np.fft.fft(ccf_ik, n=nfft2)
        Fjk = np.fft.fft(ccf_jk, n=nfft2)

        if mode_k == "correlation":
            F = Fik * np.conj(Fjk)
        elif mode_k == "convolution":
            F = Fik * Fjk
        else:
            raise ValueError(f"Unknown mode_k={mode_k}")

        ncf = np.real(np.fft.ifft(F))
        return np.fft.fftshift(ncf)

    def _lags_for_len(self, n: int) -> Array:
        return (np.arange(n) - (n // 2)) / self.sr

    def _crop_indices(self, n: int, max_lag2: float) -> Tuple[int, int]:
        half = int(round(max_lag2 * self.sr))
        center = n // 2
        start = max(0, center - half)
        end = min(n, center + half + 1)
        return start, end