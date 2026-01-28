# seismocorr/preprocessing/das_process.py

"""
DAS 预处理包：坏道处理 + 共模噪声去除 + 中值滤波 + fk滤波

支持的方法（name）：
  共模噪声去除：
    - "cm_median"  : 每时刻跨道中位数并相减（稳健，共模噪声）
    - "cm_mean"    : 每时刻跨道均值并相减
    - "cm_pca"     : PCA/SVD 低秩去除（去前 k 个主成分，激进）
    - "cm_fk"      : f-k 域去近零波数（|k| < k0），抑制空间慢变共模

  滤波：
    - "median": 时-空二维中值滤波（SciPy 实现，支持 NaN 忽略）
    - "fk_fan"     : f-k 扇形滤波（频带 + 表观速度 + 方向选择）
  坏道处理：
    - "bad_robust" : 鲁棒统计（score + robust z）检测坏道，并插值/置零/不修补

说明：
  - 输入 data 是 2D： (n_channels, n_samples) 或 (n_samples, n_channels)
    由 channel_axis 指定通道维（0/1）。
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple, Union
from scipy.ndimage import generic_filter, median_filter

ArrayLike = Union[np.ndarray, list]

# =============================================================================
# helpers
# =============================================================================
RepairMethod = Literal["interp_linear", "interp_median", "zero", "none"]
ScoreMethod = Literal["rms", "mad", "std", "kurtosis"]


def _as_2d(data: np.ndarray, channel_axis: int) -> Tuple[np.ndarray, bool]:
    """Return view as (n_channels, n_samples) and whether transposed."""
    x = np.asarray(data)
    if x.ndim != 2:
        raise ValueError(f"data must be 2D; got shape {x.shape}")
    if channel_axis not in (0, 1):
        raise ValueError("channel_axis must be 0 or 1")
    if channel_axis == 1:
        return x.T, True
    return x, False


def _robust_zscore(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """robust z = (v - median(v)) / (1.4826*MAD)"""
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return (v - med) / (1.4826 * mad + eps)


def _cosine_taper_1d(n: int, frac: float) -> np.ndarray:
    """简单余弦 taper（两端各 frac*n 点渐变），frac∈[0,0.5)"""
    frac = float(frac)
    if frac <= 0:
        return np.ones(n, dtype=np.float64)
    frac = min(frac, 0.499)
    m = int(np.floor(frac * n))
    if m < 1:
        return np.ones(n, dtype=np.float64)

    w = np.ones(n, dtype=np.float64)
    x = np.linspace(0, np.pi / 2, m, endpoint=False)
    ramp = np.sin(x) ** 2  # 0->1
    w[:m] = ramp
    w[-m:] = ramp[::-1]
    return w


def _fk_filter_das_fan(
    das_ct: np.ndarray,
    dt: float,
    dx: float,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mode: Literal["pass", "reject"] = "pass",
    direction: Literal["both", "pos_k", "neg_k"] = "both",
    taper_time: float = 0.05,
    taper_space: float = 0.05,
    pad_t: int = 0,
    pad_x: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    对 DAS (channels, time) 做 f-k 滤波（速度扇形 fan filter + 频带 + 方向）。
    频率 f 轴单位 Hz；波数 k 轴单位 cycles/m（np.fft.fftfreq 定义）。
    表观速度 v_app = |f| / |k| （m/s），k≈0 用 eps 保护。
    """
    if das_ct.ndim != 2:
        raise ValueError("das_ct 必须是 2D，形状 (channels, time)")
    if dt <= 0 or dx <= 0:
        raise ValueError("dt 和 dx 必须为正数")

    x = np.asarray(das_ct, dtype=np.float64)
    nx, nt = x.shape

    # taper 减少泄漏
    wt = _cosine_taper_1d(nt, taper_time)
    wx = _cosine_taper_1d(nx, taper_space)
    xw = (x * wx[:, None]) * wt[None, :]

    # padding
    pad_t = int(pad_t)
    pad_x = int(pad_x)
    nxp = nx + 2 * pad_x
    ntp = nt + 2 * pad_t
    if pad_x > 0 or pad_t > 0:
        xp = np.zeros((nxp, ntp), dtype=np.float64)
        xp[pad_x:pad_x + nx, pad_t:pad_t + nt] = xw
    else:
        xp = xw

    # FFT2 -> (k, f)
    X = np.fft.fft2(xp)
    Xs = np.fft.fftshift(X, axes=(0, 1))

    f = np.fft.fftshift(np.fft.fftfreq(ntp, d=dt))  # Hz
    k = np.fft.fftshift(np.fft.fftfreq(nxp, d=dx))  # cycles/m
    K, F = np.meshgrid(k, f, indexing="ij")         # (nxp, ntp)

    mask = np.ones((nxp, ntp), dtype=bool)

    if fmin is not None:
        mask &= (np.abs(F) >= float(fmin))
    if fmax is not None:
        mask &= (np.abs(F) <= float(fmax))

    if direction == "pos_k":
        mask &= (K > 0)
    elif direction == "neg_k":
        mask &= (K < 0)

    if (vmin is not None) or (vmax is not None):
        vapp = np.abs(F) / (np.abs(K) + eps)  # m/s
        if vmin is not None:
            mask &= (vapp >= float(vmin))
        if vmax is not None:
            mask &= (vapp <= float(vmax))

    Y = Xs * mask if mode == "pass" else Xs * (~mask)

    Y = np.fft.ifftshift(Y, axes=(0, 1))
    y = np.fft.ifft2(Y).real

    if pad_x > 0 or pad_t > 0:
        y = y[pad_x:pad_x + nx, pad_t:pad_t + nt]

    return y


def _nanmed(vec: np.ndarray) -> float:
    return float(np.nanmedian(vec))


def _channel_score(x: np.ndarray, method: ScoreMethod) -> np.ndarray:
    """score per channel; x: (n_ch, n_t) -> (n_ch,)"""
    if method == "rms":
        return np.sqrt(np.mean(x * x, axis=1))
    if method == "std":
        return np.std(x, axis=1)
    if method == "mad":
        med = np.median(x, axis=1, keepdims=True)
        return np.median(np.abs(x - med), axis=1)
    if method == "kurtosis":
        mu = np.mean(x, axis=1, keepdims=True)
        y = x - mu
        v = np.mean(y * y, axis=1) + 1e-12
        m4 = np.mean(y ** 4, axis=1)
        return m4 / (v * v) - 3.0
    raise ValueError("Unknown score method")


def _repair_bad_channels(x: np.ndarray, bad: np.ndarray, method: RepairMethod) -> np.ndarray:
    """Repair along channel axis."""
    if method == "none":
        return x.copy()

    y = x.copy()
    n_ch, _n_t = y.shape

    if method == "zero":
        y[bad, :] = 0.0
        return y

    good_idx = np.where(~bad)[0]
    bad_idx = np.where(bad)[0]
    if good_idx.size == 0:
        y[bad, :] = 0.0
        return y
    if bad_idx.size == 0:
        return y

    for i in bad_idx:
        left = good_idx[good_idx < i]
        right = good_idx[good_idx > i]
        il = left[-1] if left.size else None
        ir = right[0] if right.size else None

        if il is None and ir is None:
            y[i, :] = 0.0
            continue
        if il is None:
            y[i, :] = y[ir, :]
            continue
        if ir is None:
            y[i, :] = y[il, :]
            continue

        if method == "interp_linear":
            w = (i - il) / (ir - il)
            y[i, :] = (1.0 - w) * y[il, :] + w * y[ir, :]
        elif method == "interp_median":
            y[i, :] = np.median(np.stack([y[il, :], y[ir, :]], axis=0), axis=0)
        else:
            raise ValueError("Unknown repair method")

    return y


# =============================================================================
# basic class
# =============================================================================
class DASPreprocessor(ABC):
    """
    DAS 预处理基类（只返回处理后数组；诊断信息写到 self.info）

    子类实现 apply()：
      - 输入：np.ndarray（2D）
      - 输出：np.ndarray（与输入同形状）
    """

    def __init__(self, channel_axis: int = 0, dtype: Optional[np.dtype] = None):
        self.channel_axis = channel_axis
        self.dtype = dtype
        self.info: Dict[str, Any] = {}  # 每次 apply 后更新

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: ArrayLike) -> np.ndarray:
        y = self.apply(np.asarray(x))
        return y


# =============================================================================
# Common-mode removers
# =============================================================================
class CMMedian(DASPreprocessor):
    """共模：每时刻跨道中位数 cm(t)，所有通道减去 cm(t)"""

    def __init__(self, channel_axis: int = 0, dtype: Optional[np.dtype] = None, return_cm: bool = False):
        super().__init__(channel_axis=channel_axis, dtype=dtype)
        self.return_cm = return_cm

    def apply(self, x: np.ndarray) -> np.ndarray:
        X, transposed = _as_2d(x, self.channel_axis)
        if self.dtype is not None:
            X = X.astype(self.dtype, copy=False)
        if not np.isfinite(X).all():
            raise ValueError("data contains nan/inf; clean it before calling.")

        cm = np.median(X, axis=0)
        Y = X - cm[None, :]

        self.info = {"method": "cm_median"}
        if self.return_cm:
            self.info["cm"] = cm

        return Y.T if transposed else Y


class CMMean(DASPreprocessor):
    """共模：每时刻跨道均值 cm(t)，所有通道减去 cm(t)"""

    def __init__(self, channel_axis: int = 0, dtype: Optional[np.dtype] = None, return_cm: bool = False):
        super().__init__(channel_axis=channel_axis, dtype=dtype)
        self.return_cm = return_cm

    def apply(self, x: np.ndarray) -> np.ndarray:
        X, transposed = _as_2d(x, self.channel_axis)
        if self.dtype is not None:
            X = X.astype(self.dtype, copy=False)
        if not np.isfinite(X).all():
            raise ValueError("data contains nan/inf; clean it before calling.")

        cm = np.mean(X, axis=0)
        Y = X - cm[None, :]

        self.info = {"method": "cm_mean"}
        if self.return_cm:
            self.info["cm"] = cm

        return Y.T if transposed else Y


class CMPCA(DASPreprocessor):
    """
    共模：PCA/SVD 低秩去除（更激进，可能削弱真实相干波场）

    参数：
      n_components: 去掉前 k 个主成分
      center: "median"|"mean"|"none"（按时间点跨道减去中心轨迹）
    """

    def __init__(
        self,
        n_components: int = 1,
        channel_axis: int = 0,
        center: Literal["median", "mean", "none"] = "median",
        dtype: Optional[np.dtype] = None,
        return_info: bool = True,
    ):
        super().__init__(channel_axis=channel_axis, dtype=dtype)

        self.n_components = int(n_components)
        self.center = center
        self.return_info = return_info
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if center not in ("median", "mean", "none"):
            raise ValueError("center must be 'median', 'mean', or 'none'")
    def apply(self, x: np.ndarray) -> np.ndarray:
        X, transposed = _as_2d(x, self.channel_axis)
        if self.dtype is not None:
            X = X.astype(self.dtype, copy=False)
        X = X.astype(float, copy=False)
        if not np.isfinite(X).all():
            raise ValueError("data contains nan/inf; clean it before calling.")

        if self.center == "median":
            cm0 = np.median(X, axis=0)
            X0 = X - cm0[None, :]
        elif self.center == "mean":
            cm0 = np.mean(X, axis=0)
            X0 = X - cm0[None, :]
        elif self.center == "none":
            cm0 = None
            X0 = X
        else:
            raise ValueError("center must be 'median', 'mean', or 'none'")

        U, S, Vt = np.linalg.svd(X0, full_matrices=False)
        k = min(self.n_components, S.size)

        low_rank = (U[:, :k] * S[:k][None, :]) @ Vt[:k, :]
        clean = X0 - low_rank

        self.info = {"method": "cm_pca"}
        if self.return_info:
            total = float((S * S).sum() + 1e-12)
            removed = float((S[:k] * S[:k]).sum())
            self.info["info"] = {
                "rank_removed": int(k),
                "explained_energy_fraction": removed / total,
                "singular_values": S,
                "center": self.center,
                "initial_center_trace": cm0,
            }

        return clean.T if transposed else clean


class CMFK(DASPreprocessor):
    """
    共模：f-k 域去近零波数（|k| < k0）

    参数：
      dx: 通道间距（m）
      k0: 截止波数（rad/m）
      taper: 过渡带宽（rad/m），0 表示硬切
    """

    def __init__(
        self,
        dx: float,
        k0: float,
        channel_axis: int = 0,
        taper: float = 0.0,
        dtype: Optional[np.dtype] = None,
        return_info: bool = True,
    ):
        super().__init__(channel_axis=channel_axis, dtype=dtype)
        dx = float(dx)
        k0 = float(k0)
        taper = float(taper)
        if dx <= 0:
            raise ValueError("dx must be > 0")
        if k0 <= 0:
            raise ValueError("k0 must be > 0")
        if taper < 0:
            raise ValueError("taper must be >= 0")
        self.dx = dx
        self.k0 = k0
        self.taper = taper
        self.return_info = return_info

    def apply(self, x: np.ndarray) -> np.ndarray:
        X, transposed = _as_2d(x, self.channel_axis)
        if self.dtype is not None:
            X = X.astype(self.dtype, copy=False)
        X = X.astype(float, copy=False)
        if not np.isfinite(X).all():
            raise ValueError("data contains nan/inf; clean it before calling.")

        n_ch, _n_t = X.shape
        F = np.fft.fft2(X)

        k_cycles = np.fft.fftfreq(n_ch, d=self.dx)  # cycles/m
        k = 2.0 * np.pi * k_cycles                 # rad/m
        abs_k = np.abs(k)

        mask = np.ones((n_ch, 1), dtype=float)
        if self.taper == 0.0:
            mask[abs_k < self.k0, 0] = 0.0
        else:
            mask[abs_k < self.k0, 0] = 0.0
            mid = (abs_k >= self.k0) & (abs_k <= (self.k0 + self.taper))
            mask[mid, 0] = 0.5 * (1 - np.cos(np.pi * (abs_k[mid] - self.k0) / self.taper))
            mask[abs_k > (self.k0 + self.taper), 0] = 1.0

        Y = np.fft.ifft2(F * mask).real

        self.info = {"method": "cm_fk"}
        if self.return_info:
            self.info["info"] = {"dx": self.dx, "k0": self.k0, "taper": self.taper}

        return Y.T if transposed else Y


# =============================================================================
# fk滤波
# =============================================================================
class FKFanFilter(DASPreprocessor):
    """
    f-k 扇形滤波（频带 + 表观速度扇形 + 传播方向选择）

    用于在 f-k 域中按「频率 + 表观速度（v = |f| / |k|）」
    保留或抑制能量，常用于：
      - 去慢速滚动噪声
      - 提取特定速度范围的波场
      - 分离传播方向（正/负 k）

    Args：
      dt : float
        时间采样间隔（秒）

      dx : float
        道间距（米）

      channel_axis : int, default=0
        通道所在维度：
          - 0 : (channels, time)
          - 1 : (time, channels)

      dtype : np.dtype | None
        内部/输出使用的数据类型；None 表示保持输入 dtype

      fmin, fmax : float | None
        频率范围（Hz）：
          - None 表示不限制
          - 实际使用 |f| ∈ [fmin, fmax]

      vmin, vmax : float | None
        表观速度范围（m/s）：
          v_app = |f| / |k|
        常用于定义扇形滤波器（fan filter）

      mode : {"pass", "reject"}, default="pass"
        - "pass"   : 保留掩膜内能量
        - "reject" : 抑制掩膜内能量

      direction : {"both", "pos_k", "neg_k"}, default="both"
        传播方向选择：
          - "both"  : 保留 ±k
          - "pos_k" : 仅保留 k > 0（单向传播）
          - "neg_k" : 仅保留 k < 0

      taper_time : float, default=0.05
        时间维 taper 比例（0~0.5），用于减少频谱泄漏

      taper_space : float, default=0.05
        空间维 taper 比例（0~0.5）

      pad_t : int, default=0
        时间维补零点数（两侧各 pad_t）

      pad_x : int, default=0
        空间维补零点数（两侧各 pad_x）

      eps : float, default=1e-12
        防止 k≈0 时除零的稳定项

    Returns：
      - 与输入形状相同的 2D 数组
      - 滤波后的时空域 DAS 数据

    备注：
      - 内部使用 float64 进行 FFT 计算
      - 若数据含 NaN/Inf，请先进行坏道处理
    """

    def __init__(
        self,
        dt: float,
        dx: float,
        channel_axis: int = 0,
        dtype: Optional[np.dtype] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        mode: Literal["pass", "reject"] = "pass",
        direction: Literal["both", "pos_k", "neg_k"] = "both",
        taper_time: float = 0.05,
        taper_space: float = 0.05,
        pad_t: int = 0,
        pad_x: int = 0,
        eps: float = 1e-12,
        return_info: bool = True,
    ):
        super().__init__(channel_axis=channel_axis, dtype=dtype)
        self.dt = float(dt)
        self.dx = float(dx)
        if self.dt <= 0 or self.dx <= 0:
            raise ValueError("dt/dx must be > 0")

        self.fmin = fmin
        self.fmax = fmax
        self.vmin = vmin
        self.vmax = vmax
        self.mode = mode
        self.direction = direction
        self.taper_time = float(taper_time)
        self.taper_space = float(taper_space)
        self.pad_t = int(pad_t)
        self.pad_x = int(pad_x)
        self.eps = float(eps)
        self.return_info = return_info
        self.mode = mode
        self.direction = direction

        if self.pad_t < 0 or self.pad_x < 0:
            raise ValueError("pad_t/pad_x 必须 >= 0")
        if self.eps <= 0:
            raise ValueError("eps 必须 > 0")
        if isinstance(pad_t, bool) or isinstance(pad_x, bool) or isinstance(eps, bool):
            raise TypeError("pad_t/pad_x/eps 类型有误")

        if self.mode not in ("pass", "reject"):
            raise ValueError('mode 只能是 "pass" 或 "reject"')
        if self.direction not in ("both", "pos_k", "neg_k"):
            raise ValueError('direction 只能是 "both" / "pos_k" / "neg_k"')

    def apply(self, x: np.ndarray) -> np.ndarray:
        X, transposed = _as_2d(x, self.channel_axis)

        # dtype 处理：内部用 float64，最后再 cast 回输入 dtype（或用户指定 dtype）
        x_in_dtype = X.dtype
        if self.dtype is not None:
            X = X.astype(self.dtype, copy=False)

        if not np.isfinite(X).all():
            raise ValueError("data contains nan/inf; clean it before calling.")

        Y = _fk_filter_das_fan(
            das_ct=X,
            dt=self.dt,
            dx=self.dx,
            fmin=self.fmin,
            fmax=self.fmax,
            vmin=self.vmin,
            vmax=self.vmax,
            mode=self.mode,
            direction=self.direction,
            taper_time=self.taper_time,
            taper_space=self.taper_space,
            pad_t=self.pad_t,
            pad_x=self.pad_x,
            eps=self.eps,
        )

        out_dtype = x_in_dtype if self.dtype is None else self.dtype
        Y = Y.astype(out_dtype, copy=False)

        self.info = {"method": "fk_fan"}
        if self.return_info:
            self.info["info"] = {
                "dt": self.dt,
                "dx": self.dx,
                "fmin": self.fmin,
                "fmax": self.fmax,
                "vmin": self.vmin,
                "vmax": self.vmax,
                "mode": self.mode,
                "direction": self.direction,
                "taper_time": self.taper_time,
                "taper_space": self.taper_space,
                "pad_t": self.pad_t,
                "pad_x": self.pad_x,
            }

        return Y.T if transposed else Y


# =============================================================================
# 中值滤波
# =============================================================================
class MedianFilterSciPy(DASPreprocessor):
    """
    DAS 二维中值滤波（SciPy 实现）

    沿「通道 × 时间」二维窗口做中值滤波，
    用于：
      - 抑制脉冲噪声
      - 去除孤立坏点
      - 平滑高频随机噪声（比均值滤波更稳健）

    Args：
      k_time : int, default=5
        时间维窗口长度（必须为正奇数）
        k_time=1 表示不沿时间滤波

      k_chan : int, default=3
        空间/通道维窗口长度（必须为正奇数）
        k_chan=1 表示不沿空间滤波

      channel_axis : int, default=0
        通道所在维度：
          - 0 : (channels, time)
          - 1 : (time, channels)

      dtype : np.dtype | None
        输出数据类型；None 表示保持输入 dtype

      mode : str, default="reflect"
        边界处理方式（传给 SciPy）：
          - "reflect", "nearest", "constant" 等

      nan_policy : {"propagate", "omit"}, default="propagate"
        NaN 处理策略：
          - "propagate" : 窗口内有 NaN 则输出 NaN
          - "omit"      : 忽略 NaN 计算中值（适合 DAS 掉点）

    Returns：
      - 与输入形状相同的 2D 数组

    """

    def __init__(
        self,
        k_time: int = 5,
        k_chan: int = 3,
        channel_axis: int = 0,
        dtype: Optional[np.dtype] = None,
        mode: str = "reflect",
        nan_policy: Literal["propagate", "omit"] = "propagate",
        return_info: bool = True,
    ):
        super().__init__(channel_axis=channel_axis, dtype=dtype)
        if k_time % 2 == 0 or k_chan % 2 == 0 or k_time < 1 or k_chan < 1:
            raise ValueError("k_time/k_chan 必须是正奇数（如 3/5/7）")
        if nan_policy not in ("propagate", "omit"):
            raise ValueError('nan_policy 只能是 "propagate" 或 "omit"')

        self.k_time = int(k_time)
        self.k_chan = int(k_chan)
        self.mode = mode
        self.nan_policy = nan_policy
        self.return_info = return_info

    def apply(self, x: np.ndarray) -> np.ndarray:
        if median_filter is None:
            raise ImportError("scipy is required for MedianFilterSciPy (pip install scipy).")

        X, transposed = _as_2d(x, self.channel_axis)

        x_in_dtype = X.dtype
        if self.dtype is not None:
            X = X.astype(self.dtype, copy=False)

        # 注意：这里不强制拒绝 NaN/Inf，因为用户可能就是想用 omit 模式处理 NaN
        if self.nan_policy == "omit":
            if generic_filter is None:
                raise ImportError("scipy is required for nan-aware median filter (pip install scipy).")
            Y = generic_filter(X, function=_nanmed, size=(self.k_chan, self.k_time), mode=self.mode)
        else:
            Y = median_filter(X, size=(self.k_chan, self.k_time), mode=self.mode)

        out_dtype = x_in_dtype if self.dtype is None else self.dtype
        Y = Y.astype(out_dtype, copy=False)

        self.info = {"method": "median_scipy"}
        if self.return_info:
            self.info["info"] = {
                "k_time": self.k_time,
                "k_chan": self.k_chan,
                "mode": self.mode,
                "nan_policy": self.nan_policy,
            }

        return Y.T if transposed else Y
    

# =============================================================================
# Bad-channel processing
# =============================================================================
class BadRobust(DASPreprocessor):
    """坏道检测 + 修补（鲁棒统计）。

    对 DAS 数据按“道/通道”进行坏道检测，并按指定策略修补。内部会把输入统一为
    2D 视图 `(n_channels, n_samples)` 处理；若 `channel_axis=1`（输入为
    `(time, channels)`），则会内部转置，输出再转置回原布局。
    
    检测规则：
        1) 任意包含 NaN/Inf 的道 -> bad
        2) 计算每道 score（由 `score_method` 决定）
        3) 计算 score 的鲁棒 z-score：`abs(z) > z_thresh` -> bad
        4) 可选：饱和比例检测：`sat_frac > sat_frac_thresh` -> bad
    Args：
        channel_axis (int, default=0):
            通道所在维度：
              - 0：输入形状为 `(channels, time)`
              - 1：输入形状为 `(time, channels)`

        dtype (np.dtype | None, default=None):
            可选的内部处理 dtype。若不为 None，会先 cast 到该 dtype，
            随后为了鲁棒统计计算会再转为 float。

        score_method (ScoreMethod, default="rms"):
            每道指标（score）的计算方式，由 `_channel_score` 支持，例如 "rms"。

        z_thresh (float, default=6.0):
            鲁棒 z-score 阈值。满足 `abs(robust_z) > z_thresh` 的道判为坏道。

        repair (RepairMethod, default="interp_linear"):
            修补策略：
              - "interp_linear"：用最近两条好道线性插值修补
              - "interp_median"：用最近两条好道做稳健/中值插值修补（取决于实现）
              - "zero"：坏道置 0
              - "none"：不修补，仅做 finite-clean（NaN/Inf -> 0），并输出 bad_mask

        sat_value (float | None, default=None):
            可选饱和判定幅值阈值。需要与 `sat_frac_thresh` 同时提供才生效。
            饱和判定为：`abs(x) >= sat_value`。

        sat_frac_thresh (float | None, default=None):
            可选饱和比例阈值。需要与 `sat_value` 同时提供才生效。
            若某道饱和比例 `sat_frac > sat_frac_thresh`，该道判为坏道。

    Returns：
        np.ndarray:
            修补后的数组，形状与输入 `x` 相同。
            当前实现会将数据转为 float 进行处理，输出也为 float。

    """

    def __init__(
        self,
        channel_axis: int = 0,
        dtype: Optional[np.dtype] = None,
        score_method: ScoreMethod = "rms",
        z_thresh: float = 6.0,
        repair: RepairMethod = "interp_linear",
        sat_value: Optional[float] = None,
        sat_frac_thresh: Optional[float] = None,
    ):
        super().__init__(channel_axis=channel_axis, dtype=dtype)
        self.score_method = score_method
        self.z_thresh = float(z_thresh)
        self.repair = repair
        self.sat_value = sat_value
        self.sat_frac_thresh = sat_frac_thresh

    def apply(self, x: np.ndarray) -> np.ndarray:
        X, transposed = _as_2d(x, self.channel_axis)
        if self.dtype is not None:
            X = X.astype(self.dtype, copy=False)
        X = X.astype(float, copy=False)

        finite = np.isfinite(X)
        bad_nan = ~finite.all(axis=1)

        X2 = X.copy()
        X2[~finite] = 0.0

        score = _channel_score(X2, self.score_method)
        rz = _robust_zscore(score)
        bad_score = np.abs(rz) > self.z_thresh

        bad_sat = np.zeros(X2.shape[0], dtype=bool)
        sat_frac = None
        if self.sat_value is not None and self.sat_frac_thresh is not None:
            sat = np.abs(X2) >= float(self.sat_value)
            sat_frac = np.mean(sat, axis=1)
            bad_sat = sat_frac > float(self.sat_frac_thresh)

        bad = bad_nan | bad_score | bad_sat
        Y = _repair_bad_channels(X2, bad, self.repair)

        self.info = {
            "method": "bad_robust",
            "bad_mask": bad,
            "score": score,
            "robust_z": rz,
            "score_method": self.score_method,
            "z_thresh": self.z_thresh,
            "repair": self.repair,
        }
        if sat_frac is not None:
            self.info["sat_value"] = float(self.sat_value)
            self.info["sat_frac"] = sat_frac
            self.info["sat_frac_thresh"] = float(self.sat_frac_thresh)

        return Y.T if transposed else Y


# =============================================================================
# Factory
# =============================================================================
_PREPROCESSOR_MAP = {
    # common-mode
    "cm_median": CMMedian,
    "cm_mean": CMMean,
    "cm_pca": CMPCA,
    "cm_fk": CMFK,

    # new: fk fan filter
    "fk_fan": FKFanFilter,

    # bad-channel
    "bad_robust": BadRobust,

    # new: median filter
    "median": MedianFilterSciPy,
}


def get_das_preprocessor(name: str, **kwargs) -> DASPreprocessor:
    """
    工厂函数：创建 DAS 预处理器实例
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be str, got {type(name).__name__}: {name!r}")
    name = name.strip().lower()
    if not name:
        raise ValueError("name cannot be empty string")

    cls = _PREPROCESSOR_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown DAS preprocessor: '{name}'. Choose from {list(_PREPROCESSOR_MAP.keys())}"
        )
    return cls(**kwargs)
