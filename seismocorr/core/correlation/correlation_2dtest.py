# seismocorr/core/correlation_2d.py

"""
2D Cross-Correlation Core Module

提供灵活高效的二维互相关计算接口，支持：
- 时域 / 频域算法选择
- 多种归一化与滤波预处理（二维矩阵版本）
- 多道批量输入（二维数组形式）
- 返回标准 CCF 结构（lags, ccf_matrix）

不包含文件 I/O 或任务调度 —— 这些由 pipeline 层管理。
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.fftpack import fft, ifft, next_fast_len
from scipy.signal import correlate

# 二维预处理模块
from seismocorr.preprocessing.normal_func2d import bandpass as bandpass_2d
from seismocorr.preprocessing.normal_func2d import demean as demean_2d
from seismocorr.preprocessing.normal_func2d import detrend as detrend_2d
from seismocorr.preprocessing.normal_func2d import taper as taper_2d
from seismocorr.preprocessing.time_norm2d import get_time_normalizer_2d
from seismocorr.config.default import SUPPORTED_METHODS, NORMALIZATION_OPTIONS

# 优化库导入
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(func=None, **kwargs):
        """Numba不可用时的回退装饰器"""
        if func is None:
            return lambda f: f
        return func

    def prange(n):
        """Numba不可用时的回退prange"""
        return range(n)


# -----------------------------# 类型定义
# -----------------------------
ArrayLike = Union[np.ndarray, List[float]]
LagsAndCCF2D = Tuple[
    np.ndarray, np.ndarray
]  # (lags, ccf_matrix)，ccf_matrix形状为(n_signals, 2*max_lag_samples+1)


# -----------------------------# 主要函数：compute_cross_correlation_2d
# -----------------------------


def compute_cross_correlation_2d(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    sampling_rate: float,
    method: str = "time-domain",
    time_normalize: str = "one-bit",
    freq_normalize: str = "no",
    freq_band: Optional[Tuple[float, float]] = None,
    max_lag: Optional[Union[float, int]] = None,
    nfft: Optional[int] = None,
    time_norm_kwargs: Optional[Dict[str, Any]] = None,
    freq_norm_kwargs: Optional[Dict[str, Any]] = None,
) -> LagsAndCCF2D:
    """
    计算二维数组中对应的时间序列对的互相关函数（CCF）

    Args:
        x_matrix: 输入信号矩阵A，形状为 (n_signals, n_samples)
        y_matrix: 输入信号矩阵B，形状为 (n_signals, n_samples)
        sampling_rate: 采样率 (Hz)
        method: 计算方法 ('time-domain', 'freq-domain', 'deconv', 'coherency')
        time_normalize: 时域归一化方法
        freq_normalize: 频域归一化方法
        freq_band: 带通滤波范围 (fmin, fmax)，单位 Hz
        max_lag: 最大滞后时间（秒）；若为 None，则使用 min(len(x), len(y))
        nfft: FFT 长度，自动补零到 next_fast_len

    Returns:
        lags: 时间滞后数组 (单位：秒)，形状为 (2*max_lag_samples+1,)
        ccf_matrix: 互相关函数值矩阵，形状为 (n_signals, 2*max_lag_samples+1)
    """

    if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, (int, float)):
        raise TypeError(
            f"sampling_rate 必须是数字(Hz)，当前为 {type(sampling_rate).__name__}: {sampling_rate!r}"
        )
    if not np.isfinite(float(sampling_rate)) or float(sampling_rate) <= 0:
        raise ValueError(f"sampling_rate 必须为有限且 > 0 的数(Hz)，当前为: {sampling_rate!r}")
    sampling_rate = float(sampling_rate)

    if not isinstance(method, str):
        raise TypeError(f"method 必须是 str，当前为 {type(method).__name__}: {method!r}")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"不支持的计算方法: {method!r}。请从 {SUPPORTED_METHODS} 中选择")

    if not isinstance(time_normalize, str):
        raise TypeError(
            f"time_normalize 必须是 str，当前为 {type(time_normalize).__name__}: {time_normalize!r}"
        )
    if time_normalize not in NORMALIZATION_OPTIONS:
        raise ValueError(
            f"time_normalize 必须是 {NORMALIZATION_OPTIONS} 之一，当前为: {time_normalize!r}"
        )
    if not isinstance(freq_normalize, str):
        raise TypeError(
            f"freq_normalize 必须是 str，当前为 {type(freq_normalize).__name__}: {freq_normalize!r}"
        )
    if freq_normalize not in NORMALIZATION_OPTIONS:
        raise ValueError(
            f"freq_normalize 必须是 {NORMALIZATION_OPTIONS} 之一，当前为: {freq_normalize!r}"
        )
    if freq_norm_kwargs not in (None,) and not isinstance(freq_norm_kwargs, dict):
        raise TypeError(
            f"freq_norm_kwargs 必须是 dict 或 None，当前为 {type(freq_norm_kwargs).__name__}"
        )

    if time_norm_kwargs not in (None,) and not isinstance(time_norm_kwargs, dict):
        raise TypeError(
            f"time_norm_kwargs 必须是 dict 或 None，当前为 {type(time_norm_kwargs).__name__}"
        )

    if freq_band is not None:
        if not (isinstance(freq_band, (tuple, list)) and len(freq_band) == 2):
            raise TypeError(f"freq_band 必须是 (fmin, fmax) 或 None，当前为: {freq_band!r}")
        fmin, fmax = freq_band
        fmin = float(fmin)
        fmax = float(fmax)
        if (not np.isfinite(fmin)) or (not np.isfinite(fmax)):
            raise ValueError(f"freq_band 必须是有限数值，当前为: {freq_band!r}")
        if fmin <= 0 or fmax <= 0:
            raise ValueError(f"freq_band 频率必须 > 0，当前为: {freq_band!r}")
        if fmin >= fmax:
            raise ValueError(f"freq_band 要求 fmin < fmax，当前为: {freq_band!r}")
        nyquist = sampling_rate / 2.0
        if fmax >= nyquist:
            raise ValueError(
                f"freq_band 的 fmax 必须 < Nyquist({nyquist:g}Hz)，当前为: {fmax!r}"
            )

    if max_lag is not None:
        if isinstance(max_lag, bool) or not isinstance(max_lag, (int, float)):
            raise TypeError(
                f"max_lag 必须是 float/int 或 None，当前为 {type(max_lag).__name__}: {max_lag!r}"
            )
        max_lag = float(max_lag)
        if not np.isfinite(max_lag) or max_lag < 0:
            raise ValueError(f"max_lag 必须为有限且 >= 0 的秒数，当前为: {max_lag!r}")
    if nfft is not None:
        if isinstance(nfft, bool) or not isinstance(nfft, int):
            raise TypeError(f"nfft 必须是 int 或 None，当前为 {type(nfft).__name__}: {nfft!r}")
        if nfft <= 0:
            raise ValueError(f"nfft 必须是正整数，当前为: {nfft!r}")

    if not isinstance(x_matrix, np.ndarray):
        raise TypeError(f"x_matrix 必须是 np.ndarray，当前为 {type(x_matrix).__name__}")
    if not isinstance(y_matrix, np.ndarray):
        raise TypeError(f"y_matrix 必须是 np.ndarray，当前为 {type(y_matrix).__name__}")

    if x_matrix.ndim != 2:
        raise ValueError(f"x_matrix 必须是二维数组 (n_signals, n_samples)，当前 ndim={x_matrix.ndim}")
    if y_matrix.ndim != 2:
        raise ValueError(f"y_matrix 必须是二维数组 (n_signals, n_samples)，当前 ndim={y_matrix.ndim}")

    if not np.issubdtype(x_matrix.dtype, np.number):
        raise TypeError(f"x_matrix dtype 必须是数值类型，当前为 {x_matrix.dtype}")
    if not np.issubdtype(y_matrix.dtype, np.number):
        raise TypeError(f"y_matrix dtype 必须是数值类型，当前为 {y_matrix.dtype}")

    if not np.all(np.isfinite(x_matrix)):
        raise ValueError("输入 x_matrix 包含 NaN/Inf")
    if not np.all(np.isfinite(y_matrix)):
        raise ValueError("输入 y_matrix 包含 NaN/Inf")

    # 验证输入矩阵形状
    if x_matrix.shape != y_matrix.shape:
        raise ValueError(
            f"输入矩阵形状不匹配: x_matrix.shape={x_matrix.shape}, y_matrix.shape={y_matrix.shape}"
        )

    n_signals, n_samples = x_matrix.shape

    if n_signals == 0 or n_samples == 0:
        return (
            np.array([], dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
        )

    # 确定最大滞后
    if max_lag is None:
        max_lag = n_samples / sampling_rate

    max_lag_samples = int(max_lag * sampling_rate)

    # max_lag 上限不能超过 (n_samples - 1)
    if max_lag_samples > (n_samples - 1):
        raise ValueError(
            f"max_lag 太大：对应样本数 {max_lag_samples} > n_samples-1({n_samples-1})。"
            f"请调小 max_lag 或增大窗口长度。"
        )

    # 初始化参数字典
    time_norm_kwargs = time_norm_kwargs or {}
    freq_norm_kwargs = freq_norm_kwargs or {}

    # 对x_matrix和y_matrix进行二维预处理
    # 1. 去趋势
    x_processed = detrend_2d(x_matrix, type="linear")
    y_processed = detrend_2d(y_matrix, type="linear")

    # 2. 去均值
    x_processed = demean_2d(x_processed)
    y_processed = demean_2d(y_processed)

    # 3. 加窗
    x_processed = taper_2d(x_processed, width=0.05)
    y_processed = taper_2d(y_processed, width=0.05)

    # 4. 滤波（如果需要）
    if freq_band is not None:
        x_processed = bandpass_2d(
            x_processed,
            freq_band[0],
            freq_band[1],
            sr=sampling_rate,
            order=4,
            zero_phase=True,
        )
        y_processed = bandpass_2d(
            y_processed,
            freq_band[0],
            freq_band[1],
            sr=sampling_rate,
            order=4,
            zero_phase=True,
        )

    # 5. 时域归一化（二维版本）
    time_norm_kwargs_with_fs = {
        **time_norm_kwargs,
        "Fs": sampling_rate,
        "npts": n_samples,
    }
    normalizer = get_time_normalizer_2d(time_normalize, **time_norm_kwargs_with_fs)
    x_processed = normalizer.apply(x_processed)
    y_processed = normalizer.apply(y_processed)

    # 选择方法
    if method == "time-domain":
        lags, ccf_matrix = _xcorr_time_domain_2d(
            x_processed, y_processed, sampling_rate, max_lag
        )
    elif method in ["freq-domain", "deconv"]:
        lags, ccf_matrix = _xcorr_freq_domain_2d(
            x_processed,
            y_processed,
            sampling_rate,
            max_lag,
            nfft,
            deconv=method == "deconv",
        )
    elif method == "coherency":
        lags, ccf_matrix = _coherency_2d(
            x_processed, y_processed, sampling_rate, max_lag, nfft
        )
    else:
        raise ValueError(
            f"Unsupported method: {method}. Choose from {SUPPORTED_METHODS}"
        )

    lags = np.asarray(lags, dtype=np.float64)
    ccf_matrix = np.asarray(ccf_matrix, dtype=np.float64)

    if lags.ndim != 1:
        raise RuntimeError(f"lags 必须是一维数组，当前 ndim={lags.ndim}")
    if ccf_matrix.ndim != 2:
        raise RuntimeError(f"ccf_matrix 必须是二维数组，当前 ndim={ccf_matrix.ndim}")
    if ccf_matrix.shape[0] != n_signals:
        raise RuntimeError(
            f"ccf_matrix 第一维必须等于 n_signals={n_signals}，当前为: {ccf_matrix.shape}"
        )
    if ccf_matrix.shape[1] != len(lags):
        raise RuntimeError(
            f"输出不一致：ccf_matrix.shape[1]={ccf_matrix.shape[1]} != len(lags)={len(lags)}"
        )

    return lags, ccf_matrix


# -----------------------------# 内部实现函数
# -----------------------------


@njit(cache=True, fastmath=True, nogil=True)
def _as_float_array_2d(x: np.ndarray) -> np.ndarray:
    """将输入转换为二维浮点数组"""
    return np.asarray(x, dtype=np.float64)


def _xcorr_time_domain_2d(
    x: np.ndarray, y: np.ndarray, sr: float, max_lag: float
) -> LagsAndCCF2D:
    """
    二维时域互相关计算

    Args:
        x, y: 输入信号矩阵，形状为 (n_signals, n_samples)
        sr: 采样率 (Hz)
        max_lag: 最大滞后时间（秒）

    Returns:
        lags: 时间滞后数组 (单位：秒)，形状为 (2*max_lag_samples+1,)
        ccf_matrix: 互相关函数值矩阵，形状为 (n_signals, 2*max_lag_samples+1)
    """
    n_signals, n_samples = x.shape
    max_lag_samples = int(max_lag * sr)

    # 计算滞后时间数组
    lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    lags = lags_samples / sr

    # 初始化结果矩阵
    ccf_matrix = np.zeros((n_signals, len(lags)), dtype=np.float64)

    # 预计算归一化因子，避免在循环中重复计算
    x_energy = np.sum(x**2, axis=1)
    y_energy = np.sum(y**2, axis=1)
    norm_factors = np.sqrt(x_energy * y_energy)

    # 使用FFT方法进行时域互相关，提高计算效率
    for i in range(n_signals):
        # 使用scipy的correlate函数，method='fft'利用快速傅里叶变换加速计算
        ccf_full = correlate(x[i], y[i], mode="full", method="fft")

        # 截取所需的滞后范围
        center = len(ccf_full) // 2
        start_idx = center - max_lag_samples
        end_idx = center + max_lag_samples + 1
        ccf_result = ccf_full[start_idx:end_idx]

        # 归一化互相关
        if norm_factors[i] > 0:
            ccf_matrix[i] = ccf_result / norm_factors[i]
        else:
            ccf_matrix[i] = 0.0

    return lags, ccf_matrix


def _xcorr_freq_domain_2d(
    x: np.ndarray,
    y: np.ndarray,
    sr: float,
    max_lag: float,
    nfft: Optional[int],
    deconv: bool = False,
) -> LagsAndCCF2D:
    """
    二维频域互相关/去卷积计算

    Args:
        x, y: 输入信号矩阵，形状为 (n_signals, n_samples)
        sr: 采样率 (Hz)
        max_lag: 最大滞后时间（秒）
        nfft: FFT长度
        deconv: 如果为True，执行去卷积；如果为False，执行标准互相关

    Returns:
        lags: 时间滞后数组（秒），形状为 (2*max_lag_samples+1,)
        ccf_matrix: 互相关/去卷积结果矩阵，形状为 (n_signals, 2*max_lag_samples+1)
    """
    n_signals, n_samples = x.shape

    # 计算最大滞后样本数
    max_lag_samples = int(max_lag * sr)

    # 计算滞后时间数组
    lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    lags = lags_samples / sr

    # 初始化结果矩阵
    ccf_matrix = np.zeros((n_signals, len(lags)), dtype=np.float64)

    # 计算统一的FFT长度（假设所有信号长度相同，这在二维输入中通常成立）
    # 对于不同长度的信号，可以考虑分组处理，但通常输入矩阵是规整的
    if nfft is None:
        from scipy.fftpack import next_fast_len

        nfft_i = next_fast_len(n_samples)
    else:
        nfft_i = nfft

    # 向量化计算FFT，对整个二维数组进行操作
    X = fft(x, n=nfft_i, axis=1)
    Y = fft(y, n=nfft_i, axis=1)

    if deconv:
        # 向量化反卷积计算
        eps = np.median(np.abs(X), axis=1, keepdims=True) * 1e-6
        Sxy = Y / (X + eps)
    else:
        # 向量化交叉谱计算
        Sxy = np.conj(X) * Y

    # 向量化计算逆FFT
    ccf_full = np.real(ifft(Sxy, axis=1))

    # 向量化循环互相关移位
    ccf_shifted = np.fft.ifftshift(ccf_full, axes=1)

    # 提取所需的滞后范围
    center = nfft_i // 2
    start = center - max_lag_samples
    end = center + max_lag_samples + 1

    # 向量化提取结果
    ccf_result = ccf_shifted[:, start:end]

    return lags, ccf_result


def _coherency_2d(
    x: np.ndarray, y: np.ndarray, sr: float, max_lag: float, nfft: Optional[int]
) -> LagsAndCCF2D:
    """使用相干性作为权重的二维互相关"""
    # 首先计算原始互相关
    lags, ccf_raw = _xcorr_freq_domain_2d(x, y, sr, max_lag, nfft, deconv=False)

    # 在频域计算相位一致性（相干性权重）
    n_signals, n_samples = x.shape
    ccf_matrix = np.zeros_like(ccf_raw)

    for i in range(n_signals):
        # 对每个互相关结果计算相干性权重
        Cxy = fft(ccf_raw[i])
        phase = np.exp(1j * np.angle(Cxy))
        coh = np.abs(np.mean(phase)) ** 4  # 权重
        ccf_matrix[i] = ccf_raw[i] * coh

    return lags, ccf_matrix
