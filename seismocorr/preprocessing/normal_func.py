# seismocorr/preprocessing/normal_func.py

"""
Unified Preprocessing Toolkit

提供完整的信号预处理功能，适用于地震背景噪声互相关分析。
支持：
- 趋势移除（detrend, demean）
- 滤波（带通、低通、高通）
- 时域 / 频域归一化
- 分段 + FFT 流水线
- 批量处理接口

设计原则：
    - 函数式接口为主，便于组合
    - 支持配置驱动（config['filter'] = 'bandpass'）
    - 内存友好，支持 chunked 处理
"""

import numpy as np
from typing import Dict, Any, Union, Optional, List
from scipy.signal import butter, filtfilt, detrend as scipy_detrend



# =============================================================================
# 🛠 基础预处理函数
# =============================================================================

def demean(x: np.ndarray) -> np.ndarray:
    """去除均值"""
    x = np.asarray(x)
    if x.size == 0:
        return x.copy()
    return x - np.mean(x)


def detrend(x: np.ndarray, type: str = 'linear') -> np.ndarray:
    """
    去除趋势

    Args:
        x: 输入数组
        type: 'constant'（去均值）、'linear'（去线性趋势）

    Returns:
        去趋势后的数组
    """
    x = np.asarray(x)
    if x.size == 0:
        return x.copy()
    if not isinstance(type, str):
        raise TypeError(f"type 类型应为 str，当前为 {type(type).__name__}: {type!r}")
    type = type.strip().lower()
    if type not in ("constant", "linear"):
        raise ValueError(f"type 只能是 'constant' 或 'linear'，当前为 {type!r}")

    return scipy_detrend(x, type=type)


def taper(x: np.ndarray, width: float = 0.05) -> np.ndarray:
    """
    对信号加窗（汉宁窗），减少边缘效应

    Args:
        x: 输入数组
        width: 窗口比例（默认首尾 5% 加窗）

    Returns:
        加窗后的数组
    """
    x = np.asarray(x)
    if x.size == 0:
        return x.copy()
    if isinstance(width, bool) or not isinstance(width, (int, float)):
        raise TypeError(f"width 应为数值类型，当前为 {type(width).__name__}: {width!r}")
    width = float(width)
    if not (0.0 <= width <= 0.5):
        raise ValueError(f"width 建议在 [0, 0.5]，当前为 {width!r}")

    window = int(len(x) * width)
    if window == 0:
        return x.copy()
    y = x.copy()
    y[:window] *= np.hanning(2 * window)[:window]
    y[-window:] *= np.hanning(2 * window)[window:]
    return y


# =============================================================================
# 🔧 滤波函数
# =============================================================================

def _butter_filter(
    data: np.ndarray,
    sampling_rate: float,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    通用 Butterworth 滤波器

    Args:
        data: 输入时间序列
        sampling_rate: 采样率 (Hz)
        freq_min: 高通频率（Hz）
        freq_max: 低通频率（Hz）
        order: 滤波阶数
        zero_phase: 是否零相位滤波（前后各一次）

    Returns:
        滤波后的时间序列
    """
    data = np.asarray(data)
    if data.size == 0:
        return data.copy()
    if data.ndim != 1:
        raise ValueError(f"data 应为一维时间序列，当前 shape={data.shape}")

    if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, (int, float)):
        raise TypeError(f"sampling_rate 应为数值类型，当前为 {type(sampling_rate).__name__}: {sampling_rate!r}")
    sampling_rate = float(sampling_rate)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(f"sampling_rate 应 > 0 且为有限数，当前为 {sampling_rate!r}")

    if freq_min is not None:
        if isinstance(freq_min, bool) or not isinstance(freq_min, (int, float)):
            raise TypeError(f"freq_min 应为数值或 None，当前为 {type(freq_min).__name__}: {freq_min!r}")
        freq_min = float(freq_min)
        if not np.isfinite(freq_min) or freq_min <= 0:
            raise ValueError(f"freq_min 应 > 0 且为有限数，当前为 {freq_min!r}")

    if freq_max is not None:
        if isinstance(freq_max, bool) or not isinstance(freq_max, (int, float)):
            raise TypeError(f"freq_max 应为数值或 None，当前为 {type(freq_max).__name__}: {freq_max!r}")
        freq_max = float(freq_max)
        if not np.isfinite(freq_max) or freq_max <= 0:
            raise ValueError(f"freq_max 应 > 0 且为有限数，当前为 {freq_max!r}")

    if (freq_min is not None) and (freq_max is not None) and (freq_min >= freq_max):
        raise ValueError(f"freq_min 应 < freq_max，当前为 freq_min={freq_min}, freq_max={freq_max}")

    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError(f"order 应为 int，当前为 {type(order).__name__}: {order!r}")
    if order < 1:
        raise ValueError(f"order 应 >= 1，当前为 {order!r}")

    nyquist = sampling_rate / 2.0

    # 设计滤波器
    if (freq_min is not None) and (freq_max is not None):
        btype = 'band'
        critical = [freq_min / nyquist, freq_max / nyquist]
    elif freq_min is not None:
        btype = 'high'
        critical = [freq_min / nyquist]
    elif freq_max is not None:
        btype = 'low'
        critical = [freq_max / nyquist]
    else:
        return data.copy()  # 无滤波要求

    # 防止超 Nyquist
    critical = [c for c in critical if c < 1.0]
    if not critical:
        return data.copy()

    b, a = butter(order, critical, btype=btype)

    filtered = filtfilt(b, a, data) if zero_phase else np.apply_along_axis(lambda x: np.convolve(x, b, mode='same'), 0, data)
    return filtered


def bandpass(
    x: np.ndarray,
    fmin: float,
    fmax: float,
    sr: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """带通滤波"""
    return _butter_filter(x, sr, freq_min=fmin, freq_max=fmax, order=order, zero_phase=zero_phase)


def lowpass(
    x: np.ndarray,
    fmax: float,
    sr: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """低通滤波"""
    return _butter_filter(x, sr, freq_max=fmax, order=order, zero_phase=zero_phase)


def highpass(
    x: np.ndarray,
    fmin: float,
    sr: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """高通滤波"""
    return _butter_filter(x, sr, freq_min=fmin, order=order, zero_phase=zero_phase)