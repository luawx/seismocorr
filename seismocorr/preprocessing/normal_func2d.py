# seismocorr/preprocessing/matrix_normal_func.py

"""
Unified Preprocessing Toolkit

提供完整的信号预处理功能的矩阵版本，适用于地震背景噪声互相关分析。
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
    - 矩阵化操作，提高计算效率
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.signal import butter
from scipy.signal import detrend as scipy_detrend
from scipy.signal import filtfilt, lfilter

# =============================================================================
# 🛠 基础预处理函数 - 矩阵版本
# =============================================================================


def demean(x: np.ndarray) -> np.ndarray:
    """去除均值

    Args:
        x: 输入数组，形状为 (n_signals, n_samples)

    Returns:
        去均值后的数组，形状为 (n_signals, n_samples)
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x 应为二维数组 (n_signals, n_samples)，当前 shape={x.shape}")
    if x.size == 0:
        return x.copy()
    return x - np.mean(x, axis=1, keepdims=True)


def detrend(x: np.ndarray, type: str = "linear") -> np.ndarray:
    """
    去除趋势

    Args:
        x: 输入数组，形状为 (n_signals, n_samples)
        type: 'constant'（去均值）、'linear'（去线性趋势）

    Returns:
        去趋势后的数组，形状为 (n_signals, n_samples)
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x 应为二维数组 (n_signals, n_samples)，当前 shape={x.shape}")
    if x.size == 0:
        return x.copy()
    if type not in ("constant", "linear"):
        raise ValueError('type 只能是 "constant" 或 "linear"')
    return scipy_detrend(x, type=type, axis=1)


def taper(x: np.ndarray, width: float = 0.05) -> np.ndarray:
    """
    对信号加窗（汉宁窗），减少边缘效应

    Args:
        x: 输入数组，形状为 (n_signals, n_samples)
        width: 窗口比例（默认首尾 5% 加窗）

    Returns:
        加窗后的数组，形状为 (n_signals, n_samples)
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x 应为二维数组 (n_signals, n_samples)，当前 shape={x.shape}")
    if x.size == 0:
        return x.copy()
    width = float(width)
    if not (0.0 <= width < 0.5):
        raise ValueError("width 必须在 [0, 0.5) 范围内")

    n_samples = x.shape[1]
    window = int(n_samples * width)
    if window == 0:
        return x.copy()

    y = x.copy()

    # 创建汉宁窗
    hanning_window = np.hanning(2 * window)
    left_window = hanning_window[:window]
    right_window = hanning_window[window:]

    # 应用窗到所有信号
    y[:, :window] *= left_window
    y[:, -window:] *= right_window

    return y


# =============================================================================
# 🔧 滤波函数 - 矩阵版本
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
    Butterworth 滤波器

    Args:
        data: 输入时间序列矩阵，形状为 (n_signals, n_samples)
        sampling_rate: 采样率 (Hz)
        freq_min: 高通频率（Hz）
        freq_max: 低通频率（Hz）
        order: 滤波阶数
        zero_phase: 是否零相位滤波（前后各一次）

    Returns:
        滤波后的时间序列矩阵，形状为 (n_signals, n_samples)
    """

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data 应为二维数组 (n_signals, n_samples)，当前 shape={data.shape}")
    if data.size == 0:
        return data.copy()

    sampling_rate = float(sampling_rate)
    if sampling_rate <= 0:
        raise ValueError("sampling_rate 必须 > 0")

    if freq_min is not None:
        freq_min = float(freq_min)
        if freq_min <= 0:
            raise ValueError("freq_min 必须 > 0")
    if freq_max is not None:
        freq_max = float(freq_max)
        if freq_max <= 0:
            raise ValueError("freq_max 必须 > 0")

    if (freq_min is not None) and (freq_max is not None) and (freq_min >= freq_max):
        raise ValueError("freq_min 必须小于 freq_max")

    # 早期返回：无滤波要求
    if freq_min is None and freq_max is None:
        return data.copy()

    nyquist = sampling_rate / 2.0
    btype = None
    critical = []

    # 简化滤波器设计逻辑
    if (freq_min is not None) and (freq_max is not None):
        btype = "bandpass"
        Wn = [freq_min / nyquist, freq_max / nyquist]
    elif freq_min is not None:
        btype = "highpass"
        Wn = freq_min / nyquist
    else:  # only freq_max
        btype = "lowpass"
        Wn = freq_max / nyquist

    # 检查频率范围是否有效
    if isinstance(Wn, list):
        if any(w >= 1.0 for w in Wn):
            return data.copy()
    elif Wn >= 1.0:
        return data.copy()

    # 设计滤波器
    b, a = butter(order, Wn, btype=btype)

    # 直接处理2D数组
    if zero_phase:
        # filtfilt直接支持axis参数
        return filtfilt(b, a, data, axis=1)
    else:
        # lfilter也支持axis参数
        return lfilter(b, a, data, axis=1)


def bandpass(
    x: np.ndarray,
    fmin: float,
    fmax: float,
    sr: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """带通滤波

    Args:
        x: 输入时间序列矩阵，形状为 (n_signals, n_samples)
        fmin: 高通频率（Hz）
        fmax: 低通频率（Hz）
        sr: 采样率 (Hz)
        order: 滤波阶数
        zero_phase: 是否零相位滤波

    Returns:
        滤波后的时间序列矩阵，形状为 (n_signals, n_samples)
    """
    return _butter_filter(
        x, sr, freq_min=fmin, freq_max=fmax, order=order, zero_phase=zero_phase
    )


def lowpass(
    x: np.ndarray,
    fmax: float,
    sr: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """低通滤波

    Args:
        x: 输入时间序列矩阵，形状为 (n_signals, n_samples)
        fmax: 低通频率（Hz）
        sr: 采样率 (Hz)
        order: 滤波阶数
        zero_phase: 是否零相位滤波

    Returns:
        滤波后的时间序列矩阵，形状为 (n_signals, n_samples)
    """
    return _butter_filter(x, sr, freq_max=fmax, order=order, zero_phase=zero_phase)


def highpass(
    x: np.ndarray,
    fmin: float,
    sr: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """高通滤波

    Args:
        x: 输入时间序列矩阵，形状为 (n_signals, n_samples)
        fmin: 高通频率（Hz）
        sr: 采样率 (Hz)
        order: 滤波阶数
        zero_phase: 是否零相位滤波

    Returns:
        滤波后的时间序列矩阵，形状为 (n_signals, n_samples)
    """
    return _butter_filter(x, sr, freq_min=fmin, order=order, zero_phase=zero_phase)
