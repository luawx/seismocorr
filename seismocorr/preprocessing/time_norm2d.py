# seismocorr/preprocessing/matrix_time_norm.py

"""
时域归一化方法（Time-domain Normalization）

在 FFT 前对时间序列矩阵进行预处理。

设计原则：
    - 保持与原版本相同的接口
    - 矩阵化操作，提高计算效率
    - 支持批量处理多个信号
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from scipy.signal import convolve2d

ArrayLike = Union[np.ndarray, list]


def moving_ave2D(A: np.ndarray, N: int) -> np.ndarray:
    """
    移动平均计算的矩阵版本，输入N为平滑的完整窗口长度

    PARAMETERS:
    ---------------------
    A: 2-D array of data to be smoothed, shape (n_signals, n_samples)
    N: integer, 定义平滑的完整窗口长度

    RETURNS:
    ---------------------
    B: 2-D array with smoothed data, shape (n_signals, n_samples)
    """
    # 获取输入矩阵的形状
    n_signals, n_samples = A.shape

    # 定义一个扩展数组，每侧添加N个样本
    temp = np.zeros((n_signals, n_samples + 2 * N))

    # 将原始数组放置在扩展数组的中心
    temp[:, N:-N] = A

    # 前导样本：等于实际数组的第一个样本
    temp[:, 0:N] = temp[:, N].reshape(-1, 1)

    # 尾随样本：等于实际数组的最后一个样本
    temp[:, -N:] = temp[:, -N - 1].reshape(-1, 1)

    # 与boxcar卷积并归一化，只使用结果的中心部分
    # 长度等于原始数组，丢弃添加的前导和尾随样本
    # 对每个信号应用卷积
    B = np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(N) / N, mode="same"), 1, temp
    )[:, N:-N]

    return B


class TimeNormalizer2D(ABC):
    """时域归一化抽象基类 - 矩阵版本"""

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: 输入信号矩阵，形状为 (n_signals, n_samples)

        Returns:
            归一化后的信号矩阵，形状为 (n_signals, n_samples)
        """
        pass

    def __call__(self, x):
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"x 应为二维数组 (n_signals, n_samples)，当前 shape={x.shape}")
        if x.size == 0:
            return x.copy()
        return self.apply(x)


class ZScoreNormalizer(TimeNormalizer2D):
    """Z-Score 标准化: (x - μ) / σ - 矩阵版本"""

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: 输入信号矩阵，形状为 (n_signals, n_samples)

        Returns:
            Z-Score 标准化后的信号矩阵，形状为 (n_signals, n_samples)
        """
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)

        # 处理标准差为0的情况
        std[std == 0] = 1

        return (x - mean) / std


class OneBitNormalizer(TimeNormalizer2D):
    """1-bit 归一化: sign(x) - 矩阵版本"""

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: 输入信号矩阵，形状为 (n_signals, n_samples)

        Returns:
            1-bit 归一化后的信号矩阵，形状为 (n_signals, n_samples)
        """
        return np.sign(x)


class RMSNormalizer(TimeNormalizer2D):
    """RMS 归一化: x / sqrt(mean(x²)) - 矩阵版本"""

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: 输入信号矩阵，形状为 (n_signals, n_samples)

        Returns:
            RMS 归一化后的信号矩阵，形状为 (n_signals, n_samples)
        """
        rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True))

        # 处理RMS为0的情况
        rms[rms == 0] = 1

        return x / rms


class ClipNormalizer(TimeNormalizer2D):
    """截幅归一化：限制最大值 - 矩阵版本"""

    def __init__(self, clip_val: float = 3.0):
        clip_val = float(clip_val)
        if clip_val <= 0:
            raise ValueError("clip_val 必须 > 0")
        self.clip_val = clip_val

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: 输入信号矩阵，形状为 (n_signals, n_samples)

        Returns:
            截幅归一化后的信号矩阵，形状为 (n_signals, n_samples)
        """
        return np.clip(x, -self.clip_val, self.clip_val)


class NoTimeNorm(TimeNormalizer2D):
    """无操作 - 矩阵版本"""

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x.copy()


class RAMNormalizer(TimeNormalizer2D):
    """RAM 归一化: x / mean(|x|) - 矩阵版本"""

    def __init__(self, fmin, Fs, norm_win=0.5):
        fmin = float(fmin)
        Fs = float(Fs)
        norm_win = float(norm_win)
        if fmin <= 0:
            raise ValueError("fmin 必须 > 0")
        if Fs <= 0:
            raise ValueError("Fs 必须 > 0")
        if norm_win <= 0:
            raise ValueError("norm_win 必须 > 0")
        self.fmin = fmin
        self.Fs = Fs
        self.norm_win = norm_win

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: 输入信号矩阵，形状为 (n_signals, n_samples)

        Returns:
            RAM 归一化后的信号矩阵，形状为 (n_signals, n_samples)
        """
        period = 1 / self.fmin
        lwin = int(period * self.Fs * self.norm_win)
        N = 2 * lwin + 1

        # 对每个信号应用RAM归一化
        abs_x = np.abs(x)
        moving_avg = moving_ave2D(abs_x, N)

        # 避免除以零
        moving_avg[moving_avg == 0] = 1

        return x / moving_avg


_MATRIX_TIME_NORM_MAP = {
    "zscore": ZScoreNormalizer,
    "one-bit": OneBitNormalizer,
    "rms": RMSNormalizer,
    "clip": ClipNormalizer,  # 直接使用类，而不是lambda
    "no": NoTimeNorm,
    "ramn": RAMNormalizer,
}


def get_time_normalizer_2d(name: str, **kwargs) -> TimeNormalizer2D:
    """
    获取时域归一化器实例 - 矩阵版本

    Args:
        name: 方法名 ('zscore', 'one-bit', 'rms', 'clip', 'ramn', 'no')
        **kwargs: 传递给特定类的参数

    Returns:
        TimeNormalizer2D 实例
    """
    if not isinstance(name, str):
        raise TypeError(f"name 类型应为 str，当前为 {type(name).__name__}: {name!r}")
    if not name.strip():
        raise ValueError("name 不能为空字符串")
    name_lower = name.strip().lower()

    cls = _MATRIX_TIME_NORM_MAP.get(name_lower)

    if cls is None:
        raise ValueError(
            f"Unknown time normalization method: '{name}'. "
            f"Choose from {list(_MATRIX_TIME_NORM_MAP.keys())}"
        )

    # 根据方法名传递特定参数
    if name_lower == "clip":
        clip_val = kwargs.get("clip_val", 3.0)
        return cls(clip_val=clip_val)
    elif name_lower == "ramn":
        # RAMNormalizer 需要特定参数
        fmin = kwargs.get("fmin")
        Fs = kwargs.get("Fs")
        norm_win = kwargs.get("norm_win", 0.5)

        if fmin is None or Fs is None:
            raise ValueError("RAMNormalizer requires fmin and Fs parameters")

        return cls(fmin=fmin, Fs=Fs, norm_win=norm_win)
    else:
        # 其他归一化方法使用默认构造函数
        return cls()
