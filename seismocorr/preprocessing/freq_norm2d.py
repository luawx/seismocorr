# seismocorr/preprocessing/matrix_freq_norm.py

"""
频域归一化方法（Frequency-domain Normalization）

接收时域信号矩阵，进行频域白化处理，返回时域信号矩阵。
参考：Bensen et al., 2007; Denolle et al., 2013

设计原则：
    - 保持与原版本相同的接口
    - 矩阵化操作，提高计算效率
    - 支持批量处理多个信号
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from scipy.signal import fftconvolve

ArrayLike = Union[np.ndarray, list]


class FreqNormalizer2D(ABC):
    """频域归一化抽象基类"""

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: 时域信号矩阵，形状为 (n_signals, n_samples)

        Returns:
            归一化后的时域数据矩阵，形状为 (n_signals, n_samples)
        """
        pass

    def __call__(self, data):
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"data 必须是二维矩阵 (n_signals, n_samples)，当前 shape={data.shape}")
        return self.apply(data)

    def _check_empty_input(self, data: np.ndarray) -> bool:
        """检查输入是否为空数组"""
        data = np.asarray(data)
        return data.size == 0


class SpectralWhitening2D(FreqNormalizer2D):
    """谱白化（Spectral Whitening）"""

    def __init__(self, smooth_win: int = 20):
        if isinstance(smooth_win, bool) or not isinstance(smooth_win, int):
            raise TypeError(f"smooth_win 类型应为 int，当前为 {type(smooth_win).__name__}: {smooth_win!r}")
        if smooth_win < 1:
            raise ValueError(f"smooth_win 应 >= 1，当前为 {smooth_win!r}")

        self.smooth_win = smooth_win
        self.epsilon = 1e-10  # 用于数值稳定性的小常数

    def _smooth(self, x: np.ndarray) -> np.ndarray:
        """移动平均平滑，确保输出长度与输入相同 - 优化版"""
        n_signals, n_samples = x.shape
        if n_samples < self.smooth_win:
            return x

        # 使用 FFT 卷积，速度更快，特别是大窗口
        kernel = np.ones(self.smooth_win) / self.smooth_win

        # 向量化处理所有信号，利用 fftconvolve 的高效实现
        smoothed = fftconvolve(x, kernel[np.newaxis, :], mode="same", axes=1)

        return smoothed

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        对时域信号矩阵进行谱白化
        """
        if data.size == 0:  # 更简洁的空数组检查
            return data.copy()

        n_signals, n_samples = data.shape

        # 计算FFT - 矩阵版本
        FFT = np.fft.fft(data, axis=1)

        # 获取幅度谱和相位谱 - 矩阵版本
        amplitude = np.abs(FFT)
        # phase = np.angle(FFT)  # 注意：相位在后续计算中未直接使用，可删除

        # 对幅度谱进行平滑 - 优化版
        smoothed_amp = self._smooth(amplitude)

        # 避免除以零 - 优化版，添加 epsilon 提高稳定性
        weight = np.where(
            smoothed_amp > self.epsilon, 1.0 / (smoothed_amp + self.epsilon), 0.0
        )

        # 应用白化权重 - 矩阵版本，直接修改 FFT 数组
        FFT *= weight

        # 计算IFFT返回时域信号 - 矩阵版本
        whitened_data = np.fft.ifft(FFT, axis=1).real

        return whitened_data


class BandWhitening2D(FreqNormalizer2D):
    """频带白化（Band Whitening）"""

    def __init__(self, freq_min: float, freq_max: float, Fs: float):
        for name, v in [("freq_min", freq_min), ("freq_max", freq_max), ("Fs", Fs)]:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"{name} 应为数值类型，当前为 {type(v).__name__}: {v!r}")
            if not np.isfinite(v):
                raise ValueError(f"{name} 不能是 NaN/Inf，当前为 {v!r}")
        Fs = float(Fs)
        if Fs <= 0:
            raise ValueError(f"Fs 必须 > 0，当前为 {Fs!r}")

        self.fmin = freq_min
        self.fmax = freq_max
        self.Fs = Fs
        if self.fmin >= self.fmax:
            raise ValueError("freq_min must be less than freq_max")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        对时域信号矩阵进行频带白化

        Args:
            data: 时域信号矩阵，形状为 (n_signals, n_samples)

        Returns:
            白化后的时域信号矩阵，形状为 (n_signals, n_samples)
        """
        if self._check_empty_input(data):
            return data.copy()

        n_signals, n_samples = data.shape
        if n_samples < 2:
            return data.copy()

        # 计算频率范围
        frange = float(self.fmax) - float(self.fmin)
        nsmo = int(np.fix(min(0.01, 0.5 * frange) * float(n_samples) / self.Fs))

        # 计算频率轴
        f = np.arange(n_samples) * self.Fs / (n_samples - 1.0)

        # 找到目标频带
        JJ = ((f > float(self.fmin)) & (f < float(self.fmax))).nonzero()[0]

        if len(JJ) == 0:
            # 如果没有找到目标频带，返回原始信号
            return data.copy()

        # 计算FFT
        FFTs = np.fft.fft(data, axis=1)
        FFTsW = np.zeros_like(FFTs, dtype=complex)

        # 创建白化滤波器
        # 频带外设置为0
        FFTsW[:] = 0.0

        # 频带内设置过渡带
        if nsmo > 0 and len(JJ) > 2 * nsmo:
            # 左过渡带（余弦锥）
            smo1 = np.cos(np.linspace(np.pi / 2, np.pi, nsmo + 1)) ** 2

            # 对每个信号应用左过渡带
            FFTsW[:, JJ[0] : JJ[0] + nsmo + 1] = smo1 * np.exp(
                1j * np.angle(FFTs[:, JJ[0] : JJ[0] + nsmo + 1])
            )

            # 频带内完全白化
            FFTsW[:, JJ[0] + nsmo + 1 : JJ[-1] - nsmo] = np.exp(
                1j * np.angle(FFTs[:, JJ[0] + nsmo + 1 : JJ[-1] - nsmo])
            )

            # 右过渡带（余弦锥）
            smo2 = np.cos(np.linspace(0.0, np.pi / 2.0, nsmo + 1)) ** 2

            # 对每个信号应用右过渡带
            FFTsW[:, JJ[-1] - nsmo : JJ[-1] + 1] = smo2 * np.exp(
                1j * np.angle(FFTs[:, JJ[-1] - nsmo : JJ[-1] + 1])
            )
        else:
            # 如果频带太窄，直接在整个频带内白化
            FFTsW[:, JJ] = np.exp(1j * np.angle(FFTs[:, JJ]))

        # 计算IFFT - 矩阵版本
        whitedata = 2.0 * np.fft.ifft(FFTsW, axis=1).real

        return whitedata


class RmaFreqNorm2D(FreqNormalizer2D):
    """递归移动平均白化 (Recursive Moving Average Whitening) - 矩阵版本"""

    def __init__(self, alpha: float = 0.9):
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha 应为数值类型，当前为 {type(alpha).__name__}: {alpha!r}")
        alpha = float(alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha 建议在 (0,1) 内，当前为 {alpha!r}")

        self.alpha = alpha  # 平滑系数
        self.avg_power = None

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        对时域信号矩阵进行RMA白化

        Args:
            data: 时域信号矩阵，形状为 (n_signals, n_samples)

        Returns:
            白化后的时域信号矩阵，形状为 (n_signals, n_samples)
        """
        if self._check_empty_input(data):
            return data.copy()

        n_signals, n_samples = data.shape

        # 计算FFT - 矩阵版本
        FFT = np.fft.fft(data, axis=1)

        # 计算功率谱 - 矩阵版本
        power = np.abs(FFT) ** 2

        # 更新平均功率（递归移动平均）- 矩阵版本
        if self.avg_power is None or self.avg_power.shape != power.shape:
            self.avg_power = power
        else:
            self.avg_power = self.alpha * self.avg_power + (1 - self.alpha) * power

        # 避免除以零
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_factor = np.where(
                self.avg_power > 0, 1.0 / np.sqrt(self.avg_power), 0.0
            )

        # 应用归一化 - 矩阵版本
        whitened_FFT = FFT * norm_factor

        # 计算IFFT - 矩阵版本
        whitened_data = np.fft.ifft(whitened_FFT, axis=1).real

        return whitened_data


class NoFreqNorm2D(FreqNormalizer2D):
    """无频域归一化"""

    def apply(self, data: np.ndarray) -> np.ndarray:
        return data.copy()


# 更新映射表 - 矩阵版本
_MATRIX_FREQ_NORM_MAP = {
    "whiten": SpectralWhitening2D,
    "rma": RmaFreqNorm2D,
    "bandwhiten": BandWhitening2D,
    "no": NoFreqNorm2D,
}


def get_freq_normalizer_2d(name: str, **kwargs) -> FreqNormalizer2D:
    """
    获取频域归一化器实例

    Args:
        name: 方法名 ('whiten', 'rma', 'bandwhiten', 'no')
        **kwargs: 如 smooth_win, alpha, freq_min, freq_max, Fs

    Returns:
        FreqNormalizer2D 实例
    """
    if not isinstance(name, str):
        raise TypeError(f"name 类型应为 str，当前为 {type(name).__name__}: {name!r}")
    if not name.strip():
        raise ValueError("name 不能为空字符串")
    name_lower = name.strip().lower()

    cls = _MATRIX_FREQ_NORM_MAP.get(name_lower)
    if cls is None:
        raise ValueError(
            f"Unknown frequency normalization method: '{name}'. "
            f"Choose from {list(_MATRIX_FREQ_NORM_MAP.keys())}"
        )

    # 根据方法名传递特定参数
    if name_lower == "whiten":
        smooth_win = kwargs.get("smooth_win", 20)
        return cls(smooth_win=smooth_win)
    elif name_lower == "rma":
        alpha = kwargs.get("alpha", 0.9)
        return cls(alpha=alpha)
    elif name_lower == "bandwhiten":
        # BandWhitening 需要特定参数
        freq_min = kwargs.get("freq_min")
        freq_max = kwargs.get("freq_max")
        Fs = kwargs.get("Fs")

        if freq_min is None or freq_max is None or Fs is None:
            raise ValueError(
                "BandWhitening requires freq_min, freq_max, and Fs parameters"
            )

        return cls(freq_min=freq_min, freq_max=freq_max, Fs=Fs)
    else:
        # 其他归一化方法使用默认构造函数
        return cls()
