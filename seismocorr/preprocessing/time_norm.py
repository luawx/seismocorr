# seismocorr/preprocessing/time_norm.py

"""
时域归一化方法（Time-domain Normalization）

在 FFT 前对时间序列进行预处理。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union
import pywt

ArrayLike = Union[np.ndarray, list]


class TimeNormalizer(ABC):
    """时域归一化抽象基类"""
    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x):
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"x 应为一维数组，当前 shape={x.shape}")
        if x.size == 0:
            return x.copy()
        return self.apply(x)


class ZScoreNormalizer(TimeNormalizer):
    """Z-Score 标准化: (x - μ) / σ"""
    def apply(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        if std == 0:
            return x - mean
        return (x - mean) / std


class OneBitNormalizer(TimeNormalizer):
    """1-bit 归一化: sign(x)"""
    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x)


class RMSNormalizer(TimeNormalizer):
    """RMS 归一化: x / sqrt(mean(x²))"""
    def apply(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2))
        if rms == 0:
            return x.copy()
        return x / rms


class ClipNormalizer(TimeNormalizer):
    """截幅归一化：限制最大值"""
    def __init__(self, clip_val: float = 3.0):
        clip_val = float(clip_val)
        if clip_val <= 0:
            raise ValueError("clip_val 必须 > 0")
        self.clip_val = clip_val

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, -self.clip_val, self.clip_val)


class NoTimeNorm(TimeNormalizer):
    """无操作"""
    def apply(self, x: np.ndarray) -> np.ndarray:
        return x.copy()
    

class RAMNormalizer(TimeNormalizer):
    """RAM 归一化: x / mean(|x|)"""
    def __init__(self,fmin,Fs,norm_win=0.5):
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

    @staticmethod
    def _moving_ave(A, N):
        '''
        Alternative function for moving average for an array.
        PARAMETERS:
        ---------------------
        A: 1-D array of data to be smoothed
        N: integer, it defines the full!! window length to smooth
        RETURNS:
        ---------------------
        B: 1-D array with smoothed data
        '''
        # defines an array with N extra samples at either side
        temp = np.zeros(len(A) + 2 * N)
        # set the central portion of the array to A
        temp[N: -N] = A
        # leading samples: equal to first sample of actual array
        temp[0: N] = temp[N]
        # trailing samples: Equal to last sample of actual array
        temp[-N:] = temp[-N-1]
        # convolve with a boxcar and normalize, and use only central portion of the result
        # with length equal to the original array, discarding the added leading and trailing samples
        B = np.convolve(temp, np.ones(N)/N, mode='same')[N: -N]
        return B

    def apply(self, x: np.ndarray) -> np.ndarray:
        period = 1 / self.fmin
        lwin = int(period * self.Fs * self.norm_win)
        N = 2*lwin+1
        x = x/self._moving_ave(np.abs(x),N)
        return x.copy()


class WaterLevelNormalizer(TimeNormalizer):
    """
    Water-level normalization（按窗口 RMS 做"截顶"缩放）

    
    Args:
        Fs : float
            采样率 (Hz)。
        win_length : float
            窗口长度（秒）。窗口样本数 win_n = round(win_length * Fs)。
            常用 1~10 s。
        water_level_factor : float
            水位系数。水位 W = water_level_factor * global_rms。
        n_iter : int
            迭代次数。每次迭代都会基于当前信号重新计算水位。
        eps : float
            数值稳定项。
        dynamic_factor_decay : float, optional
            水位系数衰减因子。每次迭代时 water_level_factor *= dynamic_factor_decay
            默认 None 表示不衰减。
    Returns:
        归一化后数据
    """

    def __init__(
        self,
        Fs: float,
        win_length: float = 10.0,
        water_level_factor: float = 1.0,
        n_iter: int = 1,
        eps: float = 1e-10,
        dynamic_factor_decay: float = None,
    ):
        self.Fs = float(Fs)
        self.win_length = float(win_length)
        self.water_level_factor = float(water_level_factor)
        self.n_iter = int(n_iter)
        self.eps = float(eps)
        self.dynamic_factor_decay = dynamic_factor_decay

        if self.Fs <= 0:
            raise ValueError("Fs 必须 > 0")
        if self.win_length <= 0:
            raise ValueError("win_length 必须 > 0")
        if self.n_iter < 1:
            raise ValueError("n_iter 必须 >= 1")
        if self.eps <= 0:
            raise ValueError("eps 必须 > 0")
        if self.dynamic_factor_decay is not None:
            self.dynamic_factor_decay = float(self.dynamic_factor_decay)
            if not (0.0 < self.dynamic_factor_decay <= 1.0):
                raise ValueError("dynamic_factor_decay 必须在 (0, 1] 内")

    def apply(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=float).copy()
        npts = y.size
        if npts == 0:
            return y

        # 初始化当前水位系数
        current_factor = self.water_level_factor
        
        for iteration in range(self.n_iter):
            # 1. 计算当前信号的全局RMS和水位
            global_rms = np.sqrt(np.mean(y * y)) + self.eps
            W = current_factor * global_rms

            # 2. 分窗口
            win_n = int(round(self.win_length * self.Fs))
            if win_n <= 0:
                raise ValueError("win_length * Fs must be >= 1 sample.")

            n_win = npts // win_n
            n_full = n_win * win_n

            # 完整窗口
            if n_win > 0:
                Xw = y[:n_full].reshape(n_win, win_n)
                # 计算每个窗口的RMS
                rms = np.sqrt(np.mean(Xw * Xw, axis=1)) + self.eps
                # 计算缩放因子（只压制超过水位的窗口）
                scale = np.minimum(1.0, W / rms)
                # 应用缩放
                Xw *= scale[:, None]

            # 尾部处理
            tail = y[n_full:] if n_full < npts else None
            if tail is not None and tail.size > 0:
                tail_rms = np.sqrt(np.mean(tail * tail)) + self.eps
                if tail_rms > W:
                    tail *= (W / tail_rms)

            # 3. 如果启用水位系数衰减，更新因子
            if self.dynamic_factor_decay is not None:
                current_factor *= self.dynamic_factor_decay

        return y


class CWTSoftThreshold1D(TimeNormalizer):
    """
    基于连续小波变换 (CWT) 的软阈值处理（designal / denoise），仅保留 apply 输出处理后信号。
    思路简述
    1) 对信号做 CWT 得到复系数 W(scale, t)
    2) 在 noise_idx 指定的“纯噪声时窗”内，按每个尺度统计幅值分位数阈值 beta(scale)
    3) 对每个尺度进行 soft-threshold：
       - designal：压制“过强瞬态”，把超过阈值的幅值截到 beta
       - denoise ：抑制噪声，把系数做 (mag - beta) 的软阈值，小于阈值置 0
    4) 用与你原函数一致的“定性重建”：对所有尺度的实部求和得到 x_out
    参考自Yang et al., GJI (2020)

    Args：
        fs : float
            采样率 (Hz)。决定 dt=1/fs，以及默认最高频 f_max=fs/2。
        noise_idx : slice | ndarray[bool] | ndarray[int]
            “纯噪声”时间窗索引，用来估计每个尺度的噪声幅值阈值 beta。
            例：
              - slice(0, 2000)
              - bool mask: mask.shape == (n_samples,)
              - int 索引数组: np.array([0,1,2,...])
        mode : {"designal", "denoise"}
            - "designal": 压制强瞬态（超过阈值的幅值被截到 beta）
            - "denoise" : 去噪软阈值（超过阈值的幅值减去 beta，小于阈值置 0）
        wavelet : str
            PyWavelets 支持的小波名。默认复 Morlet： "cmor1.5-1.0"。
            不同参数会影响时频分辨率与幅值分布。
        voices_per_octave : int
            每个倍频程使用多少个尺度（尺度密度）。
            越大：尺度更密、计算更慢、频率分辨率更高。
        quantile : float
            噪声幅值阈值的分位数（0~1）。
            例如 0.99 表示用噪声幅值的 99% 分位作为阈值（更“严格”）。
        f_min : float
            构造尺度使用的最低频 (Hz)。越低意味着尺度范围更大、计算更慢。
        f_max : float | None
            构造尺度使用的最高频 (Hz)。默认 None -> fs/2（奈奎斯特频率）。
        normalize : bool
            是否做幅值缩放，使输出的最大绝对值与输入最大绝对值相当：
                x_out = x_out / max(|x_out|) * max(|x|)
            这与你原函数保持一致（用于“视觉/定性”重建的幅值对齐）。
        eps : float
            数值稳定项，用于避免除 0（如 mag=0 时的相位计算、归一化分母）。
    Returns：
        归一化后的一维数据

    """

    def __init__(
        self,
        fs: float,
        noise_idx,
        mode: str = "designal",
        wavelet: str = "cmor1.5-1.0",
        voices_per_octave: int = 16,
        quantile: float = 0.99,
        f_min: float = 0.01,
        f_max: float | None = None,
        normalize: bool = True,
        eps: float = 1e-12,
    ): 

        self.fs = float(fs)
        self.noise_idx = noise_idx
        self.mode = mode
        self.wavelet = wavelet
        self.voices_per_octave = int(voices_per_octave)
        self.quantile = float(quantile)
        self.f_min = float(f_min)
        self.f_max = float(fs / 2) if f_max is None else float(f_max)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        # 预计算 scales（同一实例多次 apply 会更快）
        self._dt = 1.0 / self.fs
        self._scales = self._build_scales()

        if self.fs <= 0:
            raise ValueError("fs 必须 > 0")
        if self.voices_per_octave < 1:
            raise ValueError("voices_per_octave 必须 >= 1")
        if not (0.0 < self.quantile < 1.0):
            raise ValueError("quantile 必须在 (0, 1) 内")
        if not isinstance(self.wavelet, str) or not self.wavelet.strip():
            raise TypeError("wavelet 必须是非空字符串")


    def _build_scales(self) -> np.ndarray:
        if self.f_min <= 0 or self.f_max <= 0 or self.f_max <= self.f_min:
            raise ValueError("Require 0 < f_min < f_max.")

        n_octaves = np.log2(self.f_max / self.f_min)
        n_scales = max(1, int(self.voices_per_octave * n_octaves))
        freqs = np.geomspace(self.f_max, self.f_min, n_scales)  # Hz
        return pywt.frequency2scale(self.wavelet, freqs * self._dt)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.size
        if n == 0:
            return x.copy()

        # CWT
        W, _ = pywt.cwt(x, self._scales, self.wavelet, sampling_period=self._dt)
        mag = np.abs(W)

        idx = self.noise_idx
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = n if idx.stop is None else idx.stop
            if stop <= start:
                raise ValueError("noise_idx slice 为空或无效")
        else:
            idx = np.asarray(idx)
            if idx.size == 0:
                raise ValueError("noise_idx 不能为空")
            if idx.dtype == bool:
                if idx.shape[0] != n:
                    raise ValueError("noise_idx 为 bool mask 时长度必须等于信号长度")
            else:
                if np.any((idx < 0) | (idx >= n)):
                    raise ValueError("noise_idx 存在越界索引")

        # beta: 一次性 quantile（向量化提速）
        beta = np.quantile(mag[:, self.noise_idx], self.quantile, axis=1)  # (n_scales,)
        beta2 = beta[:, None]

        # phase：避免除零
        phase = W / (mag + self.eps)

        # soft-threshold（向量化）
        if self.mode == "designal":
            W_new = np.where(mag >= beta2, phase * beta2, W)
        elif self.mode == "denoise":
            W_new = np.where(mag >= beta2, phase * (mag - beta2), 0.0)
        else:
            raise ValueError("mode must be 'designal' or 'denoise'")

        # “定性”逆变换：按尺度求和
        x_out = np.sum(np.real(W_new), axis=0)

        if self.normalize:
            in_rms = np.sqrt(np.mean(x * x)) + self.eps
            out_rms = np.sqrt(np.mean(x_out * x_out)) + self.eps
            x_out = x_out * (in_rms / out_rms)

        return x_out


_TIME_NORM_MAP = {
    'zscore': ZScoreNormalizer,
    'one-bit': OneBitNormalizer,
    'rms': RMSNormalizer,
    'clip': ClipNormalizer,  # 直接使用类，而不是lambda
    'no': NoTimeNorm,
    'ramn': RAMNormalizer,
    'waterlevel': WaterLevelNormalizer,
    'cwt-soft': CWTSoftThreshold1D,
}


def get_time_normalizer(name: str, **kwargs) -> TimeNormalizer:
    """
    获取时域归一化器实例

    Args:
        name: 方法名 (
            'zscore', 'one-bit', 'rms', 'clip',
            'ramn', 'waterlevel', 'cwt-soft', 'no'
        )
        **kwargs: 传递给特定类的参数

    Returns:
        TimeNormalizer 实例
    """
    if not isinstance(name, str):
        raise TypeError(f"name 类型应为 str，当前为 {type(name).__name__}: {name!r}")
    if not name.strip():
        raise ValueError("name 不能为空字符串")
    name_lower = name.strip().lower()

    cls = _TIME_NORM_MAP.get(name_lower)
    if cls is None:
        available_methods = list(_TIME_NORM_MAP.keys())
        raise ValueError(f"未知的时域归一化方法: '{name}'. 请从以下方法中选择: {', '.join(available_methods)}")
    
    # 根据方法名传递特定参数
    if name_lower == 'clip':
        clip_val = kwargs.get('clip_val', 3.0)
        return cls(clip_val=clip_val)
    elif name_lower == 'ramn':
        # RAMNormalizer 需要特定参数
        fmin = kwargs.get('fmin')
        Fs = kwargs.get('Fs')
        norm_win = kwargs.get('norm_win', 0.5)
        
        if fmin is None or Fs is None:
            raise ValueError("RAMNormalizer requires fmin and Fs parameters")
            
        return cls(fmin=fmin, Fs=Fs, norm_win=norm_win)

    elif name_lower == 'waterlevel':
        # Water-level normalization（按窗口 RMS 做能量压制）
        Fs = kwargs.get('Fs')
        if Fs is None:
            raise ValueError("WaterLevelNormalizer requires Fs parameter")

        win_length = kwargs.get('win_length', 10.0)
        water_level_factor = kwargs.get('water_level_factor', 1.0)
        n_iter = kwargs.get('n_iter', 1)
        eps = kwargs.get('eps', 1e-10)

        return cls(
            Fs=Fs,
            win_length=win_length,
            water_level_factor=water_level_factor,
            n_iter=n_iter,
            eps=eps,
        )

    elif name_lower == 'cwt-soft':
        # 基于 CWT 的软阈值处理（designal / denoise）
        Fs = kwargs.get('Fs')
        noise_idx = kwargs.get('noise_idx')

        if Fs is None or noise_idx is None:
            raise ValueError("CWTSoftThreshold1D requires fs and noise_idx")

        mode = kwargs.get('mode', 'designal')
        wavelet = kwargs.get('wavelet', 'cmor1.5-1.0')
        voices_per_octave = kwargs.get('voices_per_octave', 16)
        quantile = kwargs.get('quantile', 0.99)
        f_min = kwargs.get('f_min', 0.01)
        f_max = kwargs.get('f_max', None)
        normalize = kwargs.get('normalize', True)
        eps = kwargs.get('eps', 1e-12)

        return cls(
            fs=Fs,
            noise_idx=noise_idx,
            mode=mode,
            wavelet=wavelet,
            voices_per_octave=voices_per_octave,
            quantile=quantile,
            f_min=f_min,
            f_max=f_max,
            normalize=normalize,
            eps=eps,
        )

    else:
        # 其他归一化方法使用默认构造函数
        # (zscore, one-bit, rms, no)
        return cls()