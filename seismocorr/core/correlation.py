# seismocorr/core/correlation.py

"""
Cross-Correlation Core Module

提供灵活高效的互相关计算接口，支持：
- 时域 / 频域算法选择
- 多种归一化与滤波预处理
- 单道对或多道批量输入
- 返回标准 CCF 结构（lags, ccf）

不包含文件 I/O 或任务调度 —— 这些由 pipeline 层管理。
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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

from scipy.fftpack import fft, ifft, next_fast_len

from seismocorr.preprocessing.freq_norm import get_freq_normalizer
from seismocorr.preprocessing.normal_func import bandpass
from seismocorr.preprocessing.time_norm import get_time_normalizer

# -----------------------------# 类型定义# -----------------------------
ArrayLike = Union[np.ndarray, List[float]]
LagsAndCCF = Tuple[np.ndarray, np.ndarray]
BatchResult = Dict[str, LagsAndCCF]  # {channel_pair: (lags, ccf)}

# -----------------------------# 核心算法枚举# -----------------------------
SUPPORTED_METHODS = ["time-domain", "freq-domain", "deconv", "coherency"]
NORMALIZATION_OPTIONS = ["zscore", "one-bit", "rms", "no"]


# -----------------------------# 辅助函数# -----------------------------
def list_available_normalization_methods() -> Dict[str, List[str]]:
    """
    列出所有可用的归一化方法
    
    Returns:
        Dict[str, List[str]]: 包含时域和频域归一化方法的字典
    """
    return {
        "time-domain": ['zscore', 'one-bit', 'rms', 'clip', 'no', 'ramn', 'waterlevel', 'cwt-soft'],
        "freq-domain": ['no', 'whiten', 'rma', 'bandwhiten', 'powerlaw', 'bandwise', 'refspectrum', 'clipwhiten']
    }


def get_normalization_method_details(method_type: str, method_name: str) -> Dict[str, Any]:
    """
    获取特定归一化方法的详细信息，包括所需参数
    
    Args:
        method_type: 方法类型 ('time-domain' 或 'freq-domain')
        method_name: 方法名称
    
    Returns:
        Dict[str, Any]: 包含方法描述和参数信息的字典
    """
    # 时域归一化方法详情
    time_method_details = {
        'zscore': {
            'description': 'Z-Score 标准化，将信号转换为均值为0、标准差为1',
            'required_params': [],
            'optional_params': {} 
        },
        'one-bit': {
            'description': '1-bit 归一化，将信号转换为±1，仅保留相位信息',
            'required_params': [],
            'optional_params': {} 
        },
        'rms': {
            'description': 'RMS 归一化，将信号除以其均方根值',
            'required_params': [],
            'optional_params': {} 
        },
        'clip': {
            'description': '截幅归一化，将信号值限制在指定范围内',
            'required_params': [],
            'optional_params': {'clip_val': '截幅值，默认: 3.0'} 
        },
        'no': {
            'description': '无归一化，直接使用原始信号',
            'required_params': [],
            'optional_params': {} 
        },
        'ramn': {
            'description': 'RAM 归一化，使用移动平均进行归一化',
            'required_params': ['fmin', 'Fs'],
            'optional_params': {'norm_win': '归一化窗口大小，默认: 0.5'} 
        },
        'waterlevel': {
            'description': '水位归一化，按窗口 RMS 做"截顶"缩放',
            'required_params': ['Fs'],
            'optional_params': {
                'win_length': '窗口长度（秒），默认: 10.0',
                'water_level_factor': '水位系数，默认: 1.0',
                'n_iter': '迭代次数，默认: 1'
            } 
        },
        'cwt-soft': {
            'description': 'CWT 软阈值归一化，基于连续小波变换进行信号处理',
            'required_params': ['Fs', 'noise_idx'],
            'optional_params': {
                'mode': '处理模式 (designal/denoise)，默认: designal',
                'wavelet': '小波名称，默认: cmor1.5-1.0'
            } 
        }
    }
    
    # 频域归一化方法详情
    freq_method_details = {
        'no': {
            'description': '无频域归一化',
            'required_params': [],
            'optional_params': {} 
        },
        'whiten': {
            'description': '谱白化，平坦化信号的频谱',
            'required_params': [],
            'optional_params': {'smooth_win': '平滑窗口大小，默认: 20'} 
        },
        'rma': {
            'description': '递归移动平均白化，使用递归平均进行频谱平滑',
            'required_params': [],
            'optional_params': {'alpha': '平滑系数，默认: 0.9'} 
        },
        'bandwhiten': {
            'description': '频带白化，仅在指定频带内进行白化',
            'required_params': ['freq_min', 'freq_max', 'Fs'],
            'optional_params': {} 
        },
        'powerlaw': {
            'description': '幂谱白化，使用幂律函数调整频谱',
            'required_params': [],
            'optional_params': {'alpha': '幂指数，默认: 0.5'} 
        },
        'bandwise': {
            'description': '频带分段归一化，将频谱划分为多个子频带进行归一化',
            'required_params': ['bands', 'Fs'],
            'optional_params': {'method': '归一化方法 (rms/mean)，默认: rms'} 
        },
        'refspectrum': {
            'description': '参考谱归一化，使用参考频谱进行归一化',
            'required_params': ['ref_spectrum'],
            'optional_params': {} 
        },
        'clipwhiten': {
            'description': '自适应频谱截断白化，对白化权重设置上下限',
            'required_params': [],
            'optional_params': {
                'smooth_win': '平滑窗口大小，默认: 20',
                'min_weight': '最小权重，默认: 0.1',
                'max_weight': '最大权重，默认: 10.0'
            } 
        }
    }
    
    if method_type == 'time-domain':
        if method_name not in time_method_details:
            raise ValueError(f"不支持的时域归一化方法: {method_name}")
        return time_method_details[method_name]
    elif method_type == 'freq-domain':
        if method_name not in freq_method_details:
            raise ValueError(f"不支持的频域归一化方法: {method_name}")
        return freq_method_details[method_name]
    else:
        raise ValueError(f"不支持的方法类型: {method_type}")


class CorrelationConfig:
    """
    互相关计算配置类
    
    可用的时域归一化方法：
    - 'zscore': Z-Score 标准化，参数：无
    - 'one-bit': 1-bit 归一化，参数：无
    - 'rms': RMS 归一化，参数：无
    - 'clip': 截幅归一化，参数：clip_val (默认: 3.0)
    - 'no': 无归一化，参数：无
    - 'ramn': RAM 归一化，参数：fmin, Fs, norm_win (默认: 0.5)
    - 'waterlevel': 水位归一化，参数：Fs, win_length (默认: 10.0), water_level_factor (默认: 1.0), n_iter (默认: 1)
    - 'cwt-soft': CWT 软阈值归一化，参数：Fs, noise_idx, mode (默认: 'designal'), wavelet (默认: 'cmor1.5-1.0')
    
    可用的频域归一化方法：
    - 'no': 无归一化，参数：无
    - 'whiten': 谱白化，参数：smooth_win (默认: 20)
    - 'rma': 递归移动平均白化，参数：alpha (默认: 0.9)
    - 'bandwhiten': 频带白化，参数：freq_min, freq_max, Fs
    - 'powerlaw': 幂谱白化，参数：alpha (默认: 0.5), eps (默认: 1e-10)
    - 'bandwise': 频带分段归一化，参数：bands, Fs, method (默认: 'rms')
    - 'refspectrum': 参考谱归一化，参数：ref_spectrum
    - 'clipwhiten': 自适应频谱截断白化，参数：smooth_win (默认: 20), min_weight (默认: 0.1), max_weight (默认: 10.0)
    """
    def __init__(
        self,
        method: str = "time-domain",
        time_normalize: str = "one-bit",
        freq_normalize: str = "no",
        freq_band: Optional[Tuple[float, float]] = None,
        max_lag: Optional[Union[float, int]] = None,
        nfft: Optional[int] = None,
        time_norm_kwargs: Optional[Dict[str, Any]] = None,
        freq_norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化互相关配置
        
        Args:
            method: 计算方法 ('time-domain', 'freq-domain', 'deconv', 'coherency')
            time_normalize: 时域归一化方法
            freq_normalize: 频域归一化方法
            freq_band: 带通滤波范围 (fmin, fmax)，单位 Hz
            max_lag: 最大滞后时间（秒）
            nfft: FFT长度
            time_norm_kwargs: 时域归一化参数，根据选择的方法不同，需要的参数不同
            freq_norm_kwargs: 频域归一化参数，根据选择的方法不同，需要的参数不同
        """
        self.method = method
        self.time_normalize = time_normalize
        self.freq_normalize = freq_normalize
        self.freq_band = freq_band
        self.max_lag = max_lag
        self.nfft = nfft
        self.time_norm_kwargs = time_norm_kwargs or {}
        self.freq_norm_kwargs = freq_norm_kwargs or {}
        
        # 验证配置
        self._validate()
    
    def _validate(self):
        """验证配置参数的有效性"""
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"不支持的计算方法: {self.method}。请从 {SUPPORTED_METHODS} 中选择")
        
        # 验证时域归一化方法
        valid_time_methods = ['zscore', 'one-bit', 'rms', 'clip', 'no', 'ramn', 'waterlevel', 'cwt-soft']
        if self.time_normalize not in valid_time_methods:
            raise ValueError(f"不支持的时域归一化方法: {self.time_normalize}。请从 {valid_time_methods} 中选择")
        
        # 验证频域归一化方法
        valid_freq_methods = ['no', 'whiten', 'rma', 'bandwhiten', 'powerlaw', 'bandwise', 'refspectrum', 'clipwhiten']
        if self.freq_normalize not in valid_freq_methods:
            raise ValueError(f"不支持的频域归一化方法: {self.freq_normalize}。请从 {valid_freq_methods} 中选择")
        
        # 验证时域归一化参数
        self._validate_time_norm_params()
        
        # 验证频域归一化参数
        self._validate_freq_norm_params()
    
    def _validate_time_norm_params(self):
        """验证时域归一化参数"""
        method = self.time_normalize
        kwargs = self.time_norm_kwargs
        
        if method == 'ramn':
            if 'fmin' not in kwargs or kwargs['fmin'] is None:
                raise ValueError(f"时域归一化方法 '{method}' 需要 'fmin' 参数")
            if 'Fs' not in kwargs or kwargs['Fs'] is None:
                raise ValueError(f"时域归一化方法 '{method}' 需要 'Fs' 参数")
        elif method == 'waterlevel':
            if 'Fs' not in kwargs or kwargs['Fs'] is None:
                raise ValueError(f"时域归一化方法 '{method}' 需要 'Fs' 参数")
        elif method == 'cwt-soft':
            if 'Fs' not in kwargs or kwargs['Fs'] is None:
                raise ValueError(f"时域归一化方法 '{method}' 需要 'Fs' 参数")
            if 'noise_idx' not in kwargs or kwargs['noise_idx'] is None:
                raise ValueError(f"时域归一化方法 '{method}' 需要 'noise_idx' 参数")
        elif method == 'clip':
            # clip_val 是可选参数，有默认值
            pass
    
    def _validate_freq_norm_params(self):
        """验证频域归一化参数"""
        method = self.freq_normalize
        kwargs = self.freq_norm_kwargs
        
        if method == 'bandwhiten':
            if 'freq_min' not in kwargs or kwargs['freq_min'] is None:
                raise ValueError(f"频域归一化方法 '{method}' 需要 'freq_min' 参数")
            if 'freq_max' not in kwargs or kwargs['freq_max'] is None:
                raise ValueError(f"频域归一化方法 '{method}' 需要 'freq_max' 参数")
            if 'Fs' not in kwargs or kwargs['Fs'] is None:
                raise ValueError(f"频域归一化方法 '{method}' 需要 'Fs' 参数")
        elif method == 'bandwise':
            if 'bands' not in kwargs or kwargs['bands'] is None:
                raise ValueError(f"频域归一化方法 '{method}' 需要 'bands' 参数")
            if 'Fs' not in kwargs or kwargs['Fs'] is None:
                raise ValueError(f"频域归一化方法 '{method}' 需要 'Fs' 参数")
        elif method == 'refspectrum':
            if 'ref_spectrum' not in kwargs or kwargs['ref_spectrum'] is None:
                raise ValueError(f"频域归一化方法 '{method}' 需要 'ref_spectrum' 参数")
        elif method in ['whiten', 'rma', 'powerlaw', 'clipwhiten']:
            # 这些方法的参数都是可选的，有默认值
            pass
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "method": self.method,
            "time_normalize": self.time_normalize,
            "freq_normalize": self.freq_normalize,
            "freq_band": self.freq_band,
            "max_lag": self.max_lag,
            "nfft": self.nfft,
            "time_norm_kwargs": self.time_norm_kwargs,
            "freq_norm_kwargs": self.freq_norm_kwargs,
        }


class SignalPreprocessor:
    """
    信号预处理类
    """
    @staticmethod
    @njit(cache=True, fastmath=True, nogil=True)
    def _apply_preprocessing(data: np.ndarray, window: int) -> np.ndarray:
        """
        使用Numba优化的预处理函数
        """
        # 去趋势（简化版本，仅去线性趋势）
        n = len(data)
        x = np.arange(n)
        mean_x = np.mean(x)
        mean_y = np.mean(data)

        # 计算斜率
        slope = np.sum((x - mean_x) * (data - mean_y)) / np.sum((x - mean_x) ** 2)
        intercept = mean_y - slope * mean_x

        # 去趋势
        detrended = data - (slope * x + intercept)

        # 去均值
        demeaned = detrended - np.mean(detrended)

        # 加窗
        if window > 0:
            # 创建汉宁窗
            han_window = np.hanning(2 * window)
            # 应用窗函数到数据两端
            demeaned[:window] *= han_window[:window]
            demeaned[-window:] *= han_window[window:]

        return demeaned
    
    @staticmethod
    def preprocess_signal(
        data: np.ndarray, 
        sampling_rate: float,
        freq_band: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        预处理单个信号
        
        Args:
            data: 输入信号
            sampling_rate: 采样率
            freq_band: 带通滤波范围
            
        Returns:
            预处理后的信号
        """
        # 跳过非常短的信号的预处理，避免除以零错误
        if len(data) > 2:
            window = max(1, int(len(data) * 0.05))
            data = SignalPreprocessor._apply_preprocessing(data, window)
        
        # 滤波（如果需要）
        if freq_band is not None:
            data = bandpass(data, freq_band[0], freq_band[1], sr=sampling_rate)
        
        return data
    
    @staticmethod
    def normalize_signal(
        data: np.ndarray, 
        sampling_rate: float,
        time_normalize: str = "one-bit",
        freq_normalize: str = "no",
        time_norm_kwargs: Optional[Dict[str, Any]] = None,
        freq_norm_kwargs: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        归一化信号
        
        Args:
            data: 输入信号
            sampling_rate: 采样率
            time_normalize: 时域归一化方法
            freq_normalize: 频域归一化方法
            time_norm_kwargs: 时域归一化参数
            freq_norm_kwargs: 频域归一化参数
            
        Returns:
            归一化后的信号
        """
        time_norm_kwargs = time_norm_kwargs or {}
        freq_norm_kwargs = freq_norm_kwargs or {}
        
        # 时域归一化
        time_norm_kwargs_with_fs = {**time_norm_kwargs, "Fs": sampling_rate, "npts": len(data)}
        normalizer = get_time_normalizer(time_normalize, **time_norm_kwargs_with_fs)
        data = normalizer.apply(data)
        
        # 频域归一化
        freq_norm_kwargs_with_fs = {**freq_norm_kwargs, "Fs": sampling_rate}
        normalizer = get_freq_normalizer(freq_normalize, **freq_norm_kwargs_with_fs)
        data = normalizer.apply(data)
        
        return data


class CorrelationEngine:
    """
    互相关计算引擎类
    """
    def __init__(self, config: Optional[CorrelationConfig] = None):
        """
        初始化互相关计算引擎
        
        Args:
            config: 互相关计算配置
        """
        self.config = config or CorrelationConfig()
        self.preprocessor = SignalPreprocessor()
    
    @staticmethod
    @njit(cache=True, fastmath=True, nogil=True)
    def _as_float_array(x: ArrayLike) -> np.ndarray:
        """
        转换为浮点数组
        """
        return np.asarray(x, dtype=np.float64).flatten()
    
    @staticmethod
    @njit(cache=True, fastmath=True, nogil=True)
    def _xcorr_time_domain(
        x: np.ndarray, y: np.ndarray, sr: float, max_lag: float
    ) -> LagsAndCCF:
        """
        时域互相关计算
        """
        # 确保输入信号长度相同
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]

        # 将秒转换为样本数
        max_lag_samples = int(max_lag * sr)

        # 限制最大滞后不超过信号长度
        max_lag_samples = min(max_lag_samples, min_len - 1)

        n = len(x)
        ccf_len = 2 * n - 1
        ccf_full = np.zeros(ccf_len)

        for i in range(ccf_len):
            # 计算当前滞后的起始和结束索引
            shift = i - (n - 1)
            if shift < 0:
                # 正lag（x领先y）
                start = -shift
                end = n
                x_segment = x[start:end]
                y_segment = y[: end - start]
            else:
                # 负lag（y领先x）
                start = 0
                end = n - shift
                x_segment = x[:end]
                y_segment = y[shift : shift + end]

            # 计算点积
            ccf_full[i] = np.sum(x_segment * y_segment)

        # 计算滞后对应的索引范围
        center = ccf_len // 2  # 零滞后对应的索引

        # 截取从 -max_lag_samples 到 +max_lag_samples 的部分
        start_idx = center - max_lag_samples
        end_idx = center + max_lag_samples + 1  # +1 确保包含max_lag_samples

        # 确保索引不越界
        start_idx = max(0, start_idx)
        end_idx = min(ccf_len, end_idx)

        # 提取互相关值
        ccf = ccf_full[start_idx:end_idx]

        # 计算对应的滞后时间（秒）
        lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
        lags = lags_samples / sr

        # 归一化互相关
        norm_factor = np.sqrt(np.sum(x**2) * np.sum(y**2))
        if norm_factor > 0:
            ccf = ccf / norm_factor

        return lags, ccf
    
    @staticmethod
    def _xcorr_freq_domain(
        x: np.ndarray,
        y: np.ndarray,
        sr: float,
        max_lag: float,
        nfft: Optional[int],
        deconv=False,
    ) -> LagsAndCCF:
        """
        频域互相关/去卷积计算
        """
        length = len(x)
        if nfft is None:
            nfft = next_fast_len(length)

        X = fft(x, n=nfft)
        Y = fft(y, n=nfft)

        if deconv:
            # Deconvolution: Y/X
            eps = np.median(np.abs(X)) * 1e-6
            Sxy = Y / (X + eps)
        else:
            # Cross-spectrum
            Sxy = np.conj(X) * Y

        ccf_full = np.real(ifft(Sxy))
        # 因为是循环互相关，需要移位
        ccf_shifted = np.fft.ifftshift(ccf_full)

        # 提取 ±max_lag 范围
        center = nfft // 2
        lag_in_samples = int(max_lag * sr)
        start = center - lag_in_samples
        end = center + lag_in_samples + 1
        lags = np.arange(-lag_in_samples, lag_in_samples + 1) / sr
        return lags, ccf_shifted[start:end]
    
    @staticmethod
    @njit(cache=True, fastmath=True, nogil=True)
    def _coherency(
        x: np.ndarray, y: np.ndarray, sr: float, max_lag: int, nfft: Optional[int]
    ) -> LagsAndCCF:
        """
        使用相干性作为权重的互相关（类似 PWS 的频域版本），提高互相关结果的可靠性
        """
        lags, ccf_raw = CorrelationEngine._xcorr_freq_domain(x, y, sr, max_lag, nfft, deconv=False)

        # 在频域计算相位一致性
        Cxy = fft(ccf_raw)
        phase = np.exp(1j * np.angle(Cxy))
        coh = np.abs(np.mean(phase)) ** 4  # 权重
        return lags, ccf_raw * coh
    
    def compute_cross_correlation(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sampling_rate: float,
        config: Optional[CorrelationConfig] = None
    ) -> LagsAndCCF:
        """
        计算两个时间序列的互相关函数（CCF）
        
        Args:
            x, y: 时间序列数据
            sampling_rate: 采样率 (Hz)
            config: 互相关计算配置，若提供则覆盖实例配置
            
        Returns:
            lags: 时间滞后数组 (单位：秒)
            ccf: 互相关函数值
        """
        # 使用提供的配置或实例配置
        current_config = config or self.config
        config_dict = current_config.to_dict()
        
        # 转换为浮点数组
        x = self._as_float_array(x)
        y = self._as_float_array(y)

        if len(x) == 0 or len(y) == 0:
            return np.array([]), np.array([])

        # 确定最大滞后
        max_lag = config_dict["max_lag"]
        if not max_lag:
            max_lag = min(len(x), len(y)) / sampling_rate

        # 对x和y进行预处理
        x = self.preprocessor.preprocess_signal(x, sampling_rate, config_dict["freq_band"])
        y = self.preprocessor.preprocess_signal(y, sampling_rate, config_dict["freq_band"])

        # 归一化
        x = self.preprocessor.normalize_signal(
            x, sampling_rate,
            config_dict["time_normalize"],
            config_dict["freq_normalize"],
            config_dict["time_norm_kwargs"],
            config_dict["freq_norm_kwargs"]
        )
        y = self.preprocessor.normalize_signal(
            y, sampling_rate,
            config_dict["time_normalize"],
            config_dict["freq_normalize"],
            config_dict["time_norm_kwargs"],
            config_dict["freq_norm_kwargs"]
        )

        # 截断到相同长度（避免后续处理中的不匹配）
        min_len = min(len(x), len(y))
        if len(x) > min_len:
            x = x[:min_len]
        if len(y) > min_len:
            y = y[:min_len]

        # 选择方法
        method = config_dict["method"]
        if method == "time-domain":
            lags, ccf = self._xcorr_time_domain(x, y, sampling_rate, max_lag)
        elif method in ["freq-domain", "deconv"]:
            lags, ccf = self._xcorr_freq_domain(
                x, y, sampling_rate, max_lag, config_dict["nfft"], deconv=method == "deconv"
            )
        elif method == "coherency":
            lags, ccf = self._coherency(x, y, sampling_rate, max_lag, config_dict["nfft"])
        else:
            raise ValueError(
                f"Unsupported method: {method}. Choose from {SUPPORTED_METHODS}"
            )

        return lags, ccf


class BatchCorrelator:
    """
    批量互相关计算类
    """
    def __init__(self, engine: Optional[CorrelationEngine] = None):
        """
        初始化批量互相关计算器
        
        Args:
            engine: 互相关计算引擎
        """
        self.engine = engine or CorrelationEngine()
    
    def batch_cross_correlation_sequential(
        self,
        traces: Dict[str, np.ndarray],
        pairs: List[Tuple[str, str]],
        sampling_rate: float,
        config: Optional[CorrelationConfig] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        顺序计算多个通道对之间的互相关
        
        Returns:
            lags: 一维数组，所有通道对共享的滞后时间
            ccfs: 二维数组，每行对应一个通道对的互相关函数值
            keys: 通道对名称列表，对应ccfs的每一行
        """
        valid_pairs = []
        valid_traces_a = []
        valid_traces_b = []
        
        # 首先筛选有效的通道对
        for a, b in pairs:
            if a in traces and b in traces:
                valid_pairs.append((a, b))
                valid_traces_a.append(traces[a])
                valid_traces_b.append(traces[b])
        
        if not valid_pairs:
            return np.array([]), np.array([]), []
        
        # 计算第一个通道对，获取lags
        a0, b0 = valid_pairs[0]
        lags, first_ccf = self.engine.compute_cross_correlation(
            valid_traces_a[0], valid_traces_b[0], sampling_rate, config
        )
        
        # 初始化结果数组
        n_pairs = len(valid_pairs)
        n_lags = len(lags)
        ccfs = np.zeros((n_pairs, n_lags), dtype=np.float64)
        ccfs[0] = first_ccf
        keys = [f"{a0}--{b0}"]
        
        # 计算剩余通道对
        for i in range(1, n_pairs):
            a, b = valid_pairs[i]
            _, ccf = self.engine.compute_cross_correlation(
                valid_traces_a[i], valid_traces_b[i], sampling_rate, config
            )
            ccfs[i] = ccf
            keys.append(f"{a}--{b}")

        return lags, ccfs, keys
    
    @staticmethod
    def _process_single_pair(args):
        """
        处理单个通道对的辅助函数
        """
        a, b, trace_a, trace_b, sampling_rate, config_dict = args
        engine = CorrelationEngine(CorrelationConfig(**config_dict))
        
        _, ccf = engine.compute_cross_correlation(
            trace_a, trace_b, sampling_rate
        )
        key = f"{a}--{b}"
        
        return key, ccf
    
    def batch_cross_correlation(
        self,
        traces: Dict[str, np.ndarray],
        pairs: List[Tuple[str, str]],
        sampling_rate: float,
        n_jobs: int = -1,
        parallel_backend: str = "auto",  # "auto", "process", "thread"
        config: Optional[CorrelationConfig] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        批量计算多个通道对之间的互相关（并行版本）
        
        Args:
            traces: 通道数据字典
            pairs: 通道对列表
            sampling_rate: 采样率
            n_jobs: 并行工作数，-1 表示使用所有CPU核心
            parallel_backend: 并行后端 ("auto", "process", "thread")
            config: 互相关计算配置
            
        Returns:
            lags: 一维数组，所有通道对共享的滞后时间
            ccfs: 二维数组，每行对应一个通道对的互相关函数值
            keys: 通道对名称列表，对应ccfs的每一行
        """
        # 使用提供的配置或实例配置
        current_config = config or self.engine.config
        config_dict = current_config.to_dict()
        
        # 优化1：小批量直接顺序执行，避免并行开销
        if n_jobs == 0 or n_jobs == 1 or len(pairs) <= 4:
            return self.batch_cross_correlation_sequential(
                traces, pairs, sampling_rate, current_config
            )

        # 优化3：提前验证所有通道对，避免运行时错误
        valid_tasks = []
        valid_pairs = []
        for a, b in pairs:
            if a not in traces:
                print(f"Warning: Trace for channel '{a}' not found, skipping pair ({a}, {b})")
                continue
            if b not in traces:
                print(f"Warning: Trace for channel '{b}' not found, skipping pair ({a}, {b})")
                continue
            valid_tasks.append((
                a, b, traces[a], traces[b], sampling_rate, config_dict
            ))
            valid_pairs.append((a, b))

        if not valid_tasks:
            return np.array([]), np.array([]), []
        
        # 计算第一个通道对，获取lags
        a0, b0, trace_a0, trace_b0, _, _ = valid_tasks[0]
        lags, _ = self.engine.compute_cross_correlation(
            trace_a0, trace_b0, sampling_rate, current_config
        )
        n_lags = len(lags)
        
        # 优化2：动态调整并行度，根据任务规模和硬件情况优化
        num_cores = mp.cpu_count()
        num_pairs = len(valid_pairs)

        if n_jobs == -1:
            # 根据任务规模动态调整核心数
            if num_pairs < num_cores:
                # 任务数少于核心数，使用任务数作为并行度
                n_jobs = num_pairs
            elif num_pairs < num_cores * 2:
                # 任务数适中，使用所有核心
                n_jobs = num_cores
            else:
                # 任务数较多，使用核心数的2倍，充分利用硬件资源
                n_jobs = min(int(num_cores * 2), num_pairs)

        # 限制最大并行度，避免过多的线程/进程切换开销
        n_jobs = min(n_jobs, num_pairs, num_cores * 4)

        # 智能选择最佳并行后端
        avg_data_size = 0
        for a, b, trace_a, trace_b, _, _ in valid_tasks:
            avg_data_size += len(trace_a) + len(trace_b)
        avg_data_size /= len(valid_tasks)

        # 根据方法类型判断任务特性
        method = config_dict.get("method", "time-domain")
        is_cpu_intensive = method in ["time-domain", "freq-domain"]

        # 自动选择最佳并行后端
        if parallel_backend == "auto":
            if is_cpu_intensive and avg_data_size < 5e5:
                # 计算密集型且数据量较小，优先使用进程池
                parallel_backend = "process"
            else:
                # 数据量较大或IO密集型，优先使用线程池
                parallel_backend = "thread"

        # 选择执行器
        if parallel_backend == "process":
            Executor = ProcessPoolExecutor
        else:
            Executor = ThreadPoolExecutor

        # 并行计算所有通道对的ccf
        pair_results = []
        with Executor(max_workers=n_jobs) as executor:
            # 使用map代替submit，减少任务创建和管理开销
            pair_results = list(executor.map(self._process_single_pair, valid_tasks))
        
        # 整理结果
        keys = []
        ccfs = np.zeros((num_pairs, n_lags), dtype=np.float64)
        
        for i, (key, ccf) in enumerate(pair_results):
            if key is not None and ccf is not None:
                keys.append(key)
                ccfs[i] = ccf

        return lags, ccfs, keys

