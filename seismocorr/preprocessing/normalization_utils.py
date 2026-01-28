
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from numba import njit

from seismocorr.preprocessing.freq_norm import get_freq_normalizer
from seismocorr.preprocessing.normal_func import bandpass
from seismocorr.preprocessing.time_norm import get_time_normalizer
from seismocorr.config.default import SUPPORTED_METHODS, NORMALIZATION_OPTIONS


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
    if not isinstance(method_type, str) or not method_type.strip():
        raise TypeError("method_type 必须是非空字符串")
    if not isinstance(method_name, str) or not method_name.strip():
        raise TypeError("method_name 必须是非空字符串")

    method_type = method_type.strip().lower()
    method_name = method_name.strip().lower()
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
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError(f"data 应为一维数组，当前 shape={data.shape}")

        sampling_rate = float(sampling_rate)
        if sampling_rate <= 0:
            raise ValueError("sampling_rate 应 > 0")

        if data.size == 0:
            return data.copy()

        if not np.all(np.isfinite(data)):
            raise ValueError("data 包含 NaN/Inf")

        if freq_band is not None:
            if (not isinstance(freq_band, (tuple, list))) or len(freq_band) != 2:
                raise TypeError("freq_band 必须是 (fmin, fmax) 二元组或长度为2的list")
            fmin, fmax = float(freq_band[0]), float(freq_band[1])
            if fmin <= 0 or fmax <= 0 or fmin >= fmax:
                raise ValueError("freq_band 需要满足 0 < fmin < fmax")
            if fmax >= 0.5 * sampling_rate:
                raise ValueError("freq_band 的 fmax 不能 >= Nyquist (0.5*sampling_rate)")


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
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError(f"data 应为一维数组，当前 shape={data.shape}")
        if data.size == 0:
            return data.copy()
        if not np.all(np.isfinite(data)):
            raise ValueError("data 包含 NaN/Inf")

        sampling_rate = float(sampling_rate)
        if sampling_rate <= 0:
            raise ValueError("sampling_rate 必须 > 0")

        if not isinstance(time_normalize, str) or not time_normalize.strip():
            raise TypeError("time_normalize 必须是非空字符串")
        if not isinstance(freq_normalize, str) or not freq_normalize.strip():
            raise TypeError("freq_normalize 必须是非空字符串")

        time_normalize = time_normalize.strip().lower()
        freq_normalize = freq_normalize.strip().lower()

        if time_norm_kwargs is not None and not isinstance(time_norm_kwargs, dict):
            raise TypeError("time_norm_kwargs 必须是 dict 或 None")
        if freq_norm_kwargs is not None and not isinstance(freq_norm_kwargs, dict):
            raise TypeError("freq_norm_kwargs 必须是 dict 或 None")

        time_norm_kwargs = time_norm_kwargs or {}
        freq_norm_kwargs = freq_norm_kwargs or {}
        
        # 时域归一化
        time_norm_kwargs_with_fs = {**time_norm_kwargs, "Fs": sampling_rate, "npts": len(data)}
        normalizer = get_time_normalizer(time_normalize, **time_norm_kwargs_with_fs)
        data = normalizer(data)
        
        # 频域归一化
        freq_norm_kwargs_with_fs = {**freq_norm_kwargs, "Fs": sampling_rate}
        normalizer = get_freq_normalizer(freq_normalize, **freq_norm_kwargs_with_fs)
        data = normalizer(data)
        
        return data