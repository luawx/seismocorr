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
from scipy.fftpack import fft, ifft, next_fast_len
import numpy as np
from seismocorr.config.default import SUPPORTED_METHODS

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


# -----------------------------# 类型定义# -----------------------------#
ArrayLike = Union[np.ndarray, List[float]]
LagsAndCCF = Tuple[np.ndarray, np.ndarray]
BatchResult = Dict[str, LagsAndCCF]  # {channel_pair: (lags, ccf)}


class CorrelationConfig:
    """
    互相关计算配置类
    """
    def __init__(
        self,
        method: str = "time-domain",
        max_lag: Optional[Union[float, int]] = None,
        nfft: Optional[int] = None,
    ):
        """
        初始化互相关配置
        
        Args:
            method: 计算方法 ('time-domain', 'freq-domain', 'deconv', 'coherency')
            max_lag: 最大滞后时间（秒）
            nfft: FFT长度
        """
        self.method = method
        self.max_lag = max_lag
        self.nfft = nfft
        
        # 验证配置
        self._validate()


    def _validate(self):
        """验证配置参数的有效性"""

        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"不支持的计算方法: {self.method!r}。请从 {SUPPORTED_METHODS} 中选择"
            )

        if self.max_lag is not None:
            if isinstance(self.max_lag, bool) or not isinstance(self.max_lag, (int, float)):
                raise TypeError(
                    f"max_lag 类型应为 float/int 或 None，当前为 {type(self.max_lag).__name__}: {self.max_lag!r}"
                )
            if not np.isfinite(float(self.max_lag)):
                raise ValueError(f"max_lag 应为有限数值，当前为: {self.max_lag!r}")
            if float(self.max_lag) < 0:
                raise ValueError(f"max_lag 应该 >= 0，当前为: {self.max_lag!r}")

        if self.nfft is not None:
            if isinstance(self.nfft, bool) or not isinstance(self.nfft, int):
                raise TypeError(
                    f"nfft 类型应为 int 或 None，当前为 {type(self.nfft).__name__}: {self.nfft!r}"
                )
            if self.nfft <= 0:
                raise ValueError(f"nfft 应为正整数，当前为: {self.nfft!r}")
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "method": self.method,
            "max_lag": self.max_lag,
            "nfft": self.nfft
        }


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
    def _coherency(
        x: np.ndarray,
        y: np.ndarray,
        sr: float,
        max_lag: float,
        nfft: Optional[int]
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

        if config is not None and not isinstance(config, CorrelationConfig):
            raise TypeError(f"config 类型应为 CorrelationConfig 或 None，当前为 {type(config).__name__}")
        # 使用提供的配置或实例配置
        current_config = config or self.config
        config_dict = current_config.to_dict()

        if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, (int, float)):
            raise TypeError(f"sampling_rate 类型有误，当前为 {type(sampling_rate).__name__}: {sampling_rate!r}")
        if not np.isfinite(float(sampling_rate)) or float(sampling_rate) <= 0:
            raise ValueError(f"sampling_rate 必须为有限且 > 0 的数(Hz)，当前为: {sampling_rate!r}")
        sampling_rate = float(sampling_rate)
        
        # 转换为浮点数组
        x = self._as_float_array(x)
        y = self._as_float_array(y)

        if len(x) == 0 or len(y) == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        if not np.all(np.isfinite(x)):
            raise ValueError("输入 x 包含 NaN/Inf")
        if not np.all(np.isfinite(y)):
            raise ValueError("输入 y 包含 NaN/Inf")

        # 确定最大滞后
        max_lag = config_dict["max_lag"]
        if max_lag is None:
            max_lag = min(len(x), len(y)) / sampling_rate
        else:
            max_lag = float(max_lag)

        if not np.isfinite(max_lag) or max_lag < 0:
            raise ValueError(f"max_lag 必须为有限且 >= 0 的秒数，当前为: {max_lag!r}")

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

        if len(lags) != len(ccf):
            raise RuntimeError(
                f"输出不一致：len(lags)={len(lags)} != len(ccf)={len(ccf)}。"
                f"method={method!r}, sampling_rate={sampling_rate!r}, max_lag={max_lag!r}"
            )
        lags = np.asarray(lags, dtype=np.float64)
        ccf = np.asarray(ccf, dtype=np.float64)

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

        if config is not None and not isinstance(config, CorrelationConfig):
            raise TypeError(f"config 类型应为 CorrelationConfig 或 None，当前为 {type(config).__name__}")
        if not isinstance(traces, dict):
            raise TypeError(f"traces 类型应为 Dict[str, np.ndarray]，当前为 {type(traces).__name__}")
        if not isinstance(pairs, list):
            raise TypeError(f"pairs 类型应为 List[Tuple[str,str]]，当前为 {type(pairs).__name__}")

        if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, (int, float)):
            raise TypeError(f"sampling_rate 类型有误，当前为 {type(sampling_rate).__name__}: {sampling_rate!r}")
        if not np.isfinite(float(sampling_rate)) or float(sampling_rate) <= 0:
            raise ValueError(f"sampling_rate 必须为有限且 > 0 的数，当前为: {sampling_rate!r}")
        sampling_rate = float(sampling_rate)

        for item in pairs:
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                raise TypeError(f"pairs 元素必须是长度为2的 tuple/list，当前为: {item!r}")
            if not (isinstance(item[0], str) and isinstance(item[1], str)):
                raise TypeError(
                    f"pairs 中通道名类型应为 str，当前为: ({type(item[0]).__name__}, {type(item[1]).__name__})")

        # 首先筛选有效的通道对
        for a, b in pairs:
            if a in traces and b in traces:
                valid_pairs.append((a, b))
                valid_traces_a.append(traces[a])
                valid_traces_b.append(traces[b])

        if not valid_pairs:
            return (
                np.array([], dtype=np.float64),
                np.zeros((0, 0), dtype=np.float64),
                []
            )

        
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
            if len(ccf) != n_lags:
                raise RuntimeError(
                    f"通道对 {a}--{b} 的 ccf 长度 {len(ccf)} 与 lags 长度 {n_lags} 不一致，无法组成统一 batch 输出"
                )
            ccfs[i] = ccf
            keys.append(f"{a}--{b}")

        lags = np.asarray(lags, dtype=np.float64)
        ccfs = np.asarray(ccfs, dtype=np.float64)

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

        if config is not None and not isinstance(config, CorrelationConfig):
            raise TypeError(f"config 类型应为 CorrelationConfig 或 None，当前为 {type(config).__name__}")
        # 使用提供的配置或实例配置
        current_config = config or self.engine.config
        config_dict = current_config.to_dict()

        if parallel_backend not in ("auto", "process", "thread"):
            raise ValueError(f"parallel_backend 类型应为 'auto'/'process'/'thread'，当前为: {parallel_backend!r}")

        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
            raise TypeError(f"n_jobs 类型应为 int，当前为 {type(n_jobs).__name__}: {n_jobs!r}")
        if not isinstance(traces, dict):
            raise TypeError(f"traces 类型应为 Dict[str, np.ndarray]，当前为 {type(traces).__name__}")
        if not isinstance(pairs, list):
            raise TypeError(f"pairs 类型应为 List[Tuple[str,str]]，当前为 {type(pairs).__name__}")

        if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, (int, float)):
            raise TypeError(f"sampling_rate 类型有误，当前为 {type(sampling_rate).__name__}: {sampling_rate!r}")
        if not np.isfinite(float(sampling_rate)) or float(sampling_rate) <= 0:
            raise ValueError(f"sampling_rate 必须为有限且 > 0 的数，当前为: {sampling_rate!r}")
        sampling_rate = float(sampling_rate)

        for item in pairs:
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                raise TypeError(f"pairs 元素必须是长度为2的 tuple/list，当前为: {item!r}")
            if not (isinstance(item[0], str) and isinstance(item[1], str)):
                raise TypeError(
                    f"pairs 中通道名类型应为 str，当前为: ({type(item[0]).__name__}, {type(item[1]).__name__})")
        
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
            return (
                np.array([], dtype=np.float64),
                np.zeros((0, 0), dtype=np.float64),
                []
            )
        
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

                if len(ccf) != n_lags:
                    raise RuntimeError(
                        f"{key} 的 ccf 长度 {len(ccf)} 与 lags 长度 {n_lags} 不一致，无法组成统一 batch 输出"
                    )
                ccfs[i] = ccf

        lags = np.asarray(lags, dtype=np.float64)
        ccfs = np.asarray(ccfs, dtype=np.float64)

        return lags, ccfs, keys