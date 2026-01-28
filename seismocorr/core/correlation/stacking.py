# seismocorr/core/stacking.py

"""
Staking Strategies for Cross-Correlation Functions (CCFs)

支持多种叠加方法，用于提升信噪比（SNR）。
所有方法均接受一个 CCF 列表（List[np.ndarray]），返回一个叠加后的 CCF。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Callable
from scipy.signal import hilbert

# 类型别名
ArrayLike = Union[np.ndarray, List[float], List[np.ndarray]]


class StackingStrategy(ABC):
    """
    叠加策略抽象基类
    所有具体策略需继承并实现 stack 方法
    """

    @abstractmethod
    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        """
        对一组 CCF 进行叠加

        Args:
            ccf_list: 多个互相关函数，形状应相同 [n_lags]

        Returns:
            stacked_ccf: 叠加后的互相关函数
        """
        pass

    def __call__(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        if not isinstance(ccf_list, (list, tuple)):
            raise TypeError(f"ccf_list 必须是 list/tuple，当前为 {type(ccf_list).__name__}")

        if len(ccf_list) == 0:
            raise ValueError("ccf_list 不能为空")

        checked = []
        n_lags = None
        for i, c in enumerate(ccf_list):
            a = np.asarray(c)

            if a.ndim != 1:
                raise ValueError(f"ccf_list[{i}] 类型应为一维数组，当前 shape={a.shape}")
            if a.size == 0:
                raise ValueError(f"ccf_list[{i}] 不能为空数组")
            if not np.issubdtype(a.dtype, np.number):
                raise TypeError(f"ccf_list[{i}] dtype 类型应为数值类型，当前为 {a.dtype}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"ccf_list[{i}] 包含 NaN/Inf")
            if n_lags is None:
                n_lags = a.shape[0]
            elif a.shape[0] != n_lags:
                raise ValueError(
                    f"ccf_list 中各 CCF 长度需一致：期望 {n_lags}，但 ccf_list[{i}] 长度为 {a.shape[0]}"
                )
            checked.append(a.astype(np.float64, copy=False))

        out = self.stack(checked)
        out = np.asarray(out, dtype=np.float64)
        if out.ndim != 1:
            raise RuntimeError(f"stack 输出形状应为一维数组，当前 shape={out.shape}")
        if out.shape[0] != n_lags:
            raise RuntimeError(f"stack 输出长度应为 {n_lags}，当前为 {out.shape[0]}")
        if not np.all(np.isfinite(out)):
            raise RuntimeError("stack 输出包含 NaN/Inf")

        return out


class LinearStack(StackingStrategy):
    """线性叠加：最简单的平均"""
    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        arr = np.array(ccf_list)
        return np.mean(arr, axis=0)


class SelectiveStack(StackingStrategy):
    """选择叠加：将与平均值相关性低的剔除后再叠加"""
    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        arr = np.array(ccf_list)
        cc = np.ones(arr.shape[0])
        new_stack = np.mean(arr, axis=0)
        for i in range(arr.shape[0]):
            cc[i] = np.corrcoef(new_stack, arr[i])[0, 1]
        epsilon = np.median(cc)
        ik = np.where(cc>=epsilon)[0]
        new_stack = np.mean(arr[ik,:], axis=0)
        return new_stack

class NrootStack(StackingStrategy):
    """N次根叠加"""
    def __init__(self):
        self.power = 2

    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        arr = np.array(ccf_list)
        N, M = arr.shape
        dout = np.zeros(M,dtype=np.float32)

        for i in range(N):
            dat = arr[i,:]
            dout += np.sign(dat) * (np.abs(dat))**(1.0/self.power)
        dout /= N
        nstack = dout * np.abs(dout)**(self.power - 1.0)
        return nstack
        
    
class PhaseWeightedStack(StackingStrategy):
    """
    相位加权叠加（PWS）
    Ref: Schimmel and Palssen, 1997
    使用相位一致性作为权重：一致性越高，权重越大

    参数:
        power: 相位一致性的幂次，用于调整权重的非线性程度
    """
    def __init__(self,power=2):
        self.power = power

    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        ccfs = np.array(ccf_list)
        N,M = ccfs.shape
        # 计算解析信号，获取相位信息
        analytic = hilbert(ccfs, axis=1)  # 不使用next_fast_len，直接使用原始长度
        phase = np.angle(analytic)
        # 计算相位一致性
        phase_stack = np.mean(np.exp(1j*phase), axis=0)
        phase_stack = np.abs(phase_stack)**self.power
        # 应用相位权重并叠加
        weighted = np.multiply(ccfs, phase_stack)
        return np.mean(weighted, axis=0)


class RobustStack(StackingStrategy):
    """
    鲁棒叠加（Robust Stack）
    REF: Palvis and Vernon, 2010
    """
    def __init__(self,epsilon: float = 1e-8):
        self.epsilon = epsilon  # 防止除零
    def stack(self, ccf_list: List[np.ndarray]) -> np.ndarray:
        res = 9e9
        ccfs = np.array(ccf_list)
        w = np.ones(ccfs.shape[0])
        nstep = 0
        newstack = np.median(ccfs, axis=0)
        while res > self.epsilon:
            stack = newstack
            for i in range(ccfs.shape[0]):
                crap = np.multiply(stack, ccfs[i,:].T)
                crap_dot = np.sum(crap)
                di_norm = np.linalg.norm(ccfs[i,:])
                ri = ccfs[i,:] - crap_dot * stack
                ri_norm = np.linalg.norm(ri)
                w[i] = np.abs(crap_dot) / di_norm / ri_norm
            w = w / np.sum(w)
            newstack = np.sum((w*ccfs.T).T, axis=0)
            res = np.linalg.norm(newstack - stack,ord = 1) / np.linalg.norm(newstack) / len(ccfs[:,1])
            nstep += 1
            
        return newstack


# =====================================================================
# 🏭 工厂函数：根据名称创建策略实例
# =====================================================================

_STRATEGY_REGISTRY = {
    'linear': LinearStack,
    'pws': PhaseWeightedStack,
    'robust': RobustStack,
    'nroot': NrootStack,
    'selective': SelectiveStack
}

def get_stacker(name: str, **kwargs) -> StackingStrategy:
    """
    工厂函数：根据名称获取叠加器实例

    Args:
        name: 叠加方法名，如 'linear', 'pws', 'robust'
        **kwargs: 传递给具体策略的参数（如 alpha, threshold）

    Returns:
        StackingStrategy 实例

    Raises:
        ValueError: 如果方法名不支持
    """
    if not isinstance(name, str):
        raise TypeError(f"name 类型应为 str，当前为 {type(name).__name__}: {name!r}")
    if not name.strip():
        raise ValueError("name 不能为空字符串")
    cls = _STRATEGY_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown stacking method: {name}. "
                       f"Choose from {list(_STRATEGY_REGISTRY.keys())}")
    
    # 特殊处理带参数的类
    if name.lower() == 'pws':
        return cls(power=kwargs.get('power', 2.0))
    elif name.lower() == 'robust':
        return cls(
            epsilon=kwargs.get('epsilon', 1.0*1e-8)
        )
    else:
        return cls()


# =====================================================================
# 🔧 辅助函数：直接对数组列表进行叠加（简化接口）
# =====================================================================

def stack_ccfs(ccf_list: List[np.ndarray], method: str = 'linear', **kwargs) -> np.ndarray:
    """
    快捷函数：直接对一组 CCF 执行叠加

    Example:
        stacked = stack_ccfs([ccf1, ccf2, ccf3], method='pws', alpha=4)

    Args:
        ccf_list: CCF 数组列表
        method: 叠加方法名
        **kwargs: 方法参数

    Returns:
        叠加后的 CCF
    """
    if not isinstance(method, str):
        raise TypeError(f"method 类型应为 str，当前为 {type(method).__name__}: {method!r}")
    if not method.strip():
        raise ValueError("method 不能为空字符串")

    stacker = get_stacker(method, **kwargs)
    return stacker(ccf_list)
