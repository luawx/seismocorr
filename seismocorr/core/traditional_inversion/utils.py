# seismocorr/core/traditional_inversion/utils.py
'''
辅助工具函数
'''
from typing import Optional, List, Dict, Tuple

import numpy as np
from scipy.linalg import norm

from .types import ArrayLike, DispersionCurve, InversionParams

# 常量定义
DEFAULT_FITNESS_WEIGHT = 1.0
DEFAULT_RESIDUAL_THRESHOLD = 1e-4


def validate_inversion_params(params: InversionParams) -> None:
    """
    验证反演参数的合法性

    Args:
        params: 反演参数字典，需包含'thickness'和'vs'等关键键值

    Raises:
        ValueError: 参数缺失或数值不合法时抛出
        TypeError: 参数类型非数组时抛出
    """
    required_keys = ['thickness', 'vs']
    for key in required_keys:
        if key not in params:
            raise ValueError(f"反演参数缺失关键键值: {key}，必须包含{required_keys}")
    
    for key, value in params.items():
        arr = np.asarray(value)
        if arr.ndim != 1:
            raise TypeError(f"参数{key}必须为一维数组，当前维度: {arr.ndim}")
        if np.any(arr <= 0) and key in ['vs']:
            raise ValueError(f"参数{key}必须为正数，当前存在非正值")

'''
def calculate_residual(
    observed_curve: DispersionCurve,
    synthetic_curve: DispersionCurve
) -> Tuple[np.ndarray, float]:
    """
    计算观测与合成频散曲线的【有效逐点残差数组】和【残差L2范数】
    先匹配二者共有频率，再过滤nan/inf无效值，保证计算合理性，同时适配DLS和GA的需求

    Args:
        observed_curve: 观测曲线 (freq_obs, vel_obs)
        synthetic_curve: 合成曲线 (freq_syn, vel_syn)

    Returns:
        residual: 过滤后的逐点残差数组（vel_syn - vel_obs），仅包含共有频率，DLS核心计算用
        residual_norm: 残差L2范数（标量），GA适应度/收敛判断/阻尼调整用，无有效点时为inf
    """
    # 提取数据并转换为numpy数组，统一数据类型
    freq_obs, vel_obs = np.asarray(observed_curve[0]), np.asarray(observed_curve[1])
    freq_syn, vel_syn = np.asarray(synthetic_curve[0]), np.asarray(synthetic_curve[1])

    # 核心修改：提取共有频率，并获取其在原数组中的索引（关键步骤）
    # intersect1d返回有序的共有频率，return_indices=True返回对应原数组的索引
    common_freq, idx_obs, idx_syn = np.intersect1d(
        freq_obs, freq_syn, return_indices=True
    )
    
    # 无共有频率时，直接返回空数组+无穷大范数（边界场景处理）
    if len(common_freq) == 0:
        return np.array([]), float(np.inf)

    # 根据共有频率索引，提取对应位置的观测/合成速度
    vel_obs_common = vel_obs[idx_obs]
    vel_syn_common = vel_syn[idx_syn]

    # 过滤无效值（nan/inf），仅保留共有频率下的有效频散点（复用原核心逻辑）
    valid_mask = ~(np.isnan(vel_obs_common) | np.isinf(vel_obs_common) |
                   np.isnan(vel_syn_common) | np.isinf(vel_syn_common))
    if not np.any(valid_mask):
        return np.array([]), float(np.inf)  # 无有效点时返回空数组+无穷大范数

    # 计算共有频率下的有效逐点残差和L2范数
    residual = vel_syn_common[valid_mask] - vel_obs_common[valid_mask]
    residual_norm = np.linalg.norm(residual)

    return residual, residual_norm 
'''
'''
def calculate_residual(
    observed_curve: DispersionCurve,
    synthetic_curve: DispersionCurve
) -> Tuple[np.ndarray, float]:
    """
    计算观测与合成频散曲线的【有效逐点残差数组】和【残差L2范数】
    以观测频率freq_obs为基准，补齐合成速度vel_syn（缺失频率速度设0），再过滤nan/inf无效值
    保证计算合理性，同时适配DLS和GA的需求

    Args:
        observed_curve: 观测曲线 (freq_obs, vel_obs)，作为频率基准
        synthetic_curve: 合成曲线 (freq_syn, vel_syn)，按观测频率补齐

    Returns:
        residual: 过滤后的逐点残差数组（vel_syn_padded - vel_obs），DLS核心计算用
        residual_norm: 残差L2范数（标量），GA适应度/收敛判断/阻尼调整用，无有效点时为inf
    """
    # 提取数据并转换为numpy数组，统一数据类型
    freq_obs, vel_obs = np.asarray(observed_curve[0]), np.asarray(observed_curve[1])
    freq_syn, vel_syn = np.asarray(synthetic_curve[0]), np.asarray(synthetic_curve[1])

    # 核心修改：以freq_obs为基准补齐vel_syn，缺失频率速度设0（处理长度/频率不一致）
    # 1. 创建与freq_obs等长的全0数组，作为补齐后的合成速度初始值
    vel_syn_padded = np.zeros_like(freq_obs, dtype=np.float64)
    # 2. 浮点数精度安全匹配：找到freq_obs中在freq_syn里的频率索引（避免浮点数==匹配误差）
    # isclose广播匹配，any(axis=1)判断每个观测频率是否在合成频率中存在
    freq_match_mask = np.any(np.isclose(freq_obs[:, None], freq_syn[None, :]), axis=1)
    # 3. 对匹配到的频率，找到其在freq_syn中的对应索引，提取速度并填充
    if np.any(freq_match_mask):
        # 找到匹配的观测频率在合成频率中的索引
        match_syn_idx = [np.where(np.isclose(f, freq_syn))[0][0] for f in freq_obs[freq_match_mask]]
        # 将合成速度填充到补齐数组的对应位置
        vel_syn_padded[freq_match_mask] = vel_syn[match_syn_idx]

    # 过滤无效值（nan/inf），仅保留有效频散点（复用原核心逻辑）
    valid_mask = ~(np.isnan(vel_obs) | np.isinf(vel_obs) | 
                   np.isnan(vel_syn_padded) | np.isinf(vel_syn_padded))
    if not np.any(valid_mask):
        return np.array([]), float(np.inf)  # 无有效点时返回空数组+无穷大范数

    # 计算补齐后的有效逐点残差（数组）和L2范数（标量）
    residual = vel_syn_padded[valid_mask] - vel_obs[valid_mask]
    residual_norm = np.linalg.norm(residual)

    return residual, residual_norm  # 返回双值元组，保持原接口适配性
'''

def calculate_residual(
    observed_curve: DispersionCurve,
    synthetic_curve: DispersionCurve
) -> Tuple[np.ndarray, float]:
    """
    计算观测与合成频散曲线的【有效逐点残差数组】和【残差L2范数】
    ✅ 有效残差：筛选二者共有频率，过滤nan/inf后计算（合成-观测）
    ✅ L2范数：以观测频率freq_obs为基准，补齐合成速度vel_syn（缺失频率速度设0），过滤无效值后计算
    兼顾计算合理性与反演需求，适配DLS（用残差数组）和GA（用L2范数）的不同使用场景

    Args:
        observed_curve: 观测曲线 (freq_obs, vel_obs)，作为补0基准
        synthetic_curve: 合成曲线 (freq_syn, vel_syn)

    Returns:
        residual: 共有频率下的有效逐点残差数组，DLS核心计算用，无有效共有点时返回空数组
        residual_norm: 补0规则下的残差L2范数（标量），GA适应度/收敛判断/阻尼调整用，无有效点时为inf
    """
    # 1. 基础数据提取：转NumPy数组，统一数据类型（兼容列表/元组输入）
    freq_obs, vel_obs = np.asarray(observed_curve[0], dtype=np.float64), np.asarray(observed_curve[1], dtype=np.float64)
    freq_syn, vel_syn = np.asarray(synthetic_curve[0], dtype=np.float64), np.asarray(synthetic_curve[1], dtype=np.float64)

    # -------------------------- 规则1：计算【有效点残差】- 筛选共有频率 --------------------------
    # 浮点数精度安全匹配：找到观测频率在合成频率中的匹配掩码（避免==的精度误差）
    # 广播实现多对多匹配，any(axis=1)判断每个观测频率是否存在于合成频率中
    obs_in_syn_mask = np.any(np.isclose(freq_obs[:, None], freq_syn[None, :]), axis=1)
    # 合成频率在观测频率中的匹配掩码（反向匹配，保证双向共有）
    syn_in_obs_mask = np.any(np.isclose(freq_syn[:, None], freq_obs[None, :]), axis=1)
    
    # 提取共有频率对应的观测/合成速度
    vel_obs_common = vel_obs[obs_in_syn_mask]
    vel_syn_common = vel_syn[syn_in_obs_mask]
    
    # 过滤共有频率下的无效值（nan/inf）
    common_valid_mask = ~(np.isnan(vel_obs_common) | np.isinf(vel_obs_common) |
                          np.isnan(vel_syn_common) | np.isinf(vel_syn_common))
    # 计算有效点残差（无有效共有点时返回空数组）
    residual = vel_syn_common[common_valid_mask] - vel_obs_common[common_valid_mask] if np.any(common_valid_mask) else np.array([])

    # -------------------------- 规则2：计算【L2范数】- 观测频率基准补0 --------------------------
    # 初始化补0数组：与观测频率等长，缺失频率速度默认0
    vel_syn_padded = np.zeros_like(freq_obs, dtype=np.float64)
    # 利用已计算的obs_in_syn_mask，填充匹配频率的合成速度
    if np.any(obs_in_syn_mask):
        # 找到匹配观测频率在合成频率中的对应索引
        match_syn_idx = [np.where(np.isclose(f, freq_syn))[0][0] for f in freq_obs[obs_in_syn_mask]]
        vel_syn_padded[obs_in_syn_mask] = vel_syn[match_syn_idx]
    
    # 过滤补0后数组的无效值（nan/inf，0为有效值不会被过滤）
    padded_valid_mask = ~(np.isnan(vel_obs) | np.isinf(vel_obs) |
                          np.isnan(vel_syn_padded) | np.isinf(vel_syn_padded))
    # 计算补0规则下的L2范数（无有效点时返回inf，适配反演收敛判断）
    if np.any(padded_valid_mask):
        padded_residual = vel_syn_padded[padded_valid_mask] - vel_obs[padded_valid_mask]
        residual_norm = np.linalg.norm(padded_residual)
    else:
        residual_norm = float(np.inf)

    # 保持原接口：返回（有效残差数组，L2范数标量）
    return residual, residual_norm
    
def calculate_fitness(
    residual_norm: float,
    weight: float = DEFAULT_FITNESS_WEIGHT
) -> float:
    """
    计算遗传算法的适应度值（残差越小，适应度越高）

    Args:
        residual_norm: 残差的L2范数
        weight: 适应度权重

    Returns:
        fitness: 适应度值
    """
    # 避免除零，添加极小值
    fitness = weight / (residual_norm + 1e-8)
    return fitness


def normalize_params(
    params: InversionParams,
    bounds: Dict[str, List[Tuple[float, float]]]  # 修正：bounds值为二元组列表
) -> InversionParams:
    """
    将反演参数归一化到[0,1]区间（适配分层边界：逐层数按独立边界归一化）
    层数由bounds中thickness/vs的列表长度决定，params数组长度需与边界列表长度一致

    Args:
        params: 原始反演参数，如{'thickness': [10,10,...0], 'vs': [3.5,3.4,...4.75]}
        bounds: 各参数的分层上下界，如{'thickness': [(0,5), (0,5),...], 'vs': [(2,3), (2,3),...]}

    Returns:
        normalized_params: 归一化后的参数（数组长度与边界列表一致）
    """
    normalized_params = {}
    for key, value in params.items():
        # 原有异常：参数未定义上下界
        if key not in bounds:
            raise ValueError(f"参数{key}未定义上下界")
        # 转换为numpy数组，方便逐元素处理
        arr = np.asarray(value, dtype=np.float64)
        # 分层边界列表
        layer_bounds = bounds[key]
        # 新增校验：参数数组长度需与分层边界列表长度一致（层数匹配）
        if len(arr) != len(layer_bounds):
            raise ValueError(
                f"参数{key}数组长度({len(arr)})与分层边界列表长度({len(layer_bounds)})不匹配，层数需一致"
            )
        # 初始化归一化数组
        normalized = np.zeros_like(arr)
        # 逐层数按独立边界归一化
        for i in range(len(layer_bounds)):
            lower, upper = layer_bounds[i]
            # 避免除零错误（边界上下限相同，如半空间初始配置(0,0)）
            if upper - lower < 1e-12:
                normalized[i] = 0.0
            else:
                normalized[i] = (arr[i] - lower) / (upper - lower)
        normalized_params[key] = normalized
    return normalized_params


def denormalize_params(
    normalized_params: InversionParams,
    bounds: Dict[str, List[Tuple[float, float]]]  # 修正：bounds值为二元组列表
) -> InversionParams:
    """
    将归一化的参数反归一化到原始物理区间（适配分层边界）
    强制规则：thickness最后1层反归一化后为0.0（实现半空间，无视边界配置）

    Args:
        normalized_params: 归一化后的参数（[0,1]区间）
        bounds: 各参数的分层上下界

    Returns:
        denormalized_params: 反归一化后的物理参数
    """
    denormalized_params = {}
    for key, value in normalized_params.items():
        # 原有异常：参数未定义上下界
        if key not in bounds:
            raise ValueError(f"参数{key}未定义上下界")
        # 转换为numpy数组
        arr = np.asarray(value, dtype=np.float64)
        # 分层边界列表
        layer_bounds = bounds[key]
        # 新增校验：参数数组长度需与分层边界列表长度一致
        if len(arr) != len(layer_bounds):
            raise ValueError(
                f"参数{key}归一化数组长度({len(arr)})与分层边界列表长度({len(layer_bounds)})不匹配，层数需一致"
            )
        # 初始化反归一化数组
        denormalized = np.zeros_like(arr)
        # 逐层数按独立边界反归一化
        for i in range(len(layer_bounds)):
            lower, upper = layer_bounds[i]
            denormalized[i] = arr[i] * (upper - lower) + lower
        # 核心强制规则：thickness最后1层设为0.0，实现半空间
        if key == "thickness":
            denormalized[-1] = 0.0
        denormalized_params[key] = denormalized
    return denormalized_params