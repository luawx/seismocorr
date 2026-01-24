# seismocorr/core/dispersion_forward.py
"""
面波频散正演计算模块
基于 disba 库实现  (安装: pip install disba)
disba 优势：无需 Fortran 编译器，跨平台，性能优于传统 CPS 封装
"""
import numpy as np
from typing import Callable, Tuple, Dict

from disba import PhaseDispersion, GroupDispersion, DispersionError
from disba._helpers import is_sorted

from seismocorr.core.traditional_inversion.types import (
    InversionParams,
    DispersionCurve,
    ForwardFunc
)

def validate_velocity_model(thickness, vp, vs, density):
    """
    验证速度模型的合法性
    参数:
        thickness: 层厚度 (km)，array_like
        vp: 层P波速度 (km/s)，array_like
        vs: 层S波速度 (km/s)，array_like
        density: 层密度 (g/cm³)，array_like
    异常:
        ValueError: 模型参数不合法时抛出
    """
    # 转换为numpy数组
    thickness = np.asarray(thickness)
    vp = np.asarray(vp)
    vs = np.asarray(vs)
    density = np.asarray(density)

    # 检查维度一致
    if not (len(thickness) == len(vp) == len(vs) == len(density)):
        raise ValueError("速度模型各参数（厚度、Vp、Vs、密度）的层数必须一致")
    
    # 检查厚度非负（最后一层为半空间，厚度可设为0或任意非负值）
    if np.any(thickness[:-1] < 0):
        raise ValueError("除最后一层（半空间）外，层厚度必须非负")
    
    # 检查速度和密度为正
    if np.any(vp <= 0) or np.any(vs <= 0) or np.any(density <= 0):
        raise ValueError("P波速度、S波速度、密度必须为正值")
    
    return thickness, vp, vs, density


def compute_vp_from_vs_and_poisson(vs, sigma):
    """
    由横波速度和泊松比计算P波速度
    参数:
        vs: 横波速度 (km/s)，array_like
        sigma: 泊松比（0 ≤ σ < 0.5），float/array_like（可单层固定或分层设置）
    返回:
        vp: P波速度 (km/s)，numpy数组
    异常:
        ValueError: 泊松比超出合理范围（σ≥0.5或σ<0）时抛出
    """
    vs = np.asarray(vs)
    sigma = np.asarray(sigma)
    
    # 校验泊松比范围（避免分母0或根号内负数）
    if np.any(sigma >= 0.5) or np.any(sigma < 0):
        raise ValueError("泊松比σ需满足 0 ≤ σ < 0.5（避免公式分母为0或根号内负数）")
    
    # 计算vP/vs比值
    vp_vs_ratio = np.sqrt((1 - sigma) / (0.5 - sigma))
    # 计算vP
    vp = vs * vp_vs_ratio
    return vp


def compute_density_from_vp(vp):
    """
    由P波速度计算密度（经验公式：ρ=0.31*(vp*1000)^0.25）
    参数:
        vp: P波速度 (km/s)，array_like
    返回:
        density: 密度 (g/cm³)，numpy数组
    """
    vp = np.asarray(vp)
    density = 0.31 * (vp * 1000) ** 0.25
    return density


def compute_phase_dispersion(
    thickness,
    vp,
    vs,
    density,
    periods,
    wave_type="rayleigh",
    mode=0,
    algorithm="dunkin",
    dc=0.005
):
    """
    计算面波相速度频散曲线
    参数:
        thickness: 层厚度 (km)，array_like
        vp: 层P波速度 (km/s)，array_like
        vs: 层S波速度 (km/s)，array_like
        density: 层密度 (g/cm³)，array_like
        periods: 周期轴 (s)，array_like（需升序排列）
        wave_type: 波型，可选 "rayleigh"（瑞利波）、"love"（勒夫波）
        mode: 振型数，0为基阶，正整数为高阶
        algorithm: 瑞利波计算算法，可选 "dunkin"（默认）、"fast-delta"
        dc: 相速度根查找步长，默认 0.005
    返回:
        DispersionCurve: 包含 period/velocity/mode/wave/type 的命名元组
    异常:
        ValueError: 参数不合法时抛出
        DispersionError: 频散计算失败时抛出
    """
    # 验证输入模型
    thickness, vp, vs, density = validate_velocity_model(thickness, vp, vs, density)
    
    # 验证周期轴
    periods = np.asarray(periods)
    if not is_sorted(periods):
        raise ValueError("周期轴必须按升序排列")
    if np.any(periods <= 0):
        raise ValueError("周期必须为正值")
    
    # 初始化相速度计算类
    try:
        phase_disp = PhaseDispersion(
            thickness=thickness,
            velocity_p=vp,
            velocity_s=vs,
            density=density,
            algorithm=algorithm,
            dc=dc
        )
        # 计算相速度频散曲线
        dispersion_curve = phase_disp(periods, mode=mode, wave=wave_type)
        return dispersion_curve
    except DispersionError as e:
        raise DispersionError(f"相速度频散计算失败: {str(e)}")
    except Exception as e:
        raise ValueError(f"参数错误或计算异常: {str(e)}")


def compute_group_dispersion(
    thickness,
    vp,
    vs,
    density,
    periods,
    wave_type="rayleigh",
    mode=0,
    algorithm="dunkin",
    dc=0.005,
    dt=0.025
):
    """
    计算面波群速度频散曲线
    参数:
        thickness: 层厚度 (km)，array_like
        vp: 层P波速度 (km/s)，array_like
        vs: 层S波速度 (km/s)，array_like
        density: 层密度 (g/cm³)，array_like
        periods: 周期轴 (s)，array_like（需升序排列）
        wave_type: 波型，可选 "rayleigh"（瑞利波）、"love"（勒夫波）
        mode: 振型数，0为基阶，正整数为高阶
        algorithm: 瑞利波计算算法，可选 "dunkin"（默认）、"fast-delta"
        dc: 相速度根查找步长，默认 0.005
        dt: 计算群速度的频率增量 (%)，默认 0.025
    返回:
        DispersionCurve: 包含 period/velocity/mode/wave/type 的命名元组
    异常:
        ValueError: 参数不合法时抛出
        DispersionError: 频散计算失败时抛出
    """
    # 验证输入模型
    thickness, vp, vs, density = validate_velocity_model(thickness, vp, vs, density)
    
    # 验证周期轴
    periods = np.asarray(periods)
    if not is_sorted(periods):
        raise ValueError("周期轴必须按升序排列")
    if np.any(periods <= 0):
        raise ValueError("周期必须为正值")
    
    # 初始化群速度计算类
    try:
        group_disp = GroupDispersion(
            thickness=thickness,
            velocity_p=vp,
            velocity_s=vs,
            density=density,
            algorithm=algorithm,
            dc=dc,
            dt=dt
        )
        # 计算群速度频散曲线
        dispersion_curve = group_disp(periods, mode=mode, wave=wave_type)
        return dispersion_curve
    except DispersionError as e:
        raise DispersionError(f"群速度频散计算失败: {str(e)}")
    except Exception as e:
        raise ValueError(f"参数错误或计算异常: {str(e)}")


def forward_func(
    params: InversionParams,
    freq,
    sigma=0.25,
    wave_type="rayleigh",
    disp_type="phase",
    mode=0,
    algorithm="dunkin",
    dc=0.005,
    dt=0.025
):
    """
    降维后的正演模型封装（仅输入层厚度h和横波速度vs）
    参数:
        params: 反演参数，格式 {'thickness': 层厚度数组(km), 'vs': 横波速度数组(km/s)}
        freq : 频率数组
        sigma: 泊松比（可设为固定值或分层数组，默认0.25）
        wave_type: 波型 ("rayleigh"/"love")
        disp_type: 频散类型 ("phase"/"group")
        mode: 振型数（0为基阶）
        algorithm: 计算算法 ("dunkin"/"fast-delta")
        dc: 相速度根查找步长（默认0.005）
        dt: 群速度频率增量（默认0.025）
    返回:
         DispersionCurve
    异常:
        ValueError: 参数格式错误时抛出
        DispersionError: 频散计算失败时抛出
    """
    # --------------------------
    # 步骤1：解析降维后的模型参数
    # --------------------------
    periods = 1 / freq
    freq_order = np.argsort(periods)  # 周期升序对应的频率索引
    periods_sorted = np.sort(periods) # 正演要求周期升序

    # 校验输入参数完整性
    if "thickness" not in params or "vs" not in params:
        raise ValueError("InversionParams 必须包含 'thickness' 和 'vs' 键")
    
    # 解析层厚度和横波速度
    thickness = np.asarray(params["thickness"])  # 层厚度（对应 forward_func 的 h）
    vs = np.asarray(params["vs"])            # 横波速度
    
    # 校验数组长度一致（层数匹配）
    if len(thickness) != len(vs):
        raise ValueError(f"depth 长度({len(thickness)})与 vs 长度({len(vs)})必须一致")

    # --------------------------
    # 步骤2：经验公式推导vp和密度
    # --------------------------
    vp = compute_vp_from_vs_and_poisson(vs, sigma)  # 由vs+σ算vp
    density = compute_density_from_vp(vp)           # 由vp算密度

    # --------------------------
    # 步骤3：调用频散计算核心函数
    # --------------------------
    if disp_type == "phase":
        disp_curve = compute_phase_dispersion(
            thickness=thickness,
            vp=vp,
            vs=vs,
            density=density,
            periods=periods_sorted,
            wave_type=wave_type,
            mode=mode,
            algorithm=algorithm,
            dc=dc
        )
    elif disp_type == "group":
        disp_curve = compute_group_dispersion(
            thickness=thickness,
            vp=vp,
            vs=vs,
            density=density,
            periods=periods_sorted,
            wave_type=wave_type,
            mode=mode,
            algorithm=algorithm,
            dc=dc,
            dt=dt
        )
    else:
        raise ValueError(f"不支持的频散类型: {disp_type}")

    # 正演结果是按升序周期排列的，需映射回原始观测频率顺序
    vel_sorted = disp_curve.velocity
    vel_aligned = np.zeros_like(freq)
    vel_aligned[freq_order] = vel_sorted  # 按原始频率顺序填充速度
    # 
    return (freq, vel_aligned)

def make_forward_model(
    forward_func: ForwardFunc,
    observed_curve: DispersionCurve,
    sigma: float = 0.25,
    wave_type: str = "rayleigh",
    disp_type: str = "phase",
    mode: int = 0,
    algorithm: str = "dunkin",
    dc: float = 0.005,
    dt: float = 0.025
) -> Callable[[InversionParams], DispersionCurve]:
    """
    封装原始正演函数，生成符合反演框架要求的 forward_model 函数
    核心逻辑：提取观测频率 + 适配 InversionParams 输入格式
    
    Args:
        forward_func: 原始降维正演函数（如用户提供的 forward_func），需满足：
                      入参包含 model_params (h+vs 拼接数组)、freq (频率数组)，
                      输出为 DispersionCurve (频率, 相速度)
        observed_curve: 观测频散曲线，格式为 (观测频率数组, 观测速度数组)
        sigma: 泊松比（透传给 forward_func，默认0.25）
        wave_type: 波型（透传给 forward_func，默认"rayleigh"）
        disp_type: 频散类型（透传给 forward_func，默认"phase"）
        mode: 振型数（透传给 forward_func，默认0基阶）
        algorithm: 计算算法（透传给 forward_func，默认"dunkin"）
        dc: 相速度根查找步长（透传给 forward_func，默认0.005）
        dt: 群速度频率增量（透传给 forward_func，默认0.025）
    
    Returns:
        forward_model: 适配反演框架的正演函数，入参为 InversionParams，输出为 DispersionCurve
    """
    # --------------------------
    # 步骤1：提取观测曲线中的频率（核心：固定正演的频率轴）
    # --------------------------
    obs_freq, _ = observed_curve  # 从观测曲线中提取频率数组
    if not isinstance(obs_freq, np.ndarray):
        obs_freq = np.asarray(obs_freq)
    if np.any(obs_freq <= 0):
        raise ValueError("观测频率必须为正值")

    # --------------------------
    # 步骤2：定义适配反演框架的 forward_model 函数
    # --------------------------
    def forward_model(params: InversionParams) -> DispersionCurve:
        """
        适配反演框架的正演函数（最终返回的函数）
        Args:
            params: 反演参数，格式 {'thickness': 层厚度数组(km), 'vs': 横波速度数组(km/s)}
        Returns:
            DispersionCurve: 合成频散曲线 (频率数组, 相速度数组)
        """
        # --------------------------
        # 调用原始正演函数计算频散曲线
        # --------------------------
        synthetic_curve = forward_func(
            params=params,
            freq=obs_freq,
            sigma=sigma,
            wave_type=wave_type,
            disp_type=disp_type,
            mode=mode,
            algorithm=algorithm,
            dc=dc,
            dt=dt
        )
        return synthetic_curve
    
    # --------------------------
    # 步骤3：返回适配后的 forward_model 函数
    # --------------------------
    return forward_model