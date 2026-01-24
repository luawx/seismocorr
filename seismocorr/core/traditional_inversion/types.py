# seismocorr/core/traditional_inversion/types.py
'''
传统反演模块专属类型别名定义
覆盖：基础数据/正演函数/反演配置/反演结果 全流程
'''
from typing import Union, Optional, Dict, List, Tuple, Callable
import numpy as np

# 基础类型别名：兼容numpy数组/列表/元组，适配多场景输入
ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]

# 频散曲线：(频率数组, 速度数组)，物理单位严格约定
DispersionCurve = Tuple[ArrayLike, ArrayLike]
'''
频散曲线类型
元素说明：
    第一个元素：频率数组，单位Hz（与观测数据频率轴一致）
    第二个元素：相速度/群速度数组，单位km/s（与频率数组长度一致）
'''

# 反演参数：字典格式，固定键名+数组值，层数严格一致
InversionParams = Dict[str, ArrayLike]
'''
反演参数字典类型
固定键名：
    depth: 层厚度数组，单位km（最后一层为半空间，厚度设0）
    vs: 横波速度数组，单位km/s（与depth层数完全一致）
可选扩展键：
    vp: P波速度数组，rho: 密度数组（若正演需要手动传入）
'''

# 原始正演函数：匹配dispersion_forward.py中forward_func的输入输出规范
ForwardFunc = Callable[
    [ArrayLike, ArrayLike, ...],  # 必选：model_params(层厚+vs拼接), freq(频率)；后续为可选可变参数
    DispersionCurve                # 输出：合成频散曲线（与观测频率轴一致）
]
'''
降维正演函数类型（对接dispersion_forward.forward_func）
必选入参顺序不可变：
    model_params: ArrayLike - [h1,h2,...,vs1,vs2,...]，层厚(km)+横波速度(km/s)拼接数组
    freq: ArrayLike - 频率数组，单位Hz
'''

# GA遗传算法配置：键值类型约束
GAConfig = Dict[str, Union[int, float, str, bool, Dict[str, Union[float, ArrayLike]]]]
'''
GA配置参数字典
核心键名：
    pop_size: int - 种群大小
    max_generations: int - 最大迭代代数
    adapt_strategy: str - 自适应策略（如exponential/linear）
    param_bounds: Dict - 参数上下界，如{'thickness': (0,20), 'vs': (2.0,4.0)}
    cross_rate: float - 交叉概率（可选）
    mut_rate: float - 变异概率（可选）
'''

# DLS阻尼最小二乘配置：键值类型约束
DLSConfig = Dict[str, Union[int, float, bool]]
'''
DLS配置参数字典
核心键名：
    max_iter: int - 最大迭代次数
    damping_init: float - 初始阻尼系数
    converge_threshold: float - 收敛阈值（可选）
    damping_clip: Tuple[float, float] - 阻尼上下界（可选）
'''

# 联合反演整体配置：整合GA/DLS配置+公共参数
JointInversionConfig = Dict[str, Union[GAConfig, DLSConfig, Callable, bool]]
'''
联合反演总配置字典
核心键名：
    ga_config: GAConfig - GA算法配置
    dls_config: DLSConfig - DLS算法配置
    forward_model: Callable - 适配后的正演函数（由make_forward_model生成）
    verbose: bool - 是否打印迭代日志
'''

# 反演结果：整合GA/DLS参数+残差+频散曲线
InversionResult = Dict[str, Union[InversionParams, float, DispersionCurve]]
'''
联合反演结果字典
核心键名：
    ga_params: InversionParams - GA全局优化后的最优参数
    dls_params: InversionParams - DLS局部精细优化后的最终参数
    ga_residual: float - GA最优参数对应的残差范数
    dls_residual: float - DLS优化后的最终残差范数
    final_residual: float - 反演最终残差范数（同dls_residual）
    synthetic_curve: DispersionCurve - 最终参数的合成频散曲线
'''