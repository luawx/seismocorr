# seismocorr/core/inversion/__init__.py
"""
Seismocorr 传统反演模块：自适应GA + 嵌套DLS联合反演

提供地震频散曲线的联合反演功能，结合遗传算法的全局搜索能力
和阻尼最小二乘的局部精细优化能力，实现高精度的介质参数反演。
"""

from .types import (
    ArrayLike,
    DispersionCurve,
    InversionParams,
    GAConfig,
    DLSConfig,
    JointInversionConfig,
    InversionResult
)

from .utils import (
    validate_inversion_params,
    calculate_residual,
    calculate_fitness,
    normalize_params,
    denormalize_params
)

from .adaptive_ga import AdaptiveGA
from .nested_dls import NestedDLS
from .joint_inversion import JointInversion

__all__ = [
    # 类型别名
    "ArrayLike",
    "DispersionCurve",
    "InversionParams",
    "GAConfig",
    "DLSConfig",
    "JointInversionConfig",
    "InversionResult",
    # 工具函数
    "validate_inversion_params",
    "calculate_residual",
    "calculate_fitness",
    "normalize_params",
    "denormalize_params",
    # 核心类
    "AdaptiveGA",
    "NestedDLS",
    "JointInversion"
]