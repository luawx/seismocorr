# seismocorr/core/traditional_inversion/joint_inversion.py
'''
GA+DLS 联合反演主逻辑
'''
from typing import Optional, Callable, Dict, Any

import numpy as np

from .types import (
    JointInversionConfig,
    InversionParams,
    DispersionCurve,
    InversionResult
)
from .adaptive_ga import AdaptiveGA
from .nested_dls import NestedDLS
from .utils import validate_inversion_params, calculate_residual

# 联合反演默认配置
DEFAULT_JOINT_CONFIG = {
    "ga_config": {},         # GA配置
    "dls_config": {},        # DLS配置
    "forward_model": None,   # 正演函数
    "jacobian_func": None,   # 雅可比函数
    "verbose": True          # 是否打印日志
}


class JointInversion:
    """
    自适应GA + 嵌套DLS联合反演主类
    
    流程：
    1. 预处理观测频散数据
    2. GA全局搜索得到初始最优解
    3. DLS局部精细反演优化解
    4. 输出反演结果和评估指标
    """

    def __init__(self, config: Optional[JointInversionConfig] = None):
        """
        初始化联合反演

        Args:
            config: 联合反演配置字典
        """
        self.config = {**DEFAULT_JOINT_CONFIG, **(config or {})}
        self.verbose = self.config["verbose"]

        # 初始化GA和DLS模块
        self.ga = AdaptiveGA(self.config["ga_config"])
        self.dls = NestedDLS(self.config["dls_config"])

        # 设置正演和雅可比函数
        if self.config["forward_model"] is not None:
            self.set_forward_model(self.config["forward_model"])
        if self.config["jacobian_func"] is not None:
            self.dls.set_jacobian_func(self.config["jacobian_func"])

    def set_forward_model(self, forward_func: Callable[[InversionParams], DispersionCurve]) -> None:
        """
        设置频散曲线正演函数（注入外部成熟库）

        Args:
            forward_func: 正演函数
        """
        self.ga.set_forward_model(forward_func)
        self.dls.set_forward_model(forward_func)

    def set_jacobian_func(self, jacob_func: Callable[[InversionParams], np.ndarray]) -> None:
        """
        设置雅可比矩阵计算函数

        Args:
            jacob_func: 雅可比函数
        """
        self.dls.set_jacobian_func(jacob_func)

    def _preprocess_observed_curve(self, observed_curve: DispersionCurve) -> DispersionCurve:
        """
        预处理观测频散曲线：去噪、插值、筛选有效频率范围

        Args:
            observed_curve: 原始观测频散曲线

        Returns:
            processed_curve: 预处理后的频散曲线
        """
        obs_freq, obs_vel = map(np.asarray, observed_curve)

        # 去除无效值（NaN/Inf）
        valid_mask = np.isfinite(obs_freq) & np.isfinite(obs_vel) & (obs_freq > 0) & (obs_vel > 0)
        obs_freq = obs_freq[valid_mask]
        obs_vel = obs_vel[valid_mask]

        # 按频率排序
        sort_idx = np.argsort(obs_freq)
        obs_freq = obs_freq[sort_idx]
        obs_vel = obs_vel[sort_idx]

        if self.verbose:
            print(f"预处理后有效频散点数量: {len(obs_freq)} | 频率范围: {obs_freq.min():.2f} - {obs_freq.max():.2f} Hz")

        return (obs_freq, obs_vel)

    def run(self, observed_curve: DispersionCurve) -> InversionResult:
        """
        运行联合反演

        Args:
            observed_curve: 原始观测频散曲线

        Returns:
            inversion_result: 反演结果字典，包含：
                - ga_params: GA全局最优参数
                - dls_params: DLS精细优化参数
                - final_residual: 最终残差范数
                - synthetic_curve: 最终合成频散曲线
                - fitness_history: GA适应度历史
        """
        # 1. 预处理观测数据
        processed_curve = self._preprocess_observed_curve(observed_curve)

        # 2. GA全局搜索
        if self.verbose:
            print("\n=== 开始自适应GA全局反演 ===")
        ga_params = self.ga.run(processed_curve)
        validate_inversion_params(ga_params)

        # 3. DLS局部精细反演
        if self.verbose:
            print("\n=== 开始嵌套DLS局部精细反演 ===")
        dls_params = self.dls.run(ga_params, processed_curve)
        validate_inversion_params(dls_params)

        # 4. 结果评估
        synthetic_curve = self.dls.forward_model(dls_params)
        _, final_residual = calculate_residual(processed_curve, synthetic_curve)

        # 构建结果字典
        inversion_result: InversionResult = {
            "ga_params": ga_params,
            "dls_params": dls_params,
            "final_residual": final_residual,
            "synthetic_curve": synthetic_curve,
            "observed_curve": processed_curve,
            "fitness_history": self.ga.fitness_history
        }

        if self.verbose:
            print("\n=== 联合反演完成 ===")
            print(f"GA最优参数残差范数: {calculate_residual(processed_curve, self.ga.forward_model(ga_params))[1]:.6f}")
            print(f"DLS优化后残差范数: {final_residual:.6f}")

        return inversion_result