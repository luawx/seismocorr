# seismocorr/core/traditional_inversion/nested_dls.py
'''
嵌套阻尼最小二乘实现
'''
from typing import Optional, Callable, Tuple

import numpy as np
from scipy.linalg import inv, pinv
from scipy.optimize import line_search

from .types import DLSConfig, InversionParams, DispersionCurve
from .utils import validate_inversion_params, calculate_residual

# DLS默认配置
DEFAULT_DLS_CONFIG = {
    "max_iter": 50,          # 最大迭代次数
    "damping_init": 0.1,     # 初始阻尼系数
    "damping_adapt": True,   # 是否自适应调整阻尼
    "converge_threshold": 1e-4,  # 收敛阈值
    "step_size": 0.5,        # 初始步长
    "hessian_reg": 1e-6      # 海森矩阵正则化系数
}


class NestedDLS:
    """
    嵌套阻尼最小二乘(DLS)类，用于GA全局最优解的局部精细反演
    
    嵌套逻辑：外层调整阻尼系数，内层迭代更新反演参数，
    通过残差变化自适应调整阻尼，平衡收敛速度和稳定性。
    """

    def __init__(self, config: Optional[DLSConfig] = None):
        """
        初始化嵌套DLS

        Args:
            config: DLS配置字典，未指定则使用默认配置
        """
        self.config = {**DEFAULT_DLS_CONFIG, **(config or {})}
        self.max_iter = self.config["max_iter"]
        self.damping = self.config["damping_init"]
        self.damping_adapt = self.config["damping_adapt"]
        self.converge_threshold = self.config["converge_threshold"]
        self.step_size = self.config["step_size"]
        self.hessian_reg = self.config["hessian_reg"]

        # 正演函数和雅可比矩阵计算函数（外部注入）
        self.forward_model: Optional[Callable[[InversionParams], DispersionCurve]] = None
        self.jacobian_func: Optional[Callable[[InversionParams], np.ndarray]] = None

    def set_forward_model(self, forward_func: Callable[[InversionParams], DispersionCurve]) -> None:
        """
        设置频散曲线正演函数

        Args:
            forward_func: 正演函数，输入反演参数，输出频散曲线 (频率, 相速度)
        """
        self.forward_model = forward_func

    def set_jacobian_func(self, jacob_func: Callable[[InversionParams], np.ndarray]) -> None:
        """
        设置雅可比矩阵计算函数（可由正演库提供或数值近似）

        Args:
            jacob_func: 雅可比矩阵函数，输入反演参数，输出雅可比矩阵
        """
        self.jacobian_func = jacob_func

    def _compute_jacobian_numerical(
        self,
        params: InversionParams,
        h: float = 1e-5
    ) -> np.ndarray:
        """
        数值近似计算雅可比矩阵（备用方案，若无解析雅可比）

        Args:
            params: 反演参数
            h: 数值微分步长

        Returns:
            jacobian: 雅可比矩阵 (频散点数量, 参数维度)
        """
        if self.forward_model is None:
            raise RuntimeError("未设置正演函数")
        
        # 展平参数为一维数组
        param_flat = np.concatenate([params[key] for key in sorted(params.keys())])
        param_dim = len(param_flat)

        # 正演基准频散曲线
        syn_freq, syn_vel = self.forward_model(params)
        data_dim = len(syn_vel)

        # 初始化雅可比矩阵
        jacobian = np.zeros((data_dim, param_dim))

        for i in range(param_dim):
            # 扰动参数
            param_flat_h = param_flat.copy()
            param_flat_h[i] += h
            # 重构参数字典
            params_h = {}
            idx = 0
            for key in sorted(params.keys()):
                dim = len(params[key])
                params_h[key] = param_flat_h[idx:idx+dim]
                idx += dim
            # 正演扰动后的频散曲线
            _, syn_vel_h = self.forward_model(params_h)
            # 数值微分
            jacobian[:, i] = (syn_vel_h - syn_vel) / h

        return jacobian

    def _adjust_damping(self, residual_norm_prev: float, residual_norm_curr: float) -> None:
        """
        嵌套调整阻尼系数：根据残差变化动态调整

        Args:
            residual_norm_prev: 上一轮残差范数
            residual_norm_curr: 当前轮残差范数
        """
        if residual_norm_curr < residual_norm_prev:
            # 残差减小，降低阻尼以加速收敛
            self.damping *= 0.9
        else:
            # 残差增大，增大阻尼以保证稳定
            self.damping *= 1.5

        # 限制阻尼范围
        self.damping = np.clip(self.damping, 1e-4, 10.0)

    def _update_params(
        self,
        params: InversionParams,
        residual: np.ndarray,
        jacobian: np.ndarray,
        observed_vel: np.ndarray
    ) -> InversionParams:
        """
        阻尼最小二乘参数更新

        Args:
            params: 当前反演参数
            residual: 残差数组
            jacobian: 雅可比矩阵
            observed_vel: 观测频散曲线的相速度数组

        Returns:
            updated_params: 更新后的参数
        """
        # 展平参数为一维数组
        param_keys = sorted(params.keys())
        param_flat = np.concatenate([params[key] for key in param_keys])

        # 构建正则化海森矩阵
        hessian = jacobian.T @ jacobian + self.damping * np.eye(jacobian.shape[1]) + self.hessian_reg * np.eye(jacobian.shape[1])
        # 解析梯度计算（J^T @ r）
        gradient = jacobian.T @ residual

        # 阻尼最小二乘求解更新量
        try:
            delta = inv(hessian) @ gradient
        except np.linalg.LinAlgError:
            # 奇异矩阵时使用伪逆
            delta = pinv(hessian) @ gradient

        # 线搜索目标函数：严格按公式计算均方根误差（RMSE）
        def obj_func(x: np.ndarray) -> float:
            """
            目标函数：频散曲线拟合均方根误差
            公式：φ(x) = √[ (1/N) * Σ(v_c(x) - v_o)² ]
            v_c: 理论计算相速度, v_o: 观测相速度, N: 观测数据点总数
            """
            # 重构参数字典
            params_x = {}
            idx = 0
            for key in param_keys:
                dim = len(params[key])
                params_x[key] = x[idx:idx+dim]
                idx += dim
            # 正演理论相速度
            _, v_c = self.forward_model(params_x)
            # 观测相速度、数据点总数
            v_o = observed_vel
            N = len(v_o)
            # 严格匹配公式的RMSE计算
            phi_x = np.sqrt((1 / N) * np.sum((v_c - v_o) ** 2))
            return phi_x

        # 构建Scipy线搜索要求的可调用梯度函数
        def fprime(x: np.ndarray) -> np.ndarray:
            """梯度函数：返回预计算的解析梯度（线搜索局部近似为固定）"""
            return gradient

        # 线搜索求最优步长（无None参数，彻底解决TypeError）
        step = line_search(obj_func, fprime, param_flat, -delta)[0]
        # 步长失效时使用默认值
        if step is None:
            step = self.step_size

        # 更新展平参数
        param_flat_updated = param_flat - step * delta

        # 重构参数字典返回
        updated_params = {}
        idx = 0
        for key in param_keys:
            dim = len(params[key])
            updated_params[key] = param_flat_updated[idx:idx+dim]
            idx += dim

        return updated_params

    def run(
        self,
        init_params: InversionParams,
        observed_curve: DispersionCurve
    ) -> InversionParams:
        """
        运行嵌套DLS局部精细反演

        Args:
            init_params: GA输出的全局最优初始参数
            observed_curve: 观测频散曲线 (频率数组, 相速度数组)

        Returns:
            refined_params: 精细反演后的参数
        """
        if self.forward_model is None:
            raise RuntimeError("请先调用set_forward_model设置正演函数")
        
        # 验证初始参数合法性
        validate_inversion_params(init_params)
        current_params = init_params.copy()
        residual_norm_prev = np.inf
        # 提取观测相速度（关键：为_update_params准备参数）
        _, obs_vel = observed_curve

        # DLS核心迭代
        for iter_num in range(self.max_iter):
            # 正演理论频散曲线
            syn_curve = self.forward_model(current_params)
            # 计算残差和残差范数
            residual, residual_norm = calculate_residual(observed_curve, syn_curve)

            # 收敛判断：残差范数变化小于阈值则停止
            if abs(residual_norm - residual_norm_prev) < self.converge_threshold:
                print(f"DLS反演在第{iter_num}次迭代收敛 | 最终残差范数: {residual_norm:.6f}")
                break

            # 自适应调整阻尼系数（从第1次迭代后开始）
            if self.damping_adapt and iter_num > 0:
                self._adjust_damping(residual_norm_prev, residual_norm)

            # 计算雅可比矩阵（优先解析，无则数值近似）
            if self.jacobian_func is not None:
                jacobian = self.jacobian_func(current_params)
            else:
                jacobian = self._compute_jacobian_numerical(current_params)

            # 核心：参数更新（传4个参数，包含obs_vel，无参数缺失）
            current_params = self._update_params(current_params, residual, jacobian, obs_vel)
            residual_norm_prev = residual_norm

            # 每5次迭代打印一次进度
            if iter_num % 5 == 0:
                print(f"DLS迭代 {iter_num}/{self.max_iter} | 残差范数: {residual_norm:.6f} | 阻尼系数: {self.damping:.4f}")

        # 验证最终参数并返回
        validate_inversion_params(current_params)
        return current_params