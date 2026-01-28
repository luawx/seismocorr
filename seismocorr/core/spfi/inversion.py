# seismocorr/core/inversion.py

"""
SPFI Inversion Method Module

负责由子阵列相速度反演网格相速度，支持：
- 最小二乘法
- Tikhonov L2
- Lasso L1
- L1 + L2
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from seismocorr.config.default import SUPPORTED_REGULARIZATIONS

MatrixLike = Union[np.ndarray, csr_matrix]
InversionResult = Dict[str, Any]


class InversionStrategy(ABC):
    """
    反演策略抽象基类。
    具体策略需继承并实现 inversion 方法
    """

    @abstractmethod
    def inversion(
        self,
        A: MatrixLike,
        d: np.ndarray,
        x0: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> InversionResult:
        """
        Args:
            A: 设计稀疏矩阵（n_obs, n_model）
            d: 观测数据，shape = (n_obs,)
            x0: 初始/参考模型，shape = (n_model,)
            alpha: L2 正则化系数
            beta: L1 正则化系数

        Returns:
            dict: 包含反演结果的字典，具体包括模型、是否成功、目标函数值等
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> InversionResult:
        return self.inversion(*args, **kwargs)


class _LeastSquaresStrategy(InversionStrategy):
    """ 最小二乘法（无正则化），min ||Ax-d||^2 """

    def inversion(
        self,
        A: MatrixLike,
        d: np.ndarray,
        x0: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> InversionResult:
        d_vec, x0_vec = _validate_shapes(A, d, x0)

        # L-BFGS-B 迭代求解（无正则化 -> alpha=0,beta=0）
        res = _solve_with_lbfgs(A=A, d=d_vec, x0=x0_vec, alpha=0.0, beta=0.0)
        return _wrap_minimize(res, tag="none_lbfgs")


class _L2Strategy(InversionStrategy):
    """ Tikhonov L2 方法： min ||Ax-d||^2 + alpha||x-x0||^2 """

    def inversion(
        self,
        A: MatrixLike,
        d: np.ndarray,
        x0: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> InversionResult:
        d_vec, x0_vec = _validate_shapes(A, d, x0)
        if alpha < 0:
            raise ValueError("alpha(L2 系数) 必须 >= 0。")

        res = _solve_with_lbfgs(A=A, d=d_vec, x0=x0_vec, alpha=float(alpha), beta=0.0)
        return _wrap_minimize(res, tag="l2_lbfgs")


class _L1Strategy(InversionStrategy):
    """ L1方法： min ||Ax-d||^2 + beta||x-x0||_1 """

    def inversion(
        self,
        A: MatrixLike,
        d: np.ndarray,
        x0: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> InversionResult:
        d_vec, x0_vec = _validate_shapes(A, d, x0)
        if beta < 0:
            raise ValueError("beta(L1 系数) 必须 >= 0。")

        res = _solve_with_lbfgs(A=A, d=d_vec, x0=x0_vec, alpha=0.0, beta=float(beta))
        return _wrap_minimize(res, tag="l1_lbfgs")


class _L1L2Strategy(InversionStrategy):
    """ L1 + L2 联合正则化： min ||Ax-d||^2 + alpha||x-x0||^2 + beta||x-x0||_1 """

    def inversion(
        self,
        A: MatrixLike,
        d: np.ndarray,
        x0: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> InversionResult:
        d_vec, x0_vec = _validate_shapes(A, d, x0)
        if alpha < 0 or beta < 0:
            raise ValueError("alpha/beta 必须 >= 0。")

        res = _solve_with_lbfgs(A=A, d=d_vec, x0=x0_vec, alpha=float(alpha), beta=float(beta))
        return _wrap_minimize(res, tag="l1_l2_lbfgs")


# ====================
# 工厂函数
# ====================
_INVERSION_MAP = {
    "none": _LeastSquaresStrategy,
    "l2": _L2Strategy,
    "l1": _L1Strategy,
    "l1_l2": _L1L2Strategy,
}


def get_inversion(regularization: str) -> InversionStrategy:
    """根据正则类型返回反演策略实例。"""
    if not isinstance(regularization, str):
        raise TypeError(f"regularization 类型应为 str，当前为 {type(regularization).__name__}: {regularization!r}")
    regularization = regularization.strip().lower()
    if not regularization:
        raise ValueError("regularization 不能为空字符串")

    if regularization not in SUPPORTED_REGULARIZATIONS:
        raise ValueError(f"regularization={regularization} 不支持，应为 {SUPPORTED_REGULARIZATIONS}")
    return _INVERSION_MAP[regularization]()


# ====================
# 辅助函数
# ====================
def _validate_shapes(A: MatrixLike, d: np.ndarray, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ 校验输入矩阵和向量的形状，并返回转换后的 d 和 x0。"""
    d_vec = np.asarray(d, dtype=np.float64).reshape(-1)
    x0_vec = np.asarray(x0, dtype=np.float64).reshape(-1)

    if d_vec.size == 0:
        raise ValueError("d 不能为空。")
    if x0_vec.size == 0:
        raise ValueError("x0 不能为空。")
    if A.shape[0] != d_vec.size:
        raise ValueError("A 的行数必须等于 d 的长度。")
    if A.shape[1] != x0_vec.size:
        raise ValueError("A 的列数必须等于 x0 的长度。")

    return d_vec, x0_vec


# -----------------------------
# 内部求解函数
# -----------------------------
def _solve_with_lbfgs(A: MatrixLike, d: np.ndarray, x0: np.ndarray, alpha: float, beta: float):
    """
    使用 L-BFGS-B 迭代处理求解
    目标函数：
        ||Ax-d||^2 + alpha||x-x0||^2 + beta||x-x0||_1
    """

    def obj(x: np.ndarray) -> float:
        r = _matvec(A, x) - d
        loss = float(r @ r)
        if alpha > 0:
            loss += float(alpha * np.sum((x - x0) ** 2))
        if beta > 0:
            loss += float(beta * np.sum(np.abs(x - x0)))
        return loss

    def grad(x: np.ndarray) -> np.ndarray:
        r = _matvec(A, x) - d
        g = 2.0 * _rmatvec(A, r)
        if alpha > 0:
            g = g + 2.0 * alpha * (x - x0)
        if beta > 0:
            g = g + beta * np.sign(x - x0)
        return g

    return minimize(obj, x0, jac=grad, method="L-BFGS-B")


def _matvec(A: MatrixLike, x: np.ndarray) -> np.ndarray:
    """计算矩阵与向量乘积。"""
    return A @ x if sparse.issparse(A) else (np.asarray(A, dtype=np.float64) @ x)


def _rmatvec(A: MatrixLike, r: np.ndarray) -> np.ndarray:
    """计算矩阵转置与向量乘积。"""
    return (A.T @ r) if sparse.issparse(A) else (np.asarray(A, dtype=np.float64).T @ r)


def _wrap_minimize(res, tag: str) -> InversionResult:
    niter = int(res.nit) if getattr(res, "nit", None) is not None else -1
    return {
        "x": np.asarray(res.x, dtype=np.float64),
        "success": bool(res.success),
        "message": f"{tag}: {res.message}",
        "fun": float(res.fun),
        "niter": niter,
    }
