# seismocorr/spfi/core.py

"""
SPFI Core Module

参考：Zhenbo Li et al., 2025
把每个子阵列测得的平均相速度/慢度当作观测值，利用子阵列台站平均（station_avg）或射线路径平均（ray_avg）构建稀疏矩阵，
再利用不同反演方法反演精细网格的相速度/慢度，以进一步来改善阵列平均效应并提升横向分辨率。
"""

import numpy as np
from typing import Dict
from scipy.sparse import csr_matrix
from seismocorr.config.builder import SPFIConfig
from seismocorr.core.spfi.assumption import get_assumption
from seismocorr.core.spfi.inversion import get_inversion


def run_spfi(
    *,
    d_obs: np.ndarray,
    freqs: np.ndarray,
    subarray,
    sensor_xy: np.ndarray,
    cfg: SPFIConfig,
) -> Dict[str, object]:
    """
    SPFI 主流程：给定已拾取好的子阵列频散观测 d_obs，构建设计矩阵 A 并逐频反演。

    Args:
        d_obs: 子阵列观测相速度，shape=(n_freq, n_subarray), ray_avg 情况下会在内部转换为慢度观测再反演。
        freqs: 频率数组，shape=(n_freq,)
        subarray: 子阵列索引列表（List[np.ndarray]）
        sensor_xy: 传感器平面坐标：
            - cfg.geometry="2d": (n_sensors, 2)
            - cfg.geometry="1d": (n_sensors,) 或 (n_sensors,1)
        cfg: SPFIConfig（assumption/geometry/grid_x/grid_y/regularization/alpha/beta/pair_sampling/random_state）

    Returns:
        dict:
            - "A": csr_matrix 设计矩阵
            - "slowness": (n_freq, n_model) 反演慢度
            - "velocity": (n_freq, n_model) 反演速度
            - "grid_x","grid_y": 若 ray_avg&2d 则返回使用的网格中心点
            - "subarray": 原样返回
    """
    cfg.validate()

    if not isinstance(sensor_xy, np.ndarray):
        raise TypeError(f"sensor_xy 类型应为 np.ndarray，当前为 {type(sensor_xy).__name__}")
    if subarray is None:
        raise TypeError("subarray 不能为 None")
    if not isinstance(subarray, (list, tuple)):
        raise TypeError(f"subarray 类型应为 list/tuple，当前为 {type(subarray).__name__}")

    d = np.asarray(d_obs, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64).reshape(-1)

    if d.size > 0 and (not np.all(np.isfinite(d))):
        raise ValueError("d_obs 包含 NaN/Inf")
    if f.size > 0 and (not np.all(np.isfinite(f))):
        raise ValueError("freqs 包含 NaN/Inf")
    if d.ndim != 2:
        raise ValueError("d_obs 必须是二维数组，shape=(n_freq, n_subarray)。")
    if f.size != d.shape[0]:
        raise ValueError("freqs 长度必须等于 d_obs 的频率维度 n_freq。")

    # 构建设计矩阵 A
    A_builder = get_assumption(cfg.assumption)
    A = A_builder(
        subarray=subarray,
        sensor_xy=sensor_xy,
        geometry=cfg.geometry,
        grid_x=cfg.grid_x,
        grid_y=cfg.grid_y,
        pair_sampling=cfg.pair_sampling,
        random_state=cfg.random_state,
    )
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)

    # 选择反演策略
    inv = get_inversion(cfg.regularization)

    n_model = int(A.shape[1])
    slowness_out = np.zeros((f.size, n_model), dtype=np.float64)
    velocity_out = np.zeros((f.size, n_model), dtype=np.float64)

    is_station_avg = (cfg.assumption == "station_avg")
    alpha = float(cfg.alpha)
    beta = float(cfg.beta)

    # 逐频反演
    for fi in range(f.size):
        v_sub = np.asarray(d[fi, :], dtype=np.float64).reshape(-1)
        if v_sub.size != A.shape[0]:
            raise ValueError("d_obs 每一行长度必须等于子阵列数（A 的行数）。")

        if is_station_avg:
            # station_avg：速度域线性反演
            # v_sub ≈ A @ v_station
            x0 = np.full(n_model, float(np.mean(v_sub)), dtype=np.float64)

            res = inv(A=A, d=v_sub, x0=x0, alpha=alpha, beta=beta)  # <- call
            v_model = np.asarray(res["x"], dtype=np.float64).reshape(-1)

            velocity_out[fi, :] = v_model
            slowness_out[fi, :] = _safe_inverse(v_model)

        else:
            # ray_avg：慢度域线性反演
            # s_sub ≈ A @ s_grid
            s_sub = _safe_inverse(v_sub)
            x0 = np.full(n_model, float(np.mean(s_sub)), dtype=np.float64)

            res = inv(A=A, d=s_sub, x0=x0, alpha=alpha, beta=beta)  # <- call
            s_model = np.asarray(res["x"], dtype=np.float64).reshape(-1)

            slowness_out[fi, :] = s_model
            velocity_out[fi, :] = _safe_inverse(s_model)

    out = {
        "subarray": subarray,
        "A": A,
        "slowness": slowness_out,
        "velocity": velocity_out,
    }
    if cfg.assumption == "ray_avg" and cfg.geometry == "2d":
        out["grid_x"] = cfg.grid_x
        out["grid_y"] = cfg.grid_y

    return out


# ====================
# 辅助函数
# ====================

def _safe_inverse(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """安全求倒数：用于 v<->s 的互转，避免除零。"""
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / np.maximum(x, eps)
