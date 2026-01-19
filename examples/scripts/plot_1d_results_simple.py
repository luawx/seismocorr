import numpy as np
import matplotlib.pyplot as plt
from seismocorr.core.subarray import get_subarray
from seismocorr.core.spfi import run_spfi
from seismocorr.config.builder import SPFIConfigBuilder


def main() -> None:
    """
    简化版一维SPFI示例，专注于演示一维结果对比绘图
    """
    rng = np.random.default_rng(2026)

    # 1D 通道坐标
    width_m = 20000.0
    sensor_xy = _sensor_grid_1d(nx=2000, width=width_m, jitter_std=2.0, rng=rng)
    n_sensors = int(sensor_xy.shape[0])

    # 反演网格 + 真值棋盘（速度）
    grid_x = np.linspace(0.0, width_m, 401, dtype=np.float64)

    v0, dv = 3.0, 0.3
    tile_size_m = 2000.0  # 2km 一块棋盘
    v_true_x = _checkerboard_1d(grid_x, v0=v0, dv=dv, tile=tile_size_m)

    # 通道真值：把 grid_x 的真值采样到每个通道位置
    v_true_sta = _sample_nearest_1d(grid_x, v_true_x, sensor_xy)

    # 生成 1D 子阵列
    subarray = get_subarray("1d")(
        sensor_xy,
        n_realizations=4000,
        window_length=100.0,
        kmin=5,
        kmax=10,
        random_state=2026,
    )
    n_sub = len(subarray)
    print(f"[Example-1D] n_sensors={n_sensors}, n_subarray={n_sub}")

    # 单频测试
    freqs = np.array([2.0], dtype=np.float64)
    noise_frac = 0.03  # 3%

    # station_avg + L2
    d_obs = _subarray_mean(v_true_sta, subarray)
    d_obs = d_obs * rng.normal(1.0, noise_frac, size=d_obs.size)
    d_obs = d_obs.reshape(1, -1)

    cfg = (
        SPFIConfigBuilder()
        .set_geometry("1d")
        .set_assumption("station_avg")
        .set_regularization("l2")
        .set_l2(alpha=0.05)
        .build()
    )

    out = run_spfi(d_obs=d_obs, freqs=freqs, subarray=subarray, sensor_xy=sensor_xy, cfg=cfg)
    v_inv_sta = _get_velocity_row(out, row=0)
    v_inv_x = _idw_1d(sensor_xy, v_inv_sta, grid_x, power=2.0)

    # 绘制一维结果对比图
    _plot_1d_results(
        grid_x=grid_x,
        sensor_xy=sensor_xy,
        v_true_x=v_true_x,
        v_true_sta=v_true_sta,
        d_obs=d_obs,
        v_inv_x=v_inv_x,
        subarray=subarray,
        title=f"1D SPFI Results | freq={freqs[0]:.2f} Hz",
    )


# ===============
# 辅助函数
# ===============


def _sensor_grid_1d(*, nx: int, width: float, jitter_std: float, rng) -> np.ndarray:
    xs = np.linspace(0.0, float(width), int(nx), dtype=np.float64)
    if jitter_std > 0:
        xs = xs + rng.normal(0.0, float(jitter_std), size=xs.shape)
    xs = np.clip(xs, 0.0, float(width))
    return xs.reshape(-1, 1)


def _checkerboard_1d(grid_x: np.ndarray, *, v0: float, dv: float, tile: float) -> np.ndarray:
    gx = np.asarray(grid_x, dtype=np.float64).reshape(-1)
    sign = (np.floor((gx - gx.min()) / float(tile)).astype(np.int64) % 2) * 2 - 1  # -1 / +1
    return (float(v0) + float(dv) * sign).astype(np.float64)


def _sample_nearest_1d(grid_x: np.ndarray, v_grid: np.ndarray, sta_xy: np.ndarray) -> np.ndarray:
    gx = np.asarray(grid_x, dtype=np.float64).reshape(-1)
    vg = np.asarray(v_grid, dtype=np.float64).reshape(-1)
    x = np.asarray(sta_xy, dtype=np.float64).reshape(-1)
    ix = np.clip(np.searchsorted(gx, x, side="left"), 0, gx.size - 1)
    return vg[ix]


def _subarray_mean(v_station: np.ndarray, subarray) -> np.ndarray:
    v = np.asarray(v_station, dtype=np.float64).reshape(-1)
    out = np.empty(len(subarray), dtype=np.float64)
    for i, idx in enumerate(subarray):
        ii = np.asarray(idx, dtype=np.int64).reshape(-1)
        out[i] = float(np.mean(v[ii]))
    return out


def _idw_1d(sta_xy: np.ndarray, sta_val: np.ndarray, grid_x: np.ndarray, *, power: float) -> np.ndarray:
    x = np.asarray(sta_xy, dtype=np.float64).reshape(-1)
    v = np.asarray(sta_val, dtype=np.float64).reshape(-1)
    gx = np.asarray(grid_x, dtype=np.float64).reshape(-1)

    d2 = (gx[:, None] - x[None, :]) ** 2
    w = 1.0 / np.maximum(d2, 1e-12) ** (0.5 * float(power))
    out = (w @ v) / np.maximum(np.sum(w, axis=1), 1e-12)
    return out.astype(np.float64)


def _get_velocity_row(out: dict, *, row: int) -> np.ndarray:
    if "velocity" in out:
        return np.asarray(out["velocity"], dtype=np.float64)[int(row)]
    s = np.asarray(out["slowness"], dtype=np.float64)[int(row)]
    return 1.0 / np.maximum(s, 1e-12)


def _plot_1d_results(
    *,
    grid_x: np.ndarray,
    sensor_xy: np.ndarray,
    v_true_x: np.ndarray,
    v_true_sta: np.ndarray,
    d_obs: np.ndarray,
    v_inv_x: np.ndarray,
    subarray,
    title: str,
) -> None:
    """
    绘制一维结果对比：真实值、观测值+反演值、差值
    
    Args:
        grid_x: 网格中心点坐标
        sensor_xy: 传感器坐标
        v_true_x: 网格上的真实速度值
        v_true_sta: 传感器位置的真实速度值
        d_obs: 有噪声的观测值
        v_inv_x: 反演得到的速度值
        subarray: 子阵列列表
        title: 图标题
    """
    # 转换为numpy数组并确保形状正确
    gx = np.asarray(grid_x, dtype=np.float64).reshape(-1)
    vtx = np.asarray(v_true_x, dtype=np.float64).reshape(-1)
    obs = np.asarray(d_obs, dtype=np.float64).reshape(-1)
    vix = np.asarray(v_inv_x, dtype=np.float64).reshape(-1)
    sx = np.asarray(sensor_xy, dtype=np.float64).reshape(-1)
    
    # 确保数据长度一致
    if vtx.size != gx.size or vix.size != gx.size:
        raise ValueError("v_true_x / v_inv_x 的长度必须与 grid_x 一致。")
    
    # 创建子阵列中心点坐标用于观测值的横坐标
    subarray_centers = np.array([np.mean(sx[np.array(s).reshape(-1)]) for s in subarray])
    
    # 创建绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    
    # 第一行：真实值
    axes[0].plot(gx, vtx, color='black', linewidth=2, label='True Value')
    axes[0].set_ylabel('Velocity (km/s)')
    axes[0].set_title('(1) True Velocity Distribution')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 第二行：观测值和反演值
    axes[1].plot(gx, vtx, color='black', linewidth=2, label='True Value', alpha=0.5)
    axes[1].scatter(subarray_centers, obs, color='blue', s=10, alpha=0.5, label='Noisy Observation')
    axes[1].plot(gx, vix, color='red', linewidth=2, label='Inverted Value')
    axes[1].set_xlabel('Distance (m)')
    axes[1].set_ylabel('Velocity (km/s)')
    axes[1].set_title('(2) Observation vs Inversion')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 第三行：差值
    # 计算观测值与真实值的差值
    v_true_obs = np.interp(subarray_centers, gx, vtx)
    obs_diff = obs - v_true_obs
    inv_diff = vix - vtx
    
    axes[2].plot(gx, np.zeros_like(gx), color='black', linestyle='--', alpha=0.5)
    axes[2].scatter(subarray_centers, obs_diff, color='blue', s=10, alpha=0.5, label='Observation - True')
    axes[2].plot(gx, inv_diff, color='red', linewidth=2, label='Inversion - True')
    axes[2].set_xlabel('Distance (m)')
    axes[2].set_ylabel('Velocity Difference (km/s)')
    axes[2].set_title('(3) Differences from True Value')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.show()


if __name__ == "__main__":
    main()