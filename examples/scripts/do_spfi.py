"""
2D SPFI Checkerboard Test - station_avg vs ray_avg


对比两种反演：
A) station_avg + L2
   - d_obs: 子阵列速度 = 子阵列内台站速度均值 + 噪声
   - run_spfi 输出台站速度 -> IDW 插值到网格

B) ray_avg + L2
   - 真值慢度 s_true_grid=1/v_true_grid
   - 用 ray_avg 的 A 得到子阵列慢度 s_sub = A @ s_true
   - **噪声加在慢度域**：s_sub_noisy = s_sub * (1+noise)
   - 再转成速度观测 v_sub = 1/s_sub_noisy 作为 d_obs
   - run_spfi 输出网格慢度->速度
"""



import numpy as np
import matplotlib.pyplot as plt
from time import time

from seismocorr.core.subarray import get_subarray
from seismocorr.core.spfi import run_spfi
from seismocorr.config.builder import SPFIConfigBuilder
from seismocorr.core.assumption import get_assumption


def main() -> None:
    start_time = time()
    rng = np.random.default_rng(2026)

    # 台站坐标
    width_m, height_m = 1000.0, 1000.0
    sensor_xy = _sensor_grid(nx=20, ny=20, width=width_m, height=height_m, jitter_std=2.0, rng=rng)
    n_sensors = int(sensor_xy.shape[0])

    # 构建反演网格 + 真值速度棋盘
    grid_x = np.linspace(0.0, width_m, 51, dtype=np.float64)
    grid_y = np.linspace(0.0, height_m, 51, dtype=np.float64)
    v0, dv = 3.0, 0.3
    tile_size_m = 200.0
    v_true_grid = _checkerboard(grid_x, grid_y, v0=v0, dv=dv, tile=tile_size_m)

    # 真值慢度
    s_true_grid = 1.0 / np.maximum(v_true_grid, 1e-12)
    s_true_vec = s_true_grid.reshape(-1)

    # 台站真实速度值取临近网格点真值速度
    v_true_sta = _sample_nearest(grid_x, grid_y, v_true_grid, sensor_xy)

    # 生成 Voronoi 子阵列
    subarray = get_subarray("2d")(
        sensor_xy,
        n_realizations=1000,
        kmin=25,
        kmax=45,
        min_sensors=6,
        random_state=2026,
    )
    n_sub = len(subarray)
    print(f"[Example] n_sensors={n_sensors}, n_subarray={n_sub}")

    # 单频测试
    freqs = np.array([2.0], dtype=np.float64)
    noise_frac = 0.03  # 3%

    # A) station_avg + L2
    d_obs_sta = _subarray_mean(v_true_sta, subarray)
    d_obs_sta = d_obs_sta * rng.normal(1.0, noise_frac, size=d_obs_sta.size)
    d_obs_sta = d_obs_sta.reshape(1, -1)

    cfg_sta = (
        SPFIConfigBuilder()
        .set_geometry("2d")
        .set_assumption("station_avg")
        .set_regularization("l2")
        .set_l2(alpha=0.05)
        .build()
    )

    out_sta = run_spfi(d_obs=d_obs_sta, freqs=freqs, subarray=subarray, sensor_xy=sensor_xy, cfg=cfg_sta)
    v_inv_sta = _get_velocity_row(out_sta, row=0)
    v_inv_grid_sta = _idw(sensor_xy, v_inv_sta, grid_x, grid_y, power=2.0)
    print(f"[Example] SPFI station_avg + L2 time cost: {time()-start_time:.2f} s")

    # B) ray_avg + L2
    cfg_ray = (
        SPFIConfigBuilder()
        .set_geometry("2d")
        .set_assumption("ray_avg")
        .set_grid(grid_x, grid_y)
        .set_pair_sampling(None, 2026)
        .set_regularization("l2")
        .set_l2(alpha=0.3)
        .build()
    )

    # 用 ray_avg builder 构造 A（用于构造观测值）
    A_ray = get_assumption("ray_avg")(
        subarray=subarray,
        sensor_xy=sensor_xy,
        geometry="2d",
        grid_x=grid_x,
        grid_y=grid_y,
        pair_sampling=cfg_ray.pair_sampling,
        random_state=cfg_ray.random_state,
    )

    # 子阵列慢度观测
    s_sub = np.asarray(A_ray @ s_true_vec, dtype=np.float64).reshape(-1)
    s_sub_noisy = s_sub * rng.normal(1.0, noise_frac, size=s_sub.size)

    v_min = v0 - dv
    v_max = v0 + dv
    s_min = 1.0 / max(v_max, 1e-12)
    s_max = 1.0 / max(v_min, 1e-12)
    s_sub_noisy = np.clip(s_sub_noisy, s_min * 0.7, s_max * 1.3)

    # 转成相速度观测
    v_sub_noisy = 1.0 / np.maximum(s_sub_noisy, 1e-12)
    d_obs_ray = v_sub_noisy.reshape(1, -1)
    out_ray = run_spfi(d_obs=d_obs_ray, freqs=freqs, subarray=subarray, sensor_xy=sensor_xy, cfg=cfg_ray)
    v_inv_vec_ray = _get_velocity_row(out_ray, row=0)
    v_inv_grid_ray = v_inv_vec_ray.reshape(len(grid_y), len(grid_x))
    print(f"[Example] SPFI ray_avg + L2 time cost: {time()-start_time:.2f} s")

    _plot_2d_results(
        grid_x=grid_x,
        grid_y=grid_y,
        sensor_xy=sensor_xy,
        subarray=subarray,
        v_true=v_true_grid,
        v_inv=v_inv_grid_sta,
        d_obs=d_obs_sta,
        title=f"2D SPFI station_avg | freq={freqs[0]:.2f} Hz",
    )

    _plot_2d_results(
        grid_x=grid_x,
        grid_y=grid_y,
        sensor_xy=sensor_xy,
        subarray=subarray,
        v_true=v_true_grid,
        v_inv=v_inv_grid_ray,
        d_obs=d_obs_ray,
        title=f"2D SPFI ray_avg | freq={freqs[0]:.2f} Hz",
    )



# ===============
# 辅助函数
# ===============

def _sensor_grid(*, nx: int, ny: int, width: float, height: float, jitter_std: float, rng) -> np.ndarray:
    xs = np.linspace(0.0, float(width), int(nx))
    ys = np.linspace(0.0, float(height), int(ny))
    X, Y = np.meshgrid(xs, ys)
    xy = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float64)
    if jitter_std > 0:
        xy += rng.normal(0.0, float(jitter_std), size=xy.shape)
    xy[:, 0] = np.clip(xy[:, 0], 0.0, float(width))
    xy[:, 1] = np.clip(xy[:, 1], 0.0, float(height))
    return xy


def _checkerboard(grid_x: np.ndarray, grid_y: np.ndarray, *, v0: float, dv: float, tile: float) -> np.ndarray:
    gx = np.asarray(grid_x, dtype=np.float64)
    gy = np.asarray(grid_y, dtype=np.float64)
    X, Y = np.meshgrid(gx, gy)
    ix = np.floor((X - gx.min()) / float(tile)).astype(np.int64)
    iy = np.floor((Y - gy.min()) / float(tile)).astype(np.int64)
    sign = ((ix + iy) % 2) * 2 - 1
    return (float(v0) + float(dv) * sign).astype(np.float64)


def _subarray_mean(v_station: np.ndarray, subarray) -> np.ndarray:
    v = np.asarray(v_station, dtype=np.float64)
    out = np.empty(len(subarray), dtype=np.float64)
    for i, idx in enumerate(subarray):
        ii = np.asarray(idx, dtype=np.int64).reshape(-1)
        out[i] = float(np.mean(v[ii]))
    return out


def _idw(sta_xy: np.ndarray, sta_val: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, *, power: float) -> np.ndarray:
    xy = np.asarray(sta_xy, dtype=np.float64)
    v = np.asarray(sta_val, dtype=np.float64).reshape(-1)
    gx = np.asarray(grid_x, dtype=np.float64)
    gy = np.asarray(grid_y, dtype=np.float64)
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    d2 = (pts[:, None, 0] - xy[None, :, 0]) ** 2 + (pts[:, None, 1] - xy[None, :, 1]) ** 2
    w = 1.0 / np.maximum(d2, 1e-12) ** (0.5 * float(power))
    out = (w @ v) / np.maximum(np.sum(w, axis=1), 1e-12)
    return out.reshape(Y.shape).astype(np.float64)


def _get_velocity_row(out: dict, *, row: int) -> np.ndarray:
    if "velocity" in out:
        return np.asarray(out["velocity"], dtype=np.float64)[int(row)]
    s = np.asarray(out["slowness"], dtype=np.float64)[int(row)]
    return 1.0 / np.maximum(s, 1e-12)


def _subarray_centers_2d(sensor_xy: np.ndarray, subarray) -> np.ndarray:
    """
    计算每个子阵列的几何中心：把该子阵列包含的台站坐标取平均
    Returns:
        centers: (n_sub, 2)
    """
    xy = np.asarray(sensor_xy, dtype=np.float64)
    centers = np.empty((len(subarray), 2), dtype=np.float64)
    for i, idx in enumerate(subarray):
        ii = np.asarray(idx, dtype=np.int64).reshape(-1)
        centers[i, 0] = float(np.mean(xy[ii, 0]))
        centers[i, 1] = float(np.mean(xy[ii, 1]))
    return centers


def _sample_nearest(grid_x: np.ndarray, grid_y: np.ndarray, v_grid: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """
    对节点 pts_xy (n,2) 位置提取最近网格点的真值速度 v_grid
    """
    gx = np.asarray(grid_x, dtype=np.float64).reshape(-1)
    gy = np.asarray(grid_y, dtype=np.float64).reshape(-1)
    pts = np.asarray(pts_xy, dtype=np.float64)

    ix = np.clip(np.searchsorted(gx, pts[:, 0], side="left"), 0, gx.size - 1)
    iy = np.clip(np.searchsorted(gy, pts[:, 1], side="left"), 0, gy.size - 1)
    return np.asarray(v_grid, dtype=np.float64)[iy, ix]


def _plot_2d_results(*, grid_x, grid_y, sensor_xy, subarray, v_true, v_inv, d_obs, title: str) -> None:
    """
    绘制二维结果对比：真实值、观测值、反演值、真实值与反演值之差

    Args:
        grid_x: 2D 网格在 x 方向的坐标
        grid_x: 2D 网格在 y 方向的坐标
        sensor_xy: 传感器坐标
        subarray: 子阵列列表
        v_true: 网格上的真实速度值
        v_inv： 反演得到的速度值
        d_obs: 有噪声的观测值
        title: 图标题
    """
    gx = np.asarray(grid_x, dtype=np.float64).reshape(-1)
    gy = np.asarray(grid_y, dtype=np.float64).reshape(-1)
    xy = np.asarray(sensor_xy, dtype=np.float64)

    v_true = np.asarray(v_true, dtype=np.float64)
    v_inv = np.asarray(v_inv, dtype=np.float64)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 计算每个子阵列的中心点，画图时将子阵列观测值赋给其中心点位置
    obs = np.asarray(d_obs, dtype=np.float64).reshape(-1)
    centers = _subarray_centers_2d(xy, subarray)

    # 计算反演相速度与真实速度分布之差
    inv_diff = v_inv - v_true

    vmin, vmax = 2.5, 3.5
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(title)

    # 真实速度分布
    im00 = axes[0, 0].pcolormesh(gx, gy, v_true, shading="auto", vmin=vmin, vmax=vmax, cmap="jet_r")
    axes[0, 0].scatter(xy[:, 0], xy[:, 1], s=10, c="k")
    axes[0, 0].set_title("真实速度分布")
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 子阵列观测（中心点散点）
    sc01 = axes[0, 1].scatter(centers[:, 0], centers[:, 1], s=18, c=obs, vmin=vmin, vmax=vmax, cmap="jet_r")
    axes[0, 1].scatter(xy[:, 0], xy[:, 1], s=10, c="k")
    axes[0, 1].set_title("子阵列观测相速度值")
    plt.colorbar(sc01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 反演值
    im10 = axes[1, 0].pcolormesh(gx, gy, v_inv, shading="auto", vmin=vmin, vmax=vmax, cmap="jet_r")
    axes[1, 0].scatter(xy[:, 0], xy[:, 1], s=10, c="k")
    axes[1, 0].set_title("反演相速度值")
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 反演值与真实速度值之差
    im11 = axes[1, 1].pcolormesh(gx, gy, inv_diff, shading="auto", cmap="seismic")
    axes[1, 1].scatter(xy[:, 0], xy[:, 1], s=10, c="k")
    axes[1, 1].set_title("反演值与真实值的差值")
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")

    plt.show()


if __name__ == "__main__":
    main()