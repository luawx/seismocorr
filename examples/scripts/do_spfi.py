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

画图：2x3
第一行：station_avg (True / Inv / Diff)
第二行：ray_avg     (True / Inv / Diff)
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

    # 反演网格 + 真值棋盘
    grid_x = np.linspace(0.0, width_m, 51, dtype=np.float64)
    grid_y = np.linspace(0.0, height_m, 51, dtype=np.float64)

    v0, dv = 3.0, 0.3
    tile_size_m = 200.0
    v_true_grid = _checkerboard(grid_x, grid_y, v0=v0, dv=dv, tile=tile_size_m)

    # 真值慢度
    s_true_grid = 1.0 / np.maximum(v_true_grid, 1e-12)
    s_true_vec = s_true_grid.reshape(-1)

    # 台站真值
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

    _plot_compare_2x3(
        grid_x=grid_x,
        grid_y=grid_y,
        sensor_xy=sensor_xy,
        v_true=v_true_grid,
        v_sta=v_inv_grid_sta,
        v_ray=v_inv_grid_ray,
        title=f"SPFI compare | freq={freqs[0]:.2f} Hz",
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


def _sample_nearest(grid_x: np.ndarray, grid_y: np.ndarray, v_grid: np.ndarray, sta_xy: np.ndarray) -> np.ndarray:
    gx = np.asarray(grid_x, dtype=np.float64)
    gy = np.asarray(grid_y, dtype=np.float64)
    xy = np.asarray(sta_xy, dtype=np.float64)
    ix = np.clip(np.searchsorted(gx, xy[:, 0], side="left"), 0, gx.size - 1)
    iy = np.clip(np.searchsorted(gy, xy[:, 1], side="left"), 0, gy.size - 1)
    return np.asarray(v_grid, dtype=np.float64)[iy, ix]


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


def _plot_compare_2x3(*, grid_x, grid_y, sensor_xy, v_true, v_sta, v_ray, title: str) -> None:
    gx = np.asarray(grid_x, dtype=np.float64)
    gy = np.asarray(grid_y, dtype=np.float64)
    xy = np.asarray(sensor_xy, dtype=np.float64)

    v_true = np.asarray(v_true, dtype=np.float64)
    v_sta = np.asarray(v_sta, dtype=np.float64)
    v_ray = np.asarray(v_ray, dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(title)

    _p3(axes[0], gx, gy, xy, v_true, v_sta, "station_avg")
    _p3(axes[1], gx, gy, xy, v_true, v_ray, "ray_avg")

    plt.show()


def _p3(ax_row, gx, gy, xy, v_true, v_inv, tag: str) -> None:
    diff = v_inv - v_true

    im0 = ax_row[0].pcolormesh(gx, gy, v_true, shading="auto", vmin=2.5, vmax=3.5)
    ax_row[0].scatter(xy[:, 0], xy[:, 1], s=10, c="k")
    ax_row[0].set_title(f"True ({tag})")
    plt.colorbar(im0, ax=ax_row[0], fraction=0.046, pad=0.04)

    im1 = ax_row[1].pcolormesh(gx, gy, v_inv, shading="auto", vmin=2.5, vmax=3.5)
    ax_row[1].scatter(xy[:, 0], xy[:, 1], s=10, c="k")
    ax_row[1].set_title(f"Inverted ({tag})")
    plt.colorbar(im1, ax=ax_row[1], fraction=0.046, pad=0.04)

    im2 = ax_row[2].pcolormesh(gx, gy, diff, shading="auto")
    ax_row[2].scatter(xy[:, 0], xy[:, 1], s=10, c="k")
    ax_row[2].set_title(f"Diff ({tag})")
    plt.colorbar(im2, ax=ax_row[2], fraction=0.046, pad=0.04)

    for ax in ax_row:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")


if __name__ == "__main__":
    main()
