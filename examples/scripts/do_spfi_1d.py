import numpy as np
import matplotlib.pyplot as plt

from seismocorr.core.subarray import get_subarray
from seismocorr.core.spfi import run_spfi
from seismocorr.config.builder import SPFIConfigBuilder


def main() -> None:
    rng = np.random.default_rng(2026)

    # 1D 通道坐标
    width_m = 20000.0
    sensor_xy = _sensor_grid_1d(nx=2000, width=width_m, jitter_std=2.0, rng=rng)
    n_sensors = int(sensor_xy.shape[0])

    # 反演网格 + 真值棋盘（速度）
    grid_x = np.linspace(0.0, width_m, 401, dtype=np.float64)

    v0, dv = 3.0, 0.3
    tile_size_m = 2000.0  # 2km 一块棋盘（你可以改）
    v_true_x = _checkerboard_1d(grid_x, v0=v0, dv=dv, tile=tile_size_m)

    # 通道真值：把 grid_x 的真值采样到每个通道位置
    v_true_sta = _sample_nearest_1d(grid_x, v_true_x, sensor_xy)

    # 生成 1D 子阵列（随机滑窗 + 随机抽通道）
    subarray = get_subarray("1d")(
        sensor_xy,
        n_realizations=4000,
        window_length=100.0,  # 100m 窗
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

    # 反演结果插值到 grid_x 上，便于和真值对比
    v_inv_x = _idw_1d(sensor_xy, v_inv_sta, grid_x, power=2.0)

    _plot_1x3_pcolormesh(
        grid_x=grid_x,
        sensor_xy=sensor_xy,
        v_true=v_true_x,
        v_inv=v_inv_x,
        title=f"1D SPFI station_avg | freq={freqs[0]:.2f} Hz",
    )

    # ray_avg + L2
    cfg_ray = (
        SPFIConfigBuilder()
        .set_geometry("1d")
        .set_assumption("ray_avg")
        .set_regularization("l2")
        .set_l2(alpha=0.05)
        .build()
    )

    out_ray = run_spfi(d_obs=d_obs, freqs=freqs, subarray=subarray, sensor_xy=sensor_xy, cfg=cfg_ray)
    v_inv_ray = _get_velocity_row(out_ray, row=0)
    v_inv_ray_x = _idw_1d(sensor_xy, v_inv_ray, grid_x, power=2.0)


    _plot_1x3_pcolormesh(
        grid_x=grid_x,
        sensor_xy=sensor_xy,
        v_true=v_true_x,
        v_inv=v_inv_ray_x,
        title=f"1D SPFI ray_avg | freq={freqs[0]:.2f} Hz",
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


def _centers_to_edges_1d(xc: np.ndarray) -> np.ndarray:
    """把严格递增的中心点 xc (nx,) 转成边界 x_edges (nx+1,)"""
    x = np.asarray(xc, dtype=np.float64).reshape(-1)
    if x.size < 2:
        raise ValueError("grid_x 至少需要 2 个点。")
    if not np.all(np.diff(x) > 0):
        raise ValueError("grid_x 必须严格递增（中心点）。")

    edges = np.empty(x.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - (edges[1] - x[0])
    edges[-1] = x[-1] + (x[-1] - edges[-2])
    return edges


def _plot_1x3_pcolormesh(
    *,
    grid_x: np.ndarray,
    sensor_xy: np.ndarray,
    v_true: np.ndarray,
    v_inv: np.ndarray,
    title: str,
) -> None:
    gx = np.asarray(grid_x, dtype=np.float64).reshape(-1)
    vt = np.asarray(v_true, dtype=np.float64).reshape(-1)
    vi = np.asarray(v_inv, dtype=np.float64).reshape(-1)
    if vt.size != gx.size or vi.size != gx.size:
        raise ValueError("v_true / v_inv 的长度必须与 grid_x 一致。")

    diff = vi - vt

    # 统一色标
    vmin = float(min(vt.min(), vi.min()))
    vmax = float(max(vt.max(), vi.max()))

    # pcolormesh 需要 edges（nx+1, ny+1）
    x_edges = _centers_to_edges_1d(gx)
    y_edges = np.array([0.0, 1.0], dtype=np.float64)

    C_true = vt.reshape(1, -1)
    C_inv = vi.reshape(1, -1)
    C_diff = diff.reshape(1, -1)

    # s = np.asarray(sensor_xy, dtype=np.float64)
    # sx = s[:, 0] if (s.ndim == 2 and s.shape[1] == 1) else s.reshape(-1)
    # sy = np.full_like(sx, 0.5, dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    fig.suptitle(title)

    im0 = axes[0].pcolormesh(x_edges, y_edges, C_true, shading="flat", vmin=vmin, vmax=vmax)
    #axes[0].scatter(sx, sy, s=6, c="k")
    axes[0].set_title("True")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].pcolormesh(x_edges, y_edges, C_inv, shading="flat", vmin=vmin, vmax=vmax)
    #axes[1].scatter(sx, sy, s=6, c="k")
    axes[1].set_title("Inverted")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].pcolormesh(x_edges, y_edges, C_diff, shading="flat")
    #axes[2].scatter(sx, sy, s=6, c="k")
    axes[2].set_title("Diff")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (dummy)")
        ax.set_ylim(0.0, 1.0)

    plt.show()


if __name__ == "__main__":
    main()
