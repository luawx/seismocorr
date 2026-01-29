# visualization/backends/mpl/primitives.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np


def plot_heatmap(ax, z, x=None, y=None, *, cmap="", colorbar_label="" ):
    """
    绘制二维热力图（rect heatmap）
    - x/y 为坐标轴刻度（可选）
    """
    import matplotlib.pyplot as plt

    z = np.asarray(z)
    if x is None:
        x = np.arange(z.shape[1])
    if y is None:
        y = np.arange(z.shape[0])

    # extent: [xmin, xmax, ymin, ymax]
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]

    im = ax.imshow(
        z,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)


        
    if colorbar_label:
        cbar.set_label(colorbar_label)
    return {"im": im, "colorbar": cbar}


def plot_lines(ax, x, y, *, linewidth=1.0, alpha=1.0, label=None, color:str):
    """绘制折线/曲线"""
    (ln,) = ax.plot(x, y, linewidth=linewidth, alpha=alpha, label=label, color = color)
    return {"line": ln}


def plot_vlines(ax, xs, *, linewidth=1.0, alpha=1.0, label=None):
    """绘制多条竖线"""
    arts = []
    for i, xv in enumerate(xs):
        arts.append(ax.axvline(x=float(xv), linewidth=linewidth, alpha=alpha, label=(label if i == 0 else None)))
    return {"vlines": arts}


def plot_wiggle(
    ax,
    x,
    traces,
    *,
    scale=1.0,
    linewidth=0.8,
    alpha=1.0,
    labels=None,
    color: str = "k",
    highlights=None,
    sort=None,
):
    """
    绘制多道 wiggle（默认黑色），并支持：
    1) 对指定道、指定时间段高亮着色（默认红色）
    2) 按距离（或任意数值）对道排序显示
    3) y 轴可选用距离值作为纵坐标（y_mode="distance"）

    sort（可选）：
      None：不排序，y 用 index
      dict：
        {
          "by": array_like,        # 长度 n_tr，例如 distance(km)
          "ascending": True,       # 默认 True
          "y_mode": "index"|"distance",  # 默认 "index"
          "label": "Distance (km)" # 可选：y 轴标签（y_mode=distance 时建议给）
        }
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    tr = np.asarray(traces, dtype=float)
    if tr.ndim != 2:
        raise ValueError("traces 必须是二维数组 (n_traces, n_samples)")
    n_tr, n_samp = tr.shape
    if x.ndim != 1 or x.shape[0] != n_samp:
        raise ValueError("x 必须是一维，且长度等于 traces.shape[1]")

    # ============ 1) 处理排序与 y_offset ============
    order = np.arange(n_tr)
    inv_order = None

    sort_by = None
    sort_ascending = True
    y_mode = "index"
    y_label = None

    if sort is not None:
        if not isinstance(sort, dict) or "by" not in sort:
            raise ValueError("sort 必须是 dict 且包含 'by'，例如 {'by': dist, 'ascending': True, 'y_mode': 'distance'}")

        sort_by = np.asarray(sort["by"], dtype=float)
        if sort_by.ndim != 1 or sort_by.shape[0] != n_tr:
            raise ValueError("sort['by'] 必须是一维且长度等于 traces.shape[0]（道数）")

        sort_ascending = bool(sort.get("ascending", True))
        y_mode = str(sort.get("y_mode", "index"))
        y_label = sort.get("label", None)

        order = np.argsort(sort_by)
        if not sort_ascending:
            order = order[::-1]

        # 重排 traces
        tr = tr[order, :]

        # 重排 labels
        if labels is not None:
            if len(labels) != n_tr:
                raise ValueError("labels 长度必须等于道数 n_tr")
            labels = [labels[i] for i in order]

        # old_index -> new_index 映射（用于 highlights）
        inv_order = np.empty(n_tr, dtype=int)
        inv_order[order] = np.arange(n_tr)

        sort_by_sorted = sort_by[order]
    else:
        sort_by_sorted = None  # 未排序

    # 计算每道的“基线纵坐标”（关键）
    if y_mode == "distance":
        if sort_by_sorted is None:
            raise ValueError("y_mode='distance' 时必须提供 sort={'by': distance,...}")
        y0 = sort_by_sorted.astype(float)  # 每道基线就是距离值
    elif y_mode == "index":
        y0 = np.arange(n_tr, dtype=float)
    else:
        raise ValueError("sort['y_mode'] 只能是 'index' 或 'distance'")

    # ============ 2) 预处理 highlights（输入的 trace 默认按原始索引） ============
    hl_by_trace = {}
    if highlights:
        if not isinstance(highlights, (list, tuple)):
            raise ValueError("highlights 必须是 list[dict] 或 tuple[dict]")
        for item in highlights:
            if not isinstance(item, dict) or "trace" not in item:
                raise ValueError("highlights 每项必须是 dict 且包含 'trace' 字段")
            old_i = int(item["trace"])
            if old_i < 0 or old_i >= n_tr:
                raise ValueError(f"highlights.trace 越界: {old_i} (0~{n_tr-1})")

            # 如果发生排序，把 old_i 映射成新序号 new_i
            new_i = int(inv_order[old_i]) if inv_order is not None else old_i
            hl_by_trace.setdefault(new_i, []).append(item)

    arts = []
    hl_arts = []

    xmin = float(np.min(x))
    xmax = float(np.max(x))

    # ============ 3) 绘制 ============
    for i in range(n_tr):
        # ✅基线纵坐标用 y0（index 或 distance）
        y = y0[i] + tr[i] * float(scale)

        (ln,) = ax.plot(x, y, linewidth=linewidth, alpha=alpha, color=color)
        arts.append(ln)

        if i in hl_by_trace:
            for cfg in hl_by_trace[i]:
                t0 = float(cfg.get("t0", xmin))
                t1 = float(cfg.get("t1", xmax))
                if t1 < t0:
                    t0, t1 = t1, t0

                mask = (x >= t0) & (x <= t1)
                if not np.any(mask):
                    continue

                hl_color = cfg.get("color", "r")
                hl_lw = float(cfg.get("linewidth", linewidth))
                hl_alpha = float(cfg.get("alpha", 1.0))

                (hln,) = ax.plot(x[mask], y[mask], linewidth=hl_lw, alpha=hl_alpha, color=hl_color)
                hl_arts.append(hln)

    # ============ 4) y 轴刻度/标签 ============
    if y_mode == "index":
        # 还是老逻辑：小于60道时显示 labels
        if labels is not None and len(labels) == n_tr and n_tr <= 60:
            ax.set_yticks(range(n_tr))
            ax.set_yticklabels(labels, fontsize=8)
    else:
        # y_mode == "distance"：刻度用距离值（建议不要太密）
        if n_tr <= 60:
            ax.set_yticks(y0)
            # 如果用户给了 labels，用 labels；否则用距离数值
            if labels is not None and len(labels) == n_tr:
                ax.set_yticklabels(labels, fontsize=8)
            else:
                ax.set_yticklabels([f"{v:.3g}" for v in y0], fontsize=8)

        if y_label:
            ax.set_ylabel(y_label)

    return {"lines": arts, "highlight_lines": hl_arts, "order": order}


def plot_polar_heatmap(
    ax,
    theta,
    r,
    z,
    *,
    theta_unit="deg",
    cmap="viridis",
    colorbar_label="",
):
    """
    绘制极坐标热力图

    参数
    ----------
    ax : matplotlib.axes.Axes
        极坐标坐标轴（必须通过 projection='polar' 创建）
    theta : array_like
        角度坐标，形状 (n_theta,)
    r : array_like
        半径坐标，形状 (n_r,)
    z : array_like
        数据值，形状 (n_r, n_theta)
    theta_unit : str
        角度单位，'deg' 或 'rad'，默认为 'deg'
    cmap : str
        颜色映射
    colorbar_label : str
        颜色条标签

    返回
    -------
    dict
        包含绘制对象的字典
    """
    import numpy as np
    import matplotlib.pyplot as plt

    theta = np.asarray(theta, dtype=float)  # (n_theta,)
    r = np.asarray(r, dtype=float)          # (n_r,)
    z = np.asarray(z, dtype=float)          # (n_r, n_theta)

    if theta_unit == "deg":
        theta = np.deg2rad(theta)
    elif theta_unit != "rad":
        raise ValueError("theta_unit 必须是 'deg' 或 'rad'")

    if z.shape != (r.shape[0], theta.shape[0]):
        raise ValueError(f"z 形状必须是 (n_r={r.shape[0]}, n_theta={theta.shape[0]})，但得到 {z.shape}")

    # pcolormesh 需要网格边界：把中心点转成边界
    def _edges(v):
        v = np.asarray(v, float)
        dv = np.diff(v)
        if len(dv) == 0:
            return np.array([v[0]-0.5, v[0]+0.5])
        left = v[0] - dv[0] / 2
        right = v[-1] + dv[-1] / 2
        mid = (v[:-1] + v[1:]) / 2
        return np.concatenate([[left], mid, [right]])

    th_e = _edges(theta)  # (n_theta+1,)
    r_e = _edges(r)       # (n_r+1,)

    TH, RR = np.meshgrid(th_e, r_e)

    im = ax.pcolormesh(TH, RR, z, cmap=cmap, shading="auto")

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    # 常见极坐标显示设置
    ax.set_theta_zero_location("N")   # 0度在北
    ax.set_theta_direction(-1)        # 顺时针增加（和 N/E/S/W 习惯一致）

    # 优化标签显示，避免重叠
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))  # 设置极坐标角度标签的间距
    ax.set_xticklabels(
        ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], fontsize=12, rotation=45
    )  # 设置角度标签及其旋转角度

    # 优化径向坐标显示，避免重叠
    ax.set_yticks(np.linspace(0, np.max(r), 6))  # 限制显示的径向标签数量
    ax.set_yticklabels([f'{int(x)}' for x in np.linspace(0, np.max(r), 6)], fontsize=12)

    # 调整标题和标签
    ax.set_title("Beamforming Polar Heatmap", fontsize=16, pad=20)  # 增加标题的间距
    ax.set_xlabel("Azimuth (°)", fontsize=14, labelpad=15)  # 增加x轴标签的间距

    # 这里调整 ylabel 的位置
    ax.set_ylabel("Slowness (s/km)", fontsize=14, labelpad=15)  # 增加y轴标签的间距
    ax.yaxis.set_label_coords(-0.15, 0.5)  # 调整 ylabel 的位置，左移避免与图形重叠

    # 设置网格线透明度和样式
    ax.grid(True, linestyle="--", alpha=0.5)  # 更细的虚线网格，增加透明度以减少干扰

    return {"im": im, "colorbar": cbar}


