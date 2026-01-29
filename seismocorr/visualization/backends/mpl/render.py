# visualization/backends/mpl/render.py
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt

from ...types import FigureHandle, PlotSpec
from . import primitives


def is_available() -> bool:
    """判断 matplotlib 是否可用（是否安装依赖）。

    Returns:
        True 表示可导入 matplotlib；False 表示不可用。
    """
    try:
        import matplotlib  # noqa: F401

        return True
    except Exception:
        return False


def render(spec: PlotSpec) -> FigureHandle:
    """将 PlotSpec 渲染为 Matplotlib Figure。

    支持的 layer.type：
        - "heatmap"
        - "lines"
        - "wiggle"
        - "vlines"
        - "annotations"
        - "polar_heatmap"

    Args:
        spec: 后端无关的绘图说明书（PlotSpec）。

    Returns:
        FigureHandle：backend="mpl"，handle 为 matplotlib Figure。

    Raises:
        ValueError: 遇到不支持的 layer.type。
        KeyError: layer.data 缺少必要字段（如 heatmap 需要 "z" 等）。
    """
    layout = spec.layout or {}
    figsize = layout.get("figsize", (9, 4))

    fig = plt.figure(figsize=figsize)

    has_polar = any(layer.type == "polar_heatmap" for layer in spec.layers)
    ax = fig.add_subplot(111, projection="polar" if has_polar else None)

    extra: Dict[str, Any] = {"ax": ax}

    for layer in spec.layers:
        if layer.type == "heatmap":
            z = layer.data["z"]
            x = layer.data.get("x")
            y = layer.data.get("y")
            cmap = layer.style.get("cmap", "jet")
            cbar_label = layer.style.get("colorbar_label", "Energy")
            
            artist = primitives.plot_heatmap(
                ax,
                z,
                x=x,
                y=y,
                cmap=cmap,
                colorbar_label=cbar_label,  
            )
            
            extra.setdefault("artists", []).append(artist)
            continue

        if layer.type == "lines":
            artist = primitives.plot_lines(
                ax,
                layer.data["x"],
                layer.data["y"],
                linewidth=layer.style.get("linewidth", 1.0),
                alpha=layer.style.get("alpha", 1.0),
                color = layer.style.get("color"),
                label=layer.name,
            )
            extra.setdefault("artists", []).append(artist)
            continue

        if layer.type == "vlines":
            artist = primitives.plot_vlines(
                ax,
                layer.data["xs"],
                linewidth=layer.style.get("linewidth", 1.0),
                alpha=layer.style.get("alpha", 1.0),
                label=layer.name,
            )
            extra.setdefault("artists", []).append(artist)
            continue

        if layer.type == "wiggle":
            artist = primitives.plot_wiggle(
                ax,
                layer.data["x"],
                layer.data["traces"],
                scale=layer.style.get("scale", 1.0),
                linewidth=layer.style.get("linewidth", 0.8),
                alpha=layer.style.get("alpha", 1.0),
                labels=layer.data.get("labels"),
                color="k",
                highlights=layer.data.get("highlights"),
                sort=layer.data.get("sort"),
            )
            extra.setdefault("artists", []).append(artist)
            continue

        if layer.type == "annotations":
            for item in layer.data.get("texts", []):
                ax.text(float(item["x"]), float(item["y"]), str(item["text"]))
            continue

        if layer.type == "polar_heatmap":
            theta = layer.data["theta"]
            r_data = layer.data["r"]  # 避免与局部变量命名冲突
            z = layer.data["z"]
            theta_unit = layer.data.get("theta_unit", "deg")
            cmap = layer.style.get("cmap", "viridis")
            cbar_label = layer.style.get("colorbar_label", "")

            artist = primitives.plot_polar_heatmap(
                ax,
                theta,
                r_data,
                z,
                theta_unit=theta_unit,
                cmap=cmap,
                colorbar_label=cbar_label,
            )
            extra.setdefault("artists", []).append(artist)
            continue

        raise ValueError(f"mpl 后端不支持的 layer.type: {layer.type!r}。")

    ax.set_title(layout.get("title", "") or "")
    ax.set_xlabel(layout.get("x_label", "") or "")
    ax.set_ylabel(layout.get("y_label", "") or "")
    if layout.get("x_lim"):
        ax.set_xlim(layout.get("x_lim"))
    else:
        ax.set_xlim(ax.get_xlim()) 

    if layout.get("y_lim"):
        ax.set_ylim(layout.get("y_lim"))
    else:
        ax.set_ylim(ax.get_ylim()) 

    if any(layer.name for layer in spec.layers):
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best")

    fig.tight_layout()
    return FigureHandle(backend="mpl", handle=fig, spec=spec, extra=extra)
