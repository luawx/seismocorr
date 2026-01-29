#seismocorr/visualization/plugins/disper.py

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from ..types import Plugin, PlotSpec, Layer, Param

def _build_disper_energy(
    data: Any,
    *,
    title: Optional[str] = None,
    x_label: str = "frequency(Hz)",
    y_label: str = "velocity(m/s)",
    x_lim: Optional[list[float]] = None,  # [xmin, xmax]
    y_lim: Optional[list[float]] = None,  # [ymin, ymax]
    normalize: bool = True,
    cmap: str = "jet",
    colorbar_label: str = "Energy",
) -> PlotSpec:
    """
    Disper energy 插件：生成 PlotSpec（后端无关）

data 约定（dict）：
      required:
        - "E": 2D array, shape (n_v, n_f) 频散能量/幅值图（v 作为行，f 作为列）
        - "f": 1D array, shape (n_f,) 频率轴
        - "v": 1D array, shape (n_v,) 速度轴（相速度/群速度都可）
      optional:
        - "picks": 叠加曲线（等同于参数 picks，二选一）
        - "meta": 任意元信息

    picks（参数或 data["picks"]）格式：
      - dict: {"f": (n,), "v": (n,), "label": str?, "mode": any?}
      - list[dict]: 多条曲线
      也支持每条曲线额外字段：
        - "color": str
        - "linewidth": float
        - "alpha": float
        - "style": "line"|"scatter"|"both"

    Notes:
      - 本插件只负责产出 PlotSpec/Layer，不绑定 matplotlib。
      - 若后端支持 heatmap/image：建议 Layer.type 使用 "heatmap"。
    """
    if not isinstance(data, dict):
        raise TypeError("dispersion.energy 当前仅支持 dict 输入：{'E','f','v','picks?'}")

    E = np.asarray(data["E"], dtype=float)
    f = np.asarray(data["f"], dtype=float)
    v = np.asarray(data["v"], dtype=float)

    if E.ndim != 2:
        raise ValueError("E 必须是二维矩阵 (n_v, n_f)")
    if f.ndim != 1 or f.shape[0] != E.shape[1]:
        raise ValueError("f 必须是一维，且长度等于 E.shape[1]")
    if v.ndim != 1 or v.shape[0] != E.shape[0]:
        raise ValueError("v 必须是一维，且长度等于 E.shape[0]")
    if not np.isfinite(E).all():
        raise ValueError("E 包含 NaN/Inf，请先清理。")

    E_disp = E.copy()
    
    if normalize:
        pass

    #能量图图层
    heat = Layer(
        type="heatmap",

        data={
            "x": f,
            "y": v,
            "z": E,
        },

        style={
            "cmap": cmap, 
            "colorbar_label": colorbar_label
        },

        name="Disper energy"
    )

    layers: list[Layer] = [heat]
    layout = {
        "title": title or "Dispersion Energy Map",
        "x_label": x_label,
        "y_label": y_label,
        "x_lim": x_lim,
        "y_lim": y_lim
    }

    if x_lim is not None:
        layout["x_lim"] = [float(x_lim[0]), float(x_lim[1])]
    if y_lim is not None:
        layout["y_lim"] = [float(y_lim[0]), float(y_lim[1])]

    

    return PlotSpec(plot_id="disper_energy", layers=layers, layout=layout)


PLUGINS = [
    Plugin(
        id="disper_energy",
        title="频散能量图（f-v）",
        build=_build_disper_energy,
        default_layout={"figsize": (10, 6)},

        data_spec={
            "type": "dict",
            "required_keys": {
                "E": "2D array, shape (n_v, n_f) 能量/幅值图（行=v，列=f）",
                "f": "1D array, shape (n_f,) 频率轴",
                "v": "1D array, shape (n_v,) 速度轴（相/群速度）或波数轴",
            },
        },

        params={
            "title": Param("str", None, "图标题"),
            "x_label": Param("str", "Frequency (Hz)", "x轴标签"),
            "y_label": Param("str", "Velocity (m/s)", "y轴标签"),
            "x_lim": Param("list[float]", None, "x轴范围：[xmin, xmax]"),
            "y_lim": Param("list[float]", None, "y轴范围：[ymin, ymax]"),
            "cmap": Param("str", None, "热力图风格"),
            "colorbar_label": Param("str", None, "colorbar_label"),

        },
    )
]
