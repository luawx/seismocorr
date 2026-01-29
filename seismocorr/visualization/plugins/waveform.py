#seismocorr/visualization/plugins/disper.py

from __future__ import annotations

from typing import Any, Optional, List
import numpy as np

from ..types import Plugin, PlotSpec, Layer, Param

def _build_line(
    data: Any,
    *,
    title: Optional[str] = None,
    x_label: str = "Time(s)",
    y_label: str = "Amplitude (m/s)",
    x_lim: Optional[list[float]] = None,  
    y_lim: Optional[list[float]] = None,  
    colors: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
) -> PlotSpec:
    """
    线图绘制插件：可以绘制单条或多条线，生成 PlotSpec（后端无关）

    data 约定（dict）:
        required:
            - "x": 1D array, 时间轴或自变量
            - "y": 1D 或 2D array, 对应的数值（如振幅、速度等）。单条线为 1D， 多条线为 2D 数组。
    """
    if not isinstance(data, dict):
        raise TypeError("data 应该是 dict 格式，包含 'x' 和 'y' 键")

    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)

    if x.ndim != 1 or (y.ndim != 1 and y.ndim != 2):
        raise ValueError("x 必须是一维数组，而 y 必须是一维或二维数组")
    if x.shape[0] != y.shape[1] and y.ndim == 2:
        raise ValueError("x 和 y 长度必须一致，x 的长度应该等于 y 的列数")

    # 为每条线生成图层
    line_layers = []
    if y.ndim == 1:
        # 绘制单条线
        line_layer = Layer(
            type="lines",
            data={"x": x, "y": y},
            style={"linewidth": 2, "color": colors[0]},  
            name=labels[0]
        )
        line_layers.append(line_layer)
    else:
        # 绘制多条线
        for i in range(y.shape[0]):
            color = colors[min(i, len(colors) - 1)]  
            label = labels[min(i, len(labels) - 1)]
            line_layer = Layer(
                type="lines",
                data={"x": x, "y": y[i, :]},
                style={"linewidth": 2, "color": color, "label": label},
                name=label
            )
            line_layers.append(line_layer)

    layers: list[Layer] = line_layers

    layout = {
        "title": title or "Line Plot",
        "x_label": x_label,
        "y_label": y_label,
        "x_lim": x_lim,
        "y_lim": y_lim
    }

    # 设置坐标轴范围
    if x_lim is not None:
        layout["x_lim"] = [float(x_lim[0]), float(x_lim[1])]
    else:
        layout["x_lim"] = [float(np.min(x)), float(np.max(x))]
    
    if y_lim is not None:
        layout["y_lim"] = [float(y_lim[0]), float(y_lim[1])]
    else:
        layout["y_lim"] = [float(np.min(y)), float(np.max(y))]

    return PlotSpec(plot_id="line_plot", layers=layers, layout=layout)





PLUGINS = [
    Plugin(
        id="lines",
        title="绘制直线",
        build=_build_line,  # 这里修改为 _build_line_plot
        default_layout={"figsize": (10, 6)},  # 设置默认图形大小

        data_spec={  # 数据输入规格
            "type": "dict",  # 数据类型为字典
            "required_keys": {
                "x": "1D array, shape (n,) 时间轴/自变量",  # 必须包含 'x'，时间轴
                "y": "1D or 2D array, shape (n,) 或 (m, n) 数据值",  # 'y' 可以是 1D 或 2D
            },
        },

        params={  # 可选参数
            "title": Param("str", "Seismic Waveforms", "图标题"),  # 图标题
            "x_label": Param("str", "Time", "x轴标签"),  # x轴标签
            "y_label": Param("str", "Amplitude", "y轴标签"),  # y轴标签
            "x_lim": Param("list[float]", None, "x轴范围：[xmin, xmax]"),  # x轴范围
            "y_lim": Param("list[float]", None, "y轴范围：[ymin, ymax]"),  # y轴范围
            "colors": Param("list[str]", ["black"], "线条颜色(长度不一致时，后续的颜色按照最后一个颜色来)"),  
            "labels": Param("list[str]", ["line"], "线条label(长度不一致时，后续的label按照最后一个颜色来)"),  
        },
    )
]