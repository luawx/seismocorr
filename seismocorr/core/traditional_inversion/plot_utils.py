# seismocorr/core/traditional_inversion/plot_utils.py
# -*- coding: utf-8 -*-
"""
横波速度结构可视化工具模块
"""
from typing import Union, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# 全局绘图样式初始化（仅执行一次，避免多次调用重复设置）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 兼容中/英文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

def plot_vs_step_curve(
    thickness: Union[np.ndarray, list],
    vs: Union[np.ndarray, list],
    half_space_extend: float = 100.0,
    figsize: Tuple[float, float] = (8, 10),
    line_color: str = '#2E86AB',
    line_width: float = 2.5,
    ax: Optional[Axes] = None,
    label: str = '横波速度阶梯线',
    plot_scatter: bool = True
) -> Tuple[Figure, Axes]:
    """
    绘制横波速度结构阶梯线图，严格遵循指定坐标规则，支持外部传入Axes实现单图多曲线

    Args:
        thickness: 各层厚度数组，最后一层为半空间（必须设0），长度=N，单位：km
        vs: 对应各层的横波速度数组，长度必须=N，单位：km/s
        half_space_extend: 半空间延伸的最大深度，仅视觉展示（代表无限延伸），默认100.0，单位：km
        figsize: 图幅大小 (宽, 高)，仅当未传入ax时生效，默认(8,10)
        line_color: 阶梯线颜色，支持十六进制/颜色名，默认#2E86AB（藏青色）
        line_width: 阶梯线宽度，默认2.5，单位：pt
        ax: 外部传入的绘图轴对象，实现单图多曲线绘制，默认None（自动新建画布和轴）
        label: 曲线图例标签，用于区分多曲线，默认'横波速度阶梯线'

    Returns:
        fig: matplotlib图对象，传入ax时为原画布的fig，新建时为新fig
        ax: matplotlib轴对象，传入ax时返回原对象，新建时返回新ax

    Raises:
        ValueError: 层数不匹配/速度非正/普通层厚度为负/半空间厚度非0/有效层数量不足
    """
    # -------------------------- 参数类型转换与校验 --------------------------
    # 统一转换为float64数组，保证计算精度
    thickness = np.asarray(thickness, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    n_layers = len(thickness)          # 总层数（含半空间）
    n_valid_layers = n_layers - 1      # 有效层数量（前n-1层）

    # 严格参数校验，报错信息简洁规范
    if len(vs) != n_layers:
        raise ValueError(f"thickness与vs层数不匹配：{n_layers} vs {len(vs)}")
    if np.any(vs <= 0):
        raise ValueError("横波速度vs必须为正数值")
    if np.any(thickness[:-1] < 0):
        raise ValueError("除半空间外，其余层厚度必须非负")
    if not np.isclose(thickness[-1], 0, atol=1e-6):
        raise ValueError("最后一层为半空间，厚度必须设为0")
    if n_valid_layers < 1:
        raise ValueError("有效层数量至少为1（半空间除外）")

    # -------------------------- 计算深度节点（核心） --------------------------
    # 生成有效层深度累积节点：[0, h1, h1+h2, ..., Σh(1~n-1)]
    depth_cum = np.zeros(n_valid_layers + 1, dtype=np.float64)
    for i in range(1, n_valid_layers + 1):
        depth_cum[i] = depth_cum[i-1] + thickness[i-1]

    # -------------------------- 生成阶梯坐标（严格匹配规则） --------------------------
    x_coords = []  # 速度坐标（Vs）
    y_coords = []  # 深度坐标
    # 有效层：按(Vsi, h_prev)→(Vsi, h_curr)追加坐标
    for i in range(n_valid_layers):
        x_coords.extend([vs[i], vs[i]])
        y_coords.extend([depth_cum[i], depth_cum[i+1]])
    # 半空间：按(Vsn, hn-1)→(Vsn, 延伸深度)追加坐标
    x_coords.extend([vs[-1], vs[-1]])
    y_coords.extend([depth_cum[-1], half_space_extend])

    # 转换为数组，提升绘图效率
    x_coords = np.array(x_coords, dtype=np.float64)
    y_coords = np.array(y_coords, dtype=np.float64)

    # -------------------------- 绘图轴初始化 --------------------------
    if ax is None:
        # 未传入ax：新建画布，初始化地质标准坐标轴（仅执行一次）
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # 纵轴：深度向下为正（地质绘图标准）
        ax.set_ylim(0, half_space_extend)
        ax.invert_yaxis()
        # 坐标轴标签与刻度
        ax.set_xlabel('横波速度 Vs (km/s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('深度 (km)', fontsize=14, fontweight='bold')
        ax.set_yticks(np.arange(0, half_space_extend + 1, 10))
        # 网格与边框
        ax.grid(linewidth=0.8, zorder=1)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.2)
    else:
        # 传入ax：复用已有轴，获取原画布对象
        fig = ax.figure

    # -------------------------- 绘制阶梯线与散点 --------------------------
    # 绘制核心阶梯线，zorder保证层级（网格<线条<散点）
    ax.plot(
        x_coords, y_coords,
        color=line_color,
        linewidth=line_width,
        label=label,
        zorder=2
    )
    # -------------------------- 自适应调整横轴范围 --------------------------
    # 多曲线绘制时，自动扩展x轴，避免速度超出显示范围
    x_current_max = np.max(vs) + 0.6
    x_ax_max = ax.get_xlim()[1]
    ax.set_xlim(0, max(x_current_max, x_ax_max))
    ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 0.1, 0.2))

    # 新建画布时默认显示图例，多曲线时由用户统一调整
    if ax is None:
        ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    return fig, ax


# -------------------------- 模块自测代码 --------------------------
if __name__ == '__main__':
    """模块单独运行时，执行自测，验证函数功能是否正常"""
    # 测试数据1：原始反演数据
    thickness1 = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0]
    vs1 = [3.50, 3.40, 3.50, 3.80, 4.20, 4.50, 4.70, 4.80, 4.75]

    # 测试数据2：精细反演数据（对比用）
    thickness2 = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0]
    vs2 = [3.52, 3.41, 3.51, 3.82, 4.21, 4.52, 4.71, 4.81, 4.76]

    # 自测1：单曲线绘制
    # fig, ax = plot_vs_step_curve(thickness1, vs1, label='GA全局反演结果')

    # 自测2：双曲绘制（核心功能验证）
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    plot_vs_step_curve(thickness1, vs1, ax=ax, line_color='#2E86AB', label='GA全局反演结果')
    plot_vs_step_curve(thickness2, vs2, ax=ax, line_color='#E67E22', label='DLS精细反演结果')
    ax.set_title('面波频散反演速度结构对比', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()
    # plt.savefig('vs_structure_test.png', dpi=300, bbox_inches='tight')