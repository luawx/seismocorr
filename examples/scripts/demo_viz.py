from __future__ import annotations

from typing import Any, Dict

import numpy as np

from seismocorr.visualization import help_plot, plot, set_default_backend, show

# =========================
# 数据生成
# =========================
def generate_ccf_data(n_tr: int = 40, n_lags: int = 401, *, seed: int = 0) -> Dict[str, Any]:
    """生成模拟 CCF 数据。

    Args:
        n_tr: 道数（traces 数量）。
        n_lags: lag 采样点数。
        seed: 随机种子，用于可复现。

    Returns:
        dict，包含：
            - cc: (n_tr, n_lags) 的 CCF 矩阵
            - lags: (n_lags,) 的 lag 轴
            - labels: (n_tr,) 的标签列表
            - dist_km: (n_tr,) 的距离数组（用于排序示例）
    """
    rng = np.random.default_rng(seed)

    lags = np.linspace(-20.0, 20.0, int(n_lags))
    cc = 0.08 * rng.standard_normal((int(n_tr), int(n_lags)))

    center = int(n_lags) // 2
    for i in range(int(n_tr)):
        shift = int(round((i - n_tr / 2.0) * 0.5))
        idx = int(np.clip(center + shift, 0, int(n_lags) - 1))
        cc[i, idx] += 1.0

    labels = [f"STA{i:02d}" for i in range(int(n_tr))]
    dist_km = rng.uniform(10.0, 300.0, size=int(n_tr))

    return {"cc": cc, "lags": lags, "labels": labels, "dist_km": dist_km}


def generate_beamforming_data(
    n_azimuth: int = 100,
    n_radius: int = 50,
    *,
    seed: int = 1,
) -> Dict[str, Any]:
    """生成模拟波束形成功率矩阵数据。

    Args:
        n_azimuth: 方位角采样数。
        n_radius: 慢度采样数（极坐标半径方向）。
        seed: 随机种子，用于可复现。

    Returns:
        dict，包含：
            - power: (n_radius, n_azimuth) 波束功率矩阵
            - azimuth_deg: (n_azimuth,) 方位角数组（0..360）
            - slowness_s_per_m: (n_radius,) 慢度数组
    """
    rng = np.random.default_rng(seed)

    azimuth_deg = np.linspace(0.0, 360.0, int(n_azimuth))
    slowness_s_per_m = np.linspace(0.1, 1.0, int(n_radius))

    # 原始生成是 (n_azimuth, n_radius)，插件约定这里用 (n_radius, n_azimuth)
    power = rng.random((int(n_azimuth), int(n_radius))).T

    return {
        "power": power,
        "azimuth_deg": azimuth_deg,
        "slowness_s_per_m": slowness_s_per_m,
    }


def generate_dispersion_energy_data(
    n_v: int = 50, n_f: int = 100, *, seed: int = 42
) -> Dict[str, Any]:
    """生成模拟的频散能量数据（能量与频率/速度的关系）。

    Args:
        n_v: 速度采样点数（或波数轴）。
        n_f: 频率采样点数。
        seed: 随机种子，用于可复现。

    Returns:
        dict，包含：
            - E: (n_v, n_f) 的能量矩阵。
            - f: (n_f,) 的频率轴。
            - v: (n_v,) 的速度轴（或群速度/相速度轴）。
    """
    rng = np.random.default_rng(seed)

    # 频率轴，假设频率从 0.01 Hz 到 10 Hz
    f = np.logspace(-2, 1, n_f)
    
    # 速度轴，假设速度从 1000 m/s 到 5000 m/s
    v = np.linspace(1000, 5000, n_v)

    # 模拟能量矩阵 E：假设能量与频率和速度相关，使用正弦波与指数衰减相结合
    E = np.zeros((n_v, n_f))
    for i in range(n_v):
        for j in range(n_f):
            E[i, j] = np.exp(-((f[j] - 5.0)**2) / 2.0) * np.sin(v[i] * f[j] / 1000)

    return {"E": E, "f": f, "v": v}


def generate_seismic_data(
    n_tr: int = 3,  # 道数（traces）
    n_samples: int = 1000,  # 每道采样点数
    duration: float = 20.0,  # 持续时间（秒）s
    *, 
    seed: int = 42
) -> Dict[str, Any]:
    """生成模拟地震波形数据。

    Args:
        n_tr: 道数（traces 数量）。
        n_samples: 每道的采样点数。
        duration: 每个波形的持续时间。
        seed: 随机种子，用于可复现。

    Returns:
        dict，包含：
            - traces: (n_tr, n_samples) 的地震波形矩阵
            - time: (n_samples,) 时间轴
            - labels: (n_tr,) 的标签列表
    """
    rng = np.random.default_rng(seed)

    time = np.linspace(0, duration, n_samples)  # 时间轴
    traces = np.zeros((n_tr, n_samples))  # 初始化地震波形矩阵

    for i in range(n_tr):
        # 使用正弦波模拟地震波形
        frequency = rng.uniform(0.1, 1.0)  # 随机频率
        traces[i, :] = np.sin(2 * np.pi * frequency * time) + 0.1 * rng.standard_normal(n_samples)

    # 标签为道的编号
    labels = [f"STA{i:02d}" for i in range(n_tr)]

    return {"traces": traces, "time": time, "labels": labels}


# =========================
# 绘图测试
# =========================
def test_ccf_wiggle_plot() -> None:
    """测试并绘制 CCF wiggle 图。"""
    set_default_backend("mpl")

    ccf = generate_ccf_data()

    fig = plot(
        "ccf.wiggle",
        data={"cc": ccf["cc"], "lags": ccf["lags"], "labels": ccf["labels"]},
        normalize=True,
        clip=0.9,
        sort={
            "by": ccf["dist_km"],
            "ascending": True,
            "y_mode": "index",
            "label": "Distance (km)",
        },
        highlights=[
            {"trace": 5, "t0": -2.0, "t1": 2.0},
            {"trace": 10, "color": "blue"},
            {"trace": 12, "t0": 1.0, "t1": 3.5, "color": "#ff00ff", "linewidth": 3},
        ],
        scale=5,
    )

    show(fig)
    print(help_plot("ccf.wiggle"))


def test_beamforming_polar_heatmap() -> None:
    """测试并绘制波束形成极坐标热力图。"""
    data = generate_beamforming_data()

    fig = plot("beamforming.polar_heatmap", data)
    # 如需指定 plotly 后端：
    # fig = plot("beamforming.polar_heatmap", data, backend="plotly")

    show(fig)
    print(help_plot("beamforming.polar_heatmap"))


def test_dispersion_energy_plot() -> None:
    """测试并绘制频散能量图（f-v）。"""
    # 生成频散能量数据
    dispersion_data = generate_dispersion_energy_data()

    # 绘制频散能量图
    fig = plot(
        "heat_map",
        data={"E": dispersion_data["E"], "f": dispersion_data["f"], "v": dispersion_data["v"]},
        title="Dispersion Energy Map",
        x_label="Frequency (Hz)",
        y_label="Velocity (m/s)",
        x_lim=[8, 10], 
        y_lim=[3000, 5000],
        cmap = "jet",
        colorbar_label = "Energy",

    )

    # 显示图形
    show(fig)

    # 打印插件的帮助信息
    print(help_plot("heat_map"))


def test_seismic_waveform_plot() -> None:
    """测试并绘制地震波形图。"""
    set_default_backend("mpl")  # 设置默认后端为 matplotlib

    # 生成地震波形数据
    seismic_data = generate_seismic_data()

    # 绘制地震波形图
    fig = plot(
        "lines",  # 使用 line 插件绘制波形图
        data={"x": seismic_data["time"], "y": seismic_data["traces"]},
        title="Seismic Waveforms",  # 设置图标题
        x_label="Time (s)",  # 设置 x 轴标签
        y_label="Amplitude (m/s)",  # 设置 y 轴标签
        x_lim=[0, 20],  # 设置 x 轴范围
        y_lim=[-2, 2],  # 设置 y 轴范围
        colors = ["blue","black"],
        labels = ["s1", "s2"]
    )

    # 显示图形
    show(fig)

    # 打印插件的帮助信息
    print(help_plot("lines"))  # 根据插件 id 调用 help_plot 获取帮助信息


def main() -> None:
    """运行示例测试。"""
    test_ccf_wiggle_plot()
    test_beamforming_polar_heatmap()
    test_dispersion_energy_plot()
    test_seismic_waveform_plot()


if __name__ == "__main__":
    main()
