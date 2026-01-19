# seismocorr

待完善

## 项目简介

- 待完善

## 功能特点

- 待完善

## 安装方法

从 GitHub 克隆仓库并安装：

```bash
git clone https://github.com/yourusername/seismocorr.git
cd seismocorr
pip install -e .
```

### 安装可选依赖

安装测试依赖：

```bash
pip install seismocorr[test]
```

安装示例依赖（用于运行示例和可视化）：

```bash
pip install seismocorr[examples]
```

## 快速开始

### 基本用法

```python
import numpy as np
from seismocorr.core.spfi import run_spfi
from seismocorr.config.builder import SPFIConfig

# 准备数据
d_obs = np.random.rand(10, 20)  # 模拟观测数据
freqs = np.linspace(1.0, 10.0, 10)  # 频率数组
subarray = [np.arange(10) for _ in range(20)]  # 子阵列索引
sensor_xy = np.random.rand(100, 2)  # 传感器坐标

# 配置 SPFI
cfg = SPFIConfig(
    assumption="station_avg",
    geometry="2d",
    grid_x=np.linspace(0, 10, 5),
    grid_y=np.linspace(0, 10, 5),
    regularization="l2",
    alpha=0.1,
    beta=0.0,
    pair_sampling="all",
    random_state=42
)

# 运行 SPFI 反演
result = run_spfi(
    d_obs=d_obs,
    freqs=freqs,
    subarray=subarray,
    sensor_xy=sensor_xy,
    cfg=cfg
)

# 查看结果
print("反演速度形状:", result["velocity"].shape)
print("反演慢度形状:", result["slowness"].shape)
```


## 许可证

本项目采用 MIT 许可证，详情请见 [LICENSE](LICENSE) 文件。