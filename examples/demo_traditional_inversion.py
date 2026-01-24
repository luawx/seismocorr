"""
传统反演使用示例
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from seismocorr.core.traditional_inversion import JointInversion, InversionParams, DispersionCurve
from seismocorr.core.traditional_inversion.plot_utils import plot_vs_step_curve
from seismocorr.core.dispersion_forward import (
    forward_func,
    make_forward_model
)

# 1. 模拟观测频散曲线，注入外部成熟的正演函数
obs_freq = np.linspace(1, 10, 50)
observed_curve = (obs_freq, np.full_like(obs_freq, np.nan))

forward_model = make_forward_model(forward_func, observed_curve, mode=1)

thickness = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0]  # 层厚度（最后一层半空间设0）
vs = [3.50, 3.40, 3.50 ,3.80, 4.20, 4.50, 4.70, 4.80, 4.75]        # 横波速度

model_params = {'thickness': thickness, 'vs': vs}

observed_curve = forward_model(model_params)

# 2. 配置联合反演
joint_config = {
    "ga_config": {
        "pop_size": 60,
        "max_generations": 80,
        "adapt_strategy": "exponential",
        "param_bounds": {
            # thickness：9个二元组（9层），最后1个(0,0)→半空间，前8个自定义地质合理边界
            "thickness": [
                (0.0, 15.0), 
                (0.0, 15.0),
                (0.0, 15.0),
                (0.0, 15.0),
                (0.0, 15.0), 
                (0.0, 15.0), 
                (0.0, 15.0),
                (0.0, 15.0), 
                (0.0, 0.0)   # 第9层：强制0→半空间（不可修改，正演约定）
            ],
            # vs：9个二元组（9层），与thickness一一对应，自定义速度边界（km/s）
            "vs": [
                (3.0, 5.0),  # 第1层横波速度：3.0~5.0km/s
                (3.0, 5.0),  # 第2层
                (3.0, 5.0),  # 第3层
                (3.0, 5.0),  # 第4层
                (3.0, 5.0),  # 第5层
                (3.0, 5.0),  # 第6层
                (3.0, 5.0),  # 第7层
                (3.0, 5.0),  # 第8层
                (3.0, 5.0)   # 第9层（半空间）速度：可根据地质调整
            ]
        }
    },
    "dls_config": {
        "max_iter": 40,
        "damping_init": 0.05
    },
    "forward_model": forward_model,
    "verbose": True
}

# 3. 运行联合反演
if __name__ == "__main__":
    # 初始化联合反演
    joint_inv = JointInversion(joint_config)
    
    # 运行反演
    result = joint_inv.run(observed_curve)
    
    # 输出结果
    print("\n最终反演结果（DLS优化后）:")
    print(f"厚度网格: {result['dls_params']['thickness']}")
    print(f"横波速度: {result['dls_params']['vs']}")
    print(f"最终残差范数: {result['final_residual']:.6f}")

# 4. 可视化  
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))  # 统一设置图幅
    
    plot_vs_step_curve(thickness, vs, line_color='black', ax=ax, label='正演模型')
    plot_vs_step_curve(result['dls_params']['thickness'], result['dls_params']['vs'], line_color='red', ax=ax, label='反演结果')

    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True, bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(100, 0)
    
    plt.show()