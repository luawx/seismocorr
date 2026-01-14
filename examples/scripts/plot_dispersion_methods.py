from numba import config
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 用户提供的加载和提取函数
def load_ccf_results(input_file):
    """
    加载互相关结果文件
    
    Args:
        input_file: npz文件路径
        
    Returns:
        tuple: (results_dict, reference_index, sampling_rate, dx)
    """
    print(f"正在加载互相关结果: {input_file}")
    data = np.load(input_file, allow_pickle=True)
    
    results = data['results'].item()
    reference_index = int(data['reference_index'])
    sampling_rate = float(data['sampling_rate'])
    dx = float(data['dx'])
    
    print(f"- 参考道索引: {reference_index}")
    print(f"- 采样率: {sampling_rate} Hz")
    print(f"- 道间距: {dx} m")
    print(f"- 道对数量: {len(results)}")
    
    return results, reference_index, sampling_rate, dx


def extract_ccf_matrix(results, reference_index, dx):
    """
    从结果字典中提取按距离排序的互相关矩阵
    
    Args:
        results: 互相关结果字典，格式为 {ch_pair: (lags, ccf)}
        reference_index: 参考道索引
        dx: 道间距
        
    Returns:
        tuple: (distances, lags, cc_matrix)
    """
    print("正在提取互相关矩阵...")
    
    distances = []
    ccfs = []
    lags = None
    
    for key, (lags_arr, ccf_arr) in results.items():
        # 解析道索引
        ref_ch, other_ch = key.split('--')
        other_idx = int(other_ch[2:])
        
        # 计算距离
        distance = abs(other_idx - reference_index) * dx
        distances.append(distance)
        ccfs.append(ccf_arr)
        
        if lags is None:
            lags = lags_arr
    
    # 按距离排序
    sorted_indices = np.argsort(distances)
    distances = np.array(distances)[sorted_indices]
    cc_matrix = np.array(ccfs)[sorted_indices]
    
    print(f"- 距离范围: {distances.min():.2f} m 到 {distances.max():.2f} m")
    print(f"- 时移范围: {lags.min():.2f} s 到 {lags.max():.2f} s")
    print(f"- 互相关矩阵形状: {cc_matrix.shape}")
    
    return distances, lags, cc_matrix

# 定义绘图配置类
class PlotConfig:
    """绘图配置类"""
    def __init__(self):
        self.cmap = 'jet'
        self.font_size = 12
        self.fig_width = 12
        self.fig_height = 6
        self.freqmin = 0.0
        self.freqmax = 50.0

# 绘制频散谱图的辅助函数
def plot_spectrum_figure(f, c, A, ax=None, title=None, plot_config=None):
    """绘制单个频散谱图"""
    config = plot_config or PlotConfig()
    
    if ax is None:
        ax = plt.gca()
    
    # 频率范围选择
    fmin, fmax = config.freqmin, config.freqmax
    no_fmin = np.argmin(np.abs(f - fmin))
    no_fmax = np.argmin(np.abs(f - fmax))
    
    if A.shape[0] != len(f) or A.shape[1] != len(c):
        # 如果A的形状不匹配，尝试转置
        A = A.T
    
    # 确保A的形状是[nf, nc]
    if A.shape[0] != len(f):
        # 如果频率维度不匹配，调整数据
        Aplot = A[:, no_fmin:no_fmax]
        fplot = f[no_fmin:no_fmax]
        # 转置Aplot以匹配预期形状
        Aplot = Aplot.T
    else:
        # 正常情况
        Aplot = A[no_fmin:no_fmax, :]
        fplot = f[no_fmin:no_fmax]
    
    # 计算归一化因子
    max_val = np.nanmax(np.abs(Aplot))
    
    # 绘制频散谱图
    im = ax.pcolormesh(fplot, c, Aplot.T/max_val, 
                      cmap=config.cmap, vmin=0, vmax=0.5, shading='nearest')
    ax.grid(True)
    
    # 坐标轴设置
    ax.set_xticks(np.linspace(0, fmax + 0.01, 11))
    ax.set_xlabel('Frequency [Hz]', fontsize=config.font_size)
    ax.set_ylabel('Phase velocity [m/s]', fontsize=config.font_size)
    ax.set_xlim([fmin, fmax])
    
    # 图形设置
    ax.tick_params(direction='out', which='both')
    ax.tick_params(axis='both', which='major', labelsize=config.font_size)
    
    if title:
        ax.set_title(title, fontsize=config.font_size+2)
    return im

# 比较函数
def compare_dispersion_methods():
    """使用实际数据测试不同的频散成像方法"""
    print("=== 使用实际数据测试不同的频散成像方法 ===")
    
    # 1. 加载实际数据
    input_file = "stacked_cc_result.npz"
    results, reference_index, sampling_rate, dx = load_ccf_results(input_file)
    distances, lags, cc_matrix_full = extract_ccf_matrix(results, reference_index, dx)
    
    # 2. 准备计算所需的输入
    print("\n正在准备频散计算输入...")
    
    # 将时域互相关矩阵转换为频域
    n_lags = len(lags)//2
    freq = np.fft.rfftfreq(n_lags, 1/sampling_rate)
    cc_matrix = cc_matrix_full[:, :cc_matrix_full.shape[1]//2][:,::-1]

    n_lags_full = len(lags)
    freq_full = np.fft.rfftfreq(n_lags_full, 1/sampling_rate)
    # 设置频散提取配置
    config_common = {
        'freqmin': 0.5,      # 最小频率 (Hz)
        'freqmax': 40.0,     # 最大频率 (Hz)
        'vmin': 100.0,       # 最小相速度 (m/s)
        'vmax': 1000.0,      # 最大相速度 (m/s)
        'vnum': 1000,         # 试算速度点数
        'sampling_rate': sampling_rate
    }
    c = np.linspace(config_common['vmin'], config_common['vmax'], config_common['vnum'])
    # 对每个道的互相关函数进行FFT
    # 半支互相关函数（用于SLANT_STACK和MASW方法）
    cc_array_f = []
    for j in range(len(cc_matrix)):
        cc_f = np.fft.rfft(np.fft.ifftshift(cc_matrix[j]))
        cc_array_f.append(cc_f)
    cc_array_f = np.array(cc_array_f)

    # 全段互相关函数（用于FJ相关方法）
    cc_array_f_full = []
    for j in range(len(cc_matrix_full)):
        cc_f = np.fft.rfft(np.fft.ifftshift(cc_matrix_full[j]))
        cc_array_f_full.append(cc_f)
    cc_array_f_full = np.array(cc_array_f_full)
    # 应用频率筛选
    # 对半支互相关的频率进行筛选
    freq_id = np.where((freq >= config_common['freqmin']) & (freq <= config_common['freqmax']))[0]
    freq = freq[freq_id]
    cc_array_f = cc_array_f[:, freq_id]
    
    # 计算正确的试算速度数组（与SlantStack方法内部计算一致）
    v_min = config_common['vmin']
    v_max = config_common['vmax']
    v_num = config_common['vnum']
    
    # SlantStack方法内部使用的是慢度域计算，然后转换为速度域
    # 计算最大慢度
    p_max = 1.0 / v_min
    # 试算慢度值
    q = np.linspace(0, p_max, v_num + 1)
    # 移除慢度为0的分量，转换为速度
    q = q[1:]
    v_vals = 1.0 / q
    
    print(f"- 频率范围: {freq.min():.2f} Hz 到 {freq.max():.2f} Hz")
    print(f"- 频域互相关矩阵形状: {cc_array_f.shape}")
    print(f"- 相速度范围: {v_vals.min():.2f} m/s 到 {v_vals.max():.2f} m/s")
    print(f"- 相速度数量: {len(v_vals)}")
    
    # 其他参数
    nr = len(distances)  # 距离数量
    nf = len(freq)  # 频率数量
    
    print(f"- 频率范围: {freq.min():.2f} Hz 到 {freq.max():.2f} Hz")
    print(f"- 频率数量: {nf}")
    print(f"- 相速度范围: {v_min} m/s 到 {v_max} m/s")
    print(f"- 相速度数量: {v_num}")
    print(f"- 距离数量: {nr}")
    
    # 3. 导入所需的函数和类
    from seismocorr.plugins.disper import (
        DispersionAnalyzer,
        DispersionMethod,
        DispersionConfig,
        PlotConfig
    )
    
    # 4. 使用DispersionAnalyzer计算不同方法的频散谱
    print("\n=== 使用DispersionAnalyzer计算不同方法 ===")
    
    # 创建频散成像配置
    freqmin = config_common['freqmin']      # 最小频率 (Hz)
    freqmax = config_common['freqmax']     # 最大频率 (Hz)
    disper_config = DispersionConfig(
        freqmin=freqmin,
        freqmax=freqmax,
        vmin=config_common['vmin'],
        vmax=config_common['vmax'],
        vnum=config_common['vnum'],
        sampling_rate=config_common['sampling_rate']
    )
    
    print(f"\n频散提取配置:")
    print(f"- 频率范围: {disper_config.freqmin} - {disper_config.freqmax} Hz")
    print(f"- 相速度范围: {disper_config.vmin} - {disper_config.vmax} m/s")
    print(f"- 试算速度点数: {disper_config.vnum}")
    
    # 要测试的方法列表
    methods_to_test = [
        (DispersionMethod.FJ, "FJ方法"),
        (DispersionMethod.FJ_RR, "FJ_RR方法"),
        (DispersionMethod.MFJ_RR, "MFJ_RR方法"),
        (DispersionMethod.SLANT_STACK, "SLANT_STACK方法"),
        (DispersionMethod.MASW, "MASW方法")
    ]
    
    # 存储所有结果
    all_results = {}
    
    # 提取实部和虚部（根据FJ计算需要）
    U_f = np.real(cc_array_f).astype(np.float32)
    U_f_flat = U_f.flatten()
    
    
    # 使用DispersionAnalyzer计算各种方法
    for method, method_name in methods_to_test:
        print(f"\n=== 使用DispersionAnalyzer {method_name}计算 ===")
        start_time = time.time()
        
        # 创建频散分析器
        analyzer = DispersionAnalyzer(
            method=method,
            config=disper_config
        )
        
        # 执行频散成像
        cc_fft_complex = cc_array_f.astype(np.complex64)
        cc_fft_full_complex = cc_array_f_full.astype(np.complex64)
        if method == DispersionMethod.MASW or method == DispersionMethod.SLANT_STACK:
            # SLANT_STACK和MASW方法使用半支互相关
            spectrum = analyzer.analyze(cc_fft_complex, freq, distances, disper_config)
        else:
            # FJ相关方法使用全段互相关
            spectrum = analyzer.analyze(cc_fft_full_complex, freq_full, distances, disper_config)
        
        calc_time = time.time() - start_time
        print(f"✓ {method_name}计算完成，耗时: {calc_time:.6f}秒")
        all_results[method_name] = spectrum
    
    # 5. 绘制所有方法的结果
    print("\n=== 绘制所有方法的结果 ===")
    
    # 创建综合图
    num_methods = len(all_results)
    num_rows = (num_methods + 1) // 2  # 每行2个图
    num_cols = 2
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 5 * num_rows))
    fig.suptitle('Dispersion Extraction Results - Real Data', fontsize=16)
    axes = axes.ravel()  # 转换为一维数组
    
    # 绘制每个方法的结果
    for i, (method_name, spectrum) in enumerate(all_results.items()):
        ax = axes[i]
        
        # 绘制功率谱热力图
        if method_name=='SLANT_STACK方法':
            # SLANT_STACK方法使用v_vals作为速度数组
            im = ax.pcolormesh(freq, v_vals, spectrum, cmap='jet', shading='auto')
        elif method_name=='MASW方法':
            # MASW方法使用均匀分布的速度数组c
            im = ax.pcolormesh(freq, c, spectrum, cmap='jet', shading='auto')
        else:
            # FJ相关方法使用与SlantStack相同的速度数组v_vals，使用全段互相关的频率数组
            im = ax.pcolormesh(freq_full, c, spectrum/np.nanmax(np.abs(spectrum)), cmap='jet', shading='auto',vmin=0,vmax=0.5)
        ax.set_title(method_name)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase Velocity (m/s)')
        ax.set_ylim(disper_config.vmin, disper_config.vmax)
        ax.set_xlim(disper_config.freqmin, disper_config.freqmax)
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Normalized Power')
    
    # 隐藏多余的子图
    for i in range(num_methods, num_rows * num_cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('dispersion_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("对比图已保存到: dispersion_methods_comparison.png")
    plt.close()
    
    # 单独绘制每个方法的详细图
    for method_name, spectrum in all_results.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # 功率谱热力图
        if method_name=='SLANT_STACK方法':
            # SLANT_STACK方法使用v_vals作为速度数组
            im = ax1.pcolormesh(freq, v_vals, spectrum/np.nanmax(np.abs(spectrum)), cmap='jet', shading='auto',vmin=0.5,vmax=1)
        elif method_name=='MASW方法':
            # MASW方法使用均匀分布的速度数组c
            im = ax1.pcolormesh(freq, c, spectrum/np.nanmax(np.abs(spectrum)), cmap='jet', shading='auto',vmin=0.5,vmax=1)
        else:
            # FJ相关方法使用与SlantStack相同的速度数组v_vals，使用全段互相关的频率数组
            im = ax1.pcolormesh(freq_full, c, spectrum/np.nanmax(np.abs(spectrum)), cmap='jet', shading='auto',vmin=0,vmax=0.5)
        ax1.set_title(f'{method_name} - Real Data')
        ax1.set_ylabel('Phase Velocity (m/s)')
        ax1.set_ylim(disper_config.vmin, disper_config.vmax)
        ax1.set_xlim(disper_config.freqmin, disper_config.freqmax)
        
        # 颜色条
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label('Normalized Power')
        
        plt.tight_layout()
        plt.savefig(f'dispersion_{method_name}_result.png', dpi=300, bbox_inches='tight')
        print(f'{method_name}详细图已保存为: dispersion_{method_name}_result.png')
        plt.close()
    
    print(f"\n所有测试完成!")

# 主函数
if __name__ == "__main__":
    compare_dispersion_methods()