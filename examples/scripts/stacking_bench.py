# stacking_bench.py

import os
import sys
import psutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from seismocorr.utils.io import scan_h5_files, read_zdh5
from seismocorr.core.correlation import CorrelationConfig, BatchCorrelator
from seismocorr.core.stacking import stack_ccfs
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import gc

# 设置中文显示和负号支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def process_single_file(filepath):
    """
    单个H5文件处理函数：返回其所有道对的互相关结果 (dict: key -> (lags, ccf))
    """
    try:
        import time
        import gc
        start_time = time.time()
        
        # 步骤1: 读取文件
        start_read_time = time.time()
        DAS_data, x, t, begin_time, meta = read_zdh5(filepath)
        end_read_time = time.time()
        
        fs = meta['fs']
        DAS_data = DAS_data[256:270]
        nch, nt = DAS_data.shape

        
        # 设置参考通道，确保在限制的道数范围内
        reference_index = 0  # 选择中间的通道作为参考通道
        
        if reference_index >= nch:
            print(f"[警告] 文件 {filepath} 道数不足，跳过")
            return None

        # 步骤2: 构建 traces 字典
        start_trace_time = time.time()
        traces = {}
        for i in range(nch):
            traces[f'ch{i:04d}'] = DAS_data[i, :]
        end_trace_time = time.time()

        # 步骤3: 定义道对
        start_pairs_time = time.time()
        ref_ch = f'ch{reference_index:04d}'
        pairs = [(ref_ch, f'ch{i:04d}') for i in range(1,nch)]
        end_pairs_time = time.time()

        # 清理不再使用的大型数组
        del DAS_data, x, t, begin_time

        # 步骤4: 设置互相关参数，与cc_benchmark保持一致
        # 外层使用顺序处理，内层使用并行处理，避免嵌套并行
        correlation_config = CorrelationConfig(
            method='freq-domain',
            time_normalize='ramn',
            freq_normalize='whiten',
            freq_band=(0.5, 40),
            max_lag=5.0,
            nfft=None,
            time_norm_kwargs={'fmin': 1, 'Fs': fs, 'norm_win': 0.5},
            freq_norm_kwargs={'smooth_win': 20}
        )

        # 步骤5: 计算批量互相关
        start_cc_time = time.time()
        batch_correlator = BatchCorrelator()
        lags, ccfs, keys = batch_correlator.batch_cross_correlation(
            traces=traces,
            pairs=pairs,
            sampling_rate=fs,
            n_jobs=-1,  # 使用并行计算，充分利用多核CPU
            parallel_backend='thread',  # 线程后端，避免进程创建开销
            config=correlation_config
        )
        end_cc_time = time.time()

        # 将结果转换为字典格式，与原有代码兼容
        results = {}
        for i, key in enumerate(keys):
            results[key] = (lags, ccfs[i])

        # 清理不再使用的变量
        del traces, pairs, lags, ccfs, keys
        gc.collect()  # 强制垃圾回收

        return results

    except Exception as e:
        print(f"[错误] 处理文件 {filepath} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def group_and_stack_results(merged_results, lags_dict, stacking_method='pws', **stack_kwargs):
    """
    对来自多个文件的结果按道对分组，并进行叠加

    Args:
        merged_results: dict, {key: 2D_array}, 每个键值对应一个二维数组，每一行代表一次互相关结果
        lags_dict: dict, {key: lags}, 保存每个道对的lags
        stacking_method: str, 如 'linear', 'pws', 'robust'
        stack_kwargs: 叠加参数，如 power=2

    Returns:
        stacked_results: dict, {pair_key: (lags, stacked_ccf)}
    """
    # 开始叠加
    stacked_results = {}
    for key in merged_results:
        ccf_array = merged_results[key]
        if ccf_array.shape[0] == 0:
            continue
        # 使用指定方法叠加，直接传入二维数组
        stacked_ccf = stack_ccfs(ccf_array, method=stacking_method, **stack_kwargs)
        stacked_results[key] = (lags_dict[key], stacked_ccf)

    return stacked_results


def plot_stacked_cross_correlation(stacked_results, reference_index, dx, max_lag=None):
    """
    绘制最终叠加后的互相关图像
    
    Parameters:
        stacked_results: 叠加后的互相关结果字典
        reference_index: 参考道索引
        dx: 道间距
        max_lag: 最大滞后时间
    """
    if not stacked_results:
        print("[警告] 没有有效结果，跳过绘图")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 提取距离和互相关数据
    distances = []
    ccfs = []
    lags = None
    
    for key, (lags_arr, ccf_arr) in stacked_results.items():
        # 解析道索引
        ch_idx = int(key.split('--')[1][2:])
        distance = abs(ch_idx - reference_index) * dx
        distances.append(distance)
        ccfs.append(ccf_arr)
        
        if lags is None:
            lags = lags_arr
    
    if not distances or lags is None:
        print("[警告] 没有有效数据，跳过绘图")
        return

    # 按距离排序
    sorted_indices = np.argsort(distances)
    distances = np.array(distances)[sorted_indices]
    ccfs = np.array(ccfs)[sorted_indices]
    
    # 确定滞后时间范围
    if max_lag is not None:
        lag_mask = np.abs(lags) <= max_lag
        lags_plot = lags[lag_mask]
        ccfs_plot = ccfs[:, lag_mask]
    else:
        lags_plot = lags
        ccfs_plot = ccfs
    
    # 绘制互相关矩阵
    normalize = lambda x: (x) / np.max(np.abs(x), axis=-1, keepdims=True)
    im = ax1.imshow(normalize(ccfs_plot), aspect='auto', origin='lower',
                   extent=[lags_plot[0], lags_plot[-1], distances[0], distances[-1]],
                   cmap='RdBu_r', vmin=-0.8, vmax=0.8)
    ax1.set_xlabel('滞后时间 (秒)')
    ax1.set_ylabel('距参考道的距离 (米)')
    ax1.set_title(f'与参考道 {reference_index} 的叠加互相关矩阵')
    plt.colorbar(im, ax=ax1, label='归一化互相关')
    
    # 绘制几个典型距离的互相关曲线
    if len(distances) > 10:
        step = len(distances) // 10
        indices = range(0, len(distances), step)
    else:
        indices = range(len(distances))
    
    for i in indices:
        normalized_ccf = ccfs_plot[i] / np.max(np.abs(ccfs_plot[i]))
        ax2.plot(lags_plot, normalized_ccf + distances[i], 
                linewidth=1, color='black')
    
    ax2.set_xlabel('滞后时间 (秒)')
    ax2.set_ylabel('距离 (米)')
    ax2.set_title('叠加互相关波形（按距离偏移）')
    ax2.set_xlim(lags_plot[0], lags_plot[-1])
    
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：并行处理DAS H5文件并叠加互相关
    """
    parser = argparse.ArgumentParser(description="处理DAS H5文件并叠加互相关")
    parser.add_argument("--data_dir", type=str, default="../2024091912", help="H5文件所在目录")
    parser.add_argument("--pattern", type=str, default="*.h5", help="文件匹配模式")
    parser.add_argument("--stack_method", type=str, default="linear", choices=['linear', 'pws', 'robust', 'nroot', 'selective'],
                        help="叠加方法")
    parser.add_argument("--pws_power", type=float, default=2.0, help="PWS 方法中的幂指数")
    parser.add_argument("--output", type=str, default="stacked_cc_result.npz", help="输出文件路径")
    parser.add_argument("--max_files", type=int, default=0, help="限制处理的文件数量，0表示处理所有文件")
    args = parser.parse_args()

    import time
    
    # 步骤1: 扫描所有文件
    print("开始扫描文件...")
    start_scan_time = time.time()
    files = scan_h5_files(args.data_dir, pattern=args.pattern)
    end_scan_time = time.time()
    print(f"扫描文件完成，耗时: {end_scan_time - start_scan_time:.2f} 秒")
    print(f"发现 {len(files)} 个 H5 文件")

    if len(files) == 0:
        print("未找到任何文件，退出...")
        return

    # 限制处理的文件数量，方便调试
    if args.max_files > 0:
        files_to_process = files[:args.max_files]
    else:
        files_to_process = files
    print(f"将处理前 {len(files_to_process)} 个文件")
    
    # 步骤2: 顺序处理每个文件，内层互相关计算使用并行处理
    # 直接合并结果到一个字典中，每个键值对应一个二维数组
    merged_results = {}
    lags_dict = {}
    valid_files_count = 0
    
    # 内存监控函数
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024 / 1024  # MB
    
    # 使用tqdm显示进度
    for i, filepath in enumerate(tqdm(files_to_process, total=len(files_to_process), desc="Processing Files")):
        # 打印当前内存使用情况
        current_mem = get_memory_usage()
        
        start_file_time = time.time()
        result = process_single_file(filepath)
        end_file_time = time.time()
        
        if result is not None:
            valid_files_count += 1
            # 合并结果到merged_results字典，每次都转换为numpy数组
            for key, (lags, ccf) in result.items():
                ccf = np.array(ccf)  # 确保是numpy数组
                if key not in merged_results:
                    merged_results[key] = ccf[np.newaxis, :]  # 初始化为二维数组
                    lags_dict[key] = lags  # 保存lags，假设所有文件的lags相同
                else:
                    # 使用np.vstack合并，每次都保持为二维数组
                    merged_results[key] = np.vstack([merged_results[key], ccf])
            
            # 清理临时结果
            del result
            gc.collect()

    print(f"成功处理 {valid_files_count} 个文件")

    if valid_files_count == 0:
        print("没有有效结果，退出...")
        return

    # 步骤3: 合并并叠加
    print(f"开始使用 '{args.stack_method}' 方法叠加...")
    stack_kwargs = {}
    if args.stack_method == 'pws':
        stack_kwargs['power'] = args.pws_power

    final_stacked_results = group_and_stack_results(
        merged_results,
        lags_dict,
        stacking_method=args.stack_method,
        **stack_kwargs
    )
    print(f"共得到 {len(final_stacked_results)} 个道对的叠加结果")

    # 获取元信息用于绘图（从第一个有效文件读取）
    first_meta = None
    try:
        _, _, _, _, meta = read_zdh5(files[0])
        first_meta = meta
    except Exception as e:
        print(f"[警告] 读取第一个文件元信息失败: {e}")

    if first_meta is None:
        dx = 5.0  # fallback
        fs = 100.0
    else:
        dx = first_meta['dx']
        fs = first_meta['fs']

    # 步骤4: 绘图
    plot_stacked_cross_correlation(
        final_stacked_results,
        reference_index=0,
        dx=dx,
        max_lag=2.0
    )

    # 步骤5: 保存结果
    save_dict = {
        'results': final_stacked_results,
        'reference_index': 0,
        'sampling_rate': fs,
        'dx': dx,
        'files_used': [os.path.basename(f) for f in files if os.path.exists(f)],
        'stack_method': args.stack_method,
        **stack_kwargs
    }
    np.savez_compressed(args.output, **save_dict)
    print(f"✅ 叠加完成，结果已保存至: {args.output}")


if __name__ == "__main__":
    main()