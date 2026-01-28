# seismocorr/config/defaults.py

"""
默认配置模块：为整个 seismocorr 框架提供默认参数。
用户可通过 CorrelationConfigBuilder 显式覆盖这些值。
"""

# ========================================
# 🔧 基础采样与时间参数
# ========================================
DEFAULT_SAMPLING_RATE = 100.0  # Hz，典型地震数据
DEFAULT_CC_WINDOW_SECONDS = 3600  # 1小时滑动窗
DEFAULT_MAX_LAG_SECONDS = 120.0  # ±120秒用于互相关

# ========================================
# 🌐 频带滤波范围（单位：Hz）
# ========================================
DEFAULT_FREQ_MIN = 0.01
DEFAULT_FREQ_MAX = 10.0

# ========================================
# 📦 HDF5 与 I/O 设置
# ========================================
DEFAULT_HDF5_CHUNK_SIZE = 1024 * 1024        # 每次读取约 8MB (float64)
DEFAULT_BUFFER_SIZE_MB = 256                 # 内存缓冲区大小
DEFAULT_COMPRESSION = 'gzip'                 # HDF5 压缩方式
DEFAULT_DATA_PATH_PATTERN = "raw_data/*/*.h5" # 默认搜索路径模式

# ========================================
# 🧼 数据预处理默认设置
# ========================================
# 归一化方法选项
NORMALIZATION_OPTIONS = ['zscore', 'one-bit', 'rms', 'no']
DEFAULT_NORMALIZATION_METHOD = 'one-bit'  # 抗噪强，常用在噪声互相关

# 叠加方法选项
STACKING_OPTIONS = ['linear', 'pws', 'robust', 'selective']
DEFAULT_STACKING_METHOD = 'linear'

# 趋势移除
DEFAULT_DETREND = True
DEFAULT_DETREND_TYPE = 'linear'  # 或 'demean'

# 滤波类型
DEFAULT_FILTER_TYPE = 'bandpass'
DEFAULT_ZERO_PHASE = True       # 是否零相位滤波（前后两次滤波）


# ========================================
#  部分主函数默认设置
# ========================================
# 互相关方法选项
SUPPORTED_METHODS = ["time-domain", "freq-domain", "deconv", "coherency"]

# 空间相速度反演选项
SUPPORTED_ASSUMPTION = ["station_avg", "ray_avg"]
SUPPORTED_GEOMETRY = ["1d", "2d"]
SUPPORTED_REGULARIZATIONS = ["none", "l2", "l1", "l1_l2"]


# ========================================
# 💡 处理流程控制
# ========================================
DEFAULT_N_PARALLEL = 4           # 并行处理线程数（可根据 CPU 自动检测）
DEFAULT_USE_GPU = False          # 是否启用 GPU 加速（需 CuPy 安装）
DEFAULT_SAVE_INTERMEDIATE = False # 是否保存中间结果（individual CCFs）

# ========================================
# 📂 输出与日志
# ========================================
DEFAULT_OUTPUT_DIR = "./seismocorr_output"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_RESULT_FORMAT = "hdf5"   # 或 'zarr', 'netcdf'
DEFAULT_FILE_PREFIX = "ccf_"

# ========================================
# 📍 地理信息相关默认值
# ========================================
DEFAULT_COORDINATE_SYSTEM = "WGS84"
DEFAULT_EARTH_RADIUS_KM = 6371.0
DEFAULT_MIN_INTERSTATION_DISTANCE = 1.0  # km，太近的台站可能不参与

# ========================================
# 🧪 插件系统默认启用项（可选）
# ========================================
DEFAULT_PLUGINS_ENABLED = [
    # 'mfj', 
    # 'slant_stack',
    # 'coherence_filter'
]

# ========================================
# 🛠️ 高级调试与性能
# ========================================
DEFAULT_ENABLE_CACHE = True      # 启用 HDF5 数据缓存
DEFAULT_CHECKSUM_ON_READ = False # 是否校验数据完整性
DEFAULT_TIMEOUT_SECONDS = 300    # 单个 trace 处理超时时间
