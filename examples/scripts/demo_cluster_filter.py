import numpy as np
import matplotlib.pyplot as plt

from seismocorr.core.correlation.correlation import CorrelationEngine, CorrelationConfig
from seismocorr.core.correlation.stacking import stack_ccfs
from seismocorr.plugins.processing.three_stations_interferometry import ThreeStationInterferometry, ThreeStationConfig
# -----------------------------
# 1) 一个通用的 SNR 计算函数
# -----------------------------
def snr_peak_over_rms(
    lags: np.ndarray,
    ccf: np.ndarray,
    *,
    signal_win: tuple[float, float],
    noise_win: tuple[float, float],
) -> dict:
    """
    SNR 定义：SNR = max(|ccf| in signal window) / RMS(ccf in noise window)

    Args:
        signal_win: (t1, t2) 秒，比如 (-0.3, 0.3)
        noise_win : (t3, t4) 秒，比如 (1.0, 2.0) 或 (-2.0,-1.0)
                    你也可以传正窗或负窗，这里按绝对值 RMS 计算

    Returns:
        dict: snr, peak, noise_rms, peak_time
    """
    t1, t2 = signal_win
    n1, n2 = noise_win
    if t1 > t2 or n1 > n2:
        raise ValueError("window order must be (start, end) with start<=end")

    sig_mask = (lags >= t1) & (lags <= t2)
    noi_mask = (lags >= n1) & (lags <= n2)

    if not np.any(sig_mask):
        raise ValueError("signal window has no samples; adjust signal_win")
    if not np.any(noi_mask):
        raise ValueError("noise window has no samples; adjust noise_win")

    sig = ccf[sig_mask]
    noi = ccf[noi_mask]

    peak_idx = int(np.argmax(np.abs(sig)))
    peak_val = float(np.abs(sig[peak_idx]))
    peak_time = float(lags[sig_mask][peak_idx])

    noise_rms = float(np.sqrt(np.mean(noi.astype(float) ** 2)))
    snr = float(peak_val / (noise_rms + 1e-12))

    return {
        "snr": snr,
        "peak": peak_val,
        "noise_rms": noise_rms,
        "peak_time": peak_time,
    }

def align_by_common_lag(lags_a, ccf_a, lags_b, ccf_b):
    """
    把两条曲线裁剪到共同的 lag 范围，并按 lag 对齐（假设采样间隔相同）。
    适用于：两者都是均匀采样且 dt=1/sr。
    """
    dt_a = np.median(np.diff(lags_a))
    dt_b = np.median(np.diff(lags_b))
    if not np.isclose(dt_a, dt_b, rtol=1e-6, atol=1e-12):
        raise ValueError("lag sampling intervals differ; cannot align by simple slicing")

    tmin = max(lags_a.min(), lags_b.min())
    tmax = min(lags_a.max(), lags_b.max())

    mask_a = (lags_a >= tmin) & (lags_a <= tmax)
    mask_b = (lags_b >= tmin) & (lags_b <= tmax)

    lags_a2, ccf_a2 = lags_a[mask_a], ccf_a[mask_a]
    lags_b2, ccf_b2 = lags_b[mask_b], ccf_b[mask_b]

    # 再次确保长度一致（可能因端点取整差1个点）
    m = min(len(lags_a2), len(lags_b2))
    return lags_a2[:m], ccf_a2[:m], lags_b2[:m], ccf_b2[:m]

# -----------------------------
# 2) 生成/读取数据（你换成真实 traces 即可）
# -----------------------------
sr = 200.0
N = 60
T = int(sr * 60)
rng = np.random.default_rng(0)

# 演示用：带公共成分的随机信号（真实数据请直接用你的 traces）
common = rng.standard_normal(T)
traces = np.stack([0.5 * common + rng.standard_normal(T) for _ in range(N)], axis=0)

# -----------------------------
# 3) 外部互相关函数（调用你包里的 CorrelationEngine）
# -----------------------------
engine = CorrelationEngine()

# 一次互相关窗口（论文/常规 NCF 都会先截到 ±max_lag）
max_lag_1 = 2.0
cc_cfg = CorrelationConfig(method="freq-domain", max_lag=max_lag_1, nfft=None)

def xcorr_func(x: np.ndarray, y: np.ndarray, *, config: CorrelationConfig):
    return engine.compute_cross_correlation(x, y, sampling_rate=sr, config=config)

# -----------------------------
# 4) 三站干涉（AUTO 分段）对象
# -----------------------------
# 你用的是我给你的 auto 分段版（k 在中间卷积，否则相关）
tsi = ThreeStationInterferometry(
    sampling_rate=sr,
    xcorr_func=xcorr_func,
    cfg=ThreeStationConfig(mode="auto", max_lag2=2.0),  # 二次输出也裁到 ±2s，便于对齐比较
)

# -----------------------------
# 5) 选一对台站做对比
# -----------------------------
i, j = 10, 40
k_list = None  # None = 默认全部 k（排除 i/j）

# A) 直接互相关
lags_ij, ccf_ij = xcorr_func(traces[i], traces[j], config=cc_cfg)

# B) 三站干涉：得到很多条 ncf_ijk
res = tsi.compute_pair(traces, i=i, j=j, k_list=k_list, config=cc_cfg)
lags2 = res["lags2"]
ccfs_ijk = res["ccfs"]

# B-1) 叠加（只在 demo 做）：你可换 linear / pws / robust 等
stacked_3s_linear = stack_ccfs(ccfs_ijk, method="linear")
stacked_3s_pws = stack_ccfs(ccfs_ijk, method="pws", power=2)

# -----------------------------
# 6) 对齐到共同 lag 再算 SNR
# -----------------------------
l1, a1, l2, a2 = align_by_common_lag(lags_ij, ccf_ij, lags2, stacked_3s_linear)
_,  b1, _,  b2 = align_by_common_lag(lags_ij, ccf_ij, lags2, stacked_3s_pws)

# 你自己设窗：这里给一个常见例子
# signal: 中心附近（比如面波主能量在小延迟）
# noise : 远离中心的一段
signal_win = (-0.3, 0.3)
noise_win = (1.0, 2.0)

snr_direct = snr_peak_over_rms(l1, a1, signal_win=signal_win, noise_win=noise_win)
snr_3s_lin = snr_peak_over_rms(l2, a2, signal_win=signal_win, noise_win=noise_win)
snr_3s_pws = snr_peak_over_rms(l2, b2, signal_win=signal_win, noise_win=noise_win)

print("\n=== SNR comparison (peak/RMS) ===")
print(f"Direct CCF:      SNR={snr_direct['snr']:.3f}, peak={snr_direct['peak']:.4g} at {snr_direct['peak_time']:.3f}s, noise_rms={snr_direct['noise_rms']:.4g}")
print(f"3-station linear: SNR={snr_3s_lin['snr']:.3f}, peak={snr_3s_lin['peak']:.4g} at {snr_3s_lin['peak_time']:.3f}s, noise_rms={snr_3s_lin['noise_rms']:.4g}")
print(f"3-station PWS:    SNR={snr_3s_pws['snr']:.3f}, peak={snr_3s_pws['peak']:.4g} at {snr_3s_pws['peak_time']:.3f}s, noise_rms={snr_3s_pws['noise_rms']:.4g}")

# -----------------------------
# 7) 可视化对比
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(l1, a1, label=f"Direct CCF (SNR={snr_direct['snr']:.2f})")
plt.plot(l2, a2, label=f"3-station linear stack (SNR={snr_3s_lin['snr']:.2f})")
plt.plot(l2, b2, label=f"3-station PWS stack (SNR={snr_3s_pws['snr']:.2f})")
plt.axvspan(signal_win[0], signal_win[1], alpha=0.15, label="signal window")
plt.axvspan(noise_win[0], noise_win[1], alpha=0.10, label="noise window")
plt.title(f"Direct CCF vs Three-station Interferometry (pair {i}-{j})")
plt.xlabel("Lag (s)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
