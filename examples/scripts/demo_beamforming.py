import numpy as np
import matplotlib.pyplot as plt

from seismocorr.plugins.processing.beamforming import Beamformer


def band_limited_noise(rng, n, fs, fmin, fmax):
    """用频域置带生成带限噪声"""
    # rfft 频率轴
    freqs = np.fft.rfftfreq(n, d=1/fs)
    spec = rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
    band = (freqs >= fmin) & (freqs <= fmax)
    spec[~band] = 0.0
    x = np.fft.irfft(spec, n=n)
    x = x / (np.std(x) + 1e-12)
    return x


def simulate_ambient_plane_wave(
    fs: float,
    duration_s: float,
    xy_m: np.ndarray,
    azimuth_deg: float,
    slowness_s_per_m: float,
    fmin_hz: float,
    fmax_hz: float,
    snr_db: float = 0.0,
    seed: int = 0,
):
    """
    合成“背景噪声里有一个占优方向”的平面波模型：
      x_i(t) = a * n_band(t - τ_i) + noise
      τ_i = s * (r_i · u)

    n_band 是带限噪声（0.5-2.5 Hz 之类），更像环境波场。
    """
    rng = np.random.default_rng(seed)
    n_chan = xy_m.shape[0]
    n_samples = int(round(duration_s * fs))
    t = np.arange(n_samples) / fs

    az = np.deg2rad(azimuth_deg)
    u = np.array([np.sin(az), np.cos(az)])  # East, North
    tau = slowness_s_per_m * (xy_m @ u)     # (n_chan,)

    # 生成一个“源噪声”并对各通道施加不同延迟
    src = band_limited_noise(rng, n_samples, fs, fmin_hz, fmax_hz)

    sig = np.zeros((n_chan, n_samples), float)
    for c in range(n_chan):
        # 用插值实现非整数延迟
        tt = t - tau[c]
        sig[c] = np.interp(tt, t, src, left=0.0, right=0.0)

    # 加独立背景噪声控制 SNR
    signal_power = np.mean(sig**2)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_lin if snr_lin > 0 else signal_power * 10
    noise = rng.normal(scale=np.sqrt(noise_power), size=sig.shape)

    return sig + noise


def main():
    # 1) 阵列：4x4、10m 间距（演示用）
    spacing = 10.0
    grid = np.array([(i, j) for j in range(4) for i in range(4)], dtype=float)
    xy_m = (grid - grid.mean(axis=0)) * spacing  # x=East, y=North

    # 2) 城市地震背景噪声常见频段：比如 0.5-2.5 Hz（你可按数据改）
    fs = 50.0
    duration_s = 10 * 60.0  # 10 min

    fmin = 0.5
    fmax = 2.5

    # 3) 真值：一个占优噪声来向（演示）
    true_az = 60.0
    true_v = 1200.0
    true_s = 1.0 / true_v  # ~0.000833 s/m
    snr_db = -3.0          # 背景噪声里不必很高 SNR

    data = simulate_ambient_plane_wave(
        fs=fs,
        duration_s=duration_s,
        xy_m=xy_m,
        azimuth_deg=true_az,
        slowness_s_per_m=true_s,
        fmin_hz=fmin,
        fmax_hz=fmax,
        snr_db=snr_db,
        seed=1,
    )

    # 4) 扫描网格：慢度覆盖 333~3333 m/s
    az_scan = np.arange(0, 360, 2.0)
    s_scan = np.linspace(0.0003, 0.0030, 136)

    bf = Beamformer(
        fs=fs,
        fmin=fmin,
        fmax=fmax,
        frame_len_s=20.0,  # 低频要更长窗
        hop_s=10.0,
        window="hann",
        whiten=True,
    )

    res = bf.beamform(data=data, xy_m=xy_m, azimuth_deg=az_scan, slowness_s_per_m=s_scan)

    idx = np.unravel_index(np.argmax(res.power), res.power.shape)
    est_s = res.slowness_s_per_m[idx[0]]
    est_az = res.azimuth_deg[idx[1]]
    print(f"True azimuth={true_az:.1f} deg, True slowness={true_s:.6f} s/m (v={true_v:.1f} m/s)")
    print(f"Est  azimuth={est_az:.1f} deg, Est  slowness={est_s:.6f} s/m (v={1/est_s:.1f} m/s)")

    # 2D power
    plt.figure()
    plt.imshow(
        res.power,
        aspect="auto",
        origin="lower",
        extent=[res.azimuth_deg[0], res.azimuth_deg[-1], res.slowness_s_per_m[0], res.slowness_s_per_m[-1]],
    )
    plt.xlabel("Azimuth (deg, from North clockwise)")
    plt.ylabel("Slowness (s/m)")
    plt.title("Beamforming Power (ambient-like band-limited noise)")
    plt.colorbar(label="Power")
    plt.scatter([true_az], [true_s], marker="o", label="True")
    plt.scatter([est_az], [est_s], marker="x", label="Estimated")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()