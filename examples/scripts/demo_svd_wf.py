from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from seismocorr.plugins.processing.svd_wf import SVDLowRankDenoiser ,WienerFilterDenoiser


def make_synthetic_ncf_windows(
    n_windows: int = 300,
    n_lag: int = 2048,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成示例数据（窗口×lag）用于 demo。

    输出
    ----
    X_noisy : np.ndarray, shape (n_windows, n_lag)
        含干扰/噪声的输入矩阵
    X_clean : np.ndarray, shape (n_windows, n_lag)
        仅包含相干模板（用于对照）
    """
    rng = np.random.default_rng(seed)
    lag = np.linspace(-1.0, 1.0, n_lag)

    def packet(mu, f, w):
        return np.exp(-((lag - mu) / w) ** 2) * np.cos(2 * np.pi * f * (lag - mu))

    template = 1.2 * packet(0.25, f=10, w=0.08) + 1.0 * packet(-0.25, f=10, w=0.08)
    template += 0.6 * packet(0.45, f=6, w=0.12) + 0.6 * packet(-0.45, f=6, w=0.12)

    amps = 1.0 + 0.15 * rng.standard_normal(n_windows)
    X_clean = amps[:, None] * template[None, :]

    spike = np.exp(-(lag / 0.03) ** 2)
    inter_amp = np.zeros(n_windows)
    idx = rng.choice(n_windows, size=n_windows // 3, replace=False)
    inter_amp[idx] = 2.5 + 0.5 * rng.standard_normal(len(idx))
    interference = inter_amp[:, None] * spike[None, :]

    noise = 0.8 * rng.standard_normal((n_windows, n_lag))

    X_noisy = X_clean + interference + noise
    return X_noisy, X_clean


def main():
    """
    Demo 流程（与原 SVDWF 等价）：

    输入
    ----
    X : (n_windows, n_samples)

    处理
    ----
    1) SVD 低秩：在中心化域得到 X_lr0
    2) 残差估计：residual = X0 - X_lr0
    3) Wiener：用 X_lr0 与 residual 估计增益，并滤波得到 X_filt0
    4) 回到原始基线：X_out = X_filt0 + mean_

    输出
    ----
    X_out : (n_windows, n_samples)
        去噪后的矩阵
    """
    X, _ = make_synthetic_ncf_windows()

    # 1) SVD 低秩
    svd_den = SVDLowRankDenoiser(
        rank=1,            # 若需自动选 rank：设 rank=None，并配置 method/energy/thresh
        method="energy",
        energy=0.90,
        center=True,
        random_sign_fix=True,
    )
    svd_den.fit(X)

    X0 = svd_den.center_data(X)                 # (n_windows, n_samples)
    X_lr0 = svd_den.transform(X, add_mean=False)  # (n_windows, n_samples)，中心化域

    # 2) Wiener（基于低秩部分与残差估计谱）
    residual = X0 - X_lr0
    wien_den = WienerFilterDenoiser(
        psd_smooth=21,
        wiener_beta=1.0,
        gain_floor=0.03,
    )
    wien_den.fit(X_lr0, residual)
    X_filt0 = wien_den.transform(X_lr0)

    # 3) 输出：加回均值
    X_out = X_filt0 + svd_den.mean_

    # 可视化：奇异值谱
    fig1, ax1 = plt.subplots()
    svd_den.plot_spectrum(ax=ax1, log=True)

    # 可视化：Wiener 增益
    fig2, ax2 = plt.subplots()
    wien_den.plot_wiener_gain(ax=ax2)

    # 对比图：输入 / 输出 / 去除部分
    w_slice = slice(0, 120)
    lag_slice = slice(600, 1450)

    fig3, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(X[w_slice, lag_slice], aspect="auto")
    axes[0].set_title("Input X")
    axes[0].set_xlabel("Lag sample index")
    axes[0].set_ylabel("Window index")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(X_out[w_slice, lag_slice], aspect="auto")
    axes[1].set_title(f"Output (rank={svd_den.rank_})")
    axes[1].set_xlabel("Lag sample index")
    axes[1].set_ylabel("Window index")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow((X - X_out)[w_slice, lag_slice], aspect="auto")
    axes[2].set_title("Removed (X - Output)")
    axes[2].set_xlabel("Lag sample index")
    axes[2].set_ylabel("Window index")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.show()

    # 叠加曲线对比（很多流程只看 stack）
    stack_in = X.mean(axis=0)
    stack_out = X_out.mean(axis=0)

    fig4, ax4 = plt.subplots()
    ax4.plot(stack_in, label="Stacked input")
    ax4.plot(stack_out, label="Stacked output")
    ax4.legend()
    ax4.set_title("Stack comparison")
    ax4.grid(True)
    plt.show()


if __name__ == "__main__":
    main()