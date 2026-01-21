import pytest
import numpy as np
import sys
import os

from seismocorr.preprocessing.time_norm import (
    ZScoreNormalizer,
    OneBitNormalizer,
    RMSNormalizer,
    ClipNormalizer,
    NoTimeNorm,
    RAMNormalizer,
    WaterLevelNormalizer,
    CWTSoftThreshold1D,
    get_time_normalizer,
    _TIME_NORM_MAP
)


class TestZScoreNormalizer:
    """测试ZScore归一化"""
    
    def test_zscore_normalization(self, sample_signal):
        """测试ZScore归一化结果"""
        normalizer = ZScoreNormalizer()
        normalized = normalizer.apply(sample_signal)
        
        # 检查均值为0，标准差为1
        assert np.isclose(normalized.mean(), 0.0, atol=1e-10)
        assert np.isclose(normalized.std(), 1.0, atol=1e-10)
    
    def test_zscore_constant_signal(self, constant_signal):
        """测试常数信号的ZScore归一化"""
        normalizer = ZScoreNormalizer()
        normalized = normalizer.apply(constant_signal)
        
        # 常数信号归一化后应为全零
        assert np.allclose(normalized, 0.0)
    
    def test_zscore_zero_signal(self, zero_signal):
        """测试全零信号的ZScore归一化"""
        normalizer = ZScoreNormalizer()
        normalized = normalizer.apply(zero_signal)
        
        # 全零信号归一化后仍为全零
        assert np.allclose(normalized, 0.0)


class TestOneBitNormalizer:
    """测试1-bit归一化"""
    
    def test_onebit_normalization(self, sample_signal):
        """测试1-bit归一化结果"""
        normalizer = OneBitNormalizer()
        normalized = normalizer.apply(sample_signal)
        
        # 检查所有值为±1
        assert np.all(np.abs(normalized) == 1.0)
        assert set(np.unique(normalized)).issubset({-1.0, 1.0})
    
    def test_onebit_preserves_sign(self, sample_signal):
        """测试1-bit归一化保持原始符号"""
        normalizer = OneBitNormalizer()
        normalized = normalizer.apply(sample_signal)
        
        original_sign = np.sign(sample_signal)
        result_sign = np.sign(normalized)
        
        # 符号应该保持一致（除了零值）
        nonzero_mask = sample_signal != 0
        if np.any(nonzero_mask):
            assert np.array_equal(
                original_sign[nonzero_mask], 
                result_sign[nonzero_mask]
            )


class TestRMSNormalizer:
    """测试RMS归一化"""
    
    def test_rms_normalization(self, sample_signal):
        """测试RMS归一化结果"""
        normalizer = RMSNormalizer()
        normalized = normalizer.apply(sample_signal)
        
        # 检查RMS值为1
        rms = np.sqrt(np.mean(normalized ** 2))
        assert np.isclose(rms, 1.0, atol=1e-10)
    
    def test_rms_zero_signal(self, zero_signal):
        """测试全零信号的RMS归一化"""
        normalizer = RMSNormalizer()
        normalized = normalizer.apply(zero_signal)
        
        # 全零信号归一化后仍为全零
        assert np.allclose(normalized, 0.0)


class TestClipNormalizer:
    """测试截幅归一化"""
    
    def test_clip_normalization(self, outlier_signal):
        """测试截幅归一化结果"""
        clip_val = 3.0
        normalizer = ClipNormalizer(clip_val=clip_val)
        normalized = normalizer.apply(outlier_signal)
        
        # 检查所有值在[-clip_val, clip_val]范围内
        assert np.all(normalized >= -clip_val)
        assert np.all(normalized <= clip_val)
        
        # 检查异常值被正确截断
        assert np.max(normalized) <= clip_val
        assert np.min(normalized) >= -clip_val
    
    def test_clip_custom_value(self, outlier_signal):
        """测试自定义截断值"""
        custom_clip = 2.0
        normalizer = ClipNormalizer(clip_val=custom_clip)
        normalized = normalizer.apply(outlier_signal)
        
        assert np.all(normalized >= -custom_clip)
        assert np.all(normalized <= custom_clip)


class TestNoTimeNorm:
    """测试无操作归一化"""
    
    def test_no_normalization(self, sample_signal):
        """测试无操作归一化"""
        normalizer = NoTimeNorm()
        normalized = normalizer.apply(sample_signal)
        
        # 应该返回原始信号的副本
        assert np.array_equal(normalized, sample_signal)
        assert normalized is not sample_signal  # 应该是副本


class TestRAMNormalizer:
    """测试RAM归一化"""
    
    def test_ram_normalization_basic(self, sample_signal, ram_normalizer_params):
        """测试RAM归一化基本功能"""
        normalizer = RAMNormalizer(**ram_normalizer_params)
        normalized = normalizer.apply(sample_signal.copy())

        # 基本检查：输出应与输入同形状
        assert normalized.shape == sample_signal.shape
        assert not np.allclose(normalized, sample_signal)  # 应该有所改变
    
    def test_ram_normalizer_parameters(self):
        """测试RAM归一化器参数"""
        fmin, Fs, norm_win = 2.0, 200.0, 0.5
        normalizer = RAMNormalizer(fmin, Fs, norm_win)
        
        assert normalizer.fmin == fmin
        assert normalizer.Fs == Fs
        assert normalizer.norm_win == norm_win


class TestWaterLevelNormalizer:
    """测试 WaterLevelNormalizer"""

    def test_waterlevel_basic(self, waterlevel_signal, waterlevel_params):
        normalizer = WaterLevelNormalizer(**waterlevel_params)
        x = waterlevel_signal.copy()
        y = normalizer.apply(x)

        # shape 不变
        assert y.shape == x.shape
        # 输出应是新数组（不是同一对象）
        assert y is not x
        # 不应产生 NaN/Inf
        assert np.isfinite(y).all()

        # 应该确实“压制”了强能量窗口（不要求每点都变，只要求整体能量被限制）
        Fs = int(waterlevel_params["Fs"])
        win_n = int(round(waterlevel_params["win_length"] * Fs))

        # 计算处理后的全局 RMS 水位 W（注意实现用的是处理前 global rms）
        global_rms = np.sqrt(np.mean(x * x)) + waterlevel_params["eps"]
        W = waterlevel_params["water_level_factor"] * global_rms

        # 强能量窗口（第2窗）处理后 RMS 应 <= W（容忍小误差）
        w2 = y[win_n : 2 * win_n]
        w2_rms = np.sqrt(np.mean(w2 * w2))
        assert w2_rms <= W * 1.001  # 容差

    def test_waterlevel_empty(self, waterlevel_params):
        normalizer = WaterLevelNormalizer(**waterlevel_params)
        x = np.array([])
        y = normalizer.apply(x)
        assert y.shape == (0,)

    def test_waterlevel_single_point(self, waterlevel_params):
        normalizer = WaterLevelNormalizer(**waterlevel_params)
        x = np.array([5.0])
        y = normalizer.apply(x)
        assert y.shape == (1,)
        assert np.isfinite(y).all()

    def test_waterlevel_invalid_window(self):
        # win_length * Fs < 1 应该报错
        normalizer = WaterLevelNormalizer(Fs=10.0, win_length=0.0)
        with pytest.raises(ValueError, match="win_length \\* Fs must be"):
            normalizer.apply(np.random.randn(100))


class TestCWTSoftThreshold1D:
    """测试基于 CWT 的软阈值处理"""

    def test_cwt_designal_basic(self, cwt_signal, cwt_params):
        x = cwt_signal.copy()

        normalizer = CWTSoftThreshold1D(
            fs=cwt_params["Fs"],
            noise_idx=cwt_params["noise_idx"],
            mode="designal",
            wavelet=cwt_params["wavelet"],
            voices_per_octave=cwt_params["voices_per_octave"],
            quantile=cwt_params["quantile"],
            f_min=cwt_params["f_min"],
            f_max=cwt_params["f_max"],
            normalize=cwt_params["normalize"],
            eps=cwt_params["eps"],
        )

        y = normalizer.apply(x)

        # shape 不变
        assert y.shape == x.shape
        assert y is not x
        assert np.isfinite(y).all()

        # designal 的直觉：尖峰应该被压制一些（最大值降低）
        assert np.max(np.abs(y)) <= np.max(np.abs(x)) + 1e-9

    def test_cwt_denoise_reduces_noise_window(self, cwt_signal, cwt_params):
        x = cwt_signal.copy()
        noise_idx = cwt_params["noise_idx"]

        normalizer = CWTSoftThreshold1D(
            fs=cwt_params["Fs"],
            noise_idx=noise_idx,
            mode="denoise",
            wavelet=cwt_params["wavelet"],
            voices_per_octave=cwt_params["voices_per_octave"],
            quantile=cwt_params["quantile"],
            f_min=cwt_params["f_min"],
            f_max=cwt_params["f_max"],
            normalize=False,  # 为了更直接比较噪声能量，先关掉幅值对齐
            eps=cwt_params["eps"],
        )

        y = normalizer.apply(x)

        # 处理后在 noise window 的能量（RMS 或 std）应该降低或不增（容差）
        x_rms = np.sqrt(np.mean(x[noise_idx] ** 2))
        y_rms = np.sqrt(np.mean(y[noise_idx] ** 2))
        assert y_rms <= x_rms * 1.05  # 给一点容差，避免边界波动导致偶发失败

    def test_cwt_empty(self, cwt_params):
        normalizer = CWTSoftThreshold1D(
            fs=cwt_params["Fs"],
            noise_idx=cwt_params["noise_idx"],
            mode="designal",
            voices_per_octave=cwt_params["voices_per_octave"],
            f_min=cwt_params["f_min"],
            f_max=cwt_params["f_max"],
        )
        x = np.array([])
        y = normalizer.apply(x)
        assert y.shape == (0,)

    def test_cwt_invalid_mode(self, cwt_signal, cwt_params):
        normalizer = CWTSoftThreshold1D(
            fs=cwt_params["Fs"],
            noise_idx=cwt_params["noise_idx"],
            mode="invalid",
            voices_per_octave=cwt_params["voices_per_octave"],
            f_min=cwt_params["f_min"],
            f_max=cwt_params["f_max"],
        )
        with pytest.raises(ValueError, match="mode must be"):
            normalizer.apply(cwt_signal)

    def test_cwt_invalid_freqs(self, cwt_params):
        # f_min >= f_max 应该报错（在 build_scales 阶段）
        with pytest.raises(ValueError, match="Require 0 < f_min < f_max"):
            CWTSoftThreshold1D(
                fs=cwt_params["Fs"],
                noise_idx=cwt_params["noise_idx"],
                f_min=10.0,
                f_max=5.0,
            )


class TestGetTimeNormalizer:
    """测试归一化器工厂函数"""
    
    def test_get_all_normalizers(self, ram_normalizer_params,
                                waterlevel_params,
                                cwt_params):
        for name in _TIME_NORM_MAP.keys():
            if name == "ramn":
                normalizer = get_time_normalizer(name, **ram_normalizer_params)
            elif name == "waterlevel":
                normalizer = get_time_normalizer(name, **waterlevel_params)
            elif name == "cwt-soft":
                normalizer = get_time_normalizer(name, **cwt_params)
            else:
                normalizer = get_time_normalizer(name)

            assert normalizer is not None
            assert hasattr(normalizer, "apply")
    
    def test_get_normalizer_with_params(self):
        """测试带参数的归一化器获取"""
        # 测试ClipNormalizer带参数
        clip_normalizer = get_time_normalizer('clip', clip_val=2.5)
        assert isinstance(clip_normalizer, ClipNormalizer)
        assert clip_normalizer.clip_val == 2.5
        
        # 测试RAMNormalizer带参数
        ram_normalizer = get_time_normalizer('ramn', fmin=1.0, Fs=100.0, npts=1000)
        assert isinstance(ram_normalizer, RAMNormalizer)
        assert ram_normalizer.fmin == 1.0
    
    def test_get_normalizer_invalid_name(self):
        """测试无效归一化器名称"""
        with pytest.raises(ValueError, match="未知的时域归一化方法"):
            get_time_normalizer('invalid_method')
    
    def test_ram_normalizer_missing_params(self):
        """测试RAM归一化器缺少必需参数"""
        with pytest.raises(ValueError, match="RAMNormalizer requires"):
            get_time_normalizer('ramn')  # 缺少必需参数
    
    def test_normalizer_callable_interface(self, sample_signal):
        """测试归一化器的可调用接口"""
        normalizer = get_time_normalizer('zscore')
        
        # 测试apply方法和__call__方法应该一致
        result1 = normalizer.apply(sample_signal)
        result2 = normalizer(sample_signal)
        
        assert np.array_equal(result1, result2)

 
    def test_get_time_normalizer_missing_required_params_raises(self):
        """测试：需要参数的 normalizer 缺参应抛 ValueError"""
        with pytest.raises(ValueError):
            get_time_normalizer("ramn")  # 缺 fmin/Fs

        with pytest.raises(ValueError):
            get_time_normalizer("waterlevel")  # 缺 Fs

        with pytest.raises(ValueError):
            get_time_normalizer("cwt-soft")  # 缺 fs/noise_idx


class TestEdgeCases:
    """测试边界情况"""
    def _cwt_factory_params(self, cwt_params, signal_len: int):
        p = dict(cwt_params)
        # 兼容 cwt_params 里用 Fs 或 fs
        Fs = p.get("Fs", p.get("fs"))
        p["Fs"] = Fs
        p.pop("fs", None)
        p.pop("n", None)

        # 根据输入信号长度保证 noise_idx 合法
        if signal_len == 0:
            # 空信号时 noise_idx 不会被用到（apply 直接返回），但工厂函数要求非 None
            p["noise_idx"] = slice(0, 0)
        elif signal_len == 1:
            p["noise_idx"] = np.array([0], dtype=int)
        else:
            # 前 100 个点当噪声段，别超过长度
            k = min(100, signal_len)
            p["noise_idx"] = np.arange(0, k, dtype=int)

        return p
    def test_empty_signal(self, ram_normalizer_params, waterlevel_params, cwt_params):
        """测试空信号"""
        empty_signal = np.array([])
        # 把 cwt_params(含 fs) 转成工厂函数需要的 Fs
        cwt_factory_params = dict(cwt_params)
        cwt_factory_params["Fs"] = cwt_factory_params.pop("Fs")
        cwt_factory_params.pop("n", None)

        for name in ['zscore', 'one-bit', 'rms', 'clip', 'no', 'ramn', 'waterlevel', 'cwt-soft']:
            if name == 'ramn':
                normalizer = get_time_normalizer(name, **ram_normalizer_params)
            elif name == 'waterlevel':
                normalizer = get_time_normalizer(name, **waterlevel_params)
            elif name == 'cwt-soft':
                normalizer = get_time_normalizer(name, **cwt_factory_params)
            else:
                normalizer = get_time_normalizer(name)

            result = normalizer.apply(empty_signal)
            assert len(result) == 0
    
    def test_single_point_signal(self, ram_normalizer_params):
            """测试单点信号"""
            single_point = np.array([5.0])
            Fs = ram_normalizer_params["Fs"]

            # 对于 cwt-soft，noise_idx 至少要落在 [0, n-1] 内；单点信号就用 [0]
            for name in ['zscore', 'one-bit', 'rms', 'clip', 'no', 'ramn', 'waterlevel', 'cwt-soft']:
                if name == 'ramn':
                    normalizer = get_time_normalizer(name, **ram_normalizer_params)
                elif name == 'waterlevel':
                    normalizer = get_time_normalizer(name, Fs=Fs)
                elif name == 'cwt-soft':
                    normalizer = get_time_normalizer(name, Fs=Fs, noise_idx=np.array([0], dtype=int))
                else:
                    normalizer = get_time_normalizer(name)

                result = normalizer.apply(single_point)
                assert len(result) == 1

    def test_large_signal(self, ram_normalizer_params, waterlevel_params, cwt_params):
        large_signal = np.random.randn(100000)
        cwt_factory_params = self._cwt_factory_params(cwt_params, signal_len=len(large_signal))

        for name in ['zscore', 'one-bit', 'rms', 'clip', 'no', 'ramn', 'waterlevel', 'cwt-soft']:
            if name == 'ramn':
                normalizer = get_time_normalizer(name, **ram_normalizer_params)
            elif name == 'waterlevel':
                normalizer = get_time_normalizer(name, **waterlevel_params)
            elif name == 'cwt-soft':
                normalizer = get_time_normalizer(name, **cwt_factory_params)
            else:
                normalizer = get_time_normalizer(name)
            import time
            start = time.time()
            result = normalizer.apply(large_signal)
            elapsed = time.time() - start

            assert len(result) == len(large_signal)

            # cwt-soft 通常更慢，避免 CI 不稳定
            if name == "cwt-soft":
                assert elapsed < 5
            else:
                assert elapsed < 1.0




def test_normalizer_immutability(sample_signal, ram_normalizer_params):
    """测试归一化器不会修改原始信号"""
    original_copy = sample_signal.copy()
    Fs = ram_normalizer_params["Fs"]

    for name in _TIME_NORM_MAP.keys():
        if name == "ramn":
            normalizer = get_time_normalizer(name, **ram_normalizer_params)

        elif name == "waterlevel":
            normalizer = get_time_normalizer(name, Fs=Fs)

        elif name == "cwt-soft":
            normalizer = get_time_normalizer(
                name,
                Fs=Fs,
                noise_idx=np.arange(0, 100, dtype=int),
            )

        else:
            normalizer = get_time_normalizer(name)

        # apply 不应修改原始数组
        _ = normalizer.apply(sample_signal)
        assert np.array_equal(sample_signal, original_copy)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])