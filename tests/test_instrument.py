"""Tests for oscilloscope-style measurement bindings (gap-analysis #101)."""

import math

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


SAMPLE_RATE = 8000.0


def _sine(freq, n=4096, amp=1.0, phase=0.0):
    t = np.arange(n) / SAMPLE_RATE
    return amp * np.sin(2.0 * np.pi * freq * t + phase)


class TestPeakToPeak:
    def test_unit_sine_returns_two(self):
        sig = _sine(300.0, n=4096)
        assert abs(mpdsp.peak_to_peak(sig) - 2.0) < 1e-6

    def test_dc_signal_returns_zero(self):
        sig = np.full(1000, 0.5)
        assert mpdsp.peak_to_peak(sig) == 0.0

    def test_arbitrary_range(self):
        sig = np.array([-3.0, -1.0, 0.0, 4.0, 2.0])
        assert mpdsp.peak_to_peak(sig) == 7.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mpdsp.peak_to_peak(np.array([]))


class TestMean:
    def test_dc(self):
        sig = np.full(500, 2.75)
        assert abs(mpdsp.instrument_mean(sig) - 2.75) < 1e-12

    def test_zero_mean_sine(self):
        sig = _sine(200.0, n=8000)  # integer cycles
        assert abs(mpdsp.instrument_mean(sig)) < 1e-6

    def test_biased_sine(self):
        sig = _sine(200.0, n=8000) + 1.5
        assert abs(mpdsp.instrument_mean(sig) - 1.5) < 1e-6

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mpdsp.instrument_mean(np.array([]))


class TestRms:
    def test_unit_sine(self):
        sig = _sine(200.0, n=8000)
        assert abs(mpdsp.instrument_rms(sig) - 1.0 / math.sqrt(2)) < 1e-3

    def test_unit_square_wave(self):
        sig = np.where(np.arange(1000) % 2 == 0, 1.0, -1.0)
        assert abs(mpdsp.instrument_rms(sig) - 1.0) < 1e-12

    def test_dc_rms_equals_magnitude(self):
        sig = np.full(500, -2.0)
        assert abs(mpdsp.instrument_rms(sig) - 2.0) < 1e-12

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mpdsp.instrument_rms(np.array([]))


class TestRiseFallTime:
    def test_rise_time_linear_ramp_default_thresholds(self):
        # x[n] = n/(N-1) over 100 samples -> ramp from 0 to 1.
        # low_pct=0.1 crosses at sample 9.9, high_pct=0.9 at 89.1.
        # Rise time = 89.1 - 9.9 = 79.2 samples.
        n = 100
        sig = np.linspace(0.0, 1.0, n)
        rt = mpdsp.rise_time(sig)
        assert abs(rt - 79.2) < 0.5

    def test_fall_time_linear_ramp_default_thresholds(self):
        n = 100
        sig = np.linspace(1.0, 0.0, n)
        ft = mpdsp.fall_time(sig)
        assert abs(ft - 79.2) < 0.5

    def test_rise_time_nan_on_flat_signal(self):
        assert math.isnan(mpdsp.rise_time(np.zeros(100)))

    def test_fall_time_nan_on_flat_signal(self):
        assert math.isnan(mpdsp.fall_time(np.zeros(100)))

    def test_rise_time_nan_on_falling_signal(self):
        # Monotonically falling has no rising crossings.
        sig = np.linspace(1.0, 0.0, 100)
        assert math.isnan(mpdsp.rise_time(sig))

    def test_rise_time_custom_thresholds(self):
        n = 100
        sig = np.linspace(0.0, 1.0, n)
        rt_20_80 = mpdsp.rise_time(sig, low_pct=0.2, high_pct=0.8)
        # (0.8 - 0.2) * (n - 1) = 0.6 * 99 = 59.4
        assert abs(rt_20_80 - 59.4) < 0.5

    def test_rise_time_rejects_bad_thresholds(self):
        sig = np.linspace(0.0, 1.0, 100)
        with pytest.raises(ValueError):
            mpdsp.rise_time(sig, low_pct=0.9, high_pct=0.1)  # inverted
        with pytest.raises(ValueError):
            mpdsp.rise_time(sig, low_pct=-0.1, high_pct=0.9)  # below zero
        with pytest.raises(ValueError):
            mpdsp.rise_time(sig, low_pct=0.1, high_pct=1.1)  # above one


class TestPeriodFrequency:
    def test_period_recovers_sine(self):
        # 200 Hz sine at 8 kHz sample rate: period = 40 samples.
        sig = _sine(200.0, n=4096)
        p = mpdsp.period(sig, threshold=0.0)
        assert abs(p - 40.0) < 0.5

    def test_frequency_recovers_sine(self):
        sig = _sine(200.0, n=4096)
        f = mpdsp.frequency(sig, sample_rate=SAMPLE_RATE, threshold=0.0)
        assert abs(f - 200.0) < 2.0

    def test_frequency_recovers_high_frequency(self):
        sig = _sine(1500.0, n=8192)
        f = mpdsp.frequency(sig, sample_rate=SAMPLE_RATE, threshold=0.0)
        assert abs(f - 1500.0) < 5.0

    def test_period_nan_on_dc(self):
        assert math.isnan(mpdsp.period(np.full(1000, 0.5)))

    def test_frequency_nan_on_dc(self):
        result = mpdsp.frequency(np.full(1000, 0.5), sample_rate=SAMPLE_RATE)
        assert math.isnan(result)

    def test_period_with_offset_threshold(self):
        # Biased sine: rising zero-crossings shift by DC offset, but the
        # period between rising *threshold*-crossings should still recover.
        sig = _sine(200.0, n=4096) + 0.3
        p = mpdsp.period(sig, threshold=0.3)
        assert abs(p - 40.0) < 0.5

    def test_frequency_rejects_bad_sample_rate(self):
        sig = _sine(200.0, n=1024)
        with pytest.raises(ValueError):
            mpdsp.frequency(sig, sample_rate=0.0)
        with pytest.raises(ValueError):
            mpdsp.frequency(sig, sample_rate=-1.0)


class TestDtypeDispatch:
    @pytest.mark.parametrize(
        "measure",
        ["peak_to_peak", "instrument_mean", "instrument_rms"])
    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_dispatch_across_dtypes(self, measure, dtype):
        sig = _sine(200.0, n=1024) + 0.1
        fn = getattr(mpdsp, measure)
        ref = fn(sig, dtype="reference")
        out = fn(sig, dtype=dtype)
        assert math.isfinite(out)
        # Narrow dtype should stay in the same ballpark as reference.
        assert abs(out - ref) / (abs(ref) + 1e-12) < 0.1

    def test_frequency_dispatch(self):
        sig = _sine(200.0, n=4096)
        ref = mpdsp.frequency(sig, sample_rate=SAMPLE_RATE, dtype="reference")
        alt = mpdsp.frequency(sig, sample_rate=SAMPLE_RATE, dtype="posit_full")
        assert abs(alt - ref) < 2.0

    def test_unknown_dtype_raises(self):
        sig = _sine(200.0, n=64)
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.peak_to_peak(sig, dtype="not_a_dtype")
