"""Tests for spectrum-analyzer primitives (gap-analysis #104).

Covers RealtimeSpectrum (streaming FFT engine) and the five stateless
detector free functions plus the runtime-dispatch detect(mode) entry.
"""

import math

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Detector free functions.
# ---------------------------------------------------------------------------


class TestDetectPeak:
    def test_returns_max(self):
        assert mpdsp.detect_peak(np.array([1.0, -3.0, 2.5, 0.0])) == 2.5

    def test_all_negative(self):
        assert mpdsp.detect_peak(np.array([-5.0, -3.0, -8.0])) == -3.0

    def test_single_sample(self):
        assert mpdsp.detect_peak(np.array([7.5])) == 7.5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mpdsp.detect_peak(np.array([]))


class TestDetectNegativePeak:
    def test_returns_min(self):
        assert mpdsp.detect_negative_peak(np.array([1.0, -3.0, 2.5])) == -3.0

    def test_all_positive(self):
        assert mpdsp.detect_negative_peak(np.array([1.0, 5.0, 3.0])) == 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mpdsp.detect_negative_peak(np.array([]))


class TestDetectSample:
    def test_returns_first(self):
        assert mpdsp.detect_sample(np.array([-2.5, 1.0, 3.0])) == -2.5

    def test_single_sample(self):
        assert mpdsp.detect_sample(np.array([9.0])) == 9.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mpdsp.detect_sample(np.array([]))


class TestDetectAverage:
    def test_mean(self):
        assert mpdsp.detect_average(np.array([1.0, 2.0, 3.0, 4.0])) == 2.5

    def test_zero_mean(self):
        assert mpdsp.detect_average(np.array([-1.0, 1.0])) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mpdsp.detect_average(np.array([]))


class TestDetectRms:
    def test_unit_sine(self):
        t = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
        sig = np.sin(t)
        assert abs(mpdsp.detect_rms(sig) - 1.0 / math.sqrt(2)) < 1e-3

    def test_unit_square(self):
        sig = np.where(np.arange(1000) % 2 == 0, 1.0, -1.0)
        assert mpdsp.detect_rms(sig) == 1.0

    def test_dc_rms_equals_magnitude(self):
        assert mpdsp.detect_rms(np.full(500, -2.0)) == 2.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mpdsp.detect_rms(np.array([]))


class TestDetectDispatch:
    @pytest.mark.parametrize("mode,expected_fn", [
        ("peak", mpdsp.detect_peak),
        ("negative_peak", mpdsp.detect_negative_peak),
        ("sample", mpdsp.detect_sample),
        ("average", mpdsp.detect_average),
        ("rms", mpdsp.detect_rms),
    ])
    def test_dispatch_matches_named(self, mode, expected_fn):
        # Same bin fed through the dispatch entry point and the named
        # function must return identical results.
        rng = np.random.default_rng(0xC0FFEE)
        bin_samples = rng.standard_normal(64)
        assert mpdsp.detect(mode, bin_samples) == expected_fn(bin_samples)

    def test_unknown_mode_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.detect("not_a_mode", np.array([1.0, 2.0]))

    def test_empty_bin_raises(self):
        with pytest.raises(ValueError):
            mpdsp.detect("peak", np.array([]))


class TestDetectorDtypeDispatch:
    @pytest.mark.parametrize(
        "fn", [mpdsp.detect_peak, mpdsp.detect_average, mpdsp.detect_rms])
    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_dispatch_across_dtypes(self, fn, dtype):
        rng = np.random.default_rng(7)
        bin_samples = rng.standard_normal(64) * 0.1
        ref = fn(bin_samples, dtype="reference")
        out = fn(bin_samples, dtype=dtype)
        assert math.isfinite(out)
        assert abs(out - ref) / (abs(ref) + 1e-12) < 0.1


# ---------------------------------------------------------------------------
# RealtimeSpectrum — streaming FFT engine.
# ---------------------------------------------------------------------------


def _hann_window(N):
    """Local Hann window for tests — matches mpdsp.hanning conventions."""
    return mpdsp.hanning(N)


class TestRealtimeSpectrumConstruction:
    def test_construction_stores_configuration(self):
        w = _hann_window(64)
        rs = mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32, window=w)
        assert rs.fft_size == 64
        assert rs.hop_size == 32
        assert rs.total_ffts == 0
        assert rs.first_fft_ready is False
        assert rs.dtype == "reference"

    def test_rejects_non_power_of_two_fft_size(self):
        w = np.ones(65)
        with pytest.raises(ValueError):
            mpdsp.RealtimeSpectrum(fft_size=65, hop_size=32, window=w)

    def test_rejects_zero_hop_size(self):
        w = _hann_window(64)
        with pytest.raises(ValueError):
            mpdsp.RealtimeSpectrum(fft_size=64, hop_size=0, window=w)

    def test_rejects_hop_size_larger_than_fft_size(self):
        w = _hann_window(64)
        with pytest.raises(ValueError):
            mpdsp.RealtimeSpectrum(fft_size=64, hop_size=65, window=w)

    def test_rejects_wrong_window_length(self):
        with pytest.raises(ValueError):
            mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32, window=np.ones(32))


class TestRealtimeSpectrumStreaming:
    def test_no_ffts_before_fft_size_samples(self):
        rs = mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32,
                                    window=_hann_window(64))
        # Push < fft_size samples — no FFT produced yet.
        n_ffts = rs.push(np.random.default_rng(0).standard_normal(63))
        assert n_ffts == 0
        assert rs.total_ffts == 0
        assert rs.first_fft_ready is False

    def test_first_fft_at_fft_size_samples(self):
        rs = mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32,
                                    window=_hann_window(64))
        n_ffts = rs.push(np.zeros(64))
        assert n_ffts == 1
        assert rs.total_ffts == 1
        assert rs.first_fft_ready is True

    def test_subsequent_ffts_at_hop_intervals(self):
        rs = mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32,
                                    window=_hann_window(64))
        rs.push(np.zeros(64))  # first FFT
        assert rs.total_ffts == 1
        rs.push(np.zeros(32))  # one hop -> second FFT
        assert rs.total_ffts == 2
        rs.push(np.zeros(32))  # another hop -> third FFT
        assert rs.total_ffts == 3

    def test_push_batches_matches_streaming(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(256)
        w = _hann_window(64)

        rs1 = mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32, window=w)
        n1 = rs1.push(sig)

        rs2 = mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32, window=w)
        n2 = 0
        for x in sig:
            n2 += rs2.push(np.array([x]))

        assert n1 == n2
        assert rs1.total_ffts == rs2.total_ffts
        # After identical inputs the latest magnitude should agree.
        np.testing.assert_allclose(rs1.latest_magnitude_db(),
                                   rs2.latest_magnitude_db(), atol=1e-10)

    def test_latest_returns_empty_before_first_fft(self):
        rs = mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32,
                                    window=_hann_window(64))
        re, im = rs.latest_complex()
        assert re.shape == (0,)
        assert im.shape == (0,)
        assert rs.latest_magnitude_db().shape == (0,)

    def test_latest_shapes_after_fft(self):
        rs = mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32,
                                    window=_hann_window(64))
        rs.push(np.zeros(64))
        re, im = rs.latest_complex()
        assert re.shape == (64,)
        assert im.shape == (64,)
        assert rs.latest_magnitude_db().shape == (64,)

    def test_pure_sine_peaks_at_expected_bin(self):
        # 250 Hz sine at 1000 Hz sample rate, 256-point Hann FFT.
        # Bin resolution = 1000/256 ≈ 3.9 Hz; 250 Hz -> bin 64.
        fs = 1000.0
        freq = 250.0
        N = 256
        t = np.arange(2 * N) / fs
        sig = np.sin(2 * np.pi * freq * t)

        rs = mpdsp.RealtimeSpectrum(fft_size=N, hop_size=N,
                                    window=_hann_window(N))
        rs.push(sig)
        mag_db = rs.latest_magnitude_db()
        # Positive-frequency half
        half = mag_db[:N // 2]
        peak_bin = int(np.argmax(half))
        expected_bin = int(round(freq / fs * N))
        assert abs(peak_bin - expected_bin) <= 1

    def test_reset_clears_state_but_preserves_config(self):
        rs = mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32,
                                    window=_hann_window(64))
        rs.push(np.ones(100))
        assert rs.total_ffts > 0
        assert rs.first_fft_ready is True
        rs.reset()
        assert rs.total_ffts == 0
        assert rs.first_fft_ready is False
        # Config preserved
        assert rs.fft_size == 64
        assert rs.hop_size == 32
        # After reset, need fft_size samples again before an FFT fires.
        assert rs.push(np.zeros(63)) == 0
        assert rs.push(np.zeros(1)) == 1


class TestRealtimeSpectrumDtypeDispatch:
    def test_dtype_property_reflects_construction(self):
        rs = mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32,
                                    window=_hann_window(64),
                                    dtype="posit_full")
        assert rs.dtype == "posit_full"

    def test_unknown_dtype_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.RealtimeSpectrum(fft_size=64, hop_size=32,
                                    window=_hann_window(64),
                                    dtype="not_a_dtype")

    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_peak_recovery_across_dtypes(self, dtype):
        # Peak bin location is precision-insensitive at reasonable
        # amplitudes; verify the class constructs and finds the peak
        # under each dtype.
        fs = 1000.0
        freq = 200.0
        N = 128
        t = np.arange(2 * N) / fs
        sig = np.sin(2 * np.pi * freq * t)

        rs = mpdsp.RealtimeSpectrum(fft_size=N, hop_size=N,
                                    window=_hann_window(N), dtype=dtype)
        rs.push(sig)
        assert rs.first_fft_ready is True
        mag_db = rs.latest_magnitude_db()
        peak_bin = int(np.argmax(mag_db[:N // 2]))
        expected_bin = int(round(freq / fs * N))
        assert abs(peak_bin - expected_bin) <= 1
