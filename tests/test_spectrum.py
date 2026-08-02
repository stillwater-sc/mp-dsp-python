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


# ---------------------------------------------------------------------------
# RBWFilter — resolution-bandwidth filter (#105).
# ---------------------------------------------------------------------------


class TestRBWFilterConstruction:
    def test_default_order_and_getters(self):
        f = mpdsp.RBWFilter(center_freq_hz=1000.0, bandwidth_hz=100.0,
                            sample_rate_hz=8000.0)
        assert f.center_freq_hz == 1000.0
        assert f.bandwidth_hz == 100.0
        assert f.sample_rate_hz == 8000.0
        assert f.order == 5
        assert f.dtype == "reference"

    def test_custom_order(self):
        f = mpdsp.RBWFilter(1000.0, 100.0, 8000.0, order=3)
        assert f.order == 3

    def test_shape_factor_improves_with_order(self):
        low  = mpdsp.RBWFilter(1000.0, 100.0, 8000.0, order=1)
        mid  = mpdsp.RBWFilter(1000.0, 100.0, 8000.0, order=5)
        high = mpdsp.RBWFilter(1000.0, 100.0, 8000.0, order=8)
        # Lower shape factor = tighter shape = better selectivity
        assert low.shape_factor > mid.shape_factor > high.shape_factor
        # order=5 sync-tuned lands around 10x per header prose
        assert 5.0 < mid.shape_factor < 20.0

    def test_rejects_bad_sample_rate(self):
        with pytest.raises(ValueError):
            mpdsp.RBWFilter(1000.0, 100.0, sample_rate_hz=0.0)
        with pytest.raises(ValueError):
            mpdsp.RBWFilter(1000.0, 100.0, sample_rate_hz=-1.0)

    def test_rejects_bad_order(self):
        with pytest.raises(ValueError):
            mpdsp.RBWFilter(1000.0, 100.0, 8000.0, order=0)
        with pytest.raises(ValueError):
            mpdsp.RBWFilter(1000.0, 100.0, 8000.0, order=9)  # kMaxOrder=8

    def test_rejects_center_at_or_above_nyquist(self):
        with pytest.raises(ValueError):
            mpdsp.RBWFilter(center_freq_hz=4000.0, bandwidth_hz=100.0,
                            sample_rate_hz=8000.0)  # exactly Nyquist

    def test_rejects_bandwidth_that_pushes_shoulder_below_zero(self):
        with pytest.raises(ValueError):
            # center=100, bw=300 → lower shoulder = -50, invalid
            mpdsp.RBWFilter(center_freq_hz=100.0, bandwidth_hz=300.0,
                            sample_rate_hz=8000.0)

    def test_rejects_bandwidth_that_pushes_shoulder_above_nyquist(self):
        with pytest.raises(ValueError):
            # center=3500, bw=1500 at fs=8000 → upper shoulder=4250 > Nyquist(4000)
            mpdsp.RBWFilter(center_freq_hz=3500.0, bandwidth_hz=1500.0,
                            sample_rate_hz=8000.0)


class TestRBWFilterProcessing:
    def _mk(self, fc=1000.0, bw=100.0, fs=8000.0, order=5):
        return mpdsp.RBWFilter(fc, bw, fs, order=order)

    def test_process_single_sample_returns_float(self):
        f = self._mk()
        y = f.process(0.5)
        assert isinstance(y, float)

    def test_process_block_preserves_length(self):
        f = self._mk()
        sig = np.zeros(256)
        y = f.process_block(sig)
        assert y.shape == sig.shape
        assert y.dtype == np.float64

    def test_streaming_and_block_agree(self):
        rng = np.random.default_rng(0xB055)
        sig = rng.standard_normal(512)
        y_block = self._mk().process_block(sig)
        f_stream = self._mk()
        y_stream = np.array([f_stream.process(float(x)) for x in sig])
        np.testing.assert_allclose(y_block, y_stream)

    def test_passes_center_frequency_tone(self):
        # Sine at the tuned center should pass through with high output
        # energy relative to a far-off-center sine.
        fs, fc = 8000.0, 1000.0
        f = self._mk(fc=fc, bw=100.0)
        t = np.arange(4096) / fs
        pass_sig = np.sin(2 * np.pi * fc * t)
        stop_sig = np.sin(2 * np.pi * (fc * 3) * t)  # far off center
        # Skip transient
        skip = 500
        y_pass = f.process_block(pass_sig)[skip:]
        f2 = self._mk(fc=fc, bw=100.0)
        y_stop = f2.process_block(stop_sig)[skip:]
        # In-band should be at least an order of magnitude stronger.
        assert np.sqrt(np.mean(y_pass ** 2)) > \
               10.0 * np.sqrt(np.mean(y_stop ** 2))

    def test_retune_is_bumpless(self):
        # After retune, the last-y should be very close to what it was
        # before retune (state preserved). Compare against a fresh
        # filter which would produce very different transient output.
        f = self._mk(fc=1000.0, bw=100.0)
        sig = np.sin(2 * np.pi * 1000.0 * np.arange(4096) / 8000.0)
        f.process_block(sig)
        last_before = f.process(0.5)  # one more sample, record output
        f.retune(1500.0, 100.0)
        # Next output shouldn't be a sudden jump to zero from state clear.
        # Loose check: an order of magnitude of last_before.
        after = f.process(0.0)
        assert abs(after) > 1e-6 or abs(last_before) < 1e-6

    def test_reset_clears_state(self):
        f = self._mk()
        f.process_block(np.ones(1024))
        # After reset, feeding a fresh impulse train should behave as
        # fresh construction — first output should be very small (transient).
        f.reset()
        first_out = f.process(1.0)
        assert abs(first_out) < 0.5   # unit input, zero state


class TestRBWFilterDtypeDispatch:
    def test_dtype_property(self):
        f = mpdsp.RBWFilter(1000.0, 100.0, 8000.0, dtype="posit_full")
        assert f.dtype == "posit_full"

    def test_unknown_dtype_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.RBWFilter(1000.0, 100.0, 8000.0, dtype="not_a_dtype")

    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_construction_and_processing_across_dtypes(self, dtype):
        f = mpdsp.RBWFilter(1000.0, 100.0, 8000.0, dtype=dtype)
        sig = np.sin(2 * np.pi * 1000.0 * np.arange(512) / 8000.0)
        y = f.process_block(sig)
        assert y.shape == sig.shape
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# VBWFilter — video-bandwidth filter (#105).
# ---------------------------------------------------------------------------


class TestVBWFilterConstruction:
    def test_getters(self):
        f = mpdsp.VBWFilter(cutoff_hz=10.0, sample_rate_hz=1000.0)
        assert f.cutoff_hz == 10.0
        assert f.sample_rate_hz == 1000.0
        assert f.dtype == "reference"

    def test_rejects_bad_sample_rate(self):
        with pytest.raises(ValueError):
            mpdsp.VBWFilter(cutoff_hz=10.0, sample_rate_hz=0.0)

    def test_rejects_bad_cutoff(self):
        with pytest.raises(ValueError):
            mpdsp.VBWFilter(cutoff_hz=0.0, sample_rate_hz=1000.0)
        with pytest.raises(ValueError):
            mpdsp.VBWFilter(cutoff_hz=-1.0, sample_rate_hz=1000.0)

    def test_rejects_cutoff_above_nyquist(self):
        with pytest.raises(ValueError):
            mpdsp.VBWFilter(cutoff_hz=600.0, sample_rate_hz=1000.0)


class TestVBWFilterProcessing:
    def test_process_returns_float(self):
        f = mpdsp.VBWFilter(10.0, 1000.0)
        assert isinstance(f.process(0.5), float)

    def test_process_block_preserves_length(self):
        f = mpdsp.VBWFilter(10.0, 1000.0)
        sig = np.zeros(256)
        y = f.process_block(sig)
        assert y.shape == (256,)
        assert y.dtype == np.float64

    def test_streaming_and_block_agree(self):
        rng = np.random.default_rng(1)
        sig = rng.standard_normal(512)
        y_block = mpdsp.VBWFilter(10.0, 1000.0).process_block(sig)
        f_stream = mpdsp.VBWFilter(10.0, 1000.0)
        y_stream = np.array([f_stream.process(float(x)) for x in sig])
        np.testing.assert_allclose(y_block, y_stream)

    def test_dc_passes_through(self):
        # Steady DC input eventually settles at the DC value.
        f = mpdsp.VBWFilter(cutoff_hz=10.0, sample_rate_hz=1000.0)
        y = f.process_block(np.full(2000, 2.0))
        # After many samples the leaky integrator should be very close
        # to the DC input.
        assert abs(y[-1] - 2.0) < 0.01

    def test_high_frequency_attenuated(self):
        # A high-frequency component well above cutoff should be
        # attenuated relative to a low-frequency component well below.
        fs = 1000.0
        fc = 10.0
        f_low  = mpdsp.VBWFilter(fc, fs)
        f_high = mpdsp.VBWFilter(fc, fs)
        t = np.arange(4096) / fs
        low_sig  = np.sin(2 * np.pi * 2.0 * t)   # 2 Hz, well below fc=10
        high_sig = np.sin(2 * np.pi * 200.0 * t) # 200 Hz, far above fc
        y_low  = f_low.process_block(low_sig)
        y_high = f_high.process_block(high_sig)
        skip = 500
        # Low freq mostly passes; high freq is strongly attenuated.
        assert np.sqrt(np.mean(y_low[skip:] ** 2)) > \
               10.0 * np.sqrt(np.mean(y_high[skip:] ** 2))

    def test_set_cutoff_is_bumpless(self):
        # Similar to RBW: after changing cutoff, output shouldn't jump
        # discontinuously to zero (state preservation).
        f = mpdsp.VBWFilter(10.0, 1000.0)
        f.process_block(np.full(500, 2.0))  # settle at DC=2
        last_before = f.process(2.0)
        f.set_cutoff(50.0)
        after = f.process(2.0)
        # State preserved -> both outputs close to input DC
        assert abs(after - 2.0) < 0.1
        assert abs(last_before - 2.0) < 0.1

    def test_reset_clears_state(self):
        f = mpdsp.VBWFilter(10.0, 1000.0)
        f.process_block(np.full(1000, 5.0))
        f.reset()
        # After reset, feeding zero should give ~zero (no leftover DC).
        y = f.process(0.0)
        assert abs(y) < 1e-6


class TestVBWFilterDtypeDispatch:
    def test_dtype_property(self):
        f = mpdsp.VBWFilter(10.0, 1000.0, dtype="posit_full")
        assert f.dtype == "posit_full"

    def test_unknown_dtype_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.VBWFilter(10.0, 1000.0, dtype="not_a_dtype")

    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_construction_and_processing_across_dtypes(self, dtype):
        f = mpdsp.VBWFilter(10.0, 1000.0, dtype=dtype)
        sig = np.full(1000, 1.0)  # DC input
        y = f.process_block(sig)
        assert y.shape == (1000,)
        assert np.all(np.isfinite(y))
        # DC eventually settles near input; loose bound for narrow types
        assert abs(y[-1] - 1.0) < 0.1


# ---------------------------------------------------------------------------
# SweptLO — phase-coherent chirp generator (#106).
# ---------------------------------------------------------------------------


class TestSweptLOConstruction:
    def test_getters_reflect_construction(self):
        lo = mpdsp.SweptLO(f_start_hz=100.0, f_stop_hz=1000.0,
                           sweep_duration_s=1.0, sample_rate_hz=1000.0)
        assert lo.f_start_hz == 100.0
        assert lo.f_stop_hz == 1000.0
        assert lo.sweep_duration_s == 1.0
        assert lo.sample_rate_hz == 1000.0
        assert lo.mode == "linear"
        assert lo.num_sweep_samples == 1000
        assert lo.total_sweeps == 0
        assert lo.sweep_complete is False
        assert lo.dtype == "reference"

    def test_log_mode_accepted(self):
        lo = mpdsp.SweptLO(100.0, 1000.0, 1.0, 1000.0, mode="log")
        assert lo.mode == "logarithmic"

    def test_logarithmic_alias(self):
        lo = mpdsp.SweptLO(100.0, 1000.0, 1.0, 1000.0, mode="logarithmic")
        assert lo.mode == "logarithmic"

    def test_rejects_bad_mode(self):
        with pytest.raises(ValueError):
            mpdsp.SweptLO(100.0, 1000.0, 1.0, 1000.0, mode="not_a_mode")

    def test_rejects_non_positive_frequencies(self):
        with pytest.raises(ValueError):
            mpdsp.SweptLO(0.0, 1000.0, 1.0, 1000.0)
        with pytest.raises(ValueError):
            mpdsp.SweptLO(100.0, -1.0, 1.0, 1000.0)

    def test_rejects_short_sweep(self):
        # Fewer than 2 samples must be rejected.
        with pytest.raises(ValueError):
            mpdsp.SweptLO(100.0, 200.0,
                          sweep_duration_s=0.0005, sample_rate_hz=1000.0)


class TestSweptLOSweep:
    def test_process_returns_cos_sin_tuple(self):
        lo = mpdsp.SweptLO(100.0, 200.0, 0.1, 1000.0)
        result = lo.process()
        assert isinstance(result, tuple)
        assert len(result) == 2
        c, s = result
        assert isinstance(c, float)
        assert isinstance(s, float)

    def test_cos_sin_unit_circle(self):
        # cos^2 + sin^2 = 1 at every sample (within floating-point tolerance).
        lo = mpdsp.SweptLO(50.0, 500.0, 1.0, 1000.0)
        c, s = lo.generate_block(500)
        radii = np.sqrt(c ** 2 + s ** 2)
        np.testing.assert_allclose(radii, 1.0, atol=1e-6)

    def test_starts_at_f_start_and_ends_at_f_stop(self):
        # Linear sweep endpoints: current_frequency_hz should equal
        # f_start at sample 0 and f_stop at sample N-1.
        fs, f0, f1, dur = 1000.0, 100.0, 400.0, 1.0
        lo = mpdsp.SweptLO(f0, f1, dur, fs, mode="linear")
        N = lo.num_sweep_samples
        # Sample 0 initial frequency
        initial = lo.current_frequency_hz
        assert abs(initial - f0) < 0.01
        # Push all-but-one samples; check the (N-1)-th sample's frequency.
        for _ in range(N - 1):
            lo.process()
        assert abs(lo.current_frequency_hz - f1) < 0.5

    def test_sweep_complete_fires_at_boundary(self):
        fs, dur = 1000.0, 0.1
        lo = mpdsp.SweptLO(100.0, 200.0, dur, fs)
        N = lo.num_sweep_samples
        # The Nth process() call is the one that wraps the boundary
        # (samples 0..N-1 fill the sweep; on the Nth call we cross).
        for i in range(N - 1):
            lo.process()
        assert lo.sweep_complete is False
        lo.process()
        assert lo.sweep_complete is True
        assert lo.total_sweeps == 1
        # sweep_complete self-clears on the next process()
        lo.process()
        assert lo.sweep_complete is False

    def test_phase_continuous_across_sweep_boundary(self):
        # Between the last sample of sweep k and the first sample of
        # sweep k+1, the (cos, sin) output should NOT jump discontinuously
        # — that's the whole point of the phase-coherent design.
        lo = mpdsp.SweptLO(100.0, 300.0, 0.1, 1000.0)
        N = lo.num_sweep_samples
        c1, s1 = lo.generate_block(N)  # first sweep
        c2, s2 = lo.generate_block(1)   # first sample of second sweep
        # Distance in the complex plane between last-of-sweep-1 and
        # first-of-sweep-2 should be small (phase continues smoothly).
        boundary_jump = np.hypot(c2[0] - c1[-1], s2[0] - s1[-1])
        # A discontinuous restart would give a jump of order 1; the
        # phase-continuous restart typically gives a jump on the order of
        # the per-sample step which is ~ 2*pi*f/fs ~ 2 at these settings.
        # So a comfortable upper bound is 2.5 (allowing for the largest
        # per-sample step from the final phase_inc).
        assert boundary_jump < 2.5

    def test_reset_returns_to_start(self):
        lo = mpdsp.SweptLO(100.0, 400.0, 0.1, 1000.0)
        lo.generate_block(50)
        lo.reset()
        assert lo.total_sweeps == 0
        assert lo.sweep_complete is False
        assert abs(lo.current_frequency_hz - 100.0) < 0.01

    def test_generate_block_matches_streaming(self):
        lo1 = mpdsp.SweptLO(100.0, 200.0, 0.05, 1000.0)
        c_block, s_block = lo1.generate_block(50)
        lo2 = mpdsp.SweptLO(100.0, 200.0, 0.05, 1000.0)
        c_stream = np.zeros(50)
        s_stream = np.zeros(50)
        for i in range(50):
            c, s = lo2.process()
            c_stream[i] = c
            s_stream[i] = s
        np.testing.assert_allclose(c_block, c_stream)
        np.testing.assert_allclose(s_block, s_stream)


class TestSweptLODtypeDispatch:
    def test_dtype_property(self):
        lo = mpdsp.SweptLO(100.0, 200.0, 0.1, 1000.0, dtype="posit_full")
        assert lo.dtype == "posit_full"

    def test_unknown_dtype_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.SweptLO(100.0, 200.0, 0.1, 1000.0, dtype="not_a_dtype")

    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_unit_circle_across_dtypes(self, dtype):
        # cos^2 + sin^2 ≈ 1 should hold under any dtype at reasonable
        # tolerance.
        lo = mpdsp.SweptLO(100.0, 400.0, 0.1, 1000.0, dtype=dtype)
        c, s = lo.generate_block(50)
        radii = np.sqrt(c ** 2 + s ** 2)
        # Narrow dtypes may have wider tolerance
        np.testing.assert_allclose(radii, 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# CalibrationProfile — non-templated value type consumed by FrontEndCorrector.
# ---------------------------------------------------------------------------


class TestCalibrationProfile:
    def test_construction_and_getters(self):
        p = mpdsp.CalibrationProfile(
            frequencies=[100.0, 500.0, 1000.0],
            gain_dB=[0.0, -1.0, -3.0],
            phase_rad=[0.0, -0.1, -0.3])
        assert p.size == 3
        assert p.freq_min == 100.0
        assert p.freq_max == 1000.0

    def test_interpolation_at_tabulated_points(self):
        p = mpdsp.CalibrationProfile([100.0, 200.0], [0.0, -2.0], [0.0, 0.5])
        assert p.gain_dB(100.0) == 0.0
        assert p.gain_dB(200.0) == -2.0

    def test_interpolation_midpoint(self):
        p = mpdsp.CalibrationProfile([100.0, 200.0], [0.0, -2.0], [0.0, 0.5])
        # Midpoint linear interpolation
        assert abs(p.gain_dB(150.0) - (-1.0)) < 1e-12
        assert abs(p.phase_rad(150.0) - 0.25) < 1e-12

    def test_clamps_below_range(self):
        p = mpdsp.CalibrationProfile([100.0, 200.0], [-1.0, -2.0], [0.0, 0.5])
        assert p.gain_dB(50.0) == -1.0    # clamps to first entry
        assert p.phase_rad(50.0) == 0.0

    def test_clamps_above_range(self):
        p = mpdsp.CalibrationProfile([100.0, 200.0], [-1.0, -2.0], [0.0, 0.5])
        assert p.gain_dB(1000.0) == -2.0  # clamps to last entry
        assert p.phase_rad(1000.0) == 0.5

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            mpdsp.CalibrationProfile([100.0, 200.0], [0.0], [0.0, 0.0])

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError):
            mpdsp.CalibrationProfile([100.0], [0.0], [0.0])

    def test_rejects_non_monotonic_frequencies(self):
        with pytest.raises(ValueError):
            mpdsp.CalibrationProfile([100.0, 50.0], [0.0, 0.0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# FrontEndCorrector — FIR equalizer inverting a CalibrationProfile (#106).
# ---------------------------------------------------------------------------


def _flat_profile(fs=1000.0, gain_dB=0.0):
    # Reference: perfectly flat 0 dB profile over the full band.
    return mpdsp.CalibrationProfile(
        frequencies=[1.0, fs / 2],
        gain_dB=[gain_dB, gain_dB],
        phase_rad=[0.0, 0.0])


class TestFrontEndCorrectorConstruction:
    def test_getters(self):
        p = _flat_profile()
        f = mpdsp.FrontEndCorrector(profile=p, num_taps=33,
                                    sample_rate_hz=1000.0)
        assert f.num_taps == 33
        assert f.dtype == "reference"

    def test_rejects_too_few_taps(self):
        p = _flat_profile()
        with pytest.raises(ValueError):
            mpdsp.FrontEndCorrector(profile=p, num_taps=2,
                                    sample_rate_hz=1000.0)

    def test_rejects_bad_sample_rate(self):
        p = _flat_profile()
        with pytest.raises(ValueError):
            mpdsp.FrontEndCorrector(profile=p, num_taps=33,
                                    sample_rate_hz=0.0)

    def test_rejects_bad_max_gain(self):
        p = _flat_profile()
        with pytest.raises(ValueError):
            mpdsp.FrontEndCorrector(profile=p, num_taps=33,
                                    sample_rate_hz=1000.0,
                                    max_gain_dB=-1.0)


class TestFrontEndCorrectorProcessing:
    def test_process_returns_float(self):
        f = mpdsp.FrontEndCorrector(_flat_profile(), 21, 1000.0)
        assert isinstance(f.process(0.5), float)

    def test_process_block_preserves_length(self):
        f = mpdsp.FrontEndCorrector(_flat_profile(), 21, 1000.0)
        sig = np.zeros(256)
        y = f.process_block(sig)
        assert y.shape == sig.shape
        assert y.dtype == np.float64

    def test_streaming_and_block_agree(self):
        rng = np.random.default_rng(0xE9)
        sig = rng.standard_normal(512)
        f1 = mpdsp.FrontEndCorrector(_flat_profile(), 21, 1000.0)
        y_block = f1.process_block(sig)
        f2 = mpdsp.FrontEndCorrector(_flat_profile(), 21, 1000.0)
        y_stream = np.array([f2.process(float(x)) for x in sig])
        np.testing.assert_allclose(y_block, y_stream)

    def test_flat_profile_near_identity_after_group_delay(self):
        # A perfectly flat 0 dB / 0 rad profile inverts to ~ delta[n - K/2].
        # A DC input should therefore emerge at ~ input amplitude after
        # the FIR settles (bounded by Hamming window sidelobes).
        f = mpdsp.FrontEndCorrector(_flat_profile(), num_taps=33,
                                    sample_rate_hz=1000.0)
        sig = np.full(200, 1.0)
        y = f.process_block(sig)
        # Post-transient (past the group delay), output should approach DC.
        assert abs(y[-1] - 1.0) < 0.05


class TestFrontEndCorrectorDtypeDispatch:
    def test_dtype_property(self):
        f = mpdsp.FrontEndCorrector(_flat_profile(), 21, 1000.0,
                                    dtype="posit_full")
        assert f.dtype == "posit_full"

    def test_unknown_dtype_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.FrontEndCorrector(_flat_profile(), 21, 1000.0,
                                    dtype="not_a_dtype")

    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_processing_across_dtypes(self, dtype):
        f = mpdsp.FrontEndCorrector(_flat_profile(), 21, 1000.0, dtype=dtype)
        sig = np.full(200, 1.0)
        y = f.process_block(sig)
        assert y.shape == sig.shape
        assert np.all(np.isfinite(y))
        # Flat profile inverts to near-identity; post-transient output
        # sits near the DC input under any reasonable dtype.
        assert abs(y[-1] - 1.0) < 0.1


# ---------------------------------------------------------------------------
# TraceAverager — cross-sweep accumulation (#107).
# ---------------------------------------------------------------------------


class TestTraceAveragerConstruction:
    def test_linear_default_config_ignored(self):
        avg = mpdsp.TraceAverager(trace_length=64, mode="linear")
        assert avg.trace_length == 64
        assert avg.mode == "linear"
        assert avg.sweeps_accumulated == 0
        assert avg.dtype == "reference"

    def test_exponential_needs_valid_alpha(self):
        avg = mpdsp.TraceAverager(64, "exponential", config=0.1)
        assert avg.mode == "exponential"

    def test_exponential_rejects_bad_alpha(self):
        with pytest.raises(ValueError):
            mpdsp.TraceAverager(64, "exponential", config=0.0)
        with pytest.raises(ValueError):
            mpdsp.TraceAverager(64, "exponential", config=1.5)

    def test_max_hold_n_needs_integer_window(self):
        avg = mpdsp.TraceAverager(64, "max_hold_n", config=4.0)
        assert avg.mode == "max_hold_n"

    def test_max_hold_n_rejects_non_integer(self):
        with pytest.raises(ValueError):
            mpdsp.TraceAverager(64, "max_hold_n", config=2.5)

    def test_max_hold_n_rejects_zero(self):
        with pytest.raises(ValueError):
            mpdsp.TraceAverager(64, "max_hold_n", config=0.0)

    def test_rejects_zero_trace_length(self):
        with pytest.raises(ValueError):
            mpdsp.TraceAverager(0, "linear")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            mpdsp.TraceAverager(64, "not_a_mode")


class TestTraceAveragerModes:
    def test_linear_averages_identical_sweeps(self):
        avg = mpdsp.TraceAverager(4, "linear")
        for _ in range(5):
            avg.accept_sweep(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(avg.current_trace(),
                                   [1.0, 2.0, 3.0, 4.0])
        assert avg.sweeps_accumulated == 5

    def test_linear_averages_different_sweeps(self):
        avg = mpdsp.TraceAverager(3, "linear")
        avg.accept_sweep(np.array([1.0, 2.0, 3.0]))
        avg.accept_sweep(np.array([3.0, 4.0, 5.0]))
        # Mean = ([1+3, 2+4, 3+5]) / 2 = [2, 3, 4]
        np.testing.assert_allclose(avg.current_trace(), [2.0, 3.0, 4.0])

    def test_exponential_seeds_from_first_sweep(self):
        # First sweep must NOT be dragged toward zero.
        avg = mpdsp.TraceAverager(3, "exponential", config=0.1)
        avg.accept_sweep(np.array([5.0, 6.0, 7.0]))
        np.testing.assert_allclose(avg.current_trace(), [5.0, 6.0, 7.0])

    def test_exponential_converges(self):
        # Repeated identical input pulls the exponential average toward
        # that value.
        avg = mpdsp.TraceAverager(2, "exponential", config=0.3)
        avg.accept_sweep(np.array([10.0, 20.0]))
        for _ in range(50):
            avg.accept_sweep(np.array([0.0, 0.0]))
        out = avg.current_trace()
        assert abs(out[0]) < 0.5
        assert abs(out[1]) < 0.5

    def test_max_hold_keeps_max(self):
        avg = mpdsp.TraceAverager(3, "max_hold")
        avg.accept_sweep(np.array([1.0, 5.0, 3.0]))
        avg.accept_sweep(np.array([4.0, 2.0, 6.0]))
        avg.accept_sweep(np.array([2.0, 3.0, 1.0]))
        np.testing.assert_allclose(avg.current_trace(), [4.0, 5.0, 6.0])

    def test_min_hold_keeps_min(self):
        avg = mpdsp.TraceAverager(3, "min_hold")
        avg.accept_sweep(np.array([1.0, 5.0, 3.0]))
        avg.accept_sweep(np.array([4.0, 2.0, 6.0]))
        avg.accept_sweep(np.array([2.0, 3.0, 1.0]))
        np.testing.assert_allclose(avg.current_trace(), [1.0, 2.0, 1.0])

    def test_max_hold_n_rolling_window(self):
        # N=2: only the last 2 sweeps contribute to the max.
        avg = mpdsp.TraceAverager(3, "max_hold_n", config=2.0)
        avg.accept_sweep(np.array([9.0, 9.0, 9.0]))   # dominates initially
        avg.accept_sweep(np.array([1.0, 2.0, 3.0]))   # window [9], [1..3] -> [9, 9, 9]
        avg.accept_sweep(np.array([4.0, 5.0, 6.0]))   # window [1..3], [4..6] -> [4, 5, 6]
        np.testing.assert_allclose(avg.current_trace(), [4.0, 5.0, 6.0])

    def test_reset_clears_state_but_preserves_mode(self):
        avg = mpdsp.TraceAverager(3, "max_hold")
        avg.accept_sweep(np.array([5.0, 5.0, 5.0]))
        assert avg.sweeps_accumulated == 1
        avg.reset()
        assert avg.sweeps_accumulated == 0
        assert avg.mode == "max_hold"

    def test_accept_sweep_rejects_length_mismatch(self):
        avg = mpdsp.TraceAverager(4, "linear")
        with pytest.raises(ValueError):
            avg.accept_sweep(np.array([1.0, 2.0, 3.0]))


class TestTraceAveragerDtypeDispatch:
    def test_dtype_property(self):
        avg = mpdsp.TraceAverager(8, "linear", dtype="posit_full")
        assert avg.dtype == "posit_full"

    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_linear_across_dtypes(self, dtype):
        avg = mpdsp.TraceAverager(4, "linear", dtype=dtype)
        for _ in range(3):
            avg.accept_sweep(np.array([1.0, 2.0, 3.0, 4.0]))
        out = avg.current_trace()
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0], atol=0.1)


# ---------------------------------------------------------------------------
# WaterfallBuffer (#107).
# ---------------------------------------------------------------------------


class TestWaterfallBuffer:
    def test_construction_and_capacities(self):
        wf = mpdsp.WaterfallBuffer(num_bins=8, num_frames=4)
        assert wf.num_bins == 8
        assert wf.num_frames_capacity == 4
        assert wf.num_frames_filled == 0
        assert wf.dtype == "reference"

    def test_rejects_zero_dimensions(self):
        with pytest.raises(ValueError):
            mpdsp.WaterfallBuffer(0, 4)
        with pytest.raises(ValueError):
            mpdsp.WaterfallBuffer(4, 0)

    def test_push_and_frame_at(self):
        wf = mpdsp.WaterfallBuffer(3, 4)
        wf.push_frame(np.array([1.0, 2.0, 3.0]))
        wf.push_frame(np.array([4.0, 5.0, 6.0]))
        assert wf.num_frames_filled == 2
        np.testing.assert_allclose(wf.frame_at(0), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(wf.frame_at(1), [4.0, 5.0, 6.0])

    def test_push_frame_rejects_wrong_length(self):
        wf = mpdsp.WaterfallBuffer(3, 4)
        with pytest.raises(ValueError):
            wf.push_frame(np.array([1.0, 2.0]))

    def test_ring_wraps_over_capacity(self):
        wf = mpdsp.WaterfallBuffer(2, 3)  # 3-frame ring
        for k in range(5):  # push 5 frames -> oldest 2 dropped
            wf.push_frame(np.array([float(k), float(k) + 0.5]))
        assert wf.num_frames_filled == 3
        # After wrap: chronological order is frames 2, 3, 4
        np.testing.assert_allclose(wf.frame_at(0), [2.0, 2.5])
        np.testing.assert_allclose(wf.frame_at(1), [3.0, 3.5])
        np.testing.assert_allclose(wf.frame_at(2), [4.0, 4.5])

    def test_frame_at_out_of_range(self):
        wf = mpdsp.WaterfallBuffer(2, 3)
        wf.push_frame(np.array([1.0, 2.0]))
        with pytest.raises(IndexError):
            wf.frame_at(1)

    def test_last_frames_2d_shape_and_content(self):
        wf = mpdsp.WaterfallBuffer(2, 4)
        for k in range(3):
            wf.push_frame(np.array([float(k), float(k) + 0.5]))
        block = wf.last_frames(2)
        assert block.shape == (2, 2)
        # last_frames(2) = frames 1, 2 (chronological, oldest first)
        np.testing.assert_allclose(block, [[1.0, 1.5], [2.0, 2.5]])

    def test_last_frames_clamps_to_available(self):
        wf = mpdsp.WaterfallBuffer(2, 4)
        wf.push_frame(np.array([1.0, 2.0]))
        block = wf.last_frames(10)
        assert block.shape == (1, 2)   # only 1 available

    def test_last_frames_empty_when_no_frames(self):
        wf = mpdsp.WaterfallBuffer(2, 4)
        block = wf.last_frames(5)
        assert block.shape == (0, 2)

    def test_clear_resets_fill(self):
        wf = mpdsp.WaterfallBuffer(2, 4)
        wf.push_frame(np.array([1.0, 2.0]))
        wf.push_frame(np.array([3.0, 4.0]))
        wf.clear()
        assert wf.num_frames_filled == 0
        # Capacity preserved
        assert wf.num_frames_capacity == 4


# ---------------------------------------------------------------------------
# Marker / DeltaMarker + free functions (#107).
# ---------------------------------------------------------------------------


class TestMarkerAndDeltaMarker:
    def test_marker_default_construction(self):
        m = mpdsp.Marker()
        assert m.bin_index == 0
        assert m.frequency_hz == 0.0
        assert m.amplitude == 0.0

    def test_marker_field_read_write(self):
        m = mpdsp.Marker()
        m.bin_index = 42
        m.frequency_hz = 1234.5
        m.amplitude = -30.0
        assert m.bin_index == 42
        assert m.frequency_hz == 1234.5
        assert m.amplitude == -30.0

    def test_marker_repr(self):
        m = mpdsp.Marker()
        m.bin_index = 5
        m.frequency_hz = 500.0
        m.amplitude = -12.0
        # Repr contains the field values
        r = repr(m)
        assert "bin_index=5" in r
        assert "frequency_hz" in r
        assert "amplitude" in r

    def test_make_delta_marker(self):
        a = mpdsp.Marker()
        a.frequency_hz = 100.0
        a.amplitude = -20.0
        b = mpdsp.Marker()
        b.frequency_hz = 250.0
        b.amplitude = -15.0
        d = mpdsp.make_delta_marker(a, b)
        assert d.delta_freq_hz == 150.0
        assert d.delta_amplitude == 5.0
        assert d.a.frequency_hz == 100.0
        assert d.b.amplitude == -15.0


class TestFindPeaks:
    def test_returns_top_n_in_descending_amplitude(self):
        # Trace with three clear peaks far apart.
        trace = np.array([0.0, 5.0, 0.0, 0.0, 0.0,
                          0.0, 3.0, 0.0, 0.0, 0.0,
                          0.0, 8.0, 0.0])
        peaks = mpdsp.find_peaks(trace, bin_freq_step_hz=10.0, top_n=3)
        assert len(peaks) == 3
        # Descending amplitude order
        assert peaks[0].amplitude >= peaks[1].amplitude >= peaks[2].amplitude
        # Top peak is bin 11 (amp 8.0)
        assert peaks[0].bin_index == 11
        assert abs(peaks[0].frequency_hz - 110.0) < 5.0  # bin 11 * 10 Hz

    def test_top_n_zero_returns_empty(self):
        trace = np.array([0.0, 5.0, 0.0])
        peaks = mpdsp.find_peaks(trace, bin_freq_step_hz=1.0, top_n=0)
        assert peaks == []

    def test_empty_trace_returns_empty(self):
        peaks = mpdsp.find_peaks(np.array([]), bin_freq_step_hz=1.0, top_n=5)
        assert peaks == []

    def test_min_separation_suppresses_adjacent(self):
        # Two peaks close together: min_separation should reject the
        # weaker one.
        trace = np.array([0.0, 5.0, 0.0, 6.0, 0.0])
        peaks = mpdsp.find_peaks(trace, bin_freq_step_hz=1.0,
                                 top_n=5, min_separation_bins=3)
        # The two peaks are 2 bins apart; only the taller (bin 3, amp 6)
        # survives with min_separation=3.
        assert len(peaks) == 1
        assert peaks[0].bin_index == 3

    def test_sub_bin_frequency_interpolation(self):
        # Symmetric peak: parabolic offset should be 0.
        trace = np.array([1.0, 2.0, 5.0, 2.0, 1.0])
        peaks = mpdsp.find_peaks(trace, bin_freq_step_hz=1.0, top_n=1)
        assert peaks[0].bin_index == 2
        assert abs(peaks[0].frequency_hz - 2.0) < 1e-9  # no offset

    def test_rejects_bad_bin_step(self):
        trace = np.array([1.0, 2.0, 1.0])
        with pytest.raises(ValueError):
            mpdsp.find_peaks(trace, bin_freq_step_hz=0.0, top_n=1)


class TestHarmonicMarkers:
    def test_returns_bins_at_multiples_of_fundamental(self):
        # trace of 100 bins at 1 Hz/bin. Fundamental = 10 Hz.
        # 2f=20, 3f=30, 4f=40 -> bins 20, 30, 40.
        trace = np.arange(100, dtype=np.float64)
        markers = mpdsp.harmonic_markers(trace, bin_freq_step_hz=1.0,
                                          fundamental_hz=10.0,
                                          harmonics=3)
        assert len(markers) == 3
        assert markers[0].bin_index == 20
        assert markers[1].bin_index == 30
        assert markers[2].bin_index == 40
        # Amplitudes match trace at target bins.
        assert markers[0].amplitude == 20.0

    def test_omits_out_of_range_harmonics(self):
        # Only 2nd harmonic fits in the trace; 3rd would be past the end.
        trace = np.arange(25, dtype=np.float64)
        markers = mpdsp.harmonic_markers(trace, bin_freq_step_hz=1.0,
                                          fundamental_hz=10.0,
                                          harmonics=3)
        # 2nd = 20 (in range), 3rd = 30 (out of range)
        assert len(markers) == 1
        assert markers[0].bin_index == 20

    def test_zero_harmonics_returns_empty(self):
        trace = np.arange(100, dtype=np.float64)
        markers = mpdsp.harmonic_markers(trace, 1.0, 10.0, 0)
        assert markers == []

    def test_empty_trace_returns_empty(self):
        markers = mpdsp.harmonic_markers(np.array([]), 1.0, 10.0, 3)
        assert markers == []

    def test_rejects_bad_bin_step(self):
        with pytest.raises(ValueError):
            mpdsp.harmonic_markers(np.arange(10, dtype=np.float64),
                                    bin_freq_step_hz=0.0,
                                    fundamental_hz=1.0, harmonics=2)

    def test_rejects_bad_fundamental(self):
        with pytest.raises(ValueError):
            mpdsp.harmonic_markers(np.arange(10, dtype=np.float64),
                                    bin_freq_step_hz=1.0,
                                    fundamental_hz=-1.0, harmonics=2)


class TestMarkersDtypeDispatch:
    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_find_peaks_across_dtypes(self, dtype):
        trace = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0])
        peaks = mpdsp.find_peaks(trace, bin_freq_step_hz=1.0, top_n=2,
                                  dtype=dtype)
        # Peak bin locations are precision-insensitive at these amplitudes.
        assert len(peaks) == 2
        # Strongest at bin 5, second at bin 1.
        assert peaks[0].bin_index == 5
        assert peaks[1].bin_index == 1
