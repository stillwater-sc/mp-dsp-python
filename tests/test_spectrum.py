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
