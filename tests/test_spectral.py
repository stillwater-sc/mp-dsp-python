"""Tests for spectral analysis: FFT, PSD, spectrogram via nanobind."""

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


class TestFFT:
    def test_fft_returns_tuple(self):
        sig = mpdsp.sine(256, frequency=10.0, sample_rate=256.0)
        result = mpdsp.fft(sig)
        assert isinstance(result, tuple)
        assert len(result) == 2
        re, im = result
        assert re.shape == (256,)
        assert im.shape == (256,)

    def test_fft_peak_location(self):
        """FFT of a pure sine should peak at the signal frequency."""
        N = 1024
        sig = mpdsp.sine(N, frequency=100.0, sample_rate=1024.0)
        re, im = mpdsp.fft(sig)
        mag = np.sqrt(re**2 + im**2)
        # Peak in first half (positive frequencies)
        half = mag[:N // 2]
        peak_bin = np.argmax(half)
        peak_freq = peak_bin * 1024.0 / N
        assert abs(peak_freq - 100.0) < 2.0  # within 2 Hz

    def test_fft_magnitude_db(self):
        sig = mpdsp.sine(256, frequency=10.0, sample_rate=256.0)
        mag_db = mpdsp.fft_magnitude_db(sig)
        assert mag_db.shape == (256,)
        assert mag_db.dtype == np.float64
        # Should have a clear peak
        assert mag_db.max() > 0

    def test_ifft_roundtrip(self):
        """FFT → IFFT should recover the original signal."""
        sig = mpdsp.sine(256, frequency=10.0, sample_rate=256.0)
        re, im = mpdsp.fft(sig)
        recovered = mpdsp.ifft(re, im)
        np.testing.assert_allclose(sig, recovered[:len(sig)], atol=1e-12)


class TestPSD:
    def test_periodogram_shape(self):
        sig = mpdsp.sine(1024, frequency=100.0, sample_rate=1024.0)
        power = mpdsp.periodogram(sig)
        assert power.shape == (513,)  # N/2 + 1

    def test_psd_returns_freqs_and_power(self):
        sig = mpdsp.sine(1024, frequency=100.0, sample_rate=1024.0)
        freqs, power = mpdsp.psd(sig, sample_rate=1024.0)
        assert freqs.shape == power.shape
        assert freqs[0] == 0.0
        # Last frequency should be Nyquist
        assert abs(freqs[-1] - 512.0) < 1.0

    def test_psd_peak_at_signal_frequency(self):
        sig = mpdsp.sine(4096, frequency=440.0, sample_rate=44100.0)
        freqs, power = mpdsp.psd(sig, sample_rate=44100.0)
        peak_freq = freqs[np.argmax(power)]
        assert abs(peak_freq - 440.0) < 20.0  # within 20 Hz


class TestSpectrogram:
    def test_spectrogram_shape(self):
        sig = mpdsp.sine(2048, frequency=100.0, sample_rate=1024.0)
        times, freqs, mag = mpdsp.spectrogram(sig, sample_rate=1024.0,
                                               window_size=256, hop_size=64)
        assert mag.ndim == 2
        assert mag.shape[0] == len(times)
        assert mag.shape[1] == len(freqs)

    def test_spectrogram_time_axis(self):
        sig = mpdsp.sine(2048, frequency=100.0, sample_rate=1024.0)
        times, freqs, mag = mpdsp.spectrogram(sig, sample_rate=1024.0,
                                               window_size=256, hop_size=64)
        # Times should be monotonically increasing
        assert np.all(np.diff(times) > 0)
        # First time should be near half a window
        assert times[0] > 0

    def test_spectrogram_frequency_axis(self):
        sig = mpdsp.sine(2048, frequency=100.0, sample_rate=1024.0)
        times, freqs, mag = mpdsp.spectrogram(sig, sample_rate=1024.0,
                                               window_size=256, hop_size=64)
        assert freqs[0] == 0.0
        # Max frequency should be Nyquist
        assert abs(freqs[-1] - 512.0) < 5.0

    def test_spectrogram_values_are_db(self):
        sig = mpdsp.sine(2048, frequency=100.0, sample_rate=1024.0)
        times, freqs, mag = mpdsp.spectrogram(sig, sample_rate=1024.0)
        # dB values can be negative
        assert mag.min() < 0
        # But should have some positive peaks
        assert mag.max() > -20


# ---------------------------------------------------------------------------
# Dtype dispatch on all 5 spectral primitives (#54 Phase 5 PR-B).
#
# Every primitive now accepts a `dtype=` string selecting the internal
# arithmetic. Inputs and outputs stay double/complex128 at the Python layer;
# only the C++ computation runs at the target precision.
# ---------------------------------------------------------------------------


# Every primitive should accept every config listed in available_dtypes().
# Check this by construction so adding a new ArithConfig enumerator (e.g.
# #55's sensor/fpga types) automatically exercises each primitive.
ALL_DTYPES = list(mpdsp.available_dtypes())


class TestSpectralDtypeDispatch:
    """One test per primitive at `reference` + one narrow dtype, plus a
    sweep that hits every config. The sweep guards against a new config
    landing without a corresponding dispatch case."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_fft_runs_for_every_dtype(self, dtype):
        sig = mpdsp.sine(256, frequency=10.0, sample_rate=256.0)
        r, i = mpdsp.fft(sig, dtype=dtype)
        assert r.shape == i.shape == (256,)
        assert np.all(np.isfinite(r))
        assert np.all(np.isfinite(i))

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_fft_magnitude_db_runs_for_every_dtype(self, dtype):
        sig = mpdsp.sine(256, frequency=10.0, sample_rate=256.0)
        mag = mpdsp.fft_magnitude_db(sig, dtype=dtype)
        assert mag.shape == (256,)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_ifft_runs_for_every_dtype(self, dtype):
        sig = mpdsp.sine(256, frequency=10.0, sample_rate=256.0)
        r, i = mpdsp.fft(sig, dtype="reference")
        recovered = mpdsp.ifft(r, i, dtype=dtype)
        assert recovered.shape == sig.shape

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_periodogram_runs_for_every_dtype(self, dtype):
        sig = mpdsp.sine(256, frequency=10.0, sample_rate=256.0)
        p = mpdsp.periodogram(sig, dtype=dtype)
        assert p.shape == (129,)  # N/2 + 1 for real-input periodogram
        assert np.all(p >= 0)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_psd_runs_for_every_dtype(self, dtype):
        sig = mpdsp.sine(256, frequency=10.0, sample_rate=256.0)
        f, p = mpdsp.psd(sig, sample_rate=256.0, dtype=dtype)
        assert f.shape == p.shape == (129,)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_spectrogram_runs_for_every_dtype(self, dtype):
        sig = mpdsp.sine(512, frequency=10.0, sample_rate=256.0)
        t, f, m = mpdsp.spectrogram(
            sig, sample_rate=256.0, window_size=128, hop_size=32,
            dtype=dtype)
        assert m.ndim == 2
        assert m.shape == (len(t), len(f))


class TestSpectralDtypeFidelity:
    """Precision semantics: narrower dtypes should produce observably
    different (generally lower-fidelity) output than `reference`. These
    aren't looking for specific SQNR floors — just confirming dispatch is
    actually routing to different typed implementations."""

    def test_fft_narrow_dtype_differs_from_reference(self):
        # A mid-amplitude sine quantized through tiny_posit should show
        # meaningful divergence from the double-precision FFT.
        sig = mpdsp.sine(512, frequency=10.0, sample_rate=512.0,
                          amplitude=0.5)
        r_ref, i_ref = mpdsp.fft(sig, dtype="reference")
        r_narrow, i_narrow = mpdsp.fft(sig, dtype="tiny_posit")
        max_diff = max(np.max(np.abs(r_ref - r_narrow)),
                        np.max(np.abs(i_ref - i_narrow)))
        assert max_diff > 1e-3, \
            "tiny_posit FFT shouldn't be bit-identical to reference"

    def test_periodogram_narrow_dtype_differs_from_reference(self):
        sig = mpdsp.sine(512, frequency=10.0, sample_rate=512.0,
                          amplitude=0.5)
        p_ref = mpdsp.periodogram(sig, dtype="reference")
        p_narrow = mpdsp.periodogram(sig, dtype="tiny_posit")
        # Peak bin location should still agree (that's a structural
        # invariant), but magnitudes will diverge.
        assert np.argmax(p_ref) == np.argmax(p_narrow)
        assert np.max(np.abs(p_ref - p_narrow)) > 1e-6

    def test_fft_ifft_roundtrip_reference_is_near_exact(self):
        # At reference dtype, fft -> ifft must recover the input to
        # machine precision.
        sig = mpdsp.sine(256, frequency=12.0, sample_rate=256.0)
        r, i = mpdsp.fft(sig, dtype="reference")
        recovered = mpdsp.ifft(r, i, dtype="reference")
        np.testing.assert_allclose(recovered, sig, atol=1e-10)

    def test_invalid_dtype_raises(self):
        sig = mpdsp.sine(64, frequency=10.0, sample_rate=64.0)
        with pytest.raises(ValueError):
            mpdsp.fft(sig, dtype="nonexistent")
        with pytest.raises(ValueError):
            mpdsp.periodogram(sig, dtype="nonexistent")
