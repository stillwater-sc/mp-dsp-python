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
