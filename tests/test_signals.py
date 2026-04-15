"""Tests for signal generators and window functions via nanobind."""

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


class TestSignalGenerators:
    def test_sine_shape_and_dtype(self):
        sig = mpdsp.sine(1000, frequency=440.0, sample_rate=44100.0)
        assert sig.shape == (1000,)
        assert sig.dtype == np.float64

    def test_sine_range(self):
        sig = mpdsp.sine(10000, frequency=100.0, sample_rate=44100.0)
        assert sig.max() <= 1.0 + 1e-10
        assert sig.min() >= -1.0 - 1e-10

    def test_sine_amplitude(self):
        sig = mpdsp.sine(10000, frequency=100.0, sample_rate=44100.0, amplitude=0.5)
        assert sig.max() <= 0.5 + 1e-10
        assert sig.min() >= -0.5 - 1e-10

    def test_cosine(self):
        sig = mpdsp.cosine(1000, frequency=440.0, sample_rate=44100.0)
        # Cosine starts at 1.0
        assert abs(sig[0] - 1.0) < 1e-10

    def test_chirp(self):
        sig = mpdsp.chirp(2000, f_start=100.0, f_end=5000.0, sample_rate=44100.0)
        assert sig.shape == (2000,)
        assert sig.dtype == np.float64

    def test_square(self):
        sig = mpdsp.square(1000, frequency=100.0, sample_rate=44100.0)
        # Square wave should have values near +1 and -1
        assert abs(sig.max() - 1.0) < 0.1
        assert abs(sig.min() + 1.0) < 0.1

    def test_triangle(self):
        sig = mpdsp.triangle(1000, frequency=100.0, sample_rate=44100.0)
        assert sig.shape == (1000,)

    def test_sawtooth(self):
        sig = mpdsp.sawtooth(1000, frequency=100.0, sample_rate=44100.0)
        assert sig.shape == (1000,)

    def test_impulse(self):
        sig = mpdsp.impulse(100)
        assert sig[0] == 1.0
        assert sig[1] == 0.0
        assert sig[99] == 0.0

    def test_impulse_position(self):
        sig = mpdsp.impulse(100, position=50)
        assert sig[49] == 0.0
        assert sig[50] == 1.0
        assert sig[51] == 0.0

    def test_step(self):
        sig = mpdsp.step(100, position=30)
        assert sig[29] == 0.0
        assert sig[30] == 1.0
        assert sig[99] == 1.0

    def test_white_noise_deterministic(self):
        n1 = mpdsp.white_noise(1000, seed=42)
        n2 = mpdsp.white_noise(1000, seed=42)
        np.testing.assert_array_equal(n1, n2)

    def test_white_noise_range(self):
        n = mpdsp.white_noise(10000, seed=42)
        assert n.min() >= -1.0
        assert n.max() <= 1.0


class TestWindows:
    def test_hamming_shape(self):
        w = mpdsp.hamming(256)
        assert w.shape == (256,)
        assert w.dtype == np.float64

    def test_hamming_symmetric(self):
        w = mpdsp.hamming(256)
        np.testing.assert_allclose(w, w[::-1], atol=1e-10)

    def test_hanning(self):
        w = mpdsp.hanning(256)
        assert w.shape == (256,)
        # Hanning starts and ends at 0
        assert abs(w[0]) < 0.01
        assert abs(w[-1]) < 0.01

    def test_blackman(self):
        w = mpdsp.blackman(256)
        assert w.shape == (256,)

    def test_kaiser(self):
        w = mpdsp.kaiser(256, beta=8.0)
        assert w.shape == (256,)

    def test_rectangular(self):
        w = mpdsp.rectangular(100)
        np.testing.assert_allclose(w, np.ones(100), atol=1e-10)

    def test_flat_top(self):
        w = mpdsp.flat_top(256)
        assert w.shape == (256,)
