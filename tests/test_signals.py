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

    # ---- Windows added by gap-analysis Phase 1 (#98) ----

    def test_tukey_default_matches_hann_shape(self):
        # alpha=1.0 makes Tukey equivalent to Hann (raised cosine).
        tukey = mpdsp.tukey(256, alpha=1.0)
        hann = mpdsp.hanning(256)
        np.testing.assert_allclose(tukey, hann, atol=1e-10)

    def test_tukey_rectangular_at_alpha_zero(self):
        # alpha=0.0 collapses Tukey to a rectangular window.
        w = mpdsp.tukey(128, alpha=0.0)
        np.testing.assert_allclose(w, np.ones(128), atol=1e-10)

    def test_tukey_alpha_threaded_through(self):
        w_low = mpdsp.tukey(128, alpha=0.1)
        w_high = mpdsp.tukey(128, alpha=0.9)
        # Different alpha values give measurably different tapers.
        assert not np.allclose(w_low, w_high)

    def test_gaussian_shape_and_symmetry(self):
        w = mpdsp.gaussian(256, sigma=0.4)
        assert w.shape == (256,)
        np.testing.assert_allclose(w, w[::-1], atol=1e-10)

    def test_gaussian_sigma_threaded_through(self):
        narrow = mpdsp.gaussian(256, sigma=0.1)
        wide = mpdsp.gaussian(256, sigma=0.5)
        # Smaller sigma → more concentrated: narrow window has lower edges.
        assert narrow[0] < wide[0]

    def test_dolph_chebyshev_shape(self):
        w = mpdsp.dolph_chebyshev(256, attenuation_db=80.0)
        assert w.shape == (256,)
        assert np.all(np.isfinite(w))
        # Symmetric (the upstream implementation enforces symmetry explicitly).
        np.testing.assert_allclose(w, w[::-1], atol=1e-10)
        # Peak normalized to 1.0.
        assert abs(w.max() - 1.0) < 1e-10

    # NOTE on upstream limitation: the current
    # mixed-precision-dsp dolph_chebyshev_window implementation appears to
    # produce a near-constant output regardless of attenuation_db — all taps
    # sit within ~1e-14 of the peak for representative (N, atten_db) pairs.
    # We bind the function unchanged and cover the shape/symmetry/peak
    # invariants that DO hold; a "different attenuation → different taps"
    # test would be the natural addition but must wait on an upstream fix.

    def test_dolph_chebyshev_rejects_nonpositive_attenuation(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.dolph_chebyshev(256, attenuation_db=0.0)
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.dolph_chebyshev(256, attenuation_db=-10.0)

    def test_bartlett_hann_shape_and_endpoints(self):
        w = mpdsp.bartlett_hann(256)
        assert w.shape == (256,)
        # Bartlett-Hann ends approach zero (Bartlett component is triangular).
        assert abs(w[0]) < 0.01
        assert abs(w[-1]) < 0.01
        # Symmetric
        np.testing.assert_allclose(w, w[::-1], atol=1e-10)

    @pytest.mark.parametrize("name,args", [
        ("tukey", {"alpha": 0.3}),
        ("gaussian", {"sigma": 0.3}),
        ("dolph_chebyshev", {"attenuation_db": 60.0}),
        ("bartlett_hann", {}),
    ])
    def test_new_windows_accept_dtype(self, name, args):
        fn = getattr(mpdsp, name)
        ref = fn(128, **args, dtype="reference")
        posit = fn(128, **args, dtype="posit_full")
        assert posit.shape == (128,)
        assert posit.dtype == np.float64
        assert np.all(np.isfinite(posit))
        # posit_full has enough precision to closely reproduce the window.
        np.testing.assert_allclose(posit, ref, atol=1e-3)


class TestWindowsDtype:
    """Window functions accept a dtype kwarg (re-templating from upstream
    PRs #122 / #125). The default 'reference' (double) must be bitwise
    identical to the no-kwarg call; mixed-precision dtypes return a
    finite, real-valued window of the right shape that is close to but
    not necessarily equal to the reference.
    """

    def test_default_dtype_matches_no_kwarg(self):
        # Sanity: passing dtype="reference" must equal the no-kwarg call.
        np.testing.assert_array_equal(
            mpdsp.hamming(128),
            mpdsp.hamming(128, dtype="reference"),
        )

    def test_hamming_posit32_returns_float64_array(self):
        w = mpdsp.hamming(128, dtype="posit_full")
        assert w.shape == (128,)
        assert w.dtype == np.float64
        assert np.all(np.isfinite(w))

    def test_hanning_posit16_close_to_reference(self):
        ref = mpdsp.hanning(64, dtype="reference")
        p16 = mpdsp.hanning(64, dtype="posit_16_1")
        # posit<16,1> has ~12 fraction bits in the dynamic-range sweet
        # spot; should reproduce a [0, 1]-valued window to a few %.
        np.testing.assert_allclose(p16, ref, atol=1e-2)

    def test_kaiser_beta_threaded_through(self):
        w_default = mpdsp.kaiser(128, dtype="reference")
        w_narrow = mpdsp.kaiser(128, beta=2.0, dtype="reference")
        # A smaller beta gives a less-tapered window — center and edge
        # values must differ between the two beta settings.
        assert not np.allclose(w_default, w_narrow)

    def test_unknown_dtype_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.hamming(64, dtype="not_a_real_dtype")
