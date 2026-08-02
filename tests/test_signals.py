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

    # ---- Generators added by gap-analysis Phase 1 (#99) ----

    def test_ramp_default_slope(self):
        r = mpdsp.ramp(10)
        np.testing.assert_allclose(r, np.arange(10, dtype=np.float64))

    def test_ramp_custom_slope(self):
        r = mpdsp.ramp(5, slope=0.25)
        np.testing.assert_allclose(r, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_ramp_zero_length(self):
        assert mpdsp.ramp(0).shape == (0,)

    def test_multitone_shape_and_dtype(self):
        freqs = np.array([100.0, 200.0], dtype=np.float64)
        sig = mpdsp.multitone(1024, freqs, sample_rate=1024.0)
        assert sig.shape == (1024,)
        assert sig.dtype == np.float64

    def test_multitone_peaks_at_each_tone(self):
        # Place three tones far apart; PSD should peak at each.
        freqs = np.array([100.0, 250.0, 400.0], dtype=np.float64)
        sig = mpdsp.multitone(4096, freqs, sample_rate=1024.0, amplitude=1.0)
        pf, power = mpdsp.psd(sig, sample_rate=1024.0)
        for want in freqs:
            # Bin resolution ~ 1024/4096 = 0.25 Hz, but psd goes to sample_rate/2
            bin_hz = pf[1] - pf[0]
            bin_idx = int(round(want / bin_hz))
            local_peak = power[max(0, bin_idx - 3):bin_idx + 4].max()
            # Local peak near each tone must dominate the noise floor.
            assert local_peak > 10.0 * np.median(power)

    def test_multitone_empty_frequencies_is_zero(self):
        sig = mpdsp.multitone(64, np.array([], dtype=np.float64),
                              sample_rate=1024.0)
        assert sig.shape == (64,)
        np.testing.assert_array_equal(sig, np.zeros(64))

    def test_multitone_rejects_bad_sample_rate(self):
        with pytest.raises(ValueError):
            mpdsp.multitone(64, np.array([100.0]), sample_rate=0.0)

    def test_upsample_zero_insert(self):
        x = np.array([1.0, 2.0, 3.0])
        y = mpdsp.upsample(x, factor=3)
        # Every 3rd sample carries the original, the other two are zero.
        np.testing.assert_allclose(y, [1, 0, 0, 2, 0, 0, 3, 0, 0])

    def test_upsample_factor_one_is_identity(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(mpdsp.upsample(x, factor=1), x)

    def test_upsample_rejects_zero_factor(self):
        with pytest.raises(ValueError):
            mpdsp.upsample(np.array([1.0, 2.0]), factor=0)

    def test_downsample_keeps_every_nth(self):
        x = np.arange(12, dtype=np.float64)
        y = mpdsp.downsample(x, factor=3)
        np.testing.assert_allclose(y, [0.0, 3.0, 6.0, 9.0])

    def test_downsample_factor_one_is_identity(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(mpdsp.downsample(x, factor=1), x)

    def test_downsample_rejects_zero_factor(self):
        with pytest.raises(ValueError):
            mpdsp.downsample(np.array([1.0, 2.0]), factor=0)

    def test_upsample_then_downsample_roundtrip(self):
        # With no filter in between, up-N then down-N recovers the original.
        x = np.arange(1.0, 6.0)
        roundtrip = mpdsp.downsample(mpdsp.upsample(x, factor=4), factor=4)
        np.testing.assert_allclose(roundtrip, x)


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

    def test_dolph_chebyshev_peak_at_center(self):
        # Peak sits at the center of the window, not at the edges. Regression
        # guard against the upstream #200 collapse where the whole window went
        # to ~1.0 (all "peaks" everywhere).
        w = mpdsp.dolph_chebyshev(256, attenuation_db=80.0)
        assert w[128] > w[0] + 0.5

    def test_dolph_chebyshev_attenuation_threaded_through(self):
        # 40 dB and 120 dB windows are meaningfully different. Under the
        # pre-#200 upstream bug both collapsed to essentially the same
        # constant; different attenuation values are only distinguishable
        # after that fix landed.
        w_low  = mpdsp.dolph_chebyshev(256, attenuation_db=40.0)
        w_high = mpdsp.dolph_chebyshev(256, attenuation_db=120.0)
        # Sum of |diff| over the whole window — small tolerance is fine
        # because Dolph-Cheby edges differ by orders of magnitude between
        # these attenuation levels.
        assert np.sum(np.abs(w_low - w_high)) > 1.0

    def test_dolph_chebyshev_edges_below_peak(self):
        # At 100 dB attenuation the edges should be MUCH smaller than the
        # peak. Finite-N Dolph-Cheby edges don't reach the ideal
        # 10^(-atten/20) = 1e-5 asymptote, but they should easily be
        # below 0.2 of the peak for N=256.
        w = mpdsp.dolph_chebyshev(256, attenuation_db=100.0)
        assert w[0] < 0.2
        assert w[-1] < 0.2

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
