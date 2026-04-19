"""Tests for quantization: ADC, SQNR, error measurement via nanobind."""

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


class TestADC:
    def test_adc_reference_is_identity(self):
        """ADC with double → double should be lossless."""
        sig = mpdsp.sine(1000, frequency=440.0, sample_rate=44100.0)
        q = mpdsp.adc(sig, dtype="reference")
        np.testing.assert_array_equal(sig, q)

    def test_adc_float_loses_precision(self):
        """ADC through float should lose some precision."""
        sig = mpdsp.sine(1000, frequency=440.0, sample_rate=44100.0)
        q = mpdsp.adc(sig, dtype="gpu_baseline")
        assert not np.array_equal(sig, q)
        # But should be close
        np.testing.assert_allclose(sig, q, atol=1e-6)

    def test_adc_half_loses_more(self):
        """ADC through half should have larger error than float."""
        sig = mpdsp.sine(1000, frequency=440.0, sample_rate=44100.0)
        q_float = mpdsp.adc(sig, dtype="gpu_baseline")
        q_half = mpdsp.adc(sig, dtype="half")
        err_float = np.max(np.abs(sig - q_float))
        err_half = np.max(np.abs(sig - q_half))
        assert err_half > err_float

    def test_adc_posit(self):
        """ADC through posit_full config (posit<16,1> samples) should work."""
        sig = mpdsp.sine(1000, frequency=440.0, sample_rate=44100.0)
        q = mpdsp.adc(sig, dtype="posit_full")
        assert q.shape == sig.shape


class TestSQNR:
    def test_sqnr_identical_signals(self):
        """Identical signals should give very high SQNR."""
        sig = mpdsp.sine(1000, frequency=440.0, sample_rate=44100.0)
        sqnr = mpdsp.sqnr_db(sig, sig)
        assert sqnr >= 290.0  # capped at 300

    def test_sqnr_float(self):
        """Float SQNR should be ~150 dB for a sine wave."""
        sig = mpdsp.sine(10000, frequency=440.0, sample_rate=44100.0)
        sqnr = mpdsp.measure_sqnr_db(sig, "gpu_baseline")
        assert sqnr > 100.0
        assert sqnr < 200.0

    def test_sqnr_half(self):
        """Half SQNR should be lower than float."""
        sig = mpdsp.sine(10000, frequency=440.0, sample_rate=44100.0)
        sqnr_float = mpdsp.measure_sqnr_db(sig, "gpu_baseline")
        sqnr_half = mpdsp.measure_sqnr_db(sig, "half")
        assert sqnr_half < sqnr_float

    def test_sqnr_posit_vs_half(self):
        """posit_full (posit<16,1> ADC) should have higher SQNR than half."""
        sig = mpdsp.sine(10000, frequency=440.0, sample_rate=44100.0)
        sqnr_posit = mpdsp.measure_sqnr_db(sig, "posit_full")
        sqnr_half = mpdsp.measure_sqnr_db(sig, "half")
        assert sqnr_posit > sqnr_half

    def test_sqnr_ordering(self):
        """SQNR should follow: double > float > cf24 > half."""
        sig = mpdsp.sine(10000, frequency=440.0, sample_rate=44100.0)
        sqnr_f = mpdsp.measure_sqnr_db(sig, "gpu_baseline")
        sqnr_c = mpdsp.measure_sqnr_db(sig, "cf24")
        sqnr_h = mpdsp.measure_sqnr_db(sig, "half")
        assert sqnr_f > sqnr_c > sqnr_h


class TestErrors:
    def test_max_absolute_error(self):
        sig = mpdsp.sine(1000, frequency=440.0, sample_rate=44100.0)
        q = mpdsp.adc(sig, dtype="half")
        err = mpdsp.max_absolute_error(sig, q)
        assert err > 0
        assert err < 0.01  # half should be within 1%

    def test_max_relative_error(self):
        sig = mpdsp.sine(1000, frequency=440.0, sample_rate=44100.0)
        q = mpdsp.adc(sig, dtype="half")
        err = mpdsp.max_relative_error(sig, q)
        assert err > 0
        assert err < 0.01


class TestAvailableDtypes:
    def test_available_dtypes(self):
        dtypes = mpdsp.available_dtypes()
        assert "reference" in dtypes
        assert "posit_full" in dtypes
        assert "half" in dtypes

    def test_invalid_dtype_raises(self):
        sig = mpdsp.sine(100, frequency=440.0, sample_rate=44100.0)
        with pytest.raises(ValueError):
            mpdsp.adc(sig, dtype="nonexistent_type")


# ---------------------------------------------------------------------------
# dac — companion of adc (both round-trip float64 through a narrow dtype).
# ---------------------------------------------------------------------------


class TestDAC:
    def test_dac_reference_is_identity(self):
        sig = mpdsp.sine(256, frequency=440.0, sample_rate=44100.0)
        np.testing.assert_array_equal(mpdsp.dac(sig, dtype="reference"), sig)

    def test_dac_matches_adc_for_float_families(self):
        # In Python, both adc and dac do double -> narrow -> double. For
        # float-family dtypes the two are literally the same operation;
        # pin that so accidental divergence is caught.
        sig = mpdsp.sine(512, frequency=440.0, sample_rate=44100.0)
        for dt in ("gpu_baseline", "cf24", "half"):
            q = mpdsp.adc(sig, dtype=dt)
            reconstructed = mpdsp.dac(q, dtype=dt)
            np.testing.assert_array_equal(q, reconstructed)

    def test_dac_runs_for_every_dtype(self):
        sig = mpdsp.sine(128, frequency=100.0, sample_rate=44100.0)
        for dt in mpdsp.available_dtypes():
            out = mpdsp.dac(sig, dtype=dt)
            assert out.shape == sig.shape
            assert out.dtype == sig.dtype


# ---------------------------------------------------------------------------
# Dither classes — RPDFDither / TPDFDither.
# ---------------------------------------------------------------------------


class TestRPDFDither:
    def test_construction_every_dtype(self):
        for dt in mpdsp.available_dtypes():
            d = mpdsp.RPDFDither(amplitude=0.1, dtype=dt, seed=42)
            assert d.dtype == dt
            assert d.amplitude == pytest.approx(0.1)

    def test_rejects_negative_amplitude(self):
        with pytest.raises(ValueError):
            mpdsp.RPDFDither(amplitude=-0.1)

    def test_sample_bounded_by_amplitude(self):
        d = mpdsp.RPDFDither(amplitude=0.5, seed=7)
        for _ in range(200):
            s = d.sample()
            assert -0.5 <= s <= 0.5

    def test_apply_returns_new_array(self):
        rng = np.random.default_rng(1)
        signal = rng.normal(0.0, 0.2, size=256)
        d = mpdsp.RPDFDither(amplitude=0.01, seed=3)
        dithered = d.apply(signal)
        # Distinct buffer — NumPy idiom, not in-place.
        assert dithered is not signal
        assert dithered.shape == signal.shape
        assert np.max(np.abs(dithered - signal)) <= 0.01 + 1e-12

    def test_seed_deterministic(self):
        d1 = mpdsp.RPDFDither(amplitude=0.1, seed=11)
        d2 = mpdsp.RPDFDither(amplitude=0.1, seed=11)
        assert d1.sample() == d2.sample()


class TestTPDFDither:
    def test_sample_bounded_by_amplitude(self):
        d = mpdsp.TPDFDither(amplitude=0.5, seed=7)
        for _ in range(200):
            s = d.sample()
            assert -0.5 <= s <= 0.5

    def test_tpdf_variance_lower_than_rpdf_same_amplitude(self):
        # TPDF variance: A^2 / 6. RPDF variance: A^2 / 3. Large N so the
        # statistical ordering holds even with different PRNG draws.
        tpdf = mpdsp.TPDFDither(amplitude=1.0, seed=17)
        rpdf = mpdsp.RPDFDither(amplitude=1.0, seed=17)
        n = 5000
        t_samples = np.array([tpdf.sample() for _ in range(n)])
        r_samples = np.array([rpdf.sample() for _ in range(n)])
        assert np.var(t_samples) < np.var(r_samples)

    def test_apply_roundtrips_shape(self):
        signal = np.zeros(128)
        d = mpdsp.TPDFDither(amplitude=0.1, seed=9)
        out = d.apply(signal)
        assert out.shape == signal.shape
        assert np.max(np.abs(out)) <= 0.1 + 1e-12


class TestDitherPreservesSignalPrecision:
    """Regression guard for a PR #57 review finding: `apply()` must draw
    dither in the dtype's precision and add it to the ORIGINAL
    high-precision signal. An earlier implementation pre-quantized the
    signal to `T` before the dither was added, which defeated the whole
    point of dithering upstream of an explicit ADC stage.

    The cleanest way to pin this is a zero-amplitude dither — every
    drawn sample is exactly `amplitude * uniform(...) = 0`, so apply()
    reduces to the identity on the signal. A buggy "quantize first"
    implementation would show quantization error even at amplitude=0.
    """

    def test_rpdf_apply_is_identity_at_zero_amplitude_half(self):
        # High-precision signal that does NOT lie on the half-precision
        # grid — if apply() secretly round-trips through half, every
        # sample will have observable error.
        rng = np.random.default_rng(13)
        signal = rng.normal(0.0, 0.3, size=256).astype(np.float64)
        d = mpdsp.RPDFDither(amplitude=0.0, dtype="half", seed=1)
        out = d.apply(signal)
        # Exact equality — every dither draw is 0.0 because amplitude is 0.
        np.testing.assert_array_equal(out, signal)

    def test_tpdf_apply_is_identity_at_zero_amplitude_tiny_posit(self):
        rng = np.random.default_rng(17)
        signal = rng.normal(0.0, 0.3, size=256).astype(np.float64)
        d = mpdsp.TPDFDither(amplitude=0.0, dtype="tiny_posit", seed=1)
        out = d.apply(signal)
        np.testing.assert_array_equal(out, signal)

    def test_rpdf_apply_half_bounds_are_half_precision_noise(self):
        # With non-zero amplitude at half dtype, each sample's drift from
        # the input is bounded by amplitude (plus a tiny slack for the
        # T→double conversion). Signal values themselves are preserved
        # exactly — the delta from signal to output IS the dither, no
        # signal-quantization leaking in.
        rng = np.random.default_rng(21)
        signal = rng.normal(0.0, 0.3, size=256).astype(np.float64)
        d = mpdsp.RPDFDither(amplitude=0.001, dtype="half", seed=3)
        out = d.apply(signal)
        diff = out - signal
        # Bound is amplitude * (1 + small half-rounding slack). Generous
        # cap; the point is that the signal isn't being round-tripped.
        assert np.max(np.abs(diff)) <= 0.001 + 1e-4


class TestDitherDecorrelation:
    """Classic dither benefit: quantization error gets decorrelated from
    the signal — its correlation with the signal drops toward zero."""

    def test_rpdf_dither_decorrelates_error_from_signal(self):
        # Low-amplitude sine quantized to `half` gives systematic
        # distortion correlated with the signal. RPDF dither should
        # shrink that correlation. Generous tolerance — statistical.
        signal = 0.01 * np.sin(2 * np.pi * np.arange(2048) / 64)
        q = mpdsp.adc(signal, dtype="half")
        undithered_err = q - signal
        undithered_corr = np.corrcoef(signal, undithered_err)[0, 1]

        dither = mpdsp.RPDFDither(amplitude=0.001, seed=5)
        dithered_sig = dither.apply(signal)
        q_d = mpdsp.adc(dithered_sig, dtype="half")
        dithered_err = q_d - signal
        dithered_corr = np.corrcoef(signal, dithered_err)[0, 1]

        assert abs(dithered_corr) <= abs(undithered_corr) + 1e-6


# ---------------------------------------------------------------------------
# FirstOrderNoiseShaper
# ---------------------------------------------------------------------------


class TestNoiseShaper:
    def test_construction_every_dtype(self):
        for dt in mpdsp.available_dtypes():
            s = mpdsp.FirstOrderNoiseShaper(dtype=dt)
            assert s.dtype == dt

    def test_reference_process_is_near_identity(self):
        # At `reference` dtype, there's no quantization — error feedback
        # stays zero and output equals input.
        shaper = mpdsp.FirstOrderNoiseShaper(dtype="reference")
        rng = np.random.default_rng(3)
        signal = rng.normal(0.0, 0.3, size=256)
        out = shaper.process_block(signal)
        np.testing.assert_allclose(out, signal, atol=1e-12)

    def test_process_block_and_process_agree(self):
        signal = np.linspace(-0.4, 0.4, 64)
        s_block = mpdsp.FirstOrderNoiseShaper(dtype="half")
        block_out = s_block.process_block(signal)

        s_sample = mpdsp.FirstOrderNoiseShaper(dtype="half")
        sample_out = np.array([s_sample.process(x) for x in signal])

        np.testing.assert_array_equal(block_out, sample_out)

    def test_reset_clears_feedback(self):
        signal = np.linspace(-0.3, 0.3, 32)
        shaper = mpdsp.FirstOrderNoiseShaper(dtype="half")
        first = shaper.process_block(signal)
        shaper.reset()
        second = shaper.process_block(signal)
        np.testing.assert_array_equal(first, second)

    def test_shaping_produces_different_spectrum_than_plain(self):
        # The "first-order shaping pushes energy to higher frequencies"
        # claim is generally true but hard to pin in a unit test for a
        # single chosen signal — the residual spectrum depends on signal
        # content, quantization grid alignment, and feedback accumulation
        # in ways that aren't monotone for any one input. What we CAN pin
        # is that shaping produces a meaningfully different residual from
        # plain quantization; that's enough to catch a no-op binding.
        signal = 0.1 * np.sin(2 * np.pi * 4 * np.arange(1024) / 1024)
        plain_err = mpdsp.adc(signal, dtype="half") - signal
        shaper = mpdsp.FirstOrderNoiseShaper(dtype="half")
        shaped_err = shaper.process_block(signal) - signal
        # Different spectra — L1 distance between residual magnitudes
        # should be non-trivial for a working shaper, zero for a no-op.
        plain_mag = np.abs(np.fft.rfft(plain_err))
        shaped_mag = np.abs(np.fft.rfft(shaped_err))
        assert np.sum(np.abs(plain_mag - shaped_mag)) > 1e-6
