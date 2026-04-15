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
        """ADC through posit<16,1> should work."""
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
        """posit<16,1> should have higher SQNR than half (same bit width)."""
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
