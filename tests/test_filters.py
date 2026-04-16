"""Tests for IIR filter bindings (Butterworth LP/HP)."""

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


SAMPLE_RATE = 8000.0


class TestButterworthDesign:
    def test_lowpass_returns_filter_with_stages(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        assert isinstance(filt, mpdsp.IIRFilter)
        assert filt.num_stages() == 2  # order 4 -> 2 biquads

    def test_highpass_returns_filter_with_stages(self):
        filt = mpdsp.butterworth_highpass(order=6, sample_rate=SAMPLE_RATE, cutoff=500.0)
        assert filt.num_stages() == 3  # order 6 -> 3 biquads

    def test_odd_order_has_single_first_order_section(self):
        filt = mpdsp.butterworth_lowpass(order=5, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        assert filt.num_stages() == 3  # ceil(5/2) = 3 sections

    def test_coefficients_shape(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        coeffs = filt.coefficients()
        assert len(coeffs) == 2
        for stage in coeffs:
            assert len(stage) == 5  # (b0, b1, b2, a1, a2)

    def test_poles_count_matches_order(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        poles = filt.poles()
        assert len(poles) == 4
        # All poles must be inside the unit circle for a stable filter
        for p in poles:
            assert abs(p) < 1.0

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            mpdsp.butterworth_lowpass(order=0, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        with pytest.raises(ValueError):
            mpdsp.butterworth_lowpass(order=17, sample_rate=SAMPLE_RATE, cutoff=1000.0)

    def test_cutoff_above_nyquist_raises(self):
        with pytest.raises(ValueError):
            mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=5000.0)


class TestFrequencyResponse:
    def test_response_shape_and_dtype(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        freqs = np.linspace(0.0, 0.5, 64)
        h = filt.frequency_response(freqs)
        assert h.shape == (64,)
        assert h.dtype == np.complex128

    def test_lowpass_dc_passes_stopband_attenuates(self):
        filt = mpdsp.butterworth_lowpass(order=6, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        # normalized: DC -> 0.0; nyquist -> 0.5; cutoff at 1000/8000 = 0.125
        freqs = np.array([0.0, 0.125, 0.4])
        mag = np.abs(filt.frequency_response(freqs))
        # DC gain ~ 1.0
        assert abs(mag[0] - 1.0) < 1e-6
        # Butterworth is -3 dB at the cutoff (magnitude ~ 1/sqrt(2))
        assert abs(mag[1] - 1.0 / np.sqrt(2.0)) < 0.05
        # Deep stopband heavily attenuated
        assert mag[2] < 0.01

    def test_highpass_dc_rejects(self):
        filt = mpdsp.butterworth_highpass(order=6, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        freqs = np.array([0.0, 0.125, 0.4])
        mag = np.abs(filt.frequency_response(freqs))
        assert mag[0] < 0.01
        # passband at high freq approaches 1.0
        assert abs(mag[2] - 1.0) < 0.1


class TestProcessing:
    def _sine(self, freq, n=4096):
        t = np.arange(n) / SAMPLE_RATE
        return np.sin(2.0 * np.pi * freq * t)

    def test_process_shape_preserved(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = self._sine(200.0)
        y = filt.process(sig)
        assert y.shape == sig.shape
        assert y.dtype == np.float64

    def test_lowpass_passes_low_rejects_high(self):
        filt = mpdsp.butterworth_lowpass(order=6, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        low = self._sine(200.0)
        high = self._sine(3000.0)
        # Skip transient so steady-state amplitude is meaningful.
        skip = 512
        y_low = filt.process(low)[skip:]
        y_high = filt.process(high)[skip:]
        # Passband amplitude close to 1; stopband amplitude heavily attenuated.
        assert np.max(np.abs(y_low)) > 0.9
        assert np.max(np.abs(y_high)) < 0.1

    def test_dtype_reference_matches_double(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = self._sine(300.0)
        a = filt.process(sig, dtype="reference")
        b = filt.process(sig, dtype="double")
        np.testing.assert_array_equal(a, b)

    def test_dtype_dispatch_differs_from_reference(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = self._sine(300.0)
        ref = filt.process(sig, dtype="reference")
        low = filt.process(sig, dtype="half")
        # Reduced-precision arithmetic should introduce measurable error.
        assert not np.array_equal(ref, low)
        # But the output should still be close — within ~1% of reference.
        err = np.max(np.abs(ref - low)) / (np.max(np.abs(ref)) + 1e-12)
        assert err < 0.05

    def test_unknown_dtype_raises(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = self._sine(300.0)
        with pytest.raises(Exception):
            filt.process(sig, dtype="not_a_real_dtype")
