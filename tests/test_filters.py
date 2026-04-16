"""Tests for IIR filter bindings across all analog-prototype families and RBJ."""

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


# ---------------------------------------------------------------------------
# Helpers reused across families.
# ---------------------------------------------------------------------------


def _sine(freq, n=4096):
    t = np.arange(n) / SAMPLE_RATE
    return np.sin(2.0 * np.pi * freq * t)


def _assert_stable(filt):
    for p in filt.poles():
        assert abs(p) < 1.0, f"unstable pole {p}"


def _assert_lp_behavior(filt, cutoff_hz):
    freqs = np.array([0.0, cutoff_hz / SAMPLE_RATE, 0.4])
    mag = np.abs(filt.frequency_response(freqs))
    assert abs(mag[0] - 1.0) < 0.2  # DC passes (shelf/ripple may slightly offset)
    assert mag[2] < 0.2              # stopband attenuated


def _assert_hp_behavior(filt, cutoff_hz):
    freqs = np.array([0.0, cutoff_hz / SAMPLE_RATE, 0.4])
    mag = np.abs(filt.frequency_response(freqs))
    assert mag[0] < 0.2              # DC blocked
    assert abs(mag[2] - 1.0) < 0.2   # passband passes


def _assert_bp_behavior(filt, center_hz):
    f_center = center_hz / SAMPLE_RATE
    freqs = np.array([0.01, f_center, 0.48])
    mag = np.abs(filt.frequency_response(freqs))
    # center passes more strongly than far edges (DC / Nyquist)
    assert mag[1] > mag[0]
    assert mag[1] > mag[2]


def _assert_bs_behavior(filt, center_hz):
    f_center = center_hz / SAMPLE_RATE
    # use a small offset to avoid landing exactly on Nyquist when center is high
    freqs = np.array([0.01, f_center, 0.48])
    mag = np.abs(filt.frequency_response(freqs))
    # center attenuates more than the passband shoulders
    assert mag[1] < mag[0]
    assert mag[1] < mag[2]


# ---------------------------------------------------------------------------
# Analog-prototype families — Chebyshev I/II, Bessel, Legendre LP/HP/BP/BS.
# Covered parametrically to keep the test file tight; each case exercises
# design validation, stability, and expected frequency-domain behavior.
# ---------------------------------------------------------------------------


LP_HP_FAMILIES = [
    ("butterworth", {},
        lambda kind, **kw: getattr(mpdsp, f"butterworth_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0, **kw)),
    ("chebyshev1", {"ripple_db": 1.0},
        lambda kind, **kw: getattr(mpdsp, f"chebyshev1_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0, **kw)),
    ("chebyshev2", {"stopband_db": 40.0},
        lambda kind, **kw: getattr(mpdsp, f"chebyshev2_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0, **kw)),
    ("bessel", {},
        lambda kind, **kw: getattr(mpdsp, f"bessel_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0, **kw)),
    ("legendre", {},
        lambda kind, **kw: getattr(mpdsp, f"legendre_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0, **kw)),
    ("elliptic", {"ripple_db": 1.0, "rolloff": 1.0},
        lambda kind, **kw: getattr(mpdsp, f"elliptic_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0, **kw)),
]


@pytest.mark.parametrize("family,extra_kwargs,designer", LP_HP_FAMILIES,
                         ids=[f[0] for f in LP_HP_FAMILIES])
class TestLpHpFamilies:
    def test_lowpass_design(self, family, extra_kwargs, designer):
        filt = designer("lowpass", **extra_kwargs)
        assert filt.num_stages() == 2
        _assert_stable(filt)
        _assert_lp_behavior(filt, 1000.0)

    def test_highpass_design(self, family, extra_kwargs, designer):
        filt = designer("highpass", **extra_kwargs)
        assert filt.num_stages() == 2
        _assert_stable(filt)
        _assert_hp_behavior(filt, 1000.0)

    def test_lowpass_processes_signal(self, family, extra_kwargs, designer):
        filt = designer("lowpass", **extra_kwargs)
        low, high = _sine(200.0), _sine(3000.0)
        y_low = filt.process(low)[512:]
        y_high = filt.process(high)[512:]
        assert np.max(np.abs(y_low)) > 0.5
        assert np.max(np.abs(y_high)) < np.max(np.abs(y_low))


BP_BS_FAMILIES = [
    ("butterworth", {},
        lambda kind, **kw: getattr(mpdsp, f"butterworth_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, center_freq=1500.0, width_freq=800.0, **kw)),
    ("chebyshev1", {"ripple_db": 1.0},
        lambda kind, **kw: getattr(mpdsp, f"chebyshev1_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, center_freq=1500.0, width_freq=800.0, **kw)),
    ("chebyshev2", {"stopband_db": 40.0},
        lambda kind, **kw: getattr(mpdsp, f"chebyshev2_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, center_freq=1500.0, width_freq=800.0, **kw)),
    ("bessel", {},
        lambda kind, **kw: getattr(mpdsp, f"bessel_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, center_freq=1500.0, width_freq=800.0, **kw)),
    ("legendre", {},
        lambda kind, **kw: getattr(mpdsp, f"legendre_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, center_freq=1500.0, width_freq=800.0, **kw)),
    ("elliptic", {"ripple_db": 1.0, "rolloff": 1.0},
        lambda kind, **kw: getattr(mpdsp, f"elliptic_{kind}")(
            order=4, sample_rate=SAMPLE_RATE, center_freq=1500.0, width_freq=800.0, **kw)),
]


@pytest.mark.parametrize("family,extra_kwargs,designer", BP_BS_FAMILIES,
                         ids=[f[0] for f in BP_BS_FAMILIES])
class TestBpBsFamilies:
    def test_bandpass_design(self, family, extra_kwargs, designer):
        filt = designer("bandpass", **extra_kwargs)
        # Bandpass transform doubles the internal order: user order 4 -> 4 biquads
        assert filt.num_stages() == 4
        _assert_stable(filt)
        _assert_bp_behavior(filt, 1500.0)

    def test_bandstop_design(self, family, extra_kwargs, designer):
        filt = designer("bandstop", **extra_kwargs)
        assert filt.num_stages() == 4
        _assert_stable(filt)
        _assert_bs_behavior(filt, 1500.0)


# ---------------------------------------------------------------------------
# Parameter validation — per-family invariants that aren't covered above.
# ---------------------------------------------------------------------------


class TestParameterValidation:
    def test_chebyshev1_requires_positive_ripple(self):
        with pytest.raises(ValueError):
            mpdsp.chebyshev1_lowpass(order=4, sample_rate=SAMPLE_RATE,
                                     cutoff=1000.0, ripple_db=0.0)

    def test_chebyshev2_requires_positive_stopband(self):
        with pytest.raises(ValueError):
            mpdsp.chebyshev2_lowpass(order=4, sample_rate=SAMPLE_RATE,
                                     cutoff=1000.0, stopband_db=0.0)

    def test_bandpass_width_outside_range_raises(self):
        # width too large -> passband exceeds Nyquist
        with pytest.raises(ValueError):
            mpdsp.butterworth_bandpass(order=4, sample_rate=SAMPLE_RATE,
                                       center_freq=1000.0, width_freq=5000.0)

    def test_bandpass_order_above_bpbs_limit_raises(self):
        # BP/BS caps at 8 (kMaxOrderBPBS), lower than LP/HP's 16.
        with pytest.raises(ValueError):
            mpdsp.butterworth_bandpass(order=9, sample_rate=SAMPLE_RATE,
                                       center_freq=1500.0, width_freq=800.0)

    def test_elliptic_rolloff_out_of_range_raises(self):
        # Upstream validates rolloff is in [0.1, 5.0] — it's a selectivity
        # parameter, not a stopband dB. Values outside the range used to
        # produce NaN coefficients; they now raise.
        with pytest.raises(ValueError):
            mpdsp.elliptic_lowpass(order=4, sample_rate=SAMPLE_RATE,
                                   cutoff=1000.0, ripple_db=1.0, rolloff=0.0)
        with pytest.raises(ValueError):
            mpdsp.elliptic_lowpass(order=4, sample_rate=SAMPLE_RATE,
                                   cutoff=1000.0, ripple_db=1.0, rolloff=10.0)

    def test_elliptic_requires_positive_ripple(self):
        with pytest.raises(ValueError):
            mpdsp.elliptic_lowpass(order=4, sample_rate=SAMPLE_RATE,
                                   cutoff=1000.0, ripple_db=0.0, rolloff=1.0)


# ---------------------------------------------------------------------------
# RBJ — single-biquad cookbook designs, different signature (no 'order').
# ---------------------------------------------------------------------------


class TestRBJ:
    def test_lowpass_is_single_biquad(self):
        filt = mpdsp.rbj_lowpass(sample_rate=SAMPLE_RATE, cutoff=1000.0)
        assert filt.num_stages() == 1
        _assert_stable(filt)
        _assert_lp_behavior(filt, 1000.0)

    def test_highpass(self):
        filt = mpdsp.rbj_highpass(sample_rate=SAMPLE_RATE, cutoff=1000.0)
        assert filt.num_stages() == 1
        _assert_stable(filt)
        _assert_hp_behavior(filt, 1000.0)

    def test_bandpass_peaks_at_center(self):
        filt = mpdsp.rbj_bandpass(sample_rate=SAMPLE_RATE, center_freq=1000.0)
        assert filt.num_stages() == 1
        _assert_bp_behavior(filt, 1000.0)

    def test_bandstop_notches_at_center(self):
        filt = mpdsp.rbj_bandstop(sample_rate=SAMPLE_RATE, center_freq=1000.0)
        _assert_bs_behavior(filt, 1000.0)

    def test_allpass_magnitude_is_unity(self):
        filt = mpdsp.rbj_allpass(sample_rate=SAMPLE_RATE, center_freq=1000.0)
        freqs = np.linspace(0.001, 0.499, 32)
        mag = np.abs(filt.frequency_response(freqs))
        # All-pass: unit magnitude everywhere.
        assert np.allclose(mag, 1.0, atol=1e-6)

    def test_lowshelf_gain_matches_request(self):
        filt = mpdsp.rbj_lowshelf(sample_rate=SAMPLE_RATE, cutoff=500.0,
                                  gain_db=6.0)
        mag = np.abs(filt.frequency_response(np.array([0.0, 0.49])))
        # DC should be boosted by ~6 dB = factor 2.0; Nyquist ~ 0 dB (unity).
        assert mag[0] > 1.5
        assert abs(mag[1] - 1.0) < 0.2

    def test_highshelf_gain_matches_request(self):
        filt = mpdsp.rbj_highshelf(sample_rate=SAMPLE_RATE, cutoff=2000.0,
                                   gain_db=6.0)
        mag = np.abs(filt.frequency_response(np.array([0.0, 0.49])))
        assert abs(mag[0] - 1.0) < 0.2
        assert mag[1] > 1.5

    def test_processes_signal(self):
        filt = mpdsp.rbj_lowpass(sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = _sine(200.0)
        y = filt.process(sig)
        assert y.shape == sig.shape
        assert np.max(np.abs(y[512:])) > 0.5  # passband preserved

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError):
            mpdsp.rbj_lowpass(sample_rate=-1.0, cutoff=1000.0)


# ---------------------------------------------------------------------------
# Cross-family smoke tests — frequency_response and process work the same
# way on every kind of filter exposed by this module.
# ---------------------------------------------------------------------------


ALL_FILTERS = [
    lambda: mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0),
    lambda: mpdsp.chebyshev1_highpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0, ripple_db=0.5),
    lambda: mpdsp.chebyshev2_bandpass(order=4, sample_rate=SAMPLE_RATE,
                                       center_freq=1500.0, width_freq=800.0, stopband_db=40.0),
    lambda: mpdsp.bessel_bandstop(order=4, sample_rate=SAMPLE_RATE,
                                   center_freq=1500.0, width_freq=800.0),
    lambda: mpdsp.legendre_lowpass(order=3, sample_rate=SAMPLE_RATE, cutoff=1500.0),
    lambda: mpdsp.elliptic_highpass(order=4, sample_rate=SAMPLE_RATE,
                                     cutoff=1000.0, ripple_db=1.0, rolloff=1.0),
    lambda: mpdsp.rbj_allpass(sample_rate=SAMPLE_RATE, center_freq=1000.0),
]


@pytest.mark.parametrize("make", ALL_FILTERS)
def test_filter_common_api(make):
    filt = make()
    assert isinstance(filt, mpdsp.IIRFilter)
    assert filt.num_stages() >= 1

    # frequency_response returns complex128 with correct shape
    freqs = np.linspace(0.0, 0.5, 16)
    h = filt.frequency_response(freqs)
    assert h.shape == freqs.shape
    assert h.dtype == np.complex128

    # process returns float64 with same shape as input
    sig = _sine(300.0, n=1024)
    y = filt.process(sig)
    assert y.shape == sig.shape
    assert y.dtype == np.float64
