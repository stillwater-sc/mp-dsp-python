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

    def test_zeros_count_matches_order(self):
        # Butterworth LP is all-pole in the analog prototype; after bilinear
        # transform every zero maps to z = -1. The binding still reports all
        # N of them (as an N-fold cluster), matching the number of poles.
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        zeros = filt.zeros()
        assert len(zeros) == 4
        # All should be at or near z = -1 for Butterworth LP. Tolerance is
        # looser than bit-exact: Apple Silicon's Clang generates different
        # FMA sequences than x86_64 GCC/Clang, so the bilinear-transform
        # output can differ by ~1e-8 between platforms. 1e-6 is comfortably
        # above that platform noise while still pinning "clustered at -1".
        for z in zeros:
            assert abs(z - (-1 + 0j)) < 1e-6

    def test_zeros_chebyshev2_on_unit_circle(self):
        # Chebyshev II has finite stopband zeros distributed on the unit
        # circle — the canonical case where zeros are visually informative
        # (unlike Butterworth's pile-at-minus-one). Guards against a future
        # regression that throws the zeros away again.
        filt = mpdsp.chebyshev2_lowpass(order=6, sample_rate=SAMPLE_RATE,
                                         cutoff=1000.0, stopband_db=40.0)
        zeros = filt.zeros()
        # At least some zeros should be strictly off z = -1 and on the unit
        # circle (|z| ≈ 1, not clustered at the real axis).
        off_axis = [z for z in zeros
                    if abs(abs(z) - 1.0) < 1e-6 and abs(z.imag) > 1e-3]
        assert len(off_axis) > 0

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
    def test_process_shape_preserved(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = _sine(200.0)
        y = filt.process(sig)
        assert y.shape == sig.shape
        assert y.dtype == np.float64

    def test_lowpass_passes_low_rejects_high(self):
        filt = mpdsp.butterworth_lowpass(order=6, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        low = _sine(200.0)
        high = _sine(3000.0)
        # Skip transient so steady-state amplitude is meaningful.
        skip = 512
        y_low = filt.process(low)[skip:]
        y_high = filt.process(high)[skip:]
        # Passband amplitude close to 1; stopband amplitude heavily attenuated.
        assert np.max(np.abs(y_low)) > 0.9
        assert np.max(np.abs(y_high)) < 0.1

    def test_dtype_reference_matches_double(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = _sine(300.0)
        a = filt.process(sig, dtype="reference")
        b = filt.process(sig, dtype="double")
        np.testing.assert_array_equal(a, b)

    def test_dtype_dispatch_differs_from_reference(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = _sine(300.0)
        ref = filt.process(sig, dtype="reference")
        low = filt.process(sig, dtype="half")
        # Reduced-precision arithmetic should introduce measurable error.
        assert not np.array_equal(ref, low)
        # But the output should still be close — within ~1% of reference.
        err = np.max(np.abs(ref - low)) / (np.max(np.abs(ref)) + 1e-12)
        assert err < 0.05

    def test_unknown_dtype_raises(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = _sine(300.0)
        with pytest.raises(Exception):
            filt.process(sig, dtype="not_a_real_dtype")

    def test_sensor_8bit_output_stays_in_signal_range(self):
        # Sensor dtypes route samples through integer<N> with the
        # scale-quantize-unscale pipeline, so output of a unit-amplitude
        # sine must stay in roughly [-1, 1] — not collapse to zero (the
        # naive-cast failure mode) and not explode into integer units.
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = _sine(300.0)  # amplitude 1.0
        y = filt.process(sig, dtype="sensor_8bit")
        assert np.max(np.abs(y)) > 0.1   # not annihilated
        assert np.max(np.abs(y)) < 2.0   # not in integer units (would be ~127)
        # 8-bit quantization should noticeably differ from reference.
        ref = filt.process(sig, dtype="reference")
        assert not np.array_equal(y, ref)


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

    def test_non_positive_q_raises(self):
        with pytest.raises(ValueError):
            mpdsp.rbj_lowpass(sample_rate=SAMPLE_RATE, cutoff=1000.0, q=0.0)

    def test_non_positive_bandwidth_raises(self):
        with pytest.raises(ValueError):
            mpdsp.rbj_bandpass(sample_rate=SAMPLE_RATE, center_freq=1000.0,
                               bandwidth=-1.0)

    def test_non_positive_slope_raises(self):
        with pytest.raises(ValueError):
            mpdsp.rbj_lowshelf(sample_rate=SAMPLE_RATE, cutoff=1000.0,
                               gain_db=6.0, slope=0.0)


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


# ---------------------------------------------------------------------------
# Posit dtypes — now enabled for process().
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["posit_full", "tiny_posit"])
def test_iir_posit_process(dtype):
    filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
    sig = _sine(300.0)
    ref = filt.process(sig, dtype="reference")
    out = filt.process(sig, dtype=dtype)
    assert out.shape == ref.shape
    # Posit arithmetic should produce a recognizably similar result to reference.
    # posit<8,2> (tiny_posit) is quite lossy — allow a generous bound.
    err = np.max(np.abs(ref - out)) / (np.max(np.abs(ref)) + 1e-12)
    assert err < 0.5, f"{dtype} relative error too large: {err}"


# ---------------------------------------------------------------------------
# Extended diagnostics on IIRFilter.
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_stability_margin_stable_filter(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        # Butterworth is always stable; margin = 1 - max|pole| > 0
        assert filt.stability_margin() > 0.0

    def test_condition_number_finite(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        cn = filt.condition_number(num_freqs=64)
        assert np.isfinite(cn)
        assert cn > 0.0

    def test_worst_case_sensitivity_finite(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        s = filt.worst_case_sensitivity()
        assert np.isfinite(s)
        assert s >= 0.0

    def test_pole_displacement_reference_is_zero(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        assert filt.pole_displacement("reference") == 0.0

    def test_pole_displacement_half_positive(self):
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        # Half precision stores fewer coefficient bits, so poles must move.
        d = filt.pole_displacement("half")
        assert d > 0.0
        # But should still be small (filter is not marginally stable).
        assert d < 0.01

    def test_higher_order_filter_has_larger_condition(self):
        # Higher-order cascades are more sensitive to coefficient perturbation.
        low = mpdsp.butterworth_lowpass(order=2, sample_rate=SAMPLE_RATE, cutoff=500.0)
        high = mpdsp.butterworth_lowpass(order=10, sample_rate=SAMPLE_RATE, cutoff=500.0)
        assert high.condition_number(num_freqs=64) > low.condition_number(num_freqs=64)


# ---------------------------------------------------------------------------
# FIR filters.
# ---------------------------------------------------------------------------


class TestFIRDesign:
    def test_lowpass_design_shape(self):
        f = mpdsp.fir_lowpass(num_taps=51, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        assert isinstance(f, mpdsp.FIRFilter)
        assert f.num_taps() == 51
        c = f.coefficients()
        assert c.shape == (51,)
        assert c.dtype == np.float64

    def test_highpass_dc_rejects(self):
        f = mpdsp.fir_highpass(num_taps=51, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        h = f.frequency_response(np.array([0.0, 0.45]))
        assert np.abs(h[0]) < 0.05
        assert np.abs(h[1]) > 0.5

    def test_lowpass_passes_dc_rejects_high(self):
        f = mpdsp.fir_lowpass(num_taps=51, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        h = f.frequency_response(np.array([0.0, 0.45]))
        assert abs(np.abs(h[0]) - 1.0) < 0.05
        assert np.abs(h[1]) < 0.05

    def test_bandpass_peaks_between_bounds(self):
        f = mpdsp.fir_bandpass(num_taps=101, sample_rate=SAMPLE_RATE,
                                f_low=800.0, f_high=1600.0)
        freqs = np.array([100.0, 1200.0, 3500.0]) / SAMPLE_RATE
        mag = np.abs(f.frequency_response(freqs))
        assert mag[1] > mag[0]
        assert mag[1] > mag[2]

    def test_bandstop_notches_center(self):
        f = mpdsp.fir_bandstop(num_taps=101, sample_rate=SAMPLE_RATE,
                                f_low=800.0, f_high=1600.0)
        freqs = np.array([100.0, 1200.0, 3500.0]) / SAMPLE_RATE
        mag = np.abs(f.frequency_response(freqs))
        assert mag[1] < mag[0]
        assert mag[1] < mag[2]

    def test_fir_filter_from_explicit_coefficients(self):
        # Unity-impulse taps => identity filter.
        taps = np.zeros(11, dtype=np.float64)
        taps[0] = 1.0
        f = mpdsp.fir_filter(taps)
        assert f.num_taps() == 11
        sig = _sine(300.0)
        y = f.process(sig)
        # First sample in == first sample out (no delay for zero-position tap).
        assert abs(y[0] - sig[0]) < 1e-12

    def test_impulse_response_matches_taps(self):
        f = mpdsp.fir_lowpass(num_taps=21, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        ir = f.impulse_response(21)
        np.testing.assert_allclose(ir, f.coefficients())

    def test_impulse_response_padded_with_zeros(self):
        f = mpdsp.fir_lowpass(num_taps=21, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        ir = f.impulse_response(30)
        assert ir.shape == (30,)
        np.testing.assert_allclose(ir[:21], f.coefficients())
        np.testing.assert_array_equal(ir[21:], 0.0)

    def test_windows_accepts_kaiser(self):
        # Kaiser requires a beta parameter; the binding accepts it as
        # kaiser_beta and passes it through to the window.
        f = mpdsp.fir_lowpass(num_taps=51, sample_rate=SAMPLE_RATE,
                              cutoff=1000.0, window="kaiser", kaiser_beta=6.0)
        h = f.frequency_response(np.array([0.0]))
        assert abs(np.abs(h[0]) - 1.0) < 0.05

    def test_windows_rejects_unknown(self):
        with pytest.raises(ValueError):
            mpdsp.fir_lowpass(num_taps=51, sample_rate=SAMPLE_RATE,
                              cutoff=1000.0, window="not_a_window")

    def test_invalid_num_taps_raises(self):
        with pytest.raises(ValueError):
            mpdsp.fir_lowpass(num_taps=0, sample_rate=SAMPLE_RATE, cutoff=1000.0)

    def test_bandpass_requires_ordered_frequencies(self):
        with pytest.raises(ValueError):
            mpdsp.fir_bandpass(num_taps=51, sample_rate=SAMPLE_RATE,
                                f_low=1600.0, f_high=800.0)


class TestFIRDesignCoeffDtype:
    """The four window-method FIR designers accept coeff_dtype= to run
    the tap-design pipeline at a chosen precision (per upstream PR #117
    and Phase 2 / #85). The default 'reference' must be bitwise
    identical to omitting the kwarg; alternative dtypes return finite,
    real-valued taps that approximate the reference design.
    """

    def test_default_matches_no_kwarg_lowpass(self):
        f1 = mpdsp.fir_lowpass(num_taps=51, sample_rate=SAMPLE_RATE,
                               cutoff=1000.0)
        f2 = mpdsp.fir_lowpass(num_taps=51, sample_rate=SAMPLE_RATE,
                               cutoff=1000.0, coeff_dtype="reference")
        np.testing.assert_array_equal(f1.coefficients(), f2.coefficients())

    def test_lowpass_posit32_close_to_reference(self):
        ref = mpdsp.fir_lowpass(num_taps=51, sample_rate=SAMPLE_RATE,
                                cutoff=1000.0, coeff_dtype="reference")
        p32 = mpdsp.fir_lowpass(num_taps=51, sample_rate=SAMPLE_RATE,
                                cutoff=1000.0, coeff_dtype="posit_full")
        ref_taps = ref.coefficients()
        p32_taps = p32.coefficients()
        assert p32_taps.shape == ref_taps.shape
        assert np.all(np.isfinite(p32_taps))
        # posit32 has ~30 fraction bits in the dynamic-range sweet spot;
        # a 51-tap sinc-windowed lowpass should reproduce to << 1e-3.
        np.testing.assert_allclose(p32_taps, ref_taps, atol=1e-4)

    def test_highpass_dtype_propagates(self):
        f = mpdsp.fir_highpass(num_taps=51, sample_rate=SAMPLE_RATE,
                               cutoff=1000.0, coeff_dtype="posit_full")
        assert f.coefficients().shape == (51,)
        assert np.all(np.isfinite(f.coefficients()))

    def test_bandpass_dtype_propagates(self):
        f = mpdsp.fir_bandpass(num_taps=101, sample_rate=SAMPLE_RATE,
                               f_low=800.0, f_high=1600.0,
                               coeff_dtype="posit_full")
        assert f.coefficients().shape == (101,)
        assert np.all(np.isfinite(f.coefficients()))

    def test_bandstop_dtype_propagates(self):
        f = mpdsp.fir_bandstop(num_taps=101, sample_rate=SAMPLE_RATE,
                               f_low=800.0, f_high=1600.0,
                               coeff_dtype="posit_full")
        assert f.coefficients().shape == (101,)
        assert np.all(np.isfinite(f.coefficients()))

    def test_unknown_coeff_dtype_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.fir_lowpass(num_taps=21, sample_rate=SAMPLE_RATE,
                              cutoff=1000.0, coeff_dtype="not_a_dtype")


class TestFIRProcessing:
    def test_process_signal_shape(self):
        f = mpdsp.fir_lowpass(num_taps=51, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = _sine(200.0)
        y = f.process(sig)
        assert y.shape == sig.shape
        assert y.dtype == np.float64

    def test_lowpass_rejects_high_freq(self):
        f = mpdsp.fir_lowpass(num_taps=101, sample_rate=SAMPLE_RATE, cutoff=500.0)
        low = _sine(100.0)
        high = _sine(3000.0)
        # Skip group-delay transient
        skip = 200
        y_low = f.process(low)[skip:]
        y_high = f.process(high)[skip:]
        assert np.max(np.abs(y_low)) > 0.8
        assert np.max(np.abs(y_high)) < 0.1

    @pytest.mark.parametrize("dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_process_dispatch_works(self, dtype):
        f = mpdsp.fir_lowpass(num_taps=51, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = _sine(300.0)
        ref = f.process(sig, dtype="reference")
        out = f.process(sig, dtype=dtype)
        assert out.shape == ref.shape
        # Reduced precision should differ from reference but still be close.
        err = np.max(np.abs(ref - out)) / (np.max(np.abs(ref)) + 1e-12)
        assert err < 0.5


# ---------------------------------------------------------------------------
# Python helpers in mpdsp.filters.
# ---------------------------------------------------------------------------


class TestFilterHelpers:
    def test_compare_filters_runs_across_dtypes(self):
        from mpdsp.filters import compare_filters
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = _sine(300.0)
        df = compare_filters(filt, sig, dtypes=["reference", "gpu_baseline", "half"])
        # Accept either pandas.DataFrame or list-of-dicts depending on pandas
        # availability.
        rows = df.to_dict("records") if hasattr(df, "to_dict") else df
        by_dtype = {r["dtype"]: r for r in rows}
        assert set(by_dtype.keys()) == {"reference", "gpu_baseline", "half"}
        # Reference compared to itself: SQNR is capped high, error is 0.
        ref_row = by_dtype["reference"]
        assert ref_row["max_abs_error"] == 0.0
        # Reduced dtypes should have finite SQNR and positive error.
        assert np.isfinite(by_dtype["half"]["sqnr_db"])
        assert by_dtype["half"]["max_abs_error"] > 0.0

    def test_compare_filters_on_fir(self):
        from mpdsp.filters import compare_filters
        f = mpdsp.fir_lowpass(num_taps=51, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        sig = _sine(300.0)
        df = compare_filters(f, sig, dtypes=["reference", "gpu_baseline"])
        rows = df.to_dict("records") if hasattr(df, "to_dict") else df
        assert {r["dtype"] for r in rows} == {"reference", "gpu_baseline"}

    def test_plot_filter_comparison_returns_figure(self):
        pytest.importorskip("matplotlib")
        from mpdsp.filters import plot_filter_comparison
        import matplotlib
        matplotlib.use("Agg")  # headless
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        fig = plot_filter_comparison(filt, num_freqs=64,
                                     sample_rate=SAMPLE_RATE)
        # Three subplots for IIR (magnitude, phase, pole-zero)
        assert len(fig.axes) == 3
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_filter_comparison_fir_has_two_axes(self):
        pytest.importorskip("matplotlib")
        from mpdsp.filters import plot_filter_comparison
        import matplotlib
        matplotlib.use("Agg")
        f = mpdsp.fir_lowpass(num_taps=31, sample_rate=SAMPLE_RATE, cutoff=1000.0)
        fig = plot_filter_comparison(f, num_freqs=64, sample_rate=SAMPLE_RATE)
        # FIR: magnitude + phase (no pole-zero)
        assert len(fig.axes) == 2
        import matplotlib.pyplot as plt
        plt.close(fig)
