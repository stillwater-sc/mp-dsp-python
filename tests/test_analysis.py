"""Tests for the pure-Python analysis helpers in ``mpdsp.analysis``.

The helpers are thin layers over well-known biquad math and the
already-bound ``IIRFilter`` methods, so the tests focus on:

  1. Closed-form cases where the analytical answer is obvious.
  2. Consistency with ``IIRFilter.poles()`` — callers should be able
     to compare ``biquad_poles(*coeffs)`` against ``filt.poles()`` and
     agree to numerical tolerance.
  3. The safety-margin semantics of ``is_stable(..., tol=...)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import mpdsp


# ---- biquad_poles ------------------------------------------------------


def test_biquad_poles_returns_two_complex():
    poles = mpdsp.biquad_poles(1.0, 0.0, 0.0, -0.5, 0.0)
    assert len(poles) == 2
    assert all(isinstance(p, complex) for p in poles)


def test_biquad_poles_trivial_zero_denominator():
    # z² + 0 z + 0 = z² → double root at origin.
    poles = mpdsp.biquad_poles(1.0, 0.0, 0.0, 0.0, 0.0)
    assert all(abs(p) < 1e-12 for p in poles)


def test_biquad_poles_known_quadratic():
    # (z - 0.5)(z - 0.5) = z² - z + 0.25, so a1 = -1, a2 = 0.25.
    # Double roots are a numerical edge case: numpy.roots uses a companion-
    # matrix eigensolve, and at a repeated root the conditioning is
    # O(sqrt(eps)) ≈ 1.5e-8 regardless of LAPACK backend. Apple's
    # Accelerate backend reliably lands near 5e-9 here; MKL tends to
    # do a touch better. Tolerance sized to the theoretical floor plus
    # headroom rather than one specific backend.
    poles = mpdsp.biquad_poles(1.0, 0.0, 0.0, -1.0, 0.25)
    for p in poles:
        assert p.real == pytest.approx(0.5, abs=1e-6)
        assert p.imag == pytest.approx(0.0, abs=1e-6)


def test_biquad_poles_complex_conjugate_pair():
    # Choose a1, a2 giving a complex conjugate pair inside the unit
    # circle. z² + a1 z + a2 with discriminant < 0.
    a1, a2 = -0.8, 0.9
    poles = mpdsp.biquad_poles(0.0, 0.0, 0.0, a1, a2)
    # Vieta: sum of roots = -a1, product = a2.
    assert sum(p for p in poles).real == pytest.approx(-a1, abs=1e-12)
    assert (poles[0] * poles[1]).real == pytest.approx(a2, abs=1e-12)
    # Conjugate pair: one positive imag, one negative, equal magnitudes.
    assert poles[0] == pytest.approx(poles[1].conjugate(), abs=1e-12)


def test_biquad_poles_numerator_coefficients_ignored():
    # Poles depend only on the denominator; numerator should be inert.
    a1, a2 = -0.5, 0.3
    p1 = mpdsp.biquad_poles(1.0, 2.0, 3.0, a1, a2)
    p2 = mpdsp.biquad_poles(0.0, 0.0, 0.0, a1, a2)
    for r1, r2 in zip(p1, p2):
        assert r1 == pytest.approx(r2, abs=1e-12)


# ---- max_pole_radius / is_stable ---------------------------------------


def test_max_pole_radius_designed_filter_is_stable():
    filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100.0,
                                      cutoff=1000.0)
    r = mpdsp.max_pole_radius(filt)
    assert 0.0 < r < 1.0
    # Must agree with the direct computation from IIRFilter.poles().
    assert r == pytest.approx(max(abs(p) for p in filt.poles()), abs=1e-12)


def test_is_stable_true_for_designed_filter():
    filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100.0,
                                      cutoff=1000.0)
    assert mpdsp.is_stable(filt) is True


def test_is_stable_with_tolerance_rejects_near_unit_circle_poles():
    # High-order, low-cutoff Butterworth clusters poles near the unit
    # circle — small margin. A zero-tolerance check passes, a stringent
    # tolerance (larger than the actual margin) fails.
    filt = mpdsp.butterworth_lowpass(order=8, sample_rate=44100.0,
                                      cutoff=50.0)
    r = mpdsp.max_pole_radius(filt)
    actual_margin = 1.0 - r
    assert mpdsp.is_stable(filt) is True
    assert mpdsp.is_stable(filt, tol=actual_margin * 0.5) is True
    assert mpdsp.is_stable(filt, tol=actual_margin * 1.5) is False


def test_biquad_poles_agrees_with_iirfilter_poles():
    """Consistency check: concatenating per-stage biquad_poles should
    equal IIRFilter.poles() up to ordering.
    """
    filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100.0,
                                      cutoff=2000.0)
    stage_coeffs = filt.coefficients()
    manual = []
    for (b0, b1, b2, a1, a2) in stage_coeffs:
        manual.extend(mpdsp.biquad_poles(b0, b1, b2, a1, a2))
    # Sort by (real, imag) for deterministic comparison.
    manual_sorted = sorted(manual, key=lambda p: (p.real, p.imag))
    iir_sorted = sorted(filt.poles(), key=lambda p: (p.real, p.imag))
    assert len(manual_sorted) == len(iir_sorted)
    # Two independent eigensolves (numpy.roots via companion matrix in
    # Python, upstream sw::dsp in C++) may use different LAPACK backends
    # across platforms. 1e-8 leaves headroom for that without being
    # sloppy about the math.
    for a, b in zip(manual_sorted, iir_sorted):
        assert a == pytest.approx(b, abs=1e-8)


# ---- coefficient_sensitivity (C++ binding) -----------------------------


class TestCoefficientSensitivity:
    """Free-function `mpdsp.coefficient_sensitivity(b0, b1, b2, a1, a2)`
    returns the finite-difference partial derivatives of the maximum
    pole radius w.r.t. each denominator coefficient."""

    def test_returns_two_doubles(self):
        out = mpdsp.coefficient_sensitivity(1.0, 0.0, 0.0, -0.5, 0.1)
        assert len(out) == 2
        assert all(isinstance(v, float) for v in out)

    def test_numerator_coefficients_inert(self):
        # Sensitivity is a property of the denominator only. Changing b*
        # must not change the result.
        sens_a = mpdsp.coefficient_sensitivity(1.0, 2.0, 3.0, -0.5, 0.1)
        sens_b = mpdsp.coefficient_sensitivity(0.0, 0.0, 0.0, -0.5, 0.1)
        assert sens_a[0] == pytest.approx(sens_b[0], abs=1e-12)
        assert sens_a[1] == pytest.approx(sens_b[1], abs=1e-12)

    def test_complex_conjugate_a2_sensitivity_matches_analytic(self):
        # For a biquad z^2 + a1*z + a2 with complex-conjugate poles,
        # |p| = sqrt(a2). So d|p|/da2 = 1/(2*sqrt(a2)) = 1/(2r)
        # where r is the pole radius. Pin the relationship against the
        # analytic formula at three pole radii.
        import math
        for r in (0.2, 0.5, 0.8):
            theta = 0.6  # any theta giving complex poles
            a1 = -2.0 * r * math.cos(theta)
            a2 = r * r
            s = mpdsp.coefficient_sensitivity(1.0, 0.0, 0.0, a1, a2)
            expected_dp_da2 = 1.0 / (2.0 * r)
            # 1% relative tolerance — finite differences at epsilon=1e-8
            # on max_pole_radius are noisy at this level.
            assert s[1] == pytest.approx(expected_dp_da2, rel=1e-2)

    def test_complex_conjugate_a1_sensitivity_near_zero(self):
        # Still for complex-conjugate poles: |p| depends only on a2, so
        # d|p|/da1 should be ~0.
        import math
        for r in (0.2, 0.5, 0.8):
            theta = 0.6
            a1 = -2.0 * r * math.cos(theta)
            a2 = r * r
            s = mpdsp.coefficient_sensitivity(1.0, 0.0, 0.0, a1, a2)
            assert abs(s[0]) < 1e-6


# ---- biquad_condition_number (C++ binding) -----------------------------


class TestBiquadConditionNumber:
    """Free-function `mpdsp.biquad_condition_number(b0, b1, b2, a1, a2)`
    measures frequency-response sensitivity to coefficient perturbation."""

    def test_returns_finite_positive_float(self):
        cn = mpdsp.biquad_condition_number(1.0, 0.0, 0.0, -0.5, 0.1)
        assert isinstance(cn, float)
        assert cn > 0.0
        import math
        assert math.isfinite(cn)

    def test_grows_with_pole_radius(self):
        # High-Q biquads are numerically fragile: small coefficient
        # errors produce large response changes.
        def coeffs(r, theta=0.6):
            import math
            return (-2.0 * r * math.cos(theta), r * r)

        a1_low, a2_low = coeffs(0.10)
        a1_high, a2_high = coeffs(0.99)
        cn_low = mpdsp.biquad_condition_number(1.0, 0.0, 0.0, a1_low, a2_low)
        cn_high = mpdsp.biquad_condition_number(1.0, 0.0, 0.0, a1_high, a2_high)
        assert cn_high > cn_low

    def test_num_freqs_validated(self):
        with pytest.raises(ValueError):
            mpdsp.biquad_condition_number(1.0, 0.0, 0.0, -0.5, 0.1,
                                           num_freqs=0)
        with pytest.raises(ValueError):
            mpdsp.biquad_condition_number(1.0, 0.0, 0.0, -0.5, 0.1,
                                           num_freqs=-1)

    def test_default_num_freqs_is_512(self):
        # Issue #53 contract pins the default at 512. Agreement between
        # default-arg call and explicit num_freqs=512 catches a silent
        # default-value regression.
        default = mpdsp.biquad_condition_number(1.0, 0.0, 0.0, -0.5, 0.1)
        explicit = mpdsp.biquad_condition_number(1.0, 0.0, 0.0, -0.5, 0.1,
                                                   num_freqs=512)
        assert default == pytest.approx(explicit, rel=0.0, abs=0.0)


class TestCoefficientSensitivityEpsilonValidation:
    """Guard at the binding boundary: epsilon must be finite and > 0."""

    def test_rejects_zero_epsilon(self):
        with pytest.raises(ValueError):
            mpdsp.coefficient_sensitivity(1.0, 0.0, 0.0, -0.5, 0.1,
                                            epsilon=0.0)

    def test_rejects_negative_epsilon(self):
        with pytest.raises(ValueError):
            mpdsp.coefficient_sensitivity(1.0, 0.0, 0.0, -0.5, 0.1,
                                            epsilon=-1e-6)

    def test_rejects_nan_epsilon(self):
        with pytest.raises(ValueError):
            mpdsp.coefficient_sensitivity(1.0, 0.0, 0.0, -0.5, 0.1,
                                            epsilon=float("nan"))

    def test_rejects_inf_epsilon(self):
        with pytest.raises(ValueError):
            mpdsp.coefficient_sensitivity(1.0, 0.0, 0.0, -0.5, 0.1,
                                            epsilon=float("inf"))


# ---- cascade_condition_number (Python wrapper) -------------------------


class TestCascadeConditionNumber:
    """`mpdsp.cascade_condition_number(filt, num_freqs)` is the free-
    function companion to the existing `filt.condition_number(num_freqs)`
    method. They wrap the same upstream primitive, so agreement is exact."""

    def test_agrees_with_method_form(self):
        filt = mpdsp.butterworth_lowpass(order=6, sample_rate=44100.0,
                                           cutoff=1000.0)
        for nf in (128, 256, 512):
            assert (mpdsp.cascade_condition_number(filt, nf)
                    == pytest.approx(filt.condition_number(nf), abs=0.0))

    def test_high_order_exceeds_low_order(self):
        filt_lo = mpdsp.butterworth_lowpass(order=2, sample_rate=44100.0,
                                              cutoff=1000.0)
        filt_hi = mpdsp.butterworth_lowpass(order=8, sample_rate=44100.0,
                                              cutoff=1000.0)
        assert (mpdsp.cascade_condition_number(filt_hi)
                > mpdsp.cascade_condition_number(filt_lo))

    def test_default_num_freqs_is_512(self):
        # Per issue #53 the default is 512. The existing
        # IIRFilter.condition_number method keeps 256 for backwards
        # compatibility, so the wrapper default must differ from the
        # underlying method default at call-through time.
        filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100.0,
                                          cutoff=1000.0)
        default = mpdsp.cascade_condition_number(filt)
        explicit_512 = mpdsp.cascade_condition_number(filt, num_freqs=512)
        assert default == pytest.approx(explicit_512, rel=0.0, abs=0.0)
