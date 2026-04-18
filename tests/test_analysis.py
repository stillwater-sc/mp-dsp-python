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
    poles = mpdsp.biquad_poles(1.0, 0.0, 0.0, -1.0, 0.25)
    for p in poles:
        assert p.real == pytest.approx(0.5, abs=1e-10)
        assert p.imag == pytest.approx(0.0, abs=1e-10)


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
    for a, b in zip(manual_sorted, iir_sorted):
        assert a == pytest.approx(b, abs=1e-10)
