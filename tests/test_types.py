"""Tests for the TransferFunction class + projection helpers (issue #52).

Three things under test:

1. `TransferFunction` itself — constructor, properties round-trip cleanly
   through numpy, evaluate/evaluate_many/frequency_response produce the
   expected rational values, cascade composes correctly.

2. `to_transfer_function(filt)` — the pure-Python helper that folds an
   `IIRFilter` cascade into a single TF. Pinned by verifying that the
   resulting TF's `frequency_response` agrees with `IIRFilter.frequency_response`
   to numerical tolerance.

3. `project_onto` / `projection_error` — round-trip a float64 signal
   through each available dtype. Error should decrease monotonically in
   bit width; exact equivalence for the `reference` identity case.
"""

from __future__ import annotations

import numpy as np
import pytest

import mpdsp


# ---- TransferFunction: construction and property access ----------------


def test_transfer_function_constructor_stores_coefficients():
    tf = mpdsp.TransferFunction(
        numerator=np.array([1.0, 0.5, 0.25]),
        denominator=np.array([-0.8, 0.3]),
    )
    np.testing.assert_allclose(tf.numerator, [1.0, 0.5, 0.25], atol=1e-15)
    np.testing.assert_allclose(tf.denominator, [-0.8, 0.3], atol=1e-15)


def test_transfer_function_properties_are_writable():
    tf = mpdsp.TransferFunction(np.array([1.0]), np.array([]))
    tf.numerator = np.array([1.0, 0.5, 0.25])
    tf.denominator = np.array([-0.7, 0.2])
    np.testing.assert_allclose(tf.numerator, [1.0, 0.5, 0.25])
    np.testing.assert_allclose(tf.denominator, [-0.7, 0.2])


def test_transfer_function_property_returns_fresh_ndarray():
    # Regression guard for the rv_policy::take_ownership pitfall documented
    # in src/BINDING_PATTERNS.md — without the override, this access raises
    # "reference_internal policy cannot be applied". With take_ownership +
    # a capsule, NumPy's OWNDATA flag reads False (the capsule is the
    # owner, not NumPy) — the right invariant to pin is that every access
    # hands back a distinct buffer so one reader's writes can't race
    # another reader's view.
    tf = mpdsp.TransferFunction(np.array([1.0, 2.0, 3.0]), np.array([]))
    arr1 = tf.numerator
    arr2 = tf.numerator
    assert isinstance(arr1, np.ndarray)
    assert arr1.shape == (3,)
    # Different memory, identical content — exactly the fresh-capsule
    # semantics take_ownership is meant to deliver.
    assert arr1.ctypes.data != arr2.ctypes.data
    np.testing.assert_array_equal(arr1, arr2)


# ---- TransferFunction: evaluation --------------------------------------


def test_evaluate_pure_gain_at_any_z_is_the_gain():
    # H(z) = 0.5 / 1 = 0.5 at every z.
    tf = mpdsp.TransferFunction(np.array([0.5]), np.array([]))
    for z in (1.0 + 0j, 0.5j, 0.3 - 0.4j):
        v = tf.evaluate(z)
        assert v == pytest.approx(0.5 + 0j, abs=1e-12)


def test_evaluate_many_matches_evaluate_loop():
    tf = mpdsp.TransferFunction(np.array([1.0, -0.5, 0.25]),
                                 np.array([-0.8, 0.3]))
    zs = np.array([1 + 0j, 1j, -1 + 0j, 0.5 - 0.5j], dtype=complex)
    many = tf.evaluate_many(zs)
    one_by_one = np.array([tf.evaluate(z) for z in zs])
    np.testing.assert_allclose(many, one_by_one, atol=1e-12)


def test_frequency_response_matches_evaluate_on_unit_circle():
    tf = mpdsp.TransferFunction(np.array([0.1, 0.2, 0.1]),
                                 np.array([-1.0, 0.5]))
    for f in (0.0, 0.1, 0.25, 0.4, 0.5):
        direct = tf.evaluate(np.exp(2j * np.pi * f))
        via_fr = tf.frequency_response(f)
        assert via_fr == pytest.approx(direct, abs=1e-12)


def test_frequency_response_many_shape_and_values():
    tf = mpdsp.TransferFunction(np.array([1.0]), np.array([-0.5]))
    freqs = np.linspace(0.0, 0.5, 17)
    H = tf.frequency_response_many(freqs)
    assert H.shape == (17,)
    for f, h in zip(freqs, H):
        assert h == pytest.approx(tf.frequency_response(f), abs=1e-12)


# ---- TransferFunction: cascade -----------------------------------------


def test_cascade_multiplies_frequency_responses():
    # H1 and H2 are independent biquads; cascade at any frequency should
    # be the elementwise product of the two responses.
    h1 = mpdsp.TransferFunction(np.array([1.0, 0.5, 0.25]),
                                  np.array([-0.7, 0.2]))
    h2 = mpdsp.TransferFunction(np.array([0.5, -0.5]),
                                  np.array([0.3]))
    h_cascade = h1 * h2
    for f in (0.05, 0.15, 0.3):
        expected = h1.frequency_response(f) * h2.frequency_response(f)
        actual = h_cascade.frequency_response(f)
        assert actual == pytest.approx(expected, abs=1e-10)


# ---- to_transfer_function helper ---------------------------------------


def test_to_transfer_function_matches_iirfilter_frequency_response():
    filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100.0, cutoff=1000.0)
    tf = mpdsp.to_transfer_function(filt)

    # IIRFilter.frequency_response takes normalized frequencies in [0, 0.5],
    # same convention as TransferFunction.frequency_response, so we can
    # compare directly.
    freqs = np.linspace(0.01, 0.49, 64)  # avoid DC/Nyquist numerical edges
    from_filt = filt.frequency_response(freqs)
    from_tf = tf.frequency_response_many(freqs)
    # Magnitudes should agree to high precision — both compute the same
    # rational function, just via different algorithms (cascade evaluation
    # vs. polynomial evaluation of the unrolled form).
    np.testing.assert_allclose(np.abs(from_filt), np.abs(from_tf), atol=1e-10)


def test_to_transfer_function_preserves_stability():
    # Same design as above — butterworth_lowpass is stable by construction.
    filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100.0, cutoff=1000.0)
    tf = mpdsp.to_transfer_function(filt)
    assert tf.is_stable()


# ---- project_onto / projection_error -----------------------------------


def test_project_onto_reference_is_identity():
    data = np.array([0.123456789, -0.5, 0.0, 1.0, -1.0])
    projected = mpdsp.project_onto(data, "reference")
    np.testing.assert_array_equal(projected, data)
    assert mpdsp.projection_error(data, "reference") == 0.0


def test_projection_error_shape_and_sign():
    # Arbitrary signal full of irrationals so no dtype can represent it
    # exactly — every non-reference dtype should have positive error.
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 0.5, size=256)
    for dt in ("gpu_baseline", "half", "posit_full", "tiny_posit"):
        err = mpdsp.projection_error(data, dt)
        assert err > 0.0
        assert np.isfinite(err)


def test_project_onto_round_trip_matches_projection_error():
    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 0.5, size=128)
    for dt in ("gpu_baseline", "half", "posit_full", "tiny_posit"):
        projected = mpdsp.project_onto(data, dt)
        assert projected.shape == data.shape
        max_err = float(np.max(np.abs(data - projected)))
        assert max_err == pytest.approx(mpdsp.projection_error(data, dt),
                                         abs=1e-15)


def test_projection_error_narrower_worse_than_wider():
    # Pure ordering property: moving to a narrower sample scalar should
    # not *reduce* the projection error. This guards against a regression
    # where the dispatcher accidentally routed a narrow dtype through a
    # wider implementation (silently improving results).
    rng = np.random.default_rng(11)
    data = rng.normal(0.0, 0.5, size=512)
    err_gpu = mpdsp.projection_error(data, "gpu_baseline")  # 32-bit float
    err_half = mpdsp.projection_error(data, "half")          # 16-bit cfloat
    err_tiny = mpdsp.projection_error(data, "tiny_posit")    # 8-bit posit
    assert err_gpu <= err_half <= err_tiny
    # Sanity: the spread should be material.
    assert err_tiny > err_gpu * 10
