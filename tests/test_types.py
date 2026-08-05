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


# ---------------------------------------------------------------------------
# ContinuousTransferFunction (analog H(s)) — #54 Phase 5 PR-A
# ---------------------------------------------------------------------------


class TestContinuousTransferFunction:
    def test_constructor_stores_coefficients(self):
        ctf = mpdsp.ContinuousTransferFunction(
            numerator=np.array([1.0, 1.0]),
            denominator=np.array([1.0, 0.5, 1.0]),
        )
        np.testing.assert_allclose(ctf.numerator, [1.0, 1.0], atol=1e-15)
        np.testing.assert_allclose(ctf.denominator, [1.0, 0.5, 1.0], atol=1e-15)

    def test_setter_round_trip(self):
        # Pin the writable property contract: assign new arrays, read them
        # back. A broken def_prop_rw binding or ndarray-to-vector conversion
        # would slip past the constructor + getter-only tests otherwise.
        ctf = mpdsp.ContinuousTransferFunction(
            np.array([1.0]), np.array([1.0]))
        ctf.numerator = np.array([1.0, 0.5, 0.25])
        ctf.denominator = np.array([1.0, 0.3, 1.0, 0.5])
        np.testing.assert_allclose(ctf.numerator,
                                    [1.0, 0.5, 0.25], atol=1e-15)
        np.testing.assert_allclose(ctf.denominator,
                                    [1.0, 0.3, 1.0, 0.5], atol=1e-15)
        # After a setter round-trip, evaluate should reflect the new
        # coefficients, not the constructor's.
        # H(s) = (1 + 0.5s + 0.25s^2) / (1 + 0.3s + s^2 + 0.5s^3) at s=0
        # is just 1/1 = 1.
        assert ctf.evaluate(0.0 + 0j) == pytest.approx(1.0 + 0j, abs=1e-12)

    def test_evaluate_pure_gain_at_any_s_is_the_gain(self):
        ctf = mpdsp.ContinuousTransferFunction(
            np.array([2.0]), np.array([1.0]))
        for s in (1.0 + 0j, 1j, -0.3 + 0.4j):
            assert ctf.evaluate(s) == pytest.approx(2.0 + 0j, abs=1e-12)

    def test_frequency_response_matches_evaluate_on_imag_axis(self):
        # First-order lowpass H(s) = 1 / (s + 1).
        ctf = mpdsp.ContinuousTransferFunction(
            np.array([1.0]), np.array([1.0, 1.0]))
        for omega in (0.0, 0.5, 1.0, 2.0, 5.0):
            direct = ctf.evaluate(1j * omega)
            via_fr = ctf.frequency_response(omega)
            assert via_fr == pytest.approx(direct, abs=1e-12)

    def test_first_order_lowpass_magnitude_3db_at_omega_1(self):
        # |H(j*1)| = 1/sqrt(2) for H(s) = 1 / (s + 1).
        ctf = mpdsp.ContinuousTransferFunction(
            np.array([1.0]), np.array([1.0, 1.0]))
        H = ctf.frequency_response(1.0)
        assert abs(H) == pytest.approx(1.0 / np.sqrt(2), abs=1e-12)

    def test_evaluate_many_matches_loop(self):
        ctf = mpdsp.ContinuousTransferFunction(
            np.array([1.0, 0.5]), np.array([1.0, 0.3, 1.0]))
        ss = np.array([0 + 0.5j, 1j, -0.2 + 0.3j], dtype=complex)
        batch = ctf.evaluate_many(ss)
        one_by_one = np.array([ctf.evaluate(s) for s in ss])
        np.testing.assert_allclose(batch, one_by_one, atol=1e-12)

    def test_property_access_is_fresh_buffer_per_call(self):
        # Regression guard against the rv_policy pitfall in BINDING_PATTERNS.md.
        ctf = mpdsp.ContinuousTransferFunction(
            np.array([1.0, 2.0, 3.0]), np.array([1.0]))
        a = ctf.numerator
        b = ctf.numerator
        assert a.ctypes.data != b.ctypes.data
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# ztransform, freqz, group_delay (#54 Phase 5 PR-A)
# ---------------------------------------------------------------------------


class TestZTransformFreeFunctions:
    def test_ztransform_matches_evaluate_many(self):
        tf = mpdsp.TransferFunction(np.array([1.0, 0.5, 0.25]),
                                      np.array([-0.8, 0.3]))
        zs = np.exp(2j * np.pi * np.array([0.0, 0.1, 0.25, 0.4, 0.5]))
        via_free = mpdsp.ztransform(tf, zs)
        via_method = tf.evaluate_many(zs)
        np.testing.assert_array_equal(via_free, via_method)

    def test_freqz_matches_frequency_response_at_matching_points(self):
        tf = mpdsp.TransferFunction(np.array([1.0, 0.5]),
                                      np.array([-0.5, 0.1]))
        N = 64
        from_free = mpdsp.freqz(tf, num_points=N)
        freqs = np.linspace(0.0, 0.5, N, endpoint=False)
        from_method = tf.frequency_response_many(freqs)
        np.testing.assert_allclose(from_free, from_method, atol=1e-12)

    def test_freqz_default_num_points_is_512(self):
        tf = mpdsp.TransferFunction(np.array([1.0]), np.array([-0.5]))
        assert mpdsp.freqz(tf).shape == (512,)

    def test_group_delay_shape_and_type(self):
        tf = mpdsp.TransferFunction(np.array([1.0, 0.5]),
                                      np.array([-0.5, 0.1]))
        gd = mpdsp.group_delay(tf, 64)
        assert gd.shape == (64,)
        assert gd.dtype == np.float64

    def test_group_delay_zero_for_pure_gain(self):
        # A pure-gain TF has zero phase everywhere → group delay is 0.
        tf = mpdsp.TransferFunction(np.array([0.5]), np.array([]))
        gd = mpdsp.group_delay(tf, 32)
        np.testing.assert_allclose(gd, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# laplace_freqs (#54 Phase 5 PR-A)
# ---------------------------------------------------------------------------


class TestLaplaceFreqs:
    def test_matches_frequency_response_at_matching_omega(self):
        ctf = mpdsp.ContinuousTransferFunction(
            np.array([1.0]), np.array([1.0, 1.0]))  # 1/(s+1)
        N = 32
        omega_max = 5.0
        from_free = mpdsp.laplace_freqs(ctf, omega_max, N)
        omegas = np.linspace(0.0, omega_max, N, endpoint=False)
        from_method = ctf.frequency_response_many(omegas)
        np.testing.assert_allclose(from_free, from_method, atol=1e-12)

    def test_omega_max_validated(self):
        ctf = mpdsp.ContinuousTransferFunction(
            np.array([1.0]), np.array([1.0, 1.0]))
        with pytest.raises(ValueError):
            mpdsp.laplace_freqs(ctf, 0.0, 16)
        with pytest.raises(ValueError):
            mpdsp.laplace_freqs(ctf, -1.0, 16)

    def test_default_num_points_is_512(self):
        ctf = mpdsp.ContinuousTransferFunction(
            np.array([1.0]), np.array([1.0, 1.0]))
        assert mpdsp.laplace_freqs(ctf, 10.0).shape == (512,)


# ---------------------------------------------------------------------------
# Structured types (Phase 5 / #114):
# ComplexPair, PoleZeroPair, BiquadCoefficients, and
# IIRFilter.from_coefficients().
# ---------------------------------------------------------------------------


class TestComplexPair:
    def test_default_construction(self):
        cp = mpdsp.ComplexPair()
        assert cp.first == 0.0 + 0j
        assert cp.second == 0.0 + 0j

    def test_single_value_construction(self):
        cp = mpdsp.ComplexPair(3.0 + 4.0j)
        assert cp.first == 3.0 + 4.0j
        assert cp.second == 0.0 + 0j

    def test_two_value_construction(self):
        cp = mpdsp.ComplexPair(1 + 2j, 3 + 4j)
        assert cp.first == 1 + 2j
        assert cp.second == 3 + 4j

    def test_is_conjugate(self):
        cp = mpdsp.ComplexPair(0.5 + 0.8j, 0.5 - 0.8j)
        assert cp.is_conjugate() is True
        cp2 = mpdsp.ComplexPair(0.5 + 0.8j, 0.5 + 0.8j)
        assert cp2.is_conjugate() is False

    def test_is_real(self):
        real_pair = mpdsp.ComplexPair(1 + 0j, 2 + 0j)
        assert real_pair.is_real() is True
        complex_pair = mpdsp.ComplexPair(1 + 1j, 2 + 0j)
        assert complex_pair.is_real() is False

    def test_field_round_trip(self):
        cp = mpdsp.ComplexPair()
        cp.first = 5 + 6j
        cp.second = 7 + 8j
        assert cp.first == 5 + 6j
        assert cp.second == 7 + 8j


class TestPoleZeroPair:
    def test_default_construction(self):
        pz = mpdsp.PoleZeroPair()
        assert pz.is_single_pole() is True   # both entries default zero
        assert pz.poles.first == 0j
        assert pz.zeros.first == 0j

    def test_first_order_construction(self):
        pz = mpdsp.PoleZeroPair(0.5 + 0j, -1 + 0j)
        assert pz.poles.first == 0.5 + 0j
        assert pz.zeros.first == -1 + 0j
        assert pz.is_single_pole() is True

    def test_second_order_construction(self):
        pz = mpdsp.PoleZeroPair(0.5 + 0.5j, -1 + 0j,
                                 0.5 - 0.5j, -1 + 0j)
        assert pz.poles.first == 0.5 + 0.5j
        assert pz.poles.second == 0.5 - 0.5j
        assert pz.is_single_pole() is False

    def test_field_mutation(self):
        pz = mpdsp.PoleZeroPair()
        pz.poles = mpdsp.ComplexPair(0.7 + 0.7j, 0.7 - 0.7j)
        assert pz.poles.first == 0.7 + 0.7j
        assert pz.poles.second == 0.7 - 0.7j


class TestBiquadCoefficients:
    def test_default_construction(self):
        bq = mpdsp.BiquadCoefficients()
        assert bq.b0 == 0.0
        assert bq.b1 == 0.0
        assert bq.b2 == 0.0
        assert bq.a1 == 0.0
        assert bq.a2 == 0.0

    def test_explicit_construction(self):
        bq = mpdsp.BiquadCoefficients(1.0, 2.0, 3.0, 4.0, 5.0)
        assert bq.b0 == 1.0
        assert bq.b1 == 2.0
        assert bq.b2 == 3.0
        assert bq.a1 == 4.0
        assert bq.a2 == 5.0

    def test_field_mutation(self):
        bq = mpdsp.BiquadCoefficients()
        bq.b0 = 0.5
        bq.a2 = -0.25
        assert bq.b0 == 0.5
        assert bq.a2 == -0.25

    def test_set_identity(self):
        bq = mpdsp.BiquadCoefficients(0.1, 0.2, 0.3, 0.4, 0.5)
        bq.set_identity()
        assert bq.b0 == 1.0
        assert bq.b1 == 0.0
        assert bq.b2 == 0.0
        assert bq.a1 == 0.0
        assert bq.a2 == 0.0

    def test_identity_response_is_unity(self):
        bq = mpdsp.BiquadCoefficients()
        bq.set_identity()
        # H(z) = 1 -> response at every frequency is 1+0j
        for f in [0.0, 0.1, 0.25, 0.5]:
            assert bq.response(f) == pytest.approx(1.0 + 0j)

    def test_response_at_dc(self):
        # H(1) = (b0 + b1 + b2) / (1 + a1 + a2) at z=1 (DC, f=0).
        bq = mpdsp.BiquadCoefficients(1.0, 0.5, 0.25, -0.5, 0.25)
        # numerator = 1 + 0.5 + 0.25 = 1.75
        # denominator = 1 - 0.5 + 0.25 = 0.75
        assert bq.response(0.0) == pytest.approx(1.75 / 0.75 + 0j)

    def test_apply_scale(self):
        bq = mpdsp.BiquadCoefficients(1.0, 2.0, 3.0, 0.0, 0.0)
        bq.apply_scale(2.0)
        assert bq.b0 == 2.0
        assert bq.b1 == 4.0
        assert bq.b2 == 6.0
        # Denominator unchanged
        assert bq.a1 == 0.0
        assert bq.a2 == 0.0

    def test_set_one_pole(self):
        bq = mpdsp.BiquadCoefficients()
        bq.set_one_pole(pole=0.5 + 0j, zero=-1.0 + 0j)
        # b0 = 1, b1 = -zero = 1, b2 = 0
        # a1 = -pole = -0.5, a2 = 0
        assert bq.b0 == 1.0
        assert bq.b1 == 1.0
        assert bq.a1 == -0.5

    def test_repr(self):
        bq = mpdsp.BiquadCoefficients(1.0, 2.0, 3.0, 4.0, 5.0)
        s = repr(bq)
        for field in ["b0", "b1", "b2", "a1", "a2"]:
            assert field in s


class TestIIRFilterFromCoefficients:
    def test_construct_from_single_biquad(self):
        bq = mpdsp.BiquadCoefficients(1.0, 0.0, 0.0, 0.0, 0.0)   # identity
        filt = mpdsp.IIRFilter.from_coefficients([bq])
        assert isinstance(filt, mpdsp.IIRFilter)
        assert filt.num_stages() == 1

    def test_identity_biquad_is_passthrough(self):
        bq = mpdsp.BiquadCoefficients()
        bq.set_identity()
        filt = mpdsp.IIRFilter.from_coefficients([bq])
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(256)
        y = filt.process(sig)
        np.testing.assert_allclose(y, sig, atol=1e-12)

    def test_multi_stage_cascade(self):
        # Build a 3-stage cascade of identity biquads — output should
        # still equal input (3 identities in cascade == identity).
        biquads = []
        for _ in range(3):
            b = mpdsp.BiquadCoefficients()
            b.set_identity()
            biquads.append(b)
        filt = mpdsp.IIRFilter.from_coefficients(biquads)
        assert filt.num_stages() == 3
        sig = np.arange(64.0)
        np.testing.assert_allclose(filt.process(sig), sig, atol=1e-12)

    def test_roundtrip_via_coefficients(self):
        # Design a Butterworth, extract its coefficients, construct a new
        # filter from them, and verify the two produce identical output.
        original = mpdsp.butterworth_lowpass(order=4, sample_rate=8000.0,
                                              cutoff=1000.0)
        # coefficients() returns list of (b0, b1, b2, a1, a2) tuples.
        biquads = [mpdsp.BiquadCoefficients(*c) for c in original.coefficients()]
        rebuilt = mpdsp.IIRFilter.from_coefficients(biquads)
        assert rebuilt.num_stages() == original.num_stages()
        sig = np.sin(2 * np.pi * 300.0 * np.arange(2048) / 8000.0)
        np.testing.assert_allclose(rebuilt.process(sig),
                                   original.process(sig), atol=1e-12)

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError):
            mpdsp.IIRFilter.from_coefficients([])

    def test_rejects_too_many_biquads(self):
        biquads = [mpdsp.BiquadCoefficients() for _ in range(9)]  # max=8
        with pytest.raises(ValueError):
            mpdsp.IIRFilter.from_coefficients(biquads)

    def test_max_biquads_accepted(self):
        # Exactly 8 biquads should be accepted.
        biquads = []
        for _ in range(8):
            b = mpdsp.BiquadCoefficients()
            b.set_identity()
            biquads.append(b)
        filt = mpdsp.IIRFilter.from_coefficients(biquads)
        assert filt.num_stages() == 8


# =============================================================================
# Analog prototypes — s-plane pole/zero constellations (Issue #115)
#
# These are the pre-bilinear view that a designed IIRFilter's response hides:
# the digital response is frequency-warped toward Nyquist, while the analog
# prototype extends linearly in omega.
# =============================================================================

_ALL_PROTOTYPES = {
    "butterworth": lambda: mpdsp.butterworth_prototype(4, 1.0),
    "chebyshev1": lambda: mpdsp.chebyshev1_prototype(4, 1.0, 1.0),
    "chebyshev2": lambda: mpdsp.chebyshev2_prototype(4, 1.0, 40.0),
    "bessel": lambda: mpdsp.bessel_prototype(4, 1.0),
    "elliptic": lambda: mpdsp.elliptic_prototype(4, 1.0, 1.0, 0.9),
}

# Families whose transfer function is all-pole: no finite s-plane zeros.
_ALL_POLE = ["butterworth", "chebyshev1", "bessel"]
# Families that place finite zeros on the jw axis to make stopband nulls.
_WITH_ZEROS = ["chebyshev2", "elliptic"]


class TestAnalogPrototypes:
    @pytest.mark.parametrize("name", sorted(_ALL_PROTOTYPES))
    def test_order_and_stability(self, name):
        plot = _ALL_PROTOTYPES[name]()
        assert plot.design == name
        assert plot.order == 4
        assert plot.kind == "lowpass"
        assert len(plot.s_poles) == 4
        # A realizable analog prototype is strictly left-half-plane.
        assert all(p.real < 0.0 for p in plot.s_poles)
        # z-plane view is empty until apply_bilinear.
        assert list(plot.z_poles) == []
        assert list(plot.z_zeros) == []

    @pytest.mark.parametrize("name", _ALL_POLE)
    def test_all_pole_families_have_no_finite_zeros(self, name):
        assert list(_ALL_PROTOTYPES[name]().s_zeros) == []

    @pytest.mark.parametrize("name", _WITH_ZEROS)
    def test_zero_families_place_zeros_on_the_jw_axis(self, name):
        """Chebyshev II and elliptic get their stopband nulls from finite
        zeros sitting exactly on the imaginary axis."""
        zeros = np.asarray(_ALL_PROTOTYPES[name]().s_zeros)
        assert len(zeros) > 0
        np.testing.assert_allclose(zeros.real, 0.0, atol=1e-12)
        # And genuinely away from the origin — these are stopband nulls.
        assert np.min(np.abs(zeros.imag)) > 1.0

    def test_butterworth_poles_lie_on_a_circle(self):
        """Butterworth's signature: poles equidistant from the origin at
        radius 2*pi*cutoff_hz."""
        plot = mpdsp.butterworth_prototype(4, 1.0)
        radii = np.abs(np.asarray(plot.s_poles))
        np.testing.assert_allclose(radii, 2.0 * np.pi, rtol=1e-9)

    def test_chebyshev1_poles_lie_on_an_ellipse(self):
        """Chebyshev I's signature: the circle is squashed along the real
        axis, which is what buys the steeper rolloff and costs passband
        flatness. Butterworth's semi-axis ratio is 1.0; Chebyshev I at 1 dB
        ripple measures ~0.34."""
        cheby = np.asarray(mpdsp.chebyshev1_prototype(4, 1.0, 1.0).s_poles)
        butter = np.asarray(mpdsp.butterworth_prototype(4, 1.0).s_poles)

        def axis_ratio(poles):
            return np.abs(poles.real).max() / np.abs(poles.imag).max()

        assert axis_ratio(butter) == pytest.approx(1.0, abs=1e-9)
        assert axis_ratio(cheby) < 0.5

    def test_design_parameters_are_carried_through(self):
        assert mpdsp.chebyshev1_prototype(4, 1.0, 1.5).ripple_db == \
            pytest.approx(1.5)
        assert mpdsp.chebyshev2_prototype(4, 1.0, 60.0).stopband_db == \
            pytest.approx(60.0)

    def test_cutoff_scales_the_constellation(self):
        small = np.abs(np.asarray(mpdsp.butterworth_prototype(4, 1.0).s_poles))
        large = np.abs(np.asarray(mpdsp.butterworth_prototype(4, 10.0).s_poles))
        np.testing.assert_allclose(large, 10.0 * small, rtol=1e-9)

    @pytest.mark.parametrize("name", sorted(_ALL_PROTOTYPES))
    def test_repr_is_informative(self, name):
        text = repr(_ALL_PROTOTYPES[name]())
        assert "PoleZeroPlot" in text and name in text

    def test_invalid_order_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.butterworth_prototype(0, 1.0)

    def test_invalid_ripple_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.chebyshev1_prototype(4, 1.0, 0.0)

    def test_invalid_stopband_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.chebyshev2_prototype(4, 1.0, 0.0)

    def test_elliptic_order_cap_raises(self):
        # Upstream caps elliptic at order 12.
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.elliptic_prototype(13, 1.0, 1.0, 0.9)


class TestPrototypeTransforms:
    def test_lp_to_hp_preserves_order_and_moves_zeros_to_origin(self):
        lp = mpdsp.butterworth_prototype(4, 1.0)
        hp = mpdsp.lp_to_hp(lp, 1.0)
        assert hp.kind == "highpass"
        assert len(hp.s_poles) == len(lp.s_poles)
        # A highpass gets one zero at the origin per pole.
        assert len(hp.s_zeros) == 4
        np.testing.assert_allclose(np.abs(np.asarray(hp.s_zeros)), 0.0,
                                   atol=1e-12)
        assert all(p.real < 0.0 for p in hp.s_poles)

    @pytest.mark.parametrize("transform,kind", [
        (mpdsp.lp_to_bp, "bandpass"),
        (mpdsp.lp_to_bs, "bandstop"),
    ])
    def test_band_transforms_double_the_order(self, transform, kind):
        lp = mpdsp.butterworth_prototype(4, 1.0)
        out = transform(lp, 300.0, 3000.0)
        assert out.kind == kind
        # Each lowpass pole splits into a conjugate pair about the band center.
        assert len(out.s_poles) == 2 * len(lp.s_poles)
        assert out.low_hz == pytest.approx(300.0)
        assert out.high_hz == pytest.approx(3000.0)

    @pytest.mark.parametrize("transform", [mpdsp.lp_to_bp, mpdsp.lp_to_bs])
    def test_band_transforms_reject_inverted_edges(self, transform):
        lp = mpdsp.butterworth_prototype(4, 1.0)
        for low, high in [(3000.0, 300.0), (0.0, 300.0), (300.0, 300.0)]:
            with pytest.raises((ValueError, RuntimeError)):
                transform(lp, low, high)

    def test_transforms_return_a_new_plot(self):
        """Upstream mutates in place; the binding copies first.

        Python callers get value semantics, so a prototype can feed several
        transforms without being consumed by the first.
        """
        lp = mpdsp.butterworth_prototype(4, 1.0)
        hp = mpdsp.lp_to_hp(lp, 1.0)
        bp = mpdsp.lp_to_bp(lp, 300.0, 3000.0)

        assert lp.kind == "lowpass"
        assert len(lp.s_poles) == 4
        assert list(lp.s_zeros) == []
        assert hp.kind == "highpass"
        assert bp.kind == "bandpass"

    def test_apply_bilinear_maps_stable_poles_inside_unit_circle(self):
        lp = mpdsp.butterworth_prototype(4, 1000.0)
        digital = mpdsp.apply_bilinear(lp, 48000.0)
        assert digital.sample_rate_hz == pytest.approx(48000.0)
        assert len(digital.z_poles) == len(lp.s_poles)
        assert all(abs(p) < 1.0 for p in digital.z_poles)
        # Source untouched — same value semantics as the transforms.
        assert list(lp.z_poles) == []
        assert lp.sample_rate_hz == pytest.approx(0.0)

    def test_transform_then_bilinear_chains(self):
        plot = mpdsp.apply_bilinear(
            mpdsp.lp_to_bp(mpdsp.butterworth_prototype(4, 1.0), 300.0, 3000.0),
            48000.0)
        assert plot.kind == "bandpass"
        assert len(plot.z_poles) == 8
        assert all(abs(p) < 1.0 for p in plot.z_poles)


# =============================================================================
# Frequency-response shape of the prototypes and transforms
#
# The transform tests above check pole/zero *counts* and metadata. That is not
# enough: `lp_to_bp` and `lp_to_bs` pass those checks while producing
# constellations whose responses are not bandpass and bandstop at all
# (upstream mixed-precision-dsp#204). These tests pin the actual physics.
# =============================================================================

def _analog_response_db(plot, omega):
    """|H(j*omega)| in dB from the s-plane constellation, peak-normalized."""
    s = 1j * np.asarray(omega, dtype=complex)
    num = np.ones_like(s)
    for z in plot.s_zeros:
        num = num * (s - z)
    den = np.ones_like(s)
    for p in plot.s_poles:
        den = den * (s - p)
    mag = np.abs(num / den)
    mag = mag / mag.max() if mag.max() > 0 else mag
    return 20.0 * np.log10(np.maximum(mag, 1e-15))


class TestPrototypeResponseShape:
    def test_butterworth_lowpass_is_textbook(self):
        """0 dB in band, -3 dB at cutoff, -20N dB/decade after."""
        plot = mpdsp.butterworth_prototype(4, 1000.0)
        db = _analog_response_db(
            plot, 2 * np.pi * np.array([100.0, 1000.0, 10000.0]))
        assert db[0] == pytest.approx(0.0, abs=0.01)
        assert db[1] == pytest.approx(-3.01, abs=0.05)
        # Fourth order rolls off at 80 dB per decade.
        assert db[2] == pytest.approx(-80.0, abs=0.5)

    @pytest.mark.parametrize("order,expected_rolloff_db", [
        (2, -40.0), (4, -80.0), (6, -120.0),
    ])
    def test_rolloff_tracks_order(self, order, expected_rolloff_db):
        """-20N dB per decade, measured between two stopband decades.

        Not from the cutoff: the response is already -3.01 dB there, so a
        cutoff-to-decade measurement lands 3 dB shy of the asymptote and
        looks like a library error when it is a measurement error. Between
        10*fc and 100*fc the slope is exact to three decimals.
        """
        plot = mpdsp.butterworth_prototype(order, 1000.0)
        db = _analog_response_db(
            plot, 2 * np.pi * np.array([10000.0, 100000.0]))
        assert db[1] - db[0] == pytest.approx(expected_rolloff_db, abs=0.01)

    def test_lowpass_passes_low_and_stops_high(self):
        plot = mpdsp.butterworth_prototype(4, 1000.0)
        omega = 2 * np.pi * np.logspace(1, 5, 400)
        db = _analog_response_db(plot, omega)
        assert np.argmax(db) < len(db) // 4       # peak lives at the low end
        assert db[-1] < -60.0

    def test_lp_to_hp_inverts_the_shape(self):
        """The one band transform that is correct upstream."""
        lp = mpdsp.butterworth_prototype(4, 1.0)
        hp = mpdsp.lp_to_hp(lp, 1000.0)
        omega = 2 * np.pi * np.logspace(1, 5, 400)
        db = _analog_response_db(hp, omega)
        assert np.argmax(db) > 3 * len(db) // 4   # peak at the high end
        assert db[0] < -60.0

    def test_chebyshev1_ripples_within_its_spec(self):
        ripple_db = 1.0
        plot = mpdsp.chebyshev1_prototype(6, 1000.0, ripple_db)
        # Sample strictly inside the passband, away from the cutoff knee.
        db = _analog_response_db(
            plot, 2 * np.pi * np.linspace(10.0, 900.0, 500))
        assert db.max() - db.min() <= ripple_db + 0.15

    def test_chebyshev2_meets_its_stopband_spec(self):
        """Flat passband, equiripple stopband at exactly the requested depth.

        The sweep has to span the passband: peak-normalizing a
        stopband-only sweep measures everything against the stopband ripple
        instead of the passband, which makes a correct filter look broken.
        """
        stopband_db = 40.0
        plot = mpdsp.chebyshev2_prototype(6, 1000.0, stopband_db)
        omega = 2 * np.pi * np.logspace(0, 5, 20001)      # 1 Hz .. 100 kHz
        freq = omega / (2 * np.pi)
        db = _analog_response_db(plot, omega)

        in_passband = freq <= 300.0
        in_stopband = (freq >= 2000.0) & (freq <= 50000.0)

        # Chebyshev II is maximally flat in the passband.
        assert db[in_passband].max() - db[in_passband].min() < 0.01
        # ...and equiripple in the stopband, peaking at the requested depth.
        assert db[in_stopband].max() == pytest.approx(-stopband_db, abs=0.5)
        # The finite jw-axis zeros drive nulls far below that ripple.
        assert db[in_stopband].min() < db[in_stopband].max() - 40.0


class TestBandTransformResponseShape:
    """Strict xfails tracking upstream mixed-precision-dsp#204.

    `strict=True` means these turn into failures the moment upstream is
    fixed — which is exactly the notification wanted, since the dashboard
    and its warning text should be reverted at that point.
    """

    _LOW, _HIGH = 800.0, 1200.0

    @property
    def _omega0(self):
        return 2 * np.pi * np.sqrt(self._LOW * self._HIGH)

    def _sweep(self):
        return self._omega0 * np.logspace(-2, 2, 4001)

    @pytest.mark.xfail(strict=True,
                       reason="upstream mixed-precision-dsp#204: lp_to_bp "
                              "peaks at ~3.5x omega0 instead of omega0")
    def test_bandpass_peaks_at_band_centre(self):
        bp = mpdsp.lp_to_bp(mpdsp.butterworth_prototype(4, 1.0),
                            self._LOW, self._HIGH)
        omega = self._sweep()
        peak = omega[int(np.argmax(_analog_response_db(bp, omega)))]
        assert peak == pytest.approx(self._omega0, rel=0.15)

    @pytest.mark.xfail(strict=True,
                       reason="upstream mixed-precision-dsp#204: lp_to_bp "
                              "emits 2N origin zeros instead of N")
    def test_bandpass_zero_count(self):
        lp = mpdsp.butterworth_prototype(4, 1.0)
        bp = mpdsp.lp_to_bp(lp, self._LOW, self._HIGH)
        assert len(bp.s_zeros) == len(lp.s_poles)

    @pytest.mark.xfail(strict=True,
                       reason="upstream mixed-precision-dsp#204: lp_to_bs "
                              "emits no notch zeros, so there is no notch")
    def test_bandstop_notches_at_band_centre(self):
        bs = mpdsp.lp_to_bs(mpdsp.butterworth_prototype(4, 1.0),
                            self._LOW, self._HIGH)
        omega = self._sweep()
        db = _analog_response_db(bs, omega)
        null = omega[int(np.argmin(db))]
        assert null == pytest.approx(self._omega0, rel=0.15)

    @pytest.mark.xfail(strict=True,
                       reason="upstream mixed-precision-dsp#204: lp_to_bs "
                              "produces an all-pole constellation")
    def test_bandstop_has_jw_axis_zeros(self):
        bs = mpdsp.lp_to_bs(mpdsp.butterworth_prototype(4, 1.0),
                            self._LOW, self._HIGH)
        assert len(bs.s_zeros) > 0
        zeros = np.asarray(bs.s_zeros)
        np.testing.assert_allclose(zeros.real, 0.0, atol=1e-9)

    def test_transforms_still_double_the_order(self):
        """Unaffected by #204 and worth keeping green — the pole count is
        the part that is right."""
        lp = mpdsp.butterworth_prototype(4, 1.0)
        for transform in (mpdsp.lp_to_bp, mpdsp.lp_to_bs):
            out = transform(lp, self._LOW, self._HIGH)
            assert len(out.s_poles) == 2 * len(lp.s_poles)
