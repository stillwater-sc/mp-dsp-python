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


# ---------------------------------------------------------------------------
# Acquisition-pipeline precision primitives (Phase 5 / #112).
# ---------------------------------------------------------------------------


class TestEnobFromSnrDb:
    def test_known_values(self):
        # ENOB = (SNR - 1.76) / 6.02
        assert mpdsp.enob_from_snr_db(98.09) == pytest.approx(16.0, abs=0.01)
        assert mpdsp.enob_from_snr_db(73.99) == pytest.approx(12.0, abs=0.01)
        # Full-scale sine: 6.02*B + 1.76 SNR -> B ENOB.
        for bits in [8, 12, 16, 24]:
            snr = 6.02 * bits + 1.76
            assert mpdsp.enob_from_snr_db(snr) == pytest.approx(bits, abs=0.01)


class TestSnrDb:
    def test_bit_identical_returns_infinite(self):
        sig = np.sin(2 * np.pi * np.arange(1024) / 128.0)
        # Same signal against itself — no noise, +300 dB.
        assert mpdsp.snr_db(sig, sig) == pytest.approx(300.0)

    def test_scaled_noise_reduces_snr(self):
        rng = np.random.default_rng(42)
        sig = np.sin(2 * np.pi * np.arange(4096) / 128.0)
        noise = rng.standard_normal(4096)
        # Test SNR scaling: -20 dB scale factor on noise -> +20 dB SNR gain
        snr_a = mpdsp.snr_db(sig, sig + 0.1 * noise)
        snr_b = mpdsp.snr_db(sig, sig + 0.01 * noise)
        assert snr_b - snr_a == pytest.approx(20.0, abs=1.0)

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            mpdsp.snr_db(np.zeros(100), np.zeros(50))

    def test_empty_arrays_return_zero(self):
        # Upstream returns 0 for empty inputs.
        assert mpdsp.snr_db(np.array([]), np.array([])) == 0.0


class TestCICBitGrowthReport:
    def test_default_construction(self):
        r = mpdsp.CICBitGrowthReport()
        assert r.theoretical_bits == 0
        assert r.observed_bits == 0
        assert r.max_abs_output == 0.0
        # `within_theory` defaults to false (bool value-initialization); it
        # only becomes meaningful after check_bit_growth() populates the
        # report from real data.
        assert r.within_theory is False

    def test_field_round_trip(self):
        r = mpdsp.CICBitGrowthReport()
        r.theoretical_bits = 12
        r.observed_bits = 10
        r.max_abs_output = 1024.0
        r.headroom_bits = 2.0
        r.within_theory = True
        assert r.theoretical_bits == 12
        assert r.observed_bits == 10
        assert r.max_abs_output == 1024.0
        assert r.headroom_bits == 2.0
        assert r.within_theory is True


class TestCICDecimatorCheckBitGrowth:
    def test_returns_report_within_theory_for_scaled_input(self):
        # 3-stage R=8 D=1 CIC. Theoretical growth = 3 * ceil(log2(8)) = 9 bits.
        # Feed a modest-amplitude signal; observed should be well within.
        cic = mpdsp.CICDecimator(decimation_ratio=8, num_stages=3,
                                  differential_delay=1)
        rng = np.random.default_rng(0)
        input_signal = rng.standard_normal(2048) * 0.1
        report = cic.check_bit_growth(input_signal)
        assert isinstance(report, mpdsp.CICBitGrowthReport)
        assert report.theoretical_bits == 9  # 3 * ceil(log2(8)) = 3 * 3
        assert report.within_theory is True
        assert report.max_abs_output > 0.0

    def test_headroom_positive_for_small_signal(self):
        cic = mpdsp.CICDecimator(decimation_ratio=8, num_stages=3,
                                  differential_delay=1)
        # Very small input: plenty of headroom.
        input_signal = np.ones(1024) * 0.001
        report = cic.check_bit_growth(input_signal)
        assert report.headroom_bits > 0

    def test_check_bit_growth_mutates_cic_state(self):
        # Documented behaviour: check_bit_growth pushes samples through
        # the CIC and mutates its state. After it runs, the internal
        # accumulator is not at its initial value.
        cic = mpdsp.CICDecimator(decimation_ratio=4, num_stages=2)
        input_signal = np.ones(64)
        cic.check_bit_growth(input_signal)
        # The output after mutation should reflect the pushed input;
        # exact value depends on the transient, so we just check it's
        # nonzero (integrator has accumulated).
        assert cic.output != 0.0


class TestNCOMeasureSfdrDb:
    # For a meaningful SFDR measurement, the tuned frequency must land
    # on an FFT bin — otherwise spectral leakage from off-bin energy
    # dominates whatever spurs the phase accumulator produces. Use
    # frequency = sample_rate * k / fft_size for integer k.
    ON_BIN_FS   = 4096.0
    ON_BIN_N    = 4096
    ON_BIN_FREQ = 100.0    # bin 100 exactly at fs = fft_size

    def test_pure_double_nco_has_high_sfdr(self):
        # On-bin tuning: no leakage, so SFDR reflects the phase-accumulator
        # quality alone. Double-precision NCO should be very clean.
        nco = mpdsp.NCO(frequency=self.ON_BIN_FREQ,
                        sample_rate=self.ON_BIN_FS, dtype="reference")
        sfdr = nco.measure_sfdr_db(fft_size=self.ON_BIN_N)
        assert sfdr > 100.0

    def test_narrow_nco_has_lower_sfdr_than_double(self):
        # Same on-bin tuning — narrow dtype's phase-accumulator spurs
        # should degrade SFDR relative to double.
        nco_ref = mpdsp.NCO(self.ON_BIN_FREQ, self.ON_BIN_FS,
                            dtype="reference")
        nco_narrow = mpdsp.NCO(self.ON_BIN_FREQ, self.ON_BIN_FS,
                               dtype="posit_8_2")
        sfdr_ref    = nco_ref.measure_sfdr_db(fft_size=self.ON_BIN_N)
        sfdr_narrow = nco_narrow.measure_sfdr_db(fft_size=self.ON_BIN_N)
        assert sfdr_ref > sfdr_narrow

    def test_measure_sfdr_mutates_nco_phase(self):
        nco = mpdsp.NCO(self.ON_BIN_FREQ, self.ON_BIN_FS)
        phase_before = nco.phase
        # Odd fft_size so 1025 * (100/4096) = 25.024… wraps to a nonzero
        # phase — avoids the exact-integer-cycles wraparound that would
        # land back at 0 (e.g. 1024 * 100/4096 = 25.0 exactly).
        nco.measure_sfdr_db(fft_size=1025)
        assert nco.phase != phase_before

    def test_rejects_zero_fft_size(self):
        nco = mpdsp.NCO(self.ON_BIN_FREQ, self.ON_BIN_FS)
        with pytest.raises((ValueError, RuntimeError)):
            nco.measure_sfdr_db(fft_size=0)


class TestAcquisitionPrecisionRow:
    def test_default_construction_matches_schema(self):
        row = mpdsp.AcquisitionPrecisionRow()
        assert row.pipeline == ""
        assert row.total_bits == 0
        assert row.output_snr_db == 0.0
        assert row.nco_sfdr_db == -1.0   # N/A sentinel
        assert row.cic_overflow_margin_bits == -1.0

    def test_field_round_trip(self):
        row = mpdsp.AcquisitionPrecisionRow()
        row.pipeline = "ddc"
        row.config_name = "posit_full"
        row.coeff_type = "posit<32,2>"
        row.state_type = "posit<32,2>"
        row.sample_type = "posit<16,1>"
        row.total_bits = 80
        row.output_snr_db = 92.5
        row.output_enob = 15.06
        row.nco_sfdr_db = 88.3
        row.cic_overflow_margin_bits = 2.4
        assert row.pipeline == "ddc"
        assert row.total_bits == 80
        assert row.output_enob == pytest.approx(15.06)


class TestWriteAcquisitionCsv:
    def test_writes_header_and_rows(self, tmp_path):
        rows = []
        r1 = mpdsp.AcquisitionPrecisionRow()
        r1.pipeline = "nco"
        r1.config_name = "double"
        r1.coeff_type = "double"
        r1.state_type = "double"
        r1.sample_type = "double"
        r1.total_bits = 192
        r1.output_snr_db = 300.0
        r1.output_enob = 49.55
        r1.nco_sfdr_db = 250.0
        rows.append(r1)

        r2 = mpdsp.AcquisitionPrecisionRow()
        r2.pipeline = "nco"
        r2.config_name = "posit_full"
        r2.coeff_type = "posit<32,2>"
        r2.state_type = "posit<32,2>"
        r2.sample_type = "posit<32,2>"
        r2.total_bits = 96
        r2.output_snr_db = 145.2
        r2.output_enob = 23.83
        r2.nco_sfdr_db = 130.0
        rows.append(r2)

        out_path = str(tmp_path / "sweep.csv")
        mpdsp.write_acquisition_csv(out_path, rows)

        with open(out_path) as fh:
            content = fh.read()
        # Header line
        assert content.split("\n")[0] == (
            "pipeline,config_name,coeff_type,state_type,sample_type,"
            "total_bits,output_snr_db,output_enob,nco_sfdr_db,"
            "cic_overflow_margin_bits")
        # Both rows appear
        assert "nco,double,double,double,double,192" in content
        assert "\"posit<32,2>\"" in content   # csv_quote around embedded commas

    def test_empty_row_list_writes_header_only(self, tmp_path):
        out_path = str(tmp_path / "empty.csv")
        mpdsp.write_acquisition_csv(out_path, [])
        with open(out_path) as fh:
            content = fh.read()
        # Just the header + trailing newline; no data rows.
        assert content.count("\n") == 1
        assert content.startswith("pipeline,")
