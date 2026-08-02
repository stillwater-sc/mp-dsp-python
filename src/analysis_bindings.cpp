// analysis_bindings.cpp: free-function analysis primitives.
//
// Phase 4 of the 0.5.0 binding sweep (#53). Surfaces the coefficient-
// level analysis functions from upstream sw::dsp/analysis/ that operate
// on raw biquad coefficients without needing a constructed IIRFilter.
//
// The method-form equivalents already live on PyIIRFilter in
// filter_bindings.cpp (`filt.stability_margin()`, `.condition_number()`,
// `.worst_case_sensitivity()`, `.pole_displacement()`). This file adds
// the per-biquad free-function forms, which matter for design-time
// coefficient sweeps — you can evaluate a candidate set of coefficients
// without instantiating a full cascade.
//
// Bound on double only. Upstream templates on DspField T, but for
// design-time analysis the meaningful arithmetic is double — the
// interesting question is "what are the sensitivities of this
// coefficient set before I quantize it". Mixed-precision analysis of a
// constructed filter lives on the IIRFilter methods, which already
// dispatch through ArithConfig.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <sw/dsp/analysis/acquisition_precision.hpp>
#include <sw/dsp/analysis/condition.hpp>
#include <sw/dsp/analysis/sensitivity.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>

#include "_binding_helpers.hpp"

#include <cmath>
#include <span>
#include <stdexcept>

namespace nb = nanobind;

using mpdsp::bindings::np_f64_ro;

void bind_analysis(nb::module_& m) {
	using BQ = sw::dsp::BiquadCoefficients<double>;

	m.def("coefficient_sensitivity",
		[](double b0, double b1, double b2, double a1, double a2,
		   double epsilon) {
			// Guard at the binding boundary: upstream forwards epsilon
			// straight into finite-differencing, so 0 / negative / NaN /
			// inf would silently produce meaningless derivatives.
			if (!(epsilon > 0.0) || !std::isfinite(epsilon)) {
				throw std::invalid_argument(
					"coefficient_sensitivity: epsilon must be finite "
					"and strictly positive");
			}
			BQ bq(b0, b1, b2, a1, a2);
			auto s = sw::dsp::coefficient_sensitivity(bq, epsilon);
			return nb::make_tuple(s.dp_da1, s.dp_da2);
		},
		nb::arg("b0"), nb::arg("b1"), nb::arg("b2"),
		nb::arg("a1"), nb::arg("a2"),
		nb::arg("epsilon") = 1e-8,
		"Coefficient sensitivity of a biquad, as a (dp_da1, dp_da2) "
		"tuple of doubles.\n\n"
		"Returns the finite-difference derivatives of the maximum pole "
		"radius with respect to each denominator coefficient. Large "
		"magnitudes indicate coefficients whose quantization will "
		"meaningfully move the poles — a design-time signal for "
		"numerical fragility under reduced precision.\n\n"
		"Numerator coefficients (b0, b1, b2) are accepted for signature "
		"symmetry with `biquad_condition_number` (and so callers can "
		"unpack an `IIRFilter.coefficients()` tuple directly), but they "
		"don't affect pole locations, so the returned sensitivities "
		"depend only on (a1, a2).");

	m.def("biquad_condition_number",
		[](double b0, double b1, double b2, double a1, double a2,
		   int num_freqs) {
			if (num_freqs <= 0) {
				throw std::invalid_argument(
					"biquad_condition_number: num_freqs must be > 0");
			}
			BQ bq(b0, b1, b2, a1, a2);
			return sw::dsp::biquad_condition_number(bq, num_freqs);
		},
		nb::arg("b0"), nb::arg("b1"), nb::arg("b2"),
		nb::arg("a1"), nb::arg("a2"),
		nb::arg("num_freqs") = 512,
		"Condition number of a single biquad section.\n\n"
		"Sweeps the unit circle at `num_freqs` points, measuring the "
		"maximum relative change in |H(e^{j2*pi*f})| per unit "
		"perturbation of each coefficient. The perturbation is chosen "
		"to survive the round-trip through the coefficient type (no "
		"silent no-ops for narrow arithmetic).\n\n"
		"Large values mean small coefficient errors induce big frequency-"
		"response changes — the numerical signature of designed-to-the-"
		"edge biquads. For the cascade-level version use "
		"`mpdsp.cascade_condition_number(filt, num_freqs)` or the "
		"equivalent `filt.condition_number(num_freqs)` method.");

	// -----------------------------------------------------------------------
	// Acquisition-pipeline precision analysis (Phase 5 / #112)
	//
	// Scalar quality metrics + CSV-schema value types from
	// analysis/acquisition_precision.hpp. Free-function form; NCO-SFDR
	// and CIC-bit-growth measurements — which need access to a live
	// analyzer instance — land as METHODS on PyNCO / PyCICDecimator in
	// acquisition_bindings.cpp (same PR).
	//
	// snr_db is bound only for double inputs. Users who want to model
	// narrow-precision effects should quantize the arrays first via the
	// mpdsp.adc(signal, dtype=...) binding, then call snr_db on the
	// results — the SNR computation itself accumulates in double
	// regardless of input scalar type (upstream contract).
	// -----------------------------------------------------------------------

	m.def("enob_from_snr_db",
		&sw::dsp::analysis::enob_from_snr_db, nb::arg("snr_db"),
		"Effective number of bits from SNR (dB) using the standard "
		"formula ENOB = (SNR_dB - 1.76) / 6.02. Assumes a sinusoidal "
		"full-scale input with quantization-noise-dominated error.");

	m.def("snr_db",
		[](np_f64_ro reference, np_f64_ro test) {
			if (reference.shape(0) != test.shape(0)) {
				throw std::invalid_argument(
					"snr_db: reference and test must have the same length");
			}
			return sw::dsp::analysis::snr_db<double, double>(
				std::span<const double>(reference.data(), reference.shape(0)),
				std::span<const double>(test.data(), test.shape(0)));
		},
		nb::arg("reference"), nb::arg("test"),
		"Signal-to-noise ratio in dB of `test` against `reference`. Both "
		"must be 1D NumPy float64 arrays of equal length. Returns +300 dB "
		"(effectively infinite) for a bit-identical match. To assess "
		"narrow-precision effects, quantize inputs via mpdsp.adc(x, "
		"dtype=...) first, then compute snr_db on the results.");

	// CICBitGrowthReport — value struct returned by CICDecimator's
	// check_bit_growth() method. Bind with read-only fields (Python users
	// inspect, don't mutate).
	nb::class_<sw::dsp::analysis::CICBitGrowthReport>(m, "CICBitGrowthReport",
			"Result of check_cic_bit_growth: theoretical vs. observed bit "
			"growth of a CIC decimator's output. `within_theory` is True "
			"when observed <= theoretical (the normal case for well-behaved "
			"inputs).")
		.def(nb::init<>())
		.def_rw("theoretical_bits",
		     &sw::dsp::analysis::CICBitGrowthReport::theoretical_bits,
		     "M * ceil(log2(R*D)) — Hogenauer's formula.")
		.def_rw("observed_bits",
		     &sw::dsp::analysis::CICBitGrowthReport::observed_bits,
		     "ceil(log2(max |output|)) for the test input.")
		.def_rw("max_abs_output",
		     &sw::dsp::analysis::CICBitGrowthReport::max_abs_output,
		     "Raw measured peak of |output|.")
		.def_rw("headroom_bits",
		     &sw::dsp::analysis::CICBitGrowthReport::headroom_bits,
		     "theoretical - observed, both as floats (positive means "
		     "headroom remaining).")
		.def_rw("within_theory",
		     &sw::dsp::analysis::CICBitGrowthReport::within_theory,
		     "True when observed <= theoretical.");

	// AcquisitionPrecisionRow — CSV schema record for precision sweeps.
	nb::class_<sw::dsp::analysis::AcquisitionPrecisionRow>(m,
			"AcquisitionPrecisionRow",
			"Schema-compatible Pareto-row record for the acquisition-"
			"pipeline precision sweeps. Written by write_acquisition_csv "
			"into the same column layout as applications/precision_sweep/"
			"precision_sweep.csv so the existing plot_precision and "
			"plot_heatmap scripts can read either file.\n\n"
			"Fields with default -1 (nco_sfdr_db, cic_overflow_margin_bits) "
			"are N/A markers — the CSV writer emits them literally so "
			"downstream plotting can filter them out.")
		.def(nb::init<>())
		.def_rw("pipeline",
		     &sw::dsp::analysis::AcquisitionPrecisionRow::pipeline,
		     "Pipeline identifier: 'ddc' / 'decim_chain' / 'nco' / etc.")
		.def_rw("config_name",
		     &sw::dsp::analysis::AcquisitionPrecisionRow::config_name,
		     "Human-readable configuration label.")
		.def_rw("coeff_type",
		     &sw::dsp::analysis::AcquisitionPrecisionRow::coeff_type,
		     "String repr of CoeffScalar (e.g. 'double', 'posit<32,2>').")
		.def_rw("state_type",
		     &sw::dsp::analysis::AcquisitionPrecisionRow::state_type)
		.def_rw("sample_type",
		     &sw::dsp::analysis::AcquisitionPrecisionRow::sample_type)
		.def_rw("total_bits",
		     &sw::dsp::analysis::AcquisitionPrecisionRow::total_bits,
		     "Sum of bit-widths across the three scalars.")
		.def_rw("output_snr_db",
		     &sw::dsp::analysis::AcquisitionPrecisionRow::output_snr_db)
		.def_rw("output_enob",
		     &sw::dsp::analysis::AcquisitionPrecisionRow::output_enob)
		.def_rw("nco_sfdr_db",
		     &sw::dsp::analysis::AcquisitionPrecisionRow::nco_sfdr_db,
		     "Set to -1 when not applicable to the row's pipeline.")
		.def_rw("cic_overflow_margin_bits",
		     &sw::dsp::analysis::AcquisitionPrecisionRow::cic_overflow_margin_bits,
		     "Set to -1 when not applicable to the row's pipeline.");

	m.def("write_acquisition_csv",
		&sw::dsp::analysis::write_acquisition_csv,
		nb::arg("path"), nb::arg("rows"),
		"Write a list of AcquisitionPrecisionRow to CSV at the given "
		"path. Header row is emitted first; column layout matches "
		"applications/precision_sweep/precision_sweep.csv for cross-tool "
		"compatibility with the existing plot_precision / plot_heatmap "
		"scripts.");
}
