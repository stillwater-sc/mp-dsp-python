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
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

#include <sw/dsp/analysis/condition.hpp>
#include <sw/dsp/analysis/sensitivity.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>

#include <stdexcept>

namespace nb = nanobind;

void bind_analysis(nb::module_& m) {
	using BQ = sw::dsp::BiquadCoefficients<double>;

	m.def("coefficient_sensitivity",
		[](double b0, double b1, double b2, double a1, double a2,
		   double epsilon) {
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
		nb::arg("num_freqs") = 256,
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
}
