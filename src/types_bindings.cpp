// types_bindings.cpp: TransferFunction class + project_onto / projection_error.
//
// Issue #52 (Phase 3 of the 0.5.0 binding sweep, #40). Closes the Python-
// side of the upstream `sw::dsp::types/*` headers.
//
// TransferFunction is bound on double only for 0.5.0. The primary use cases —
// feeding ztransform (Phase 5 / #54) and inspecting IIR cascades — all live
// in double-precision coefficient space, so multi-dtype TransferFunction
// templating isn't worth the infrastructure cost yet. If a real use case
// surfaces, a future patch can widen the class along the same dispatcher
// pattern used by the filter / conditioning / estimation classes.
//
// The analog-prototype surface (PoleZeroPlot + the *_prototype factories +
// the LP->HP/BP/BS transforms + apply_bilinear, Issue #115) is bound here
// too: it is plain-double throughout, with no ArithConfig dispatch, and it
// composes with the ContinuousTransferFunction / laplace_freqs pair already
// living in this file. The companion Bode sweep lives in filter_bindings.cpp
// instead, because it has to drive a PyIIRFilter / PyFIRFilter through the
// same quantization path process() uses, and those types are file-local
// there.
//
// project_onto / projection_error dispatch over the dtype string — every
// arithmetic config in types.hpp works as the narrower target because
// double is always the source (wider). See `projection.hpp`'s
// ProjectableOnto<Target, Source> concept for the compile-time check.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/operators.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/spectral/laplace.hpp>
#include <sw/dsp/spectral/ztransform.hpp>
#include <sw/dsp/types/biquad_coefficients.hpp>
#include <sw/dsp/types/complex_pair.hpp>
#include <sw/dsp/types/pole_zero_pair.hpp>
#include <sw/dsp/transfer_function/pole_zero.hpp>
#include <sw/dsp/types/projection.hpp>
#include <sw/dsp/types/transfer_function.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

namespace nb = nanobind;

namespace {

using mpdsp::bindings::np_f64;
using mpdsp::bindings::np_f64_ro;
using mpdsp::bindings::make_f64_array;
using mpdsp::bindings::vec_to_numpy;
using mpdsp::bindings::numpy_to_vec_fresh;

using TF = sw::dsp::TransferFunction<double>;

// ---------------------------------------------------------------------------
// Complex-valued ndarray helpers — not general enough to live in
// _binding_helpers.hpp (only TransferFunction needs complex arrays today).
// ---------------------------------------------------------------------------

using np_complex = nb::ndarray<nb::numpy, std::complex<double>>;
using np_complex_ro = nb::ndarray<nb::numpy, const std::complex<double>,
                                    nb::ndim<1>, nb::c_contig>;

static np_complex make_complex_array(std::size_t n,
                                      std::complex<double>*& out_ptr) {
	out_ptr = new std::complex<double>[n];
	nb::capsule owner(out_ptr, [](void* p) noexcept {
		delete[] static_cast<std::complex<double>*>(p);
	});
	std::size_t shape[1] = { n };
	return np_complex(out_ptr, 1, shape, owner);
}

// ---------------------------------------------------------------------------
// Python wrapper around TransferFunction<double>.
//
// Stored by value so the getter/setter pattern mirrors KalmanFilter: return
// fresh owned ndarrays (with take_ownership), accept ro ndarrays on set
// and re-wrap into mtl::dense_vector. See BINDING_PATTERNS.md for the
// rv_policy rationale.
// ---------------------------------------------------------------------------

class PyTransferFunction {
public:
	TF inner;

	PyTransferFunction() = default;

	PyTransferFunction(np_f64_ro numerator, np_f64_ro denominator)
		: inner(numpy_to_vec_fresh<double>(numerator),
		         numpy_to_vec_fresh<double>(denominator)) {}

	explicit PyTransferFunction(TF tf) : inner(std::move(tf)) {}

	np_f64 get_numerator() const { return vec_to_numpy(inner.numerator); }
	np_f64 get_denominator() const { return vec_to_numpy(inner.denominator); }

	void set_numerator(np_f64_ro a) {
		inner.numerator = numpy_to_vec_fresh<double>(a);
	}
	void set_denominator(np_f64_ro a) {
		inner.denominator = numpy_to_vec_fresh<double>(a);
	}

	std::complex<double> evaluate(std::complex<double> z) const {
		return inner.evaluate(z);
	}

	np_complex evaluate_many(np_complex_ro zs) const {
		std::size_t n = zs.shape(0);
		std::complex<double>* out_ptr = nullptr;
		auto arr = make_complex_array(n, out_ptr);
		const auto* zp = zs.data();
		for (std::size_t i = 0; i < n; ++i) {
			out_ptr[i] = inner.evaluate(zp[i]);
		}
		return arr;
	}

	std::complex<double> frequency_response(double f) const {
		return inner.frequency_response(f);
	}

	np_complex frequency_response_many(np_f64_ro freqs) const {
		std::size_t n = freqs.shape(0);
		std::complex<double>* out_ptr = nullptr;
		auto arr = make_complex_array(n, out_ptr);
		const auto* fp = freqs.data();
		for (std::size_t i = 0; i < n; ++i) {
			out_ptr[i] = inner.frequency_response(fp[i]);
		}
		return arr;
	}

	bool is_stable() const { return inner.is_stable(); }

	// Cascade: H_self(z) * H_other(z). Returns a new PyTransferFunction
	// rather than mutating — matches the immutable-math convention the
	// Python side expects from `*`.
	PyTransferFunction cascade(const PyTransferFunction& other) const {
		return PyTransferFunction(inner * other.inner);
	}
};

// ---------------------------------------------------------------------------
// PyContinuousTransferFunction — analog-domain H(s) = N(s)/D(s).
//
// Same shape as PyTransferFunction but polynomials are in ascending powers
// of s (the natural form for Laplace-plane analysis). Used by
// `mpdsp.laplace_freqs` (registered later in this same file) for frequency-
// response evaluation of analog prototype filters.
//
// Like the discrete TransferFunction, bound on double only — multi-
// precision continuous-time analysis isn't a real-world use case today
// and would need infrastructure we don't have.
// ---------------------------------------------------------------------------

using CTF = sw::dsp::spectral::ContinuousTransferFunction<double>;

class PyContinuousTransferFunction {
public:
	CTF inner;

	PyContinuousTransferFunction() = default;

	PyContinuousTransferFunction(np_f64_ro numerator, np_f64_ro denominator) {
		inner.numerator = numpy_to_vec_fresh<double>(numerator);
		inner.denominator = numpy_to_vec_fresh<double>(denominator);
	}

	explicit PyContinuousTransferFunction(CTF ctf) : inner(std::move(ctf)) {}

	np_f64 get_numerator() const { return vec_to_numpy(inner.numerator); }
	np_f64 get_denominator() const { return vec_to_numpy(inner.denominator); }

	void set_numerator(np_f64_ro a) {
		inner.numerator = numpy_to_vec_fresh<double>(a);
	}
	void set_denominator(np_f64_ro a) {
		inner.denominator = numpy_to_vec_fresh<double>(a);
	}

	std::complex<double> evaluate(std::complex<double> s) const {
		return inner.evaluate(s);
	}

	np_complex evaluate_many(np_complex_ro s_points) const {
		std::size_t n = s_points.shape(0);
		std::complex<double>* out_ptr = nullptr;
		auto arr = make_complex_array(n, out_ptr);
		const auto* sp = s_points.data();
		for (std::size_t i = 0; i < n; ++i) {
			out_ptr[i] = inner.evaluate(sp[i]);
		}
		return arr;
	}

	std::complex<double> frequency_response(double omega) const {
		return inner.frequency_response(omega);
	}

	np_complex frequency_response_many(np_f64_ro omegas) const {
		std::size_t n = omegas.shape(0);
		std::complex<double>* out_ptr = nullptr;
		auto arr = make_complex_array(n, out_ptr);
		const auto* wp = omegas.data();
		for (std::size_t i = 0; i < n; ++i) {
			out_ptr[i] = inner.frequency_response(wp[i]);
		}
		return arr;
	}
};

// ---------------------------------------------------------------------------
// project_onto / projection_error: round-trip a float64 vector through the
// target dtype and back to float64. ProjectableOnto<Target, Source> is
// satisfied for every config in types.hpp because double is wider than all
// of them.
// ---------------------------------------------------------------------------

template <typename T>
static mtl::vec::dense_vector<double>
project_typed(const mtl::vec::dense_vector<double>& src) {
	// project_onto<T>(src) returns mtl::vec::dense_vector<T>. Then cast
	// each element back to double so the Python caller always sees
	// float64 — matches the issue's signature
	// `project_onto(ndarray, dtype: str) -> ndarray`.
	//
	// For integer<N> the direct cast collapses |x| < 1 to zero, so mirror
	// the scale-quantize-unscale path used in adc_typed: map the
	// full-scale [-1, 1] input range onto integer's representable range,
	// quantize, then scale back. See quantization_bindings.cpp for the
	// design rationale.
	if constexpr (sw::universal::is_integer<T>) {
		constexpr double fs = static_cast<double>((1LL << (T::nbits - 1)) - 1);
		mtl::vec::dense_vector<double> out(src.size());
		for (std::size_t i = 0; i < src.size(); ++i) {
			T q = static_cast<T>(src[i] * fs);
			out[i] = static_cast<double>(q) / fs;
		}
		return out;
	} else {
		auto narrowed = sw::dsp::project_onto<T>(src);
		mtl::vec::dense_vector<double> out(narrowed.size());
		for (std::size_t i = 0; i < narrowed.size(); ++i) {
			out[i] = static_cast<double>(narrowed[i]);
		}
		return out;
	}
}

static mtl::vec::dense_vector<double>
project_dispatch(const mtl::vec::dense_vector<double>& src,
                  mpdsp::ArithConfig config) {
	// Note: posit_full's "sample" scalar is posit<16,1> but its "state" is
	// posit<32,2>. The projection round-trip semantically represents the
	// ADC/sample path, so project through the sample scalar — matches what
	// `measure_sqnr_db` already does. If a future request needs
	// coefficient-path projection specifically, we add a second function.
	using mpdsp::cf24;
	using mpdsp::fx1612_t;
	using mpdsp::half_;
	using mpdsp::int6_sample_t;
	using mpdsp::int8_sample_t;
	using mpdsp::p16;
	using mpdsp::p8_0;
	using mpdsp::p8_1;
	using mpdsp::p8_2;
	using mpdsp::p16_0;
	using mpdsp::p16_1;
	using mpdsp::p16_2;
	using mpdsp::p32_0;
	using mpdsp::p32_1;
	using mpdsp::p32_2;

	switch (config) {
	case mpdsp::ArithConfig::reference:
		return src;  // double -> double is identity
	case mpdsp::ArithConfig::gpu_baseline:
		return project_typed<float>(src);
	case mpdsp::ArithConfig::ml_hw:
		return project_typed<half_>(src);
	case mpdsp::ArithConfig::posit_full:
		return project_typed<p16>(src);
	case mpdsp::ArithConfig::cf24_config:
		return project_typed<cf24>(src);
	case mpdsp::ArithConfig::half_config:
		return project_typed<half_>(src);
	case mpdsp::ArithConfig::sensor_8bit:
		return project_typed<int8_sample_t>(src);
	case mpdsp::ArithConfig::sensor_6bit:
		return project_typed<int6_sample_t>(src);
	case mpdsp::ArithConfig::fpga_fixed:
		return project_typed<fx1612_t>(src);
	// Posit taxonomy grid (#81) — project through the posit<N,es> scalar.
	// posit_8_2 also covers the tiny_posit alias (same enum value).
	case mpdsp::ArithConfig::posit_8_0:  return project_typed<p8_0>(src);
	case mpdsp::ArithConfig::posit_8_1:  return project_typed<p8_1>(src);
	case mpdsp::ArithConfig::posit_8_2:  return project_typed<p8_2>(src);
	case mpdsp::ArithConfig::posit_16_0: return project_typed<p16_0>(src);
	case mpdsp::ArithConfig::posit_16_1: return project_typed<p16_1>(src);
	case mpdsp::ArithConfig::posit_16_2: return project_typed<p16_2>(src);
	case mpdsp::ArithConfig::posit_32_0: return project_typed<p32_0>(src);
	case mpdsp::ArithConfig::posit_32_1: return project_typed<p32_1>(src);
	case mpdsp::ArithConfig::posit_32_2: return project_typed<p32_2>(src);
	}
	// Unreachable: the switch above is exhaustive over mpdsp::ArithConfig.
	return src;
}

}  // namespace

void bind_types(nb::module_& m) {
	namespace tf = sw::dsp::transfer_function;
	using PZP = tf::PoleZeroPlot;

	// -- TransferFunction --------------------------------------------------

	nb::class_<PyTransferFunction>(m, "TransferFunction",
			"Rational transfer function H(z) = B(z) / A(z).\n\n"
			"Numerator B(z) = b0 + b1*z^-1 + b2*z^-2 + ... carries all\n"
			"non-zero coefficients.  Denominator A(z) = 1 + a1*z^-1 + ...\n"
			"stores only a1, a2, ... — the leading 1 (a0) is implicit.\n\n"
			"Bound on double only in 0.5.0; mixed-precision TF is tracked\n"
			"separately if demand surfaces.")
		.def(nb::init<np_f64_ro, np_f64_ro>(),
		     nb::arg("numerator"), nb::arg("denominator"),
		     "Construct from numerator and denominator coefficient arrays.\n"
		     "Denominator is the a1, a2, ... tail (a0 = 1 is implicit).")
		// Every getter builds a fresh NumPy array via vec_to_numpy's capsule,
		// so take_ownership is required — see src/BINDING_PATTERNS.md.
		.def_prop_rw("numerator",
		              &PyTransferFunction::get_numerator,
		              &PyTransferFunction::set_numerator,
		              nb::rv_policy::take_ownership,
		              "Numerator coefficients b0, b1, b2, ... as a float64 "
		              "ndarray.")
		.def_prop_rw("denominator",
		              &PyTransferFunction::get_denominator,
		              &PyTransferFunction::set_denominator,
		              nb::rv_policy::take_ownership,
		              "Denominator coefficients a1, a2, ... as a float64 "
		              "ndarray (a0 = 1 implicit).")
		.def("evaluate", &PyTransferFunction::evaluate, nb::arg("z"),
		      "Evaluate H(z) at a single complex point. Returns complex128.")
		.def("evaluate_many", &PyTransferFunction::evaluate_many,
		      nb::arg("z"),
		      "Evaluate H(z) at each point in a complex128 ndarray. "
		      "Returns a complex128 ndarray of the same length.")
		.def("frequency_response", &PyTransferFunction::frequency_response,
		      nb::arg("f"),
		      "Evaluate H(e^{j 2*pi*f}) at normalized frequency f in [0, 0.5].")
		.def("frequency_response_many",
		      &PyTransferFunction::frequency_response_many,
		      nb::arg("freqs"),
		      "Vectorized frequency_response(...) over a float64 ndarray of "
		      "normalized frequencies. Returns complex128.")
		.def("is_stable", &PyTransferFunction::is_stable,
		      "Check stability via a 360-angle sampling of the denominator "
		      "on the unit circle. False if any sample is within 1e-6 of "
		      "zero.")
		.def("__mul__", &PyTransferFunction::cascade, nb::arg("other"),
		      "Cascade: H_self(z) * H_other(z). Returns a new "
		      "TransferFunction; self is not modified.");

	// -- ContinuousTransferFunction (analog H(s)) --------------------------

	nb::class_<PyContinuousTransferFunction>(m, "ContinuousTransferFunction",
			"Continuous-time (analog) rational transfer function "
			"H(s) = N(s) / D(s).\n\n"
			"Numerator and denominator store ascending powers of s "
			"(coeffs[0] + coeffs[1]*s + coeffs[2]*s^2 + ...) — the natural "
			"form for Laplace-plane analysis of analog prototype filters.\n\n"
			"Bound on double only; mixed-precision continuous-time analysis "
			"isn't a real-world use case today.")
		.def(nb::init<np_f64_ro, np_f64_ro>(),
		     nb::arg("numerator"), nb::arg("denominator"),
		     "Construct from numerator and denominator coefficient arrays "
		     "in ascending powers of s.")
		.def_prop_rw("numerator",
		              &PyContinuousTransferFunction::get_numerator,
		              &PyContinuousTransferFunction::set_numerator,
		              nb::rv_policy::take_ownership,
		              "Numerator coefficients in ascending powers of s.")
		.def_prop_rw("denominator",
		              &PyContinuousTransferFunction::get_denominator,
		              &PyContinuousTransferFunction::set_denominator,
		              nb::rv_policy::take_ownership,
		              "Denominator coefficients in ascending powers of s.")
		.def("evaluate", &PyContinuousTransferFunction::evaluate,
		      nb::arg("s"),
		      "Evaluate H(s) at a single complex s-plane point.")
		.def("evaluate_many",
		      &PyContinuousTransferFunction::evaluate_many,
		      nb::arg("s"),
		      "Evaluate H(s) at each point in a complex128 ndarray. "
		      "Returns a complex128 ndarray of the same length.")
		.def("frequency_response",
		      &PyContinuousTransferFunction::frequency_response,
		      nb::arg("omega"),
		      "Evaluate H(j*omega) at angular frequency omega (rad/s).")
		.def("frequency_response_many",
		      &PyContinuousTransferFunction::frequency_response_many,
		      nb::arg("omegas"),
		      "Vectorized frequency_response(...) over a float64 ndarray "
		      "of angular frequencies. Returns complex128.");

	// -- Projection round-trip free functions ------------------------------

	m.def("project_onto",
		[](np_f64_ro data, const std::string& dtype) {
			auto v = numpy_to_vec_fresh<double>(data);
			auto config = mpdsp::parse_config(dtype);
			return vec_to_numpy(project_dispatch(v, config));
		},
		nb::arg("data"), nb::arg("dtype"),
		"Project data through the sample scalar of `dtype` and back to "
		"float64. The round-trip surfaces the quantization error you'd see "
		"feeding a signal through an ADC at that precision — it's the "
		"underlying mechanic of `measure_sqnr_db`, exposed directly for "
		"when you want the quantized samples rather than just the SQNR.");

	m.def("projection_error",
		[](np_f64_ro data, const std::string& dtype) {
			auto v = numpy_to_vec_fresh<double>(data);
			auto config = mpdsp::parse_config(dtype);
			auto projected = project_dispatch(v, config);
			double max_err = 0.0;
			for (std::size_t i = 0; i < v.size(); ++i) {
				double e = std::abs(v[i] - projected[i]);
				if (e > max_err) max_err = e;
			}
			return max_err;
		},
		nb::arg("data"), nb::arg("dtype"),
		"Max absolute error between data and its round-trip through "
		"`dtype`. Equivalent to max(abs(data - project_onto(data, dtype))) "
		"but computed without allocating the intermediate ndarray.");

	// -- Z-transform free functions over TransferFunction ------------------

	m.def("ztransform",
		[](const PyTransferFunction& tf, np_complex_ro z_points) {
			// Free-function spelling of tf.evaluate_many — matches upstream
			// `sw::dsp::spectral::evaluate_at` naming for callers who
			// prefer a free-function style. Functionally identical.
			return tf.evaluate_many(z_points);
		},
		nb::arg("tf"), nb::arg("z"),
		"Evaluate H(z) at each z-plane point. Free-function spelling of "
		"`tf.evaluate_many(z)`. Returns complex128 ndarray.");

	m.def("freqz",
		[](const PyTransferFunction& tf, std::size_t num_points) {
			// Uniform sweep of [0, 0.5) at num_points frequencies. Matches
			// the shape of MATLAB/scipy's freqz.
			std::complex<double>* out_ptr = nullptr;
			auto arr = make_complex_array(num_points, out_ptr);
			for (std::size_t k = 0; k < num_points; ++k) {
				double f = static_cast<double>(k) /
				            static_cast<double>(num_points) * 0.5;
				out_ptr[k] = tf.inner.frequency_response(f);
			}
			return arr;
		},
		nb::arg("tf"), nb::arg("num_points") = 512,
		"Evaluate H(e^{j 2*pi*f}) at `num_points` uniformly spaced "
		"normalized frequencies in [0, 0.5). Returns complex128 ndarray.");

	m.def("group_delay",
		[](const PyTransferFunction& tf, std::size_t num_points) {
			// Group delay at uniform frequency sweep. Computed as
			// -d(phase)/d(omega) via central finite differences with a
			// small step. Wraps upstream sw::dsp::spectral::group_delay,
			// but call it explicitly so we can surface the result as a
			// float64 ndarray with take_ownership semantics.
			double* out_ptr = nullptr;
			auto arr = make_f64_array(num_points, out_ptr);
			auto gd = sw::dsp::spectral::group_delay(tf.inner, num_points);
			for (std::size_t k = 0; k < num_points; ++k) {
				out_ptr[k] = gd[k];
			}
			return arr;
		},
		nb::arg("tf"), nb::arg("num_points") = 512,
		"Group delay at `num_points` uniformly spaced normalized "
		"frequencies in [0, 0.5). Returns float64 ndarray (samples of "
		"-d(phase)/d(omega)).");

	// -- Laplace free function over ContinuousTransferFunction -------------

	m.def("laplace_freqs",
		[](const PyContinuousTransferFunction& tf, double omega_max,
		   std::size_t num_points) {
			// Uniform sweep of angular frequencies in [0, omega_max).
			// Companion to `freqz` but in the s-plane rather than z-plane.
			if (!(omega_max > 0.0)) {
				throw std::invalid_argument(
					"laplace_freqs: omega_max must be positive");
			}
			std::complex<double>* out_ptr = nullptr;
			auto arr = make_complex_array(num_points, out_ptr);
			for (std::size_t k = 0; k < num_points; ++k) {
				double omega = static_cast<double>(k) /
				                static_cast<double>(num_points) * omega_max;
				out_ptr[k] = tf.inner.frequency_response(omega);
			}
			return arr;
		},
		nb::arg("tf"), nb::arg("omega_max"), nb::arg("num_points") = 512,
		"Evaluate H(j*omega) at `num_points` uniformly spaced angular "
		"frequencies in [0, omega_max). Returns complex128 ndarray.");

	// -----------------------------------------------------------------------
	// Phase 5 / #114: first-class structured types.
	// Bound on double only (matches the TransferFunction rationale earlier
	// in this file). Read-write fields per the design discussion.
	// -----------------------------------------------------------------------

	using CP = sw::dsp::ComplexPair<double>;
	using PZ = sw::dsp::PoleZeroPair<double>;
	using BQ = sw::dsp::BiquadCoefficients<double>;
	using cd = std::complex<double>;

	nb::class_<CP>(m, "ComplexPair",
			"A pair of complex numbers — the building block for pole/zero "
			"representations that map directly to second-order sections. "
			"Typically holds either a conjugate pair or a pair of real "
			"values.")
		.def(nb::init<>(),
		     "Default construct (both entries zero).")
		.def(nb::init<const cd&>(), nb::arg("c1"),
		     "Construct with a single value; the second entry is zero.")
		.def(nb::init<const cd&, const cd&>(), nb::arg("c1"), nb::arg("c2"),
		     "Construct with both entries specified.")
		.def_rw("first",  &CP::first)
		.def_rw("second", &CP::second)
		.def("is_conjugate", &CP::is_conjugate,
		     "True if second == conj(first).")
		.def("is_real", &CP::is_real,
		     "True if both entries have zero imaginary part.")
		.def("is_matched_pair", &CP::is_matched_pair,
		     "True if this is either a conjugate pair or a pair of real "
		     "values where neither is zero.")
		.def("is_nan", &CP::is_nan,
		     "True if any real or imaginary component is NaN.");

	nb::class_<PZ>(m, "PoleZeroPair",
			"Poles + zeros for a single second-order section (biquad). "
			"For a first-order section, the `.second` complex value in "
			"each ComplexPair is zero (see is_single_pole()).")
		.def(nb::init<>(),
		     "Default construct (all entries zero).")
		.def(nb::init<const cd&, const cd&>(),
		     nb::arg("pole"), nb::arg("zero"),
		     "First-order section: single pole and zero.")
		.def(nb::init<const cd&, const cd&, const cd&, const cd&>(),
		     nb::arg("pole1"), nb::arg("zero1"),
		     nb::arg("pole2"), nb::arg("zero2"),
		     "Second-order section: conjugate pair of poles and zeros.")
		.def_rw("poles", &PZ::poles)
		.def_rw("zeros", &PZ::zeros)
		.def("is_single_pole", &PZ::is_single_pole,
		     "True if this represents a first-order section (second entries "
		     "of both pole and zero pairs are zero).")
		.def("is_nan", &PZ::is_nan);

// ---- Analog prototypes: s-plane pole/zero extraction (#115) ----------
	//
	// Upstream mutates a PoleZeroPlot in place (lp_to_hp(plot, f) etc.).
	// These bindings copy-then-transform and return the new plot, so a
	// Python PoleZeroPlot behaves as an immutable value like the rest of
	// the types module. Chaining reads left to right:
	//
	//     plot = mpdsp.apply_bilinear(
	//                mpdsp.lp_to_bp(mpdsp.butterworth_prototype(4, 1.0),
	//                               300.0, 3000.0),
	//                48000.0)
	nb::class_<PZP>(m, "PoleZeroPlot",
		"Analog (s-plane) prototype pole/zero constellation, optionally "
		"carrying its bilinear-transformed z-plane counterpart.\n\n"
		"Produced by the `*_prototype` factories, reshaped by `lp_to_hp` / "
		"`lp_to_bp` / `lp_to_bs`, and mapped to discrete time by "
		"`apply_bilinear`. `z_poles` / `z_zeros` are empty until "
		"`apply_bilinear` has been applied.\n\n"
		"This is the pre-bilinear view the digital frequency response "
		"hides: the analog prototype extends linearly in omega, while a "
		"designed IIRFilter's response is warped toward Nyquist.")
		.def_prop_ro("design", [](const PZP& p) { return p.design; },
		     "Family name — 'butterworth', 'chebyshev1', ...")
		.def_prop_ro("order", [](const PZP& p) { return p.order; })
		.def_prop_ro("kind", [](const PZP& p) { return p.kind; },
		     "'lowpass', 'highpass', 'bandpass', or 'bandstop'.")
		.def_prop_ro("s_poles", [](const PZP& p) { return p.s_poles; },
		     "Continuous-time poles, as a list of complex.")
		.def_prop_ro("s_zeros", [](const PZP& p) { return p.s_zeros; },
		     "Continuous-time zeros. All-pole families return an empty "
		     "list; elliptic and Chebyshev II carry finite jw-axis zeros.")
		.def_prop_ro("z_poles", [](const PZP& p) { return p.z_poles; },
		     "Discrete-time poles. Empty until apply_bilinear.")
		.def_prop_ro("z_zeros", [](const PZP& p) { return p.z_zeros; },
		     "Discrete-time zeros. Empty until apply_bilinear.")
		.def_prop_ro("cutoff_hz", [](const PZP& p) { return p.cutoff_hz; })
		.def_prop_ro("low_hz", [](const PZP& p) { return p.low_hz; },
		     "Lower band edge; set by lp_to_bp / lp_to_bs.")
		.def_prop_ro("high_hz", [](const PZP& p) { return p.high_hz; },
		     "Upper band edge; set by lp_to_bp / lp_to_bs.")
		.def_prop_ro("sample_rate_hz",
		     [](const PZP& p) { return p.sample_rate_hz; },
		     "0.0 until apply_bilinear.")
		.def_prop_ro("ripple_db", [](const PZP& p) { return p.ripple_dB; },
		     "Passband ripple, for the families that use one.")
		.def_prop_ro("stopband_db",
		     [](const PZP& p) { return p.stopband_dB; },
		     "Stopband attenuation, for the families that use one.")
		.def("__repr__", [](const PZP& p) {
			return "PoleZeroPlot(design='" + p.design + "', order="
			     + std::to_string(p.order) + ", kind='" + p.kind
			     + "', s_poles=" + std::to_string(p.s_poles.size())
			     + ", s_zeros=" + std::to_string(p.s_zeros.size())
			     + ", z_poles=" + std::to_string(p.z_poles.size())
			     + ", z_zeros=" + std::to_string(p.z_zeros.size()) + ")";
		});

	m.def("butterworth_prototype", &tf::butterworth_prototype,
		nb::arg("order"), nb::arg("cutoff_hz") = 1.0,
		"Butterworth analog prototype: `order` poles evenly spaced on the "
		"left half of a circle of radius 2*pi*cutoff_hz. All-pole — "
		"s_zeros is empty.");

	m.def("chebyshev1_prototype", &tf::chebyshev1_prototype,
		nb::arg("order"), nb::arg("cutoff_hz") = 1.0,
		nb::arg("ripple_db") = 1.0,
		"Chebyshev I analog prototype: poles on an ellipse, giving "
		"equiripple passband at the cost of a less flat response. "
		"ripple_db must be > 0. All-pole.");

	m.def("chebyshev2_prototype", &tf::chebyshev2_prototype,
		nb::arg("order"), nb::arg("cutoff_hz") = 1.0,
		nb::arg("stopband_db") = 40.0,
		"Chebyshev II (inverse Chebyshev) analog prototype: flat "
		"passband, equiripple stopband. Carries finite s_zeros on the "
		"jw axis, which is what produces the stopband nulls. "
		"stopband_db must be > 0.");

	m.def("bessel_prototype", &tf::bessel_prototype,
		nb::arg("order"), nb::arg("cutoff_hz") = 1.0,
		"Bessel analog prototype: maximally flat group delay. All-pole. "
		"The flat-delay signature is an omega-space property, which is "
		"why it reads clearly here and only approximately in a bilinear-"
		"warped digital response.");

	m.def("elliptic_prototype", &tf::elliptic_prototype,
		nb::arg("order"), nb::arg("cutoff_hz") = 1.0,
		nb::arg("ripple_db") = 1.0, nb::arg("selectivity_k") = 0.9,
		"Elliptic (Cauer) analog prototype: equiripple in both bands, "
		"the steepest transition for a given order. Carries finite "
		"s_zeros. selectivity_k in (0, 1) sets the modulus of the "
		"elliptic functions — higher is more selective. order <= 12.");

	m.def("lp_to_hp",
		[](PZP plot, double cutoff_hz) {
			tf::lp_to_hp(plot, cutoff_hz);
			return plot;
		}, nb::arg("plot"), nb::arg("cutoff_hz"),
		"Lowpass -> highpass frequency transformation. Returns a new "
		"plot; the input is left unchanged. Pole count is preserved and "
		"zeros move to the origin.");

	m.def("lp_to_bp",
		[](PZP plot, double low_hz, double high_hz) {
			tf::lp_to_bp(plot, low_hz, high_hz);
			return plot;
		}, nb::arg("plot"), nb::arg("low_hz"), nb::arg("high_hz"),
		"Lowpass -> bandpass frequency transformation. Returns a new "
		"plot; the input is left unchanged. Each prototype pole splits "
		"into two, so the resulting order is doubled. Requires "
		"0 < low_hz < high_hz.");

	m.def("lp_to_bs",
		[](PZP plot, double low_hz, double high_hz) {
			tf::lp_to_bs(plot, low_hz, high_hz);
			return plot;
		}, nb::arg("plot"), nb::arg("low_hz"), nb::arg("high_hz"),
		"Lowpass -> bandstop frequency transformation. Returns a new "
		"plot; the input is left unchanged. Order doubles, as with "
		"lp_to_bp. Requires 0 < low_hz < high_hz.");

	m.def("apply_bilinear",
		[](PZP plot, double sample_rate_hz) {
			tf::apply_bilinear(plot, sample_rate_hz);
			return plot;
		}, nb::arg("plot"), nb::arg("sample_rate_hz"),
		"Map the s-plane constellation to the z-plane via the bilinear "
		"transform, populating z_poles / z_zeros and sample_rate_hz. "
		"Returns a new plot; the input is left unchanged. Every stable "
		"analog pole (Re < 0) maps inside the unit circle.");

		nb::class_<BQ>(m, "BiquadCoefficients",
			"Coefficients for a second-order (biquad) IIR section:\n"
			"  H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)\n\n"
			"Note the a0-normalized convention: a0 is implicitly 1; only "
			"a1 and a2 are stored. Fields are read-write — construct a "
			"list of these and pass to IIRFilter.from_coefficients(list) "
			"to build a filter from raw coefficients (the workflow that "
			"was previously unreachable from Python).")
		.def(nb::init<>(),
		     "Default construct (all coefficients zero).")
		.def(nb::init<double, double, double, double, double>(),
		     nb::arg("b0"), nb::arg("b1"), nb::arg("b2"),
		     nb::arg("a1"), nb::arg("a2"),
		     "Construct with explicit coefficients.")
		.def_rw("b0", &BQ::b0)
		.def_rw("b1", &BQ::b1)
		.def_rw("b2", &BQ::b2)
		.def_rw("a1", &BQ::a1)
		.def_rw("a2", &BQ::a2)
		.def("set_identity", &BQ::set_identity,
		     "Reset to the pass-through filter H(z) = 1 "
		     "(b0=1, all others zero).")
		.def("set_one_pole", &BQ::set_one_pole,
		     nb::arg("pole"), nb::arg("zero"),
		     "Set from a first-order section (single pole, single zero).")
		.def("set_two_pole", &BQ::set_two_pole,
		     nb::arg("pole1"), nb::arg("zero1"),
		     nb::arg("pole2"), nb::arg("zero2"),
		     "Set from a conjugate pair of poles and zeros (second-order "
		     "section).")
		.def("set_from_pole_zero_pair", &BQ::set_from_pole_zero_pair,
		     nb::arg("pz"),
		     "Set from a PoleZeroPair. Dispatches to set_one_pole or "
		     "set_two_pole based on pz.is_single_pole().")
		.def("apply_scale", &BQ::apply_scale, nb::arg("scale"),
		     "Multiply the numerator coefficients (b0, b1, b2) by a "
		     "gain scale factor.")
		.def("response", &BQ::response, nb::arg("normalized_freq"),
		     "Evaluate H(e^{j*2*pi*f}) at the normalized frequency "
		     "f in [0, 0.5], where f = frequency / sample_rate. Returns "
		     "complex.")
		.def("__repr__", [](const BQ& b) {
			return "BiquadCoefficients(b0=" + std::to_string(b.b0)
			     + ", b1=" + std::to_string(b.b1)
			     + ", b2=" + std::to_string(b.b2)
			     + ", a1=" + std::to_string(b.a1)
			     + ", a2=" + std::to_string(b.a2) + ")";
		});
}
