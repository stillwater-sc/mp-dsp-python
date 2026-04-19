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
// project_onto / projection_error dispatch over the dtype string — every
// arithmetic config in types.hpp works as the narrower target because
// double is always the source (wider). See `projection.hpp`'s
// ProjectableOnto<Target, Source> concept for the compile-time check.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/operators.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <mtl/vec/dense_vector.hpp>
#include <sw/dsp/spectral/laplace.hpp>
#include <sw/dsp/spectral/ztransform.hpp>
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
	auto narrowed = sw::dsp::project_onto<T>(src);
	mtl::vec::dense_vector<double> out(narrowed.size());
	for (std::size_t i = 0; i < narrowed.size(); ++i) {
		out[i] = static_cast<double>(narrowed[i]);
	}
	return out;
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
	using mpdsp::half_;
	using mpdsp::p16;
	using tiny_posit_t = sw::universal::posit<8, 2>;

	switch (config) {
	case mpdsp::ArithConfig::reference:
		return src;  // double -> double is identity
	case mpdsp::ArithConfig::gpu_baseline:
		return project_typed<float>(src);
	case mpdsp::ArithConfig::ml_hw:
		return project_typed<half_>(src);
	case mpdsp::ArithConfig::posit_full:
		return project_typed<p16>(src);
	case mpdsp::ArithConfig::tiny_posit:
		return project_typed<tiny_posit_t>(src);
	case mpdsp::ArithConfig::cf24_config:
		return project_typed<cf24>(src);
	case mpdsp::ArithConfig::half_config:
		return project_typed<half_>(src);
	}
	// Unreachable: the switch above is exhaustive over mpdsp::ArithConfig.
	// This return exists only to keep the compiler quiet without reaching for
	// [[noreturn]] or a warning pragma. When a new ArithConfig enumerator
	// lands (e.g. sensor_8bit in #55), add a case above and this line stays
	// inert.
	return src;
}

}  // namespace

void bind_types(nb::module_& m) {
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
}
