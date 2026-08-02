// math_bindings.cpp: numerical utilities from sw::dsp/math/.
//
// Phase 5 / #113 of the bindings-gap roadmap. Wraps:
//   polynomial.hpp        - evaluate_polynomial (Horner), multiply_polynomials
//   quadratic.hpp         - solve_quadratic, solve_quadratic_1, _2
//   elliptic_integrals.hpp - elliptic_K (complete elliptic integral, first kind)
//   root_finder.hpp       - RootFinder<T, MaxDegree> (Laguerre's method)
//
// Bound on double only. Mixed-precision dispatch on these small closed-form
// utilities has marginal value — users doing mixed-precision analysis can
// quantize inputs via mpdsp.adc(x, dtype=...) before feeding them in.
// RootFinder is bound with MaxDegree = 32 (covers all reasonable filter
// design orders with headroom).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <sw/dsp/math/elliptic_integrals.hpp>
#include <sw/dsp/math/polynomial.hpp>
#include <sw/dsp/math/quadratic.hpp>
#include <sw/dsp/math/root_finder.hpp>

#include "_binding_helpers.hpp"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

using mpdsp::bindings::np_f64_ro;

// Compile-time bound on the polynomial degree supported by mpdsp.RootFinder.
// Filter-design workflows rarely exceed order 20; 32 gives comfortable
// headroom without an unreasonable stack footprint per instance
// (~32 * sizeof(complex<double>) * 3 arrays ~= 1.5 KB).
constexpr int MPDSP_ROOT_FINDER_MAX_DEGREE = 32;

using CD    = std::complex<double>;
using RF32  = sw::dsp::RootFinder<double, MPDSP_ROOT_FINDER_MAX_DEGREE>;
using np_c128_ro = nb::ndarray<nb::numpy, const std::complex<double>,
                               nb::ndim<1>, nb::c_contig>;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

static std::vector<double> to_vec_d(np_f64_ro arr) {
	std::vector<double> out(arr.shape(0));
	for (std::size_t i = 0; i < out.size(); ++i) out[i] = arr.data()[i];
	return out;
}

// PyRootFinder: encapsulates coefficient state so the Python interface is
// batch-oriented (set_coefficients / solve / roots) rather than element-wise.
// Degree is inferred from the coefficient array length passed to
// set_coefficients — no need for the caller to track it separately.
class PyRootFinder {
public:
	explicit PyRootFinder() : rf_(), degree_(0) {}

	// Set coefficients from a NumPy complex128 array. Length must be
	// degree + 1; array element i is the coefficient of x^i (ascending
	// order, matching the C++ documentation).
	void set_coefficients(np_c128_ro coeffs) {
		std::size_t n = coeffs.shape(0);
		if (n == 0) {
			throw std::invalid_argument(
				"RootFinder.set_coefficients: coefficient array must not be empty");
		}
		if (n > MPDSP_ROOT_FINDER_MAX_DEGREE + 1) {
			throw std::invalid_argument(
				"RootFinder.set_coefficients: degree " + std::to_string(n - 1) +
				" exceeds max_degree " + std::to_string(MPDSP_ROOT_FINDER_MAX_DEGREE));
		}
		degree_ = static_cast<int>(n - 1);
		for (std::size_t i = 0; i < n; ++i) {
			rf_.coef(static_cast<int>(i)) = coeffs.data()[i];
		}
	}

	void solve(bool polish, bool do_sort) {
		if (degree_ < 1) {
			throw std::runtime_error(
				"RootFinder.solve: no coefficients set — call "
				"set_coefficients() first");
		}
		rf_.solve(degree_, polish, do_sort);
	}

	// Return roots as a fresh NumPy complex128 array of length degree_.
	nb::ndarray<nb::numpy, std::complex<double>> roots() {
		if (degree_ < 1) {
			throw std::runtime_error(
				"RootFinder.roots: no coefficients set / solve not run");
		}
		auto* data = new std::complex<double>[degree_];
		for (int i = 0; i < degree_; ++i) data[i] = rf_.root(i);
		nb::capsule owner(data, [](void* p) noexcept {
			delete[] static_cast<std::complex<double>*>(p);
		});
		std::size_t shape[1] = { static_cast<std::size_t>(degree_) };
		return nb::ndarray<nb::numpy, std::complex<double>>(
			data, 1, shape, owner);
	}

	int degree() const { return degree_; }
	int max_degree() const { return MPDSP_ROOT_FINDER_MAX_DEGREE; }

private:
	RF32 rf_;
	int  degree_;
};

void bind_math(nb::module_& m) {

	// -----------------------------------------------------------------------
	// Polynomial evaluation (Horner's method) — two overloads: real x
	// returns real, complex x returns complex. Real-coeff / complex-coeff
	// dispatch happens in Python: users pass float64 or complex128 arrays.
	// -----------------------------------------------------------------------

	m.def("evaluate_polynomial",
		[](np_f64_ro coeffs, double x) {
			return sw::dsp::evaluate_polynomial(to_vec_d(coeffs), x);
		},
		nb::arg("coeffs"), nb::arg("x"),
		"Evaluate a real-coefficient polynomial at a real point via Horner's "
		"method. `coeffs` is a NumPy float64 array in ascending order of "
		"power: p(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...");

	m.def("evaluate_polynomial",
		[](np_f64_ro coeffs, CD x) {
			return sw::dsp::evaluate_polynomial<double>(to_vec_d(coeffs), x);
		},
		nb::arg("coeffs"), nb::arg("x"),
		"Evaluate a real-coefficient polynomial at a complex point. "
		"Same coefficient convention as the real-x overload. Returns "
		"complex128.");

	m.def("multiply_polynomials",
		[](np_f64_ro a, np_f64_ro b) {
			auto result = sw::dsp::multiply_polynomials(to_vec_d(a), to_vec_d(b));
			// Return as NumPy float64 array.
			std::size_t n = result.size();
			auto* data = new double[n];
			for (std::size_t i = 0; i < n; ++i) data[i] = result[i];
			nb::capsule owner(data, [](void* p) noexcept {
				delete[] static_cast<double*>(p);
			});
			std::size_t shape[1] = { n };
			return nb::ndarray<nb::numpy, double>(data, 1, shape, owner);
		},
		nb::arg("a"), nb::arg("b"),
		"Multiply two polynomials — coefficient-vector convolution. "
		"Result has degree = deg(a) + deg(b). Either input empty gives "
		"an empty result. Equivalent to numpy.convolve(a, b, mode='full') "
		"for the coefficient-in-ascending-order convention.");

	// -----------------------------------------------------------------------
	// Quadratic solver — both roots as complex (may be real).
	// -----------------------------------------------------------------------

	m.def("solve_quadratic",
		[](double a, double b, double c) {
			if (a == 0.0) {
				throw std::invalid_argument(
					"solve_quadratic: leading coefficient a must be nonzero "
					"(equation is not quadratic when a=0)");
			}
			return sw::dsp::solve_quadratic<double>(a, b, c);
		},
		nb::arg("a"), nb::arg("b"), nb::arg("c"),
		"Return both roots of a*x^2 + b*x + c = 0 as a tuple of complex "
		"numbers. Real roots are returned with zero imaginary part.");

	m.def("solve_quadratic_1",
		[](double a, double b, double c) {
			if (a == 0.0) {
				throw std::invalid_argument(
					"solve_quadratic_1: leading coefficient a must be nonzero");
			}
			return sw::dsp::solve_quadratic_1<double>(a, b, c);
		},
		nb::arg("a"), nb::arg("b"), nb::arg("c"),
		"Root with the positive discriminant sign: "
		"(-b + sqrt(b^2 - 4ac)) / (2a).");

	m.def("solve_quadratic_2",
		[](double a, double b, double c) {
			if (a == 0.0) {
				throw std::invalid_argument(
					"solve_quadratic_2: leading coefficient a must be nonzero");
			}
			return sw::dsp::solve_quadratic_2<double>(a, b, c);
		},
		nb::arg("a"), nb::arg("b"), nb::arg("c"),
		"Root with the negative discriminant sign: "
		"(-b - sqrt(b^2 - 4ac)) / (2a).");

	// -----------------------------------------------------------------------
	// Complete elliptic integral of the first kind.
	// -----------------------------------------------------------------------

	m.def("elliptic_K",
		[](double k) {
			// Domain: k must be in [0, 1). K(k) diverges at k=1.
			if (!(k >= 0.0 && k < 1.0)) {
				throw std::invalid_argument(
					"elliptic_K: modulus k must be in [0, 1) (got "
					+ std::to_string(k) + "); K(1) is unbounded");
			}
			return sw::dsp::elliptic_K<double>(k);
		},
		nb::arg("k"),
		"Complete elliptic integral of the first kind K(k) via the "
		"arithmetic-geometric mean (AGM) iteration. Modulus k must be "
		"in [0, 1). Peak error less than 2e-16. Used by Elliptic (Cauer) "
		"filter design.");

	// -----------------------------------------------------------------------
	// RootFinder — Laguerre's method for complex polynomial roots.
	// -----------------------------------------------------------------------

	nb::class_<PyRootFinder>(m, "RootFinder",
			"Complex polynomial root finder via Laguerre's method with "
			"deflation and optional polishing. Supports polynomials up to "
			"degree 32 (compile-time bound; passing a longer coefficient "
			"array raises).\n\n"
			"Usage:\n"
			"  rf = mpdsp.RootFinder()\n"
			"  rf.set_coefficients(np.array([-1, 0, 1], dtype=complex128))  # x^2 - 1\n"
			"  rf.solve()\n"
			"  print(rf.roots())  # [-1+0j, 1+0j]")
		.def(nb::init<>(),
		     "Construct with all coefficients zero-initialized. Call "
		     "set_coefficients() before solve().")
		.def("set_coefficients", &PyRootFinder::set_coefficients,
		     nb::arg("coeffs"),
		     "Set polynomial coefficients from a NumPy complex128 array. "
		     "Length is degree+1; element i is the coefficient of x^i "
		     "(ascending order). Degree is inferred from array length. "
		     "Maximum degree is 32.")
		.def("solve", &PyRootFinder::solve,
		     nb::arg("polish") = true, nb::arg("sort") = true,
		     "Find all roots of the polynomial set via set_coefficients(). "
		     "polish=True (default) refines each root using the original "
		     "(un-deflated) polynomial for accuracy. sort=True (default) "
		     "orders roots by descending imaginary part.")
		.def("roots", &PyRootFinder::roots,
		     "Return the `degree` roots as a NumPy complex128 array. "
		     "Requires solve() to have been called after set_coefficients().")
		.def_prop_ro("degree", &PyRootFinder::degree,
		     "Current polynomial degree (0 before set_coefficients()).")
		.def_prop_ro("max_degree", &PyRootFinder::max_degree,
		     "Compile-time bound on the polynomial degree (32).");
}
