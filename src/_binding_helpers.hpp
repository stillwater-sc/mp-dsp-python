#pragma once
// _binding_helpers.hpp: shared nanobind + MTL marshalling utilities.
//
// Factored out of conditioning_bindings.cpp and estimation_bindings.cpp
// (where identical copies had been duplicating by hand) so new binding
// files can pick up the same idioms without re-deriving them:
//
//   np_f64, np_f64_ro            — 1D float64 NumPy views (c_contig for ro)
//   np_f64_2d, np_f64_2d_ro      — 2D float64 NumPy views (c_contig for ro)
//   make_f64_array(n)            — owning output buffer of length n
//   make_f64_2d_array(rows, cols)— owning 2D output buffer
//   mat_to_numpy<T>              — dense2D<T> -> NumPy float64 (copy, casts)
//   numpy_to_mat<T>              — NumPy -> existing dense2D<T> (shape-checked)
//   numpy_to_mat_fresh<T>        — NumPy -> freshly-allocated dense2D<T>
//   vec_to_numpy<T>              — dense_vector<T> -> NumPy float64 (copy, casts)
//   numpy_to_vec<T>              — NumPy -> existing dense_vector<T>
//   make_impl_for_dtype<...>     — construct one Impl<T> per ArithConfig
//
// Everything lives in a private `mpdsp::bindings` namespace. Include this
// header from each binding translation unit that needs it. There is no
// corresponding .cpp — the helpers are intentionally inline template code
// so multiple TUs can instantiate them independently.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>

#include "types.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>  // for std::forward used in make_impl_for_dtype

namespace mpdsp::bindings {

namespace nb = ::nanobind;

// ---------------------------------------------------------------------------
// NumPy typedefs. c_contig on the read-only variants forces nanobind to copy
// non-contiguous inputs transparently so .data() walks the buffer linearly.
// ---------------------------------------------------------------------------

using np_f64       = nb::ndarray<nb::numpy, double>;
using np_f64_ro    = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
using np_f64_2d    = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using np_f64_2d_ro = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>;

// ---------------------------------------------------------------------------
// Owning output-buffer builders. Ownership is transferred to the nb::capsule
// only after the capsule constructor succeeds, so a throw on the capsule
// path doesn't leak the buffer.
// ---------------------------------------------------------------------------

inline np_f64 make_f64_array(std::size_t n, double*& out_ptr) {
	auto buf = std::unique_ptr<double[]>(new double[n]);
	double* data = buf.get();
	nb::capsule owner(data, [](void* p) noexcept {
		delete[] static_cast<double*>(p);
	});
	buf.release();
	out_ptr = data;
	std::size_t shape[1] = { n };
	return np_f64(data, 1, shape, owner);
}

inline np_f64_2d make_f64_2d_array(std::size_t rows, std::size_t cols,
                                   double*& out_ptr) {
	// Reject oversized dimensions that would wrap rows * cols past SIZE_MAX
	// and silently produce a too-small allocation. Image generators take
	// rows/cols from Python, so adversarial or mistyped values can reach
	// here directly.
	if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols) {
		throw std::overflow_error(
			"make_f64_2d_array: rows * cols overflows size_t");
	}
	std::size_t total = rows * cols;
	auto buf = std::unique_ptr<double[]>(new double[total]);
	double* data = buf.get();
	nb::capsule owner(data, [](void* p) noexcept {
		delete[] static_cast<double*>(p);
	});
	buf.release();
	out_ptr = data;
	std::size_t shape[2] = { rows, cols };
	return np_f64_2d(data, 2, shape, owner);
}

// ---------------------------------------------------------------------------
// 1D vector marshalling. T <-> double via static_cast at the boundary.
// ---------------------------------------------------------------------------

template <typename T>
inline np_f64 vec_to_numpy(const mtl::vec::dense_vector<T>& v) {
	std::size_t n = v.size();
	double* out_ptr = nullptr;
	auto arr = make_f64_array(n, out_ptr);
	for (std::size_t i = 0; i < n; ++i) {
		out_ptr[i] = static_cast<double>(v[i]);
	}
	return arr;
}

template <typename T>
inline void numpy_to_vec(np_f64_ro src, mtl::vec::dense_vector<T>& dst,
                         const char* name) {
	if (src.shape(0) != dst.size()) {
		throw std::invalid_argument(
			std::string(name) + ": size mismatch (expected " +
			std::to_string(dst.size()) + ", got " +
			std::to_string(src.shape(0)) + ")");
	}
	const double* data = src.data();
	for (std::size_t i = 0; i < dst.size(); ++i) {
		dst[i] = static_cast<T>(data[i]);
	}
}

// ---------------------------------------------------------------------------
// 2D matrix marshalling. Walks (row, col) pairs with a shared row-major
// linear index — MTL's dense2D defaults to row-major orientation, matching
// NumPy C-order.
// ---------------------------------------------------------------------------

template <typename T>
inline np_f64_2d mat_to_numpy(const mtl::mat::dense2D<T>& m) {
	std::size_t rows = m.num_rows();
	std::size_t cols = m.num_cols();
	double* out_ptr = nullptr;
	auto arr = make_f64_2d_array(rows, cols, out_ptr);
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			out_ptr[r * cols + c] = static_cast<double>(m(r, c));
		}
	}
	return arr;
}

template <typename T>
inline void numpy_to_mat(np_f64_2d_ro src, mtl::mat::dense2D<T>& dst,
                         const char* name) {
	if (src.shape(0) != dst.num_rows() || src.shape(1) != dst.num_cols()) {
		throw std::invalid_argument(
			std::string(name) + ": shape mismatch (expected " +
			std::to_string(dst.num_rows()) + "x" +
			std::to_string(dst.num_cols()) + ", got " +
			std::to_string(src.shape(0)) + "x" +
			std::to_string(src.shape(1)) + ")");
	}
	std::size_t cols = dst.num_cols();
	const double* data = src.data();
	for (std::size_t r = 0; r < dst.num_rows(); ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			dst(r, c) = static_cast<T>(data[r * cols + c]);
		}
	}
}

// Convenience: convert a NumPy 2D array to a freshly-allocated dense2D<T>.
// Use when the caller doesn't have a pre-allocated matrix to copy into.
template <typename T>
inline mtl::mat::dense2D<T> numpy_to_mat_fresh(np_f64_2d_ro src) {
	std::size_t rows = src.shape(0);
	std::size_t cols = src.shape(1);
	mtl::mat::dense2D<T> dst(rows, cols);
	const double* data = src.data();
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			dst(r, c) = static_cast<T>(data[r * cols + c]);
		}
	}
	return dst;
}

// ---------------------------------------------------------------------------
// Arithmetic-config dispatcher. Given a class template Impl<T> deriving from
// Base, construct the right instantiation for the requested ArithConfig and
// return a unique_ptr<Base>. Variadic Args... forwards constructor arguments
// to the T-typed Impl — used by e.g. KalmanFilter(state_dim, meas_dim, ...).
//
// A future ArithConfig enumerator that's added without extending the switch
// raises instead of silently dispatching to double.
// ---------------------------------------------------------------------------

template <template<class> class Impl, class Base, class... Args>
inline std::unique_ptr<Base>
make_impl_for_dtype(mpdsp::ArithConfig config, const char* cls, Args&&... args) {
	using mpdsp::ArithConfig;
	using mpdsp::cf24;
	using mpdsp::half_;
	using mpdsp::p32;
	using tiny_posit_t = sw::universal::posit<8, 2>;
	switch (config) {
	case ArithConfig::reference:
		return std::make_unique<Impl<double>>(std::forward<Args>(args)...);
	case ArithConfig::gpu_baseline:
		return std::make_unique<Impl<float>>(std::forward<Args>(args)...);
	case ArithConfig::ml_hw:
		return std::make_unique<Impl<half_>>(std::forward<Args>(args)...);
	case ArithConfig::cf24_config:
		return std::make_unique<Impl<cf24>>(std::forward<Args>(args)...);
	case ArithConfig::half_config:
		return std::make_unique<Impl<half_>>(std::forward<Args>(args)...);
	case ArithConfig::posit_full:
		return std::make_unique<Impl<p32>>(std::forward<Args>(args)...);
	case ArithConfig::tiny_posit:
		return std::make_unique<Impl<tiny_posit_t>>(std::forward<Args>(args)...);
	}
	throw std::invalid_argument(std::string(cls) + ": unsupported ArithConfig");
}

} // namespace mpdsp::bindings
