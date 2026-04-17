// image_bindings.cpp: 2D image processing bindings (Phase 6 scaffold).
//
// Establishes patterns for Phase 6 before replicating across the full
// generator/processing/morphology/io surface:
//
//   - Free-function bindings returning NumPy float64 2D (for generators).
//   - Free-function bindings with a dtype dispatcher for mixed-precision
//     processing (for convolve2d and the other *_typed workers that follow).
//   - BorderMode string parsing at the Python boundary.
//
// Starts with three generators (checkerboard, gaussian_blob,
// gradient_horizontal) and one dtype-dispatched processor (convolve2d).
// The rest of the phase will reuse this file's helpers.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <sw/dsp/image/convolve2d.hpp>
#include <sw/dsp/image/generators.hpp>
#include <sw/dsp/image/image.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

using mpdsp::bindings::np_f64_2d;
using mpdsp::bindings::np_f64_2d_ro;
using mpdsp::bindings::mat_to_numpy;
using mpdsp::bindings::numpy_to_mat_fresh;

namespace {

// ---------------------------------------------------------------------------
// BorderMode string parsing. Matches sw::dsp::BorderMode enumerators; the
// default "reflect_101" mirrors the upstream default.
// ---------------------------------------------------------------------------

static sw::dsp::BorderMode parse_border(const std::string& name) {
	using sw::dsp::BorderMode;
	if (name == "constant")    return BorderMode::constant;
	if (name == "replicate")   return BorderMode::replicate;
	if (name == "reflect")     return BorderMode::reflect;
	if (name == "reflect_101") return BorderMode::reflect_101;
	if (name == "wrap")        return BorderMode::wrap;
	throw std::invalid_argument(
		"Unknown border mode: " + name +
		" (expected constant, replicate, reflect, reflect_101, or wrap)");
}

// Validate that image dimensions from the Python boundary are positive.
// Zero-sized generators are well-defined but rarely useful; reject early
// to avoid constructing empty NumPy arrays that surprise callers.
static void check_dims(std::size_t rows, std::size_t cols, const char* name) {
	if (rows == 0 || cols == 0) {
		throw std::invalid_argument(
			std::string(name) + ": rows and cols must be positive");
	}
}

// ---------------------------------------------------------------------------
// Dtype dispatcher for free-function processors.
//
// Free functions like convolve2d can't use make_impl_for_dtype<Impl, Base>
// (that one constructs a class Impl<T>). Instead, a per-function dispatcher
// instantiates the upstream function template at the requested T, ferries
// NumPy input through dense2D<T>, runs, and ships the result back as
// NumPy float64. Future processing bindings use the same shape.
// ---------------------------------------------------------------------------

template <typename T>
static np_f64_2d convolve2d_typed(np_f64_2d_ro image, np_f64_2d_ro kernel,
                                   sw::dsp::BorderMode border, double pad) {
	auto in_mat  = numpy_to_mat_fresh<T>(image);
	auto kern    = numpy_to_mat_fresh<T>(kernel);
	auto result  = sw::dsp::convolve2d<T, T>(in_mat, kern, border, static_cast<T>(pad));
	return mat_to_numpy(result);
}

static np_f64_2d convolve2d_dispatch(np_f64_2d_ro image, np_f64_2d_ro kernel,
                                     const std::string& border_name,
                                     double pad,
                                     const std::string& dtype) {
	auto config = mpdsp::parse_config(dtype);
	auto border = parse_border(border_name);
	using mpdsp::ArithConfig;
	using mpdsp::cf24;
	using mpdsp::half_;
	using mpdsp::p32;
	using tiny_posit_t = sw::universal::posit<8, 2>;
	switch (config) {
	case ArithConfig::reference:
		return convolve2d_typed<double>(image, kernel, border, pad);
	case ArithConfig::gpu_baseline:
		return convolve2d_typed<float>(image, kernel, border, pad);
	case ArithConfig::ml_hw:
		return convolve2d_typed<half_>(image, kernel, border, pad);
	case ArithConfig::cf24_config:
		return convolve2d_typed<cf24>(image, kernel, border, pad);
	case ArithConfig::half_config:
		return convolve2d_typed<half_>(image, kernel, border, pad);
	case ArithConfig::posit_full:
		return convolve2d_typed<p32>(image, kernel, border, pad);
	case ArithConfig::tiny_posit:
		return convolve2d_typed<tiny_posit_t>(image, kernel, border, pad);
	}
	throw std::invalid_argument("convolve2d: unsupported ArithConfig");
}

} // namespace

void bind_image(nb::module_& m) {
	// =======================================================================
	// Generators — return NumPy float64 2D. No dtype parameter: all
	// upstream generators are mtl::mat::dense2D<T> for arbitrary T, but
	// Python-facing generators always produce float64 so they compose with
	// the dtype-dispatched processors that follow.
	// =======================================================================

	m.def("checkerboard",
		[](std::size_t rows, std::size_t cols, std::size_t block_size,
		   double low, double high) {
			check_dims(rows, cols, "checkerboard");
			auto img = sw::dsp::checkerboard<double>(rows, cols, block_size,
			                                         low, high);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"), nb::arg("block_size"),
		nb::arg("low") = 0.0, nb::arg("high") = 1.0,
		"Checkerboard of alternating `low` / `high` blocks, "
		"`block_size` pixels per square.");

	m.def("gaussian_blob",
		[](std::size_t rows, std::size_t cols, double sigma,
		   double amplitude) {
			check_dims(rows, cols, "gaussian_blob");
			if (!(sigma > 0.0)) {
				throw std::invalid_argument(
					"gaussian_blob: sigma must be positive");
			}
			auto img = sw::dsp::gaussian_blob<double>(rows, cols, sigma,
			                                           amplitude);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"), nb::arg("sigma"),
		nb::arg("amplitude") = 1.0,
		"2D Gaussian centred on the image with standard deviation `sigma`.");

	m.def("gradient_horizontal",
		[](std::size_t rows, std::size_t cols, double start, double end) {
			check_dims(rows, cols, "gradient_horizontal");
			auto img = sw::dsp::gradient_horizontal<double>(rows, cols,
			                                                start, end);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"),
		nb::arg("start") = 0.0, nb::arg("end") = 1.0,
		"Linear horizontal gradient from `start` (left) to `end` (right).");

	// =======================================================================
	// Processing — dtype-dispatched. Convolve2d is the scaffold; all other
	// processors in this phase (separable_filter, gaussian_blur, box_blur,
	// sobel_*, prewitt_*, canny, ...) follow this pattern.
	// =======================================================================

	m.def("convolve2d",
		[](np_f64_2d_ro image, np_f64_2d_ro kernel,
		   const std::string& border, double pad, const std::string& dtype) {
			if (image.shape(0) == 0 || image.shape(1) == 0) {
				throw std::invalid_argument(
					"convolve2d: image must have non-zero dimensions");
			}
			if (kernel.shape(0) == 0 || kernel.shape(1) == 0) {
				throw std::invalid_argument(
					"convolve2d: kernel must have non-zero dimensions");
			}
			return convolve2d_dispatch(image, kernel, border, pad, dtype);
		},
		nb::arg("image"), nb::arg("kernel"),
		nb::arg("border") = "reflect_101", nb::arg("pad") = 0.0,
		nb::arg("dtype") = "reference",
		"2D spatial correlation. `border` is one of constant, replicate, "
		"reflect, reflect_101, or wrap; `pad` is the fill value for "
		"border='constant'. `dtype` selects the internal arithmetic — see "
		"available_dtypes().");
}
