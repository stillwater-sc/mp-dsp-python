// image_bindings.cpp: 2D image processing bindings (Phase 6).
//
// Patterns established:
//   - Free-function generators return NumPy float64 2D (no dtype).
//   - Free-function processors with dtype dispatch via per-function
//     template + hand-written ArithConfig switch (see convolve2d).
//   - BorderMode string parsing at the Python boundary.
//
// Covered so far: all 17 generators, threshold, add_noise, convolve2d.
// Still to land: separable/blur/edge detection, morphology + elements,
// multi-channel, I/O.

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

	m.def("gradient_vertical",
		[](std::size_t rows, std::size_t cols, double start, double end) {
			check_dims(rows, cols, "gradient_vertical");
			auto img = sw::dsp::gradient_vertical<double>(rows, cols,
			                                              start, end);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"),
		nb::arg("start") = 0.0, nb::arg("end") = 1.0,
		"Linear vertical gradient from `start` (top) to `end` (bottom).");

	m.def("gradient_radial",
		[](std::size_t rows, std::size_t cols,
		   double center_val, double edge_val) {
			check_dims(rows, cols, "gradient_radial");
			auto img = sw::dsp::gradient_radial<double>(rows, cols,
			                                            center_val, edge_val);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"),
		nb::arg("center_val") = 1.0, nb::arg("edge_val") = 0.0,
		"Radial gradient: `center_val` at the image center linearly "
		"interpolated to `edge_val` at the corners.");

	m.def("stripes_horizontal",
		[](std::size_t rows, std::size_t cols, std::size_t stripe_width,
		   double low, double high) {
			check_dims(rows, cols, "stripes_horizontal");
			auto img = sw::dsp::stripes_horizontal<double>(
				rows, cols, stripe_width, low, high);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"), nb::arg("stripe_width"),
		nb::arg("low") = 0.0, nb::arg("high") = 1.0,
		"Alternating horizontal stripes of `stripe_width` rows each.");

	m.def("stripes_vertical",
		[](std::size_t rows, std::size_t cols, std::size_t stripe_width,
		   double low, double high) {
			check_dims(rows, cols, "stripes_vertical");
			auto img = sw::dsp::stripes_vertical<double>(
				rows, cols, stripe_width, low, high);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"), nb::arg("stripe_width"),
		nb::arg("low") = 0.0, nb::arg("high") = 1.0,
		"Alternating vertical stripes of `stripe_width` columns each.");

	m.def("grid",
		[](std::size_t rows, std::size_t cols, std::size_t spacing,
		   double background, double line) {
			check_dims(rows, cols, "grid");
			auto img = sw::dsp::grid<double>(rows, cols, spacing,
			                                 background, line);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"), nb::arg("spacing"),
		nb::arg("background") = 0.0, nb::arg("line") = 1.0,
		"Thin grid lines at every `spacing` pixels against a uniform "
		"background.");

	m.def("circle",
		[](std::size_t rows, std::size_t cols, std::size_t radius,
		   double foreground, double background) {
			check_dims(rows, cols, "circle");
			auto img = sw::dsp::circle<double>(rows, cols, radius,
			                                   foreground, background);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"), nb::arg("radius"),
		nb::arg("foreground") = 1.0, nb::arg("background") = 0.0,
		"Filled circle of `radius` pixels centred on the image.");

	m.def("rectangle",
		[](std::size_t rows, std::size_t cols,
		   std::size_t y, std::size_t x, std::size_t h, std::size_t w,
		   double foreground, double background) {
			check_dims(rows, cols, "rectangle");
			auto img = sw::dsp::rectangle<double>(rows, cols, y, x, h, w,
			                                      foreground, background);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"),
		nb::arg("y"), nb::arg("x"), nb::arg("h"), nb::arg("w"),
		nb::arg("foreground") = 1.0, nb::arg("background") = 0.0,
		"Filled rectangle with top-left corner at (y, x) and dimensions "
		"(h, w). Pixels outside the rectangle get `background`.");

	m.def("zone_plate",
		[](std::size_t rows, std::size_t cols, double max_freq) {
			check_dims(rows, cols, "zone_plate");
			auto img = sw::dsp::zone_plate<double>(rows, cols, max_freq);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"), nb::arg("max_freq") = 0.0,
		"Zone plate (chirp image) — radial frequency that sweeps from 0 at "
		"the center to `max_freq` (cycles/pixel) at the corners. "
		"`max_freq = 0` (default) auto-selects half-Nyquist.");

	m.def("uniform_noise_image",
		[](std::size_t rows, std::size_t cols,
		   double low, double high, unsigned seed) {
			check_dims(rows, cols, "uniform_noise_image");
			auto img = sw::dsp::uniform_noise_image<double>(
				rows, cols, low, high, seed);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"),
		nb::arg("low") = 0.0, nb::arg("high") = 1.0, nb::arg("seed") = 0u,
		"Uniform-distribution noise in [low, high].");

	m.def("gaussian_noise_image",
		[](std::size_t rows, std::size_t cols,
		   double mean, double stddev, unsigned seed) {
			check_dims(rows, cols, "gaussian_noise_image");
			if (!(stddev >= 0.0)) {
				throw std::invalid_argument(
					"gaussian_noise_image: stddev must be non-negative");
			}
			auto img = sw::dsp::gaussian_noise_image<double>(
				rows, cols, mean, stddev, seed);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"),
		nb::arg("mean") = 0.0, nb::arg("stddev") = 1.0, nb::arg("seed") = 0u,
		"Gaussian-distribution noise with the given mean and stddev.");

	m.def("salt_and_pepper",
		[](std::size_t rows, std::size_t cols, double density,
		   double low, double high, unsigned seed) {
			check_dims(rows, cols, "salt_and_pepper");
			if (!(density >= 0.0) || density > 1.0) {
				throw std::invalid_argument(
					"salt_and_pepper: density must be in [0, 1]");
			}
			auto img = sw::dsp::salt_and_pepper<double>(
				rows, cols, density, low, high, seed);
			return mat_to_numpy(img);
		},
		nb::arg("rows"), nb::arg("cols"),
		nb::arg("density") = 0.05,
		nb::arg("low") = 0.0, nb::arg("high") = 1.0, nb::arg("seed") = 0u,
		"Salt-and-pepper noise: `density` fraction of pixels randomly "
		"flipped to `low` (pepper) or `high` (salt); the rest stay at "
		"the midpoint (low+high)/2.");

	m.def("add_noise",
		[](np_f64_2d_ro image, double stddev, unsigned seed) {
			if (image.shape(0) == 0 || image.shape(1) == 0) {
				throw std::invalid_argument(
					"add_noise: image must have non-zero dimensions");
			}
			if (!(stddev >= 0.0)) {
				throw std::invalid_argument(
					"add_noise: stddev must be non-negative");
			}
			auto in_mat = numpy_to_mat_fresh<double>(image);
			auto out    = sw::dsp::add_noise(in_mat, stddev, seed);
			return mat_to_numpy(out);
		},
		nb::arg("image"), nb::arg("stddev"), nb::arg("seed") = 42u,
		"Return `image` with i.i.d. Gaussian noise of the given stddev "
		"added to each pixel.");

	m.def("threshold",
		[](np_f64_2d_ro image, double thresh, double low, double high) {
			if (image.shape(0) == 0 || image.shape(1) == 0) {
				throw std::invalid_argument(
					"threshold: image must have non-zero dimensions");
			}
			auto in_mat = numpy_to_mat_fresh<double>(image);
			auto out    = sw::dsp::threshold(in_mat, thresh, low, high);
			return mat_to_numpy(out);
		},
		nb::arg("image"), nb::arg("thresh"),
		nb::arg("low") = 0.0, nb::arg("high") = 1.0,
		"Binary threshold: pixels above `thresh` become `high`, pixels "
		"at or below become `low`.");

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
