// image_bindings.cpp: 2D image processing bindings (Phase 6).
//
// Patterns established:
//   - Free-function generators return NumPy float64 2D (no dtype).
//   - Free-function processors with dtype dispatch via per-function
//     template + hand-written ArithConfig switch (see convolve2d).
//   - BorderMode string parsing at the Python boundary.
//
// Covered so far: 14 generators (checkerboard, stripes_horizontal,
// stripes_vertical, grid, gradient_horizontal/vertical/radial,
// gaussian_blob, circle, rectangle, zone_plate, uniform_noise_image,
// gaussian_noise_image, salt_and_pepper), plus threshold, add_noise,
// and convolve2d.
// Still to land: separable/blur/edge detection, morphology + elements,
// multi-channel, I/O.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <sw/dsp/image/convolve2d.hpp>
#include <sw/dsp/image/edge.hpp>
#include <sw/dsp/image/generators.hpp>
#include <sw/dsp/image/image.hpp>
#include <sw/dsp/image/morphology.hpp>
#include <sw/dsp/image/separable.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

using mpdsp::bindings::np_f64_ro;
using mpdsp::bindings::np_f64_2d;
using mpdsp::bindings::np_f64_2d_ro;
using mpdsp::bindings::np_bool_2d;
using mpdsp::bindings::np_bool_2d_ro;
using mpdsp::bindings::mat_to_numpy;
using mpdsp::bindings::bool_mat_to_numpy;
using mpdsp::bindings::numpy_to_mat_fresh;
using mpdsp::bindings::numpy_to_bool_mat_fresh;
using mpdsp::bindings::numpy_to_vec_fresh;
using mpdsp::bindings::dispatch_dtype_fn;

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

// Processors share the pattern:
//
//   auto config = mpdsp::parse_config(dtype);
//   return dispatch_dtype_fn(config, "<name>", [&]<typename T>() {
//       auto m = numpy_to_mat_fresh<T>(image);
//       return mat_to_numpy(sw::dsp::<fn><T>(m, ...));
//   });
//
// The lambda body is where the per-function work lives; the surrounding
// dispatcher machinery is shared in _binding_helpers.hpp.

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
		nb::arg("low") = 0.0, nb::arg("high") = 1.0, nb::arg("seed") = 42u,
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
		nb::arg("mean") = 0.0, nb::arg("stddev") = 1.0, nb::arg("seed") = 42u,
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
		nb::arg("low") = 0.0, nb::arg("high") = 1.0, nb::arg("seed") = 42u,
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
		"Binary threshold: pixels greater than or equal to `thresh` "
		"become `high`; pixels strictly below become `low`.");

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
			auto config = mpdsp::parse_config(dtype);
			auto bm = parse_border(border);
			return dispatch_dtype_fn(config, "convolve2d", [&]<typename T>() {
				auto in_mat = numpy_to_mat_fresh<T>(image);
				auto kern   = numpy_to_mat_fresh<T>(kernel);
				auto result = sw::dsp::convolve2d<T, T>(
					in_mat, kern, bm, static_cast<T>(pad));
				return mat_to_numpy(result);
			});
		},
		nb::arg("image"), nb::arg("kernel"),
		nb::arg("border") = "reflect_101", nb::arg("pad") = 0.0,
		nb::arg("dtype") = "reference",
		"2D spatial correlation. `border` is one of constant, replicate, "
		"reflect, reflect_101, or wrap; `pad` is the fill value for "
		"border='constant'. `dtype` selects the internal arithmetic — see "
		"available_dtypes().");

	m.def("separable_filter",
		[](np_f64_2d_ro image, np_f64_ro row_kernel, np_f64_ro col_kernel,
		   const std::string& border, double pad, const std::string& dtype) {
			if (image.shape(0) == 0 || image.shape(1) == 0) {
				throw std::invalid_argument(
					"separable_filter: image must have non-zero dimensions");
			}
			if (row_kernel.shape(0) == 0 || col_kernel.shape(0) == 0) {
				throw std::invalid_argument(
					"separable_filter: row_kernel and col_kernel must be non-empty");
			}
			auto config = mpdsp::parse_config(dtype);
			auto bm = parse_border(border);
			return dispatch_dtype_fn(config, "separable_filter",
			                          [&]<typename T>() {
				auto in_mat = numpy_to_mat_fresh<T>(image);
				auto rk     = numpy_to_vec_fresh<T>(row_kernel);
				auto ck     = numpy_to_vec_fresh<T>(col_kernel);
				auto result = sw::dsp::separable_filter<T, T>(
					in_mat, rk, ck, bm, static_cast<T>(pad));
				return mat_to_numpy(result);
			});
		},
		nb::arg("image"), nb::arg("row_kernel"), nb::arg("col_kernel"),
		nb::arg("border") = "reflect_101", nb::arg("pad") = 0.0,
		nb::arg("dtype") = "reference",
		"Apply a row kernel then a column kernel (separable 2D filter). "
		"Equivalent to convolve2d with an outer-product kernel but cheaper "
		"for K rows * L cols -> O(K+L) per pixel instead of O(K*L).");

	m.def("gaussian_blur",
		[](np_f64_2d_ro image, double sigma, std::size_t radius,
		   const std::string& border, const std::string& dtype) {
			if (image.shape(0) == 0 || image.shape(1) == 0) {
				throw std::invalid_argument(
					"gaussian_blur: image must have non-zero dimensions");
			}
			if (!(sigma > 0.0)) {
				throw std::invalid_argument(
					"gaussian_blur: sigma must be positive");
			}
			auto config = mpdsp::parse_config(dtype);
			auto bm = parse_border(border);
			return dispatch_dtype_fn(config, "gaussian_blur",
			                          [&]<typename T>() {
				auto in_mat = numpy_to_mat_fresh<T>(image);
				auto result = sw::dsp::gaussian_blur<T>(in_mat, sigma, radius, bm);
				return mat_to_numpy(result);
			});
		},
		nb::arg("image"), nb::arg("sigma"), nb::arg("radius") = std::size_t{0},
		nb::arg("border") = "reflect_101", nb::arg("dtype") = "reference",
		"Separable Gaussian blur. `radius=0` auto-selects a radius that "
		"captures most of the Gaussian tail (usually ceil(3*sigma)).");

	m.def("box_blur",
		[](np_f64_2d_ro image, std::size_t size,
		   const std::string& border, const std::string& dtype) {
			if (image.shape(0) == 0 || image.shape(1) == 0) {
				throw std::invalid_argument(
					"box_blur: image must have non-zero dimensions");
			}
			if (size == 0) {
				throw std::invalid_argument(
					"box_blur: size must be positive");
			}
			auto config = mpdsp::parse_config(dtype);
			auto bm = parse_border(border);
			return dispatch_dtype_fn(config, "box_blur",
			                          [&]<typename T>() {
				auto in_mat = numpy_to_mat_fresh<T>(image);
				auto result = sw::dsp::box_blur<T>(in_mat, size, bm);
				return mat_to_numpy(result);
			});
		},
		nb::arg("image"), nb::arg("size"),
		nb::arg("border") = "reflect_101", nb::arg("dtype") = "reference",
		"Box-average blur with an `size x size` uniform kernel.");

	// =======================================================================
	// Edge detection — Sobel / Prewitt gradient components, their magnitude,
	// and Canny edge maps. All dtype-dispatched via the same lambda pattern.
	// =======================================================================

	auto bind_edge_op = [&m](const char* name, auto op) {
		m.def(name,
			[name, op](np_f64_2d_ro image, const std::string& border,
			            const std::string& dtype) {
				if (image.shape(0) == 0 || image.shape(1) == 0) {
					throw std::invalid_argument(
						std::string(name) + ": image must have non-zero dimensions");
				}
				auto config = mpdsp::parse_config(dtype);
				auto bm = parse_border(border);
				return dispatch_dtype_fn(config, name, [&]<typename T>() {
					auto in_mat = numpy_to_mat_fresh<T>(image);
					return mat_to_numpy(op.template operator()<T>(in_mat, bm));
				});
			},
			nb::arg("image"),
			nb::arg("border") = "reflect_101",
			nb::arg("dtype") = "reference");
	};

	bind_edge_op("sobel_x", []<typename T>(const mtl::mat::dense2D<T>& img,
	                                        sw::dsp::BorderMode bm) {
		return sw::dsp::sobel_x<T>(img, bm);
	});
	bind_edge_op("sobel_y", []<typename T>(const mtl::mat::dense2D<T>& img,
	                                        sw::dsp::BorderMode bm) {
		return sw::dsp::sobel_y<T>(img, bm);
	});
	bind_edge_op("prewitt_x", []<typename T>(const mtl::mat::dense2D<T>& img,
	                                          sw::dsp::BorderMode bm) {
		return sw::dsp::prewitt_x<T>(img, bm);
	});
	bind_edge_op("prewitt_y", []<typename T>(const mtl::mat::dense2D<T>& img,
	                                          sw::dsp::BorderMode bm) {
		return sw::dsp::prewitt_y<T>(img, bm);
	});

	m.def("gradient_magnitude",
		[](np_f64_2d_ro gx, np_f64_2d_ro gy, const std::string& dtype) {
			if (gx.shape(0) != gy.shape(0) || gx.shape(1) != gy.shape(1)) {
				throw std::invalid_argument(
					"gradient_magnitude: gx and gy must have the same shape");
			}
			if (gx.shape(0) == 0 || gx.shape(1) == 0) {
				throw std::invalid_argument(
					"gradient_magnitude: inputs must have non-zero dimensions");
			}
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "gradient_magnitude",
			                          [&]<typename T>() {
				auto gx_mat = numpy_to_mat_fresh<T>(gx);
				auto gy_mat = numpy_to_mat_fresh<T>(gy);
				return mat_to_numpy(
					sw::dsp::gradient_magnitude<T>(gx_mat, gy_mat));
			});
		},
		nb::arg("gx"), nb::arg("gy"), nb::arg("dtype") = "reference",
		"Pixel-wise sqrt(gx^2 + gy^2). Typically fed Sobel or Prewitt "
		"gradient outputs.");

	m.def("canny",
		[](np_f64_2d_ro image, double low_threshold, double high_threshold,
		   double sigma, const std::string& dtype) {
			if (image.shape(0) == 0 || image.shape(1) == 0) {
				throw std::invalid_argument(
					"canny: image must have non-zero dimensions");
			}
			if (!(sigma > 0.0)) {
				throw std::invalid_argument(
					"canny: sigma must be positive");
			}
			if (!(low_threshold >= 0.0) || !(high_threshold >= low_threshold)) {
				throw std::invalid_argument(
					"canny: thresholds must satisfy 0 <= low <= high");
			}
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "canny", [&]<typename T>() {
				auto in_mat = numpy_to_mat_fresh<T>(image);
				auto result = sw::dsp::canny<T>(in_mat, low_threshold,
				                                 high_threshold, sigma);
				return mat_to_numpy(result);
			});
		},
		nb::arg("image"), nb::arg("low_threshold"), nb::arg("high_threshold"),
		nb::arg("sigma") = 1.0, nb::arg("dtype") = "reference",
		"Canny edge detector: Gaussian smooth, Sobel gradients, non-maximum "
		"suppression, hysteresis thresholding. Returns a binary edge map "
		"(0.0 for non-edge, 1.0 for edge).");

	// =======================================================================
	// Morphology — structuring-element constructors (return NumPy bool 2D)
	// and the seven gray-level ops (image: float64 2D, element: bool 2D,
	// dtype selects internal arithmetic).
	// =======================================================================

	m.def("make_rect_element",
		[](std::size_t rows, std::size_t cols) {
			if (rows == 0 || cols == 0) {
				throw std::invalid_argument(
					"make_rect_element: rows and cols must be positive");
			}
			return bool_mat_to_numpy(sw::dsp::make_rect_element(rows, cols));
		},
		nb::arg("rows"), nb::arg("cols"),
		"Rectangular structuring element of shape (rows, cols), all True.");

	m.def("make_cross_element",
		[](std::size_t size) {
			if (size == 0) {
				throw std::invalid_argument(
					"make_cross_element: size must be positive");
			}
			return bool_mat_to_numpy(sw::dsp::make_cross_element(size));
		},
		nb::arg("size"),
		"Cross-shaped structuring element of size `size`x`size`: True along "
		"the center row and center column, False elsewhere.");

	m.def("make_ellipse_element",
		[](std::size_t size) {
			if (size == 0) {
				throw std::invalid_argument(
					"make_ellipse_element: size must be positive");
			}
			return bool_mat_to_numpy(sw::dsp::make_ellipse_element(size));
		},
		nb::arg("size"),
		"Elliptical (disk-like) structuring element of size `size`x`size`.");

	// Morphology ops share a dispatch body: fetch image + element, instantiate
	// the right T, call upstream. Wrap the repeating shell in a lambda so each
	// op registration stays one line.
	auto bind_morph_op = [&m](const char* name, auto op) {
		m.def(name,
			[name, op](np_f64_2d_ro image, np_bool_2d_ro element,
			            const std::string& dtype) {
				if (image.shape(0) == 0 || image.shape(1) == 0) {
					throw std::invalid_argument(
						std::string(name) + ": image must have non-zero dimensions");
				}
				if (element.shape(0) == 0 || element.shape(1) == 0) {
					throw std::invalid_argument(
						std::string(name) + ": element must have non-zero dimensions");
				}
				auto config = mpdsp::parse_config(dtype);
				auto elem_mat = numpy_to_bool_mat_fresh(element);
				return dispatch_dtype_fn(config, name, [&]<typename T>() {
					auto img_mat = numpy_to_mat_fresh<T>(image);
					return mat_to_numpy(op.template operator()<T>(img_mat, elem_mat));
				});
			},
			nb::arg("image"), nb::arg("element"),
			nb::arg("dtype") = "reference");
	};

	bind_morph_op("dilate", []<typename T>(const mtl::mat::dense2D<T>& img,
	                                        const mtl::mat::dense2D<bool>& elem) {
		return sw::dsp::dilate<T>(img, elem);
	});
	bind_morph_op("erode", []<typename T>(const mtl::mat::dense2D<T>& img,
	                                       const mtl::mat::dense2D<bool>& elem) {
		return sw::dsp::erode<T>(img, elem);
	});
	bind_morph_op("morphological_open",
		[]<typename T>(const mtl::mat::dense2D<T>& img,
		               const mtl::mat::dense2D<bool>& elem) {
			return sw::dsp::morphological_open<T>(img, elem);
		});
	bind_morph_op("morphological_close",
		[]<typename T>(const mtl::mat::dense2D<T>& img,
		               const mtl::mat::dense2D<bool>& elem) {
			return sw::dsp::morphological_close<T>(img, elem);
		});
	bind_morph_op("morphological_gradient",
		[]<typename T>(const mtl::mat::dense2D<T>& img,
		               const mtl::mat::dense2D<bool>& elem) {
			return sw::dsp::morphological_gradient<T>(img, elem);
		});
	bind_morph_op("tophat", []<typename T>(const mtl::mat::dense2D<T>& img,
	                                        const mtl::mat::dense2D<bool>& elem) {
		return sw::dsp::tophat<T>(img, elem);
	});
	bind_morph_op("blackhat", []<typename T>(const mtl::mat::dense2D<T>& img,
	                                          const mtl::mat::dense2D<bool>& elem) {
		return sw::dsp::blackhat<T>(img, elem);
	});

	// =======================================================================
	// Multi-channel — rgb_to_gray. (apply_per_channel is a pure-Python
	// convenience in mpdsp/image.py since it just iterates a callable.)
	// =======================================================================

	m.def("rgb_to_gray",
		[](np_f64_2d_ro r, np_f64_2d_ro g, np_f64_2d_ro b,
		   const std::string& dtype) {
			if (r.shape(0) != g.shape(0) || r.shape(0) != b.shape(0) ||
			    r.shape(1) != g.shape(1) || r.shape(1) != b.shape(1)) {
				throw std::invalid_argument(
					"rgb_to_gray: r, g, b must all have the same shape");
			}
			if (r.shape(0) == 0 || r.shape(1) == 0) {
				throw std::invalid_argument(
					"rgb_to_gray: channels must have non-zero dimensions");
			}
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "rgb_to_gray", [&]<typename T>() {
				sw::dsp::Image<T, 3> rgb;
				rgb[0] = numpy_to_mat_fresh<T>(r);
				rgb[1] = numpy_to_mat_fresh<T>(g);
				rgb[2] = numpy_to_mat_fresh<T>(b);
				return mat_to_numpy(sw::dsp::rgb_to_gray<T>(rgb));
			});
		},
		nb::arg("r"), nb::arg("g"), nb::arg("b"),
		nb::arg("dtype") = "reference",
		"Convert an RGB image (three NumPy 2D arrays) to grayscale using "
		"ITU-R BT.601 weights: Y = 0.299*R + 0.587*G + 0.114*B.");
}
