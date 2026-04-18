// signal_bindings.cpp: signal generators and window functions → NumPy

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/windows/windows.hpp>

#include <cstddef>

namespace nb = nanobind;

// Helper: convert mtl::vec::dense_vector<double> to a new NumPy array
static nb::ndarray<nb::numpy, double>
vec_to_numpy(const mtl::vec::dense_vector<double>& v) {
	std::size_t n = v.size();
	auto* data = new double[n];
	for (std::size_t i = 0; i < n; ++i) data[i] = v[i];
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	std::size_t shape[1] = { n };
	return nb::ndarray<nb::numpy, double>(data, 1, shape, owner);
}

void bind_signals(nb::module_& m) {
	using namespace sw::dsp;

	m.def("sine", [](std::size_t length, double frequency, double sample_rate,
	                  double amplitude, double phase) {
		return vec_to_numpy(sine<double>(length, frequency, sample_rate, amplitude, phase));
	}, nb::arg("length"), nb::arg("frequency"), nb::arg("sample_rate"),
	   nb::arg("amplitude") = 1.0, nb::arg("phase") = 0.0,
	   "Generate a sine wave. Returns NumPy float64 array.");

	m.def("cosine", [](std::size_t length, double frequency, double sample_rate,
	                    double amplitude, double phase) {
		return vec_to_numpy(cosine<double>(length, frequency, sample_rate, amplitude, phase));
	}, nb::arg("length"), nb::arg("frequency"), nb::arg("sample_rate"),
	   nb::arg("amplitude") = 1.0, nb::arg("phase") = 0.0,
	   "Generate a cosine wave.");

	m.def("chirp", [](std::size_t length, double f_start, double f_end,
	                   double sample_rate, double amplitude) {
		return vec_to_numpy(chirp<double>(length, f_start, f_end, sample_rate, amplitude));
	}, nb::arg("length"), nb::arg("f_start"), nb::arg("f_end"),
	   nb::arg("sample_rate"), nb::arg("amplitude") = 1.0,
	   "Generate a linear chirp (frequency sweep).");

	m.def("square", [](std::size_t length, double frequency, double sample_rate,
	                    double amplitude) {
		return vec_to_numpy(square<double>(length, frequency, sample_rate, amplitude));
	}, nb::arg("length"), nb::arg("frequency"), nb::arg("sample_rate"),
	   nb::arg("amplitude") = 1.0,
	   "Generate a square wave.");

	m.def("triangle", [](std::size_t length, double frequency, double sample_rate,
	                      double amplitude) {
		return vec_to_numpy(triangle<double>(length, frequency, sample_rate, amplitude));
	}, nb::arg("length"), nb::arg("frequency"), nb::arg("sample_rate"),
	   nb::arg("amplitude") = 1.0,
	   "Generate a triangle wave.");

	m.def("sawtooth", [](std::size_t length, double frequency, double sample_rate,
	                      double amplitude) {
		return vec_to_numpy(sawtooth<double>(length, frequency, sample_rate, amplitude));
	}, nb::arg("length"), nb::arg("frequency"), nb::arg("sample_rate"),
	   nb::arg("amplitude") = 1.0,
	   "Generate a sawtooth wave.");

	m.def("impulse", [](std::size_t length, std::size_t position) {
		return vec_to_numpy(impulse<double>(length, position));
	}, nb::arg("length"), nb::arg("position") = static_cast<std::size_t>(0),
	   "Generate an impulse (single 1.0 at position, rest 0).");

	m.def("step", [](std::size_t length, std::size_t position) {
		return vec_to_numpy(step<double>(length, position));
	}, nb::arg("length"), nb::arg("position") = static_cast<std::size_t>(0),
	   "Generate a unit step (0 before position, 1 from position onward).");

	m.def("white_noise", [](std::size_t length, double amplitude, unsigned seed) {
		return vec_to_numpy(white_noise<double>(length, amplitude, seed));
	}, nb::arg("length"), nb::arg("amplitude") = 1.0, nb::arg("seed") = 0u,
	   "Generate white noise (uniform in [-amplitude, amplitude]).");

	// Upstream's gaussian_noise templates on `amplitude` but docs the scale as
	// the standard deviation (mean=0, stddev=amplitude). Python callers expect
	// `stddev` — expose that name directly and forward to amplitude.
	m.def("gaussian_noise", [](std::size_t length, double stddev, unsigned seed) {
		return vec_to_numpy(gaussian_noise<double>(length, stddev, seed));
	}, nb::arg("length"), nb::arg("stddev") = 1.0, nb::arg("seed") = 0u,
	   "Generate Gaussian white noise (mean=0, normal distribution with given stddev).");

	m.def("pink_noise", [](std::size_t length, double amplitude, unsigned seed) {
		return vec_to_numpy(pink_noise<double>(length, amplitude, seed));
	}, nb::arg("length"), nb::arg("amplitude") = 1.0, nb::arg("seed") = 0u,
	   "Generate pink noise (1/f spectrum, Voss-McCartney algorithm).");

	// Window functions — these are in the sw::dsp namespace via windows.hpp
	m.def("hamming", [](std::size_t N) {
		return vec_to_numpy(hamming_window<double>(N));
	}, nb::arg("N"), "Hamming window of length N.");

	m.def("hanning", [](std::size_t N) {
		return vec_to_numpy(hanning_window<double>(N));
	}, nb::arg("N"), "Hanning (Hann) window of length N.");

	m.def("blackman", [](std::size_t N) {
		return vec_to_numpy(blackman_window<double>(N));
	}, nb::arg("N"), "Blackman window of length N.");

	m.def("kaiser", [](std::size_t N, double beta) {
		return vec_to_numpy(kaiser_window<double>(N, beta));
	}, nb::arg("N"), nb::arg("beta") = 5.0,
	   "Kaiser window of length N with shape parameter beta.");

	m.def("rectangular", [](std::size_t N) {
		return vec_to_numpy(rectangular_window<double>(N));
	}, nb::arg("N"), "Rectangular (boxcar) window of length N.");

	m.def("flat_top", [](std::size_t N) {
		return vec_to_numpy(flat_top_window<double>(N));
	}, nb::arg("N"), "Flat-top window of length N.");
}
