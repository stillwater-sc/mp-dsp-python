// signal_bindings.cpp: signal generators, window functions, WAV I/O → NumPy

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <sw/dsp/io/wav.hpp>
#include <sw/dsp/signals/generators.hpp>
#include <sw/dsp/windows/windows.hpp>

#include "_binding_helpers.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

// vec_to_numpy<T> comes from _binding_helpers.hpp; for double inputs it
// is a no-op cast and behaves identically to the previous local helper.

void bind_signals(nb::module_& m) {
	using namespace sw::dsp;
	using mpdsp::bindings::dispatch_dtype_fn;
	using mpdsp::bindings::vec_to_numpy;

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

	// Window functions — these are in the sw::dsp namespace via windows.hpp.
	// Following upstream PRs #122/#125, the window functions are templated
	// on T; the dtype kwarg controls the precision of the internal
	// computation. The result is always returned as a NumPy float64 array
	// — when dtype is e.g. posit32, the array contains the cast-to-double
	// values that the posit-typed window held internally.
	m.def("hamming", [](std::size_t N, const std::string& dtype) {
		auto config = mpdsp::parse_config(dtype);
		return dispatch_dtype_fn(config, "hamming", [&]<typename T>() {
			return vec_to_numpy(hamming_window<T>(N));
		});
	}, nb::arg("N"), nb::arg("dtype") = "reference",
	   "Hamming window of length N. dtype controls the internal compute "
	   "precision; result is always NumPy float64.");

	m.def("hanning", [](std::size_t N, const std::string& dtype) {
		auto config = mpdsp::parse_config(dtype);
		return dispatch_dtype_fn(config, "hanning", [&]<typename T>() {
			return vec_to_numpy(hanning_window<T>(N));
		});
	}, nb::arg("N"), nb::arg("dtype") = "reference",
	   "Hanning (Hann) window of length N.");

	m.def("blackman", [](std::size_t N, const std::string& dtype) {
		auto config = mpdsp::parse_config(dtype);
		return dispatch_dtype_fn(config, "blackman", [&]<typename T>() {
			return vec_to_numpy(blackman_window<T>(N));
		});
	}, nb::arg("N"), nb::arg("dtype") = "reference",
	   "Blackman window of length N.");

	m.def("kaiser", [](std::size_t N, double beta, const std::string& dtype) {
		auto config = mpdsp::parse_config(dtype);
		return dispatch_dtype_fn(config, "kaiser", [&]<typename T>() {
			return vec_to_numpy(kaiser_window<T>(N, beta));
		});
	}, nb::arg("N"), nb::arg("beta") = 5.0, nb::arg("dtype") = "reference",
	   "Kaiser window of length N with shape parameter beta.");

	m.def("rectangular", [](std::size_t N, const std::string& dtype) {
		auto config = mpdsp::parse_config(dtype);
		return dispatch_dtype_fn(config, "rectangular", [&]<typename T>() {
			return vec_to_numpy(rectangular_window<T>(N));
		});
	}, nb::arg("N"), nb::arg("dtype") = "reference",
	   "Rectangular (boxcar) window of length N.");

	m.def("flat_top", [](std::size_t N, const std::string& dtype) {
		auto config = mpdsp::parse_config(dtype);
		return dispatch_dtype_fn(config, "flat_top", [&]<typename T>() {
			return vec_to_numpy(flat_top_window<T>(N));
		});
	}, nb::arg("N"), nb::arg("dtype") = "reference",
	   "Flat-top window of length N.");

	// ---------------------------------------------------------------
	// WAV I/O. Binds upstream sw::dsp::io::read_wav / write_wav_channels.
	//
	// Shape convention (matches scipy.io.wavfile and librosa):
	//   - read_wav returns 1D ndarray for mono files, 2D (N, channels)
	//     for multi-channel files. The sample_rate is returned as an int
	//     alongside the audio data.
	//   - write_wav accepts 1D for mono, 2D (N, channels) for multi.
	//     Upstream only supports integer-PCM writes (8/16/24/32-bit);
	//     float32-PCM write isn't available even though float32-PCM
	//     read is. This asymmetry is documented rather than papered over.
	// ---------------------------------------------------------------

	using np_rw_1d = nb::ndarray<nb::numpy, const double,
	                               nb::ndim<1>, nb::c_contig>;
	using np_rw_2d = nb::ndarray<nb::numpy, const double,
	                               nb::ndim<2>, nb::c_contig>;
	using np_out_1d = nb::ndarray<nb::numpy, double>;
	using np_out_2d = nb::ndarray<nb::numpy, double>;

	m.def("read_wav",
		[](const std::string& path) {
			auto data = sw::dsp::io::read_wav(path);
			int nc = data.num_channels;
			std::size_t ns = data.num_samples();

			if (nc == 1) {
				// Mono — return a 1D array.
				auto* out = new double[ns];
				nb::capsule owner(out, [](void* p) noexcept {
					delete[] static_cast<double*>(p);
				});
				for (std::size_t i = 0; i < ns; ++i) out[i] = data.channels[0][i];
				std::size_t shape[1] = { ns };
				return nb::make_tuple(
					np_out_1d(out, 1, shape, owner), data.sample_rate);
			}

			// Multi-channel — return (N, channels) in C-contiguous layout,
			// which is the scipy.io.wavfile convention.
			auto* out = new double[ns * static_cast<std::size_t>(nc)];
			nb::capsule owner(out, [](void* p) noexcept {
				delete[] static_cast<double*>(p);
			});
			for (std::size_t i = 0; i < ns; ++i) {
				for (int c = 0; c < nc; ++c) {
					out[i * nc + c] = data.channels[c][i];
				}
			}
			std::size_t shape[2] = { ns, static_cast<std::size_t>(nc) };
			return nb::make_tuple(
				np_out_2d(out, 2, shape, owner), data.sample_rate);
		},
		nb::arg("path"),
		"Read a WAV file. Returns (data, sample_rate): data is a float64 "
		"ndarray normalized to [-1, 1] — shape (N,) for mono files, "
		"shape (N, channels) for multi-channel. Supports 8/16/24/32-bit "
		"integer PCM and 32-bit float PCM.");

	m.def("write_wav",
		[](const std::string& path, nb::ndarray<> data, int sample_rate,
		   int bits_per_sample) {
			if (bits_per_sample != 8 && bits_per_sample != 16 &&
			    bits_per_sample != 24 && bits_per_sample != 32) {
				throw std::invalid_argument(
					"write_wav: bits_per_sample must be 8, 16, 24, or 32");
			}
			if (!(sample_rate > 0)) {
				throw std::invalid_argument(
					"write_wav: sample_rate must be positive");
			}

			// Accept 1D (mono) and 2D (multi-channel) float64 C-contiguous
			// arrays. nb::ndarray<> is intentionally untyped so we can
			// branch on ndim after checking the dtype.
			if (data.dtype() != nb::dtype<double>()) {
				throw std::invalid_argument(
					"write_wav: data must be a float64 ndarray");
			}

			const double* base = static_cast<const double*>(data.data());

			if (data.ndim() == 1) {
				std::size_t ns = data.shape(0);
				std::span<const double> mono(base, ns);
				sw::dsp::io::write_wav(path, mono, sample_rate, bits_per_sample);
			} else if (data.ndim() == 2) {
				std::size_t ns = data.shape(0);
				std::size_t nc = data.shape(1);
				// De-interleave into per-channel contiguous vectors so we can
				// hand each one as a span to upstream. A zero-copy view isn't
				// possible because C-contiguous (N, C) has channel samples
				// interleaved, but upstream wants channel-major layout.
				std::vector<std::vector<double>> per_channel(nc);
				for (std::size_t c = 0; c < nc; ++c) {
					per_channel[c].resize(ns);
					for (std::size_t i = 0; i < ns; ++i) {
						per_channel[c][i] = base[i * nc + c];
					}
				}
				std::vector<std::span<const double>> channels;
				channels.reserve(nc);
				for (auto& ch : per_channel) channels.emplace_back(ch);
				sw::dsp::io::write_wav_channels(
					path, channels, sample_rate, bits_per_sample);
			} else {
				throw std::invalid_argument(
					"write_wav: data must be 1D (mono) or 2D (N, channels)");
			}
		},
		nb::arg("path"), nb::arg("data"), nb::arg("sample_rate"),
		nb::arg("bits_per_sample") = 16,
		"Write a WAV file. `data` is a float64 ndarray — 1D for mono or "
		"2D (N, channels) for multi-channel. Values outside [-1, 1] are "
		"clipped. bits_per_sample must be 8, 16, 24, or 32 (integer PCM "
		"only — float32-PCM write is not supported by upstream even though "
		"float32-PCM read is).");
}
