// spectral_bindings.cpp: FFT, PSD, spectrogram → NumPy, with dtype dispatch
//
// Phase 5 PR-B of the 0.5.0 binding sweep (#54). Wires `dtype=` dispatch
// through the existing spectral primitives so the Python API matches the
// rest of the library (filter / image / conditioning / estimation all
// accept a `dtype` string).
//
// Pattern for each primitive:
//   1. `fft_typed<T>(signal: dense_vector<double>)` casts signal to T,
//      calls the upstream templated function, casts the result back to
//      double or complex<double>.
//   2. `fft_dispatch(signal, config)` is a switch over ArithConfig
//      selecting which T-specialization to run.
//   3. The Python-facing lambda parses `dtype` → ArithConfig and forwards.
//
// Inputs and outputs to Python are always double/complex128; the dtype
// only affects the internal arithmetic. This matches the semantic the
// filter/image bindings already use.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <sw/dsp/spectral/spectral.hpp>
#include <sw/dsp/windows/windows.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>

#include "types.hpp"

namespace nb = nanobind;

using np_array = nb::ndarray<nb::numpy, double>;
using np_array_ro = nb::ndarray<nb::numpy, const double, nb::ndim<1>>;

// Helper: dense_vector<double> → NumPy 1D (owning capsule).
static np_array vec_to_numpy(const mtl::vec::dense_vector<double>& v) {
	std::size_t n = v.size();
	auto* data = new double[n];
	for (std::size_t i = 0; i < n; ++i) data[i] = v[i];
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	std::size_t shape[1] = { n };
	return np_array(data, 1, shape, owner);
}

// Helper: NumPy 1D → dense_vector<double>
static mtl::vec::dense_vector<double> numpy_to_vec(np_array_ro arr) {
	std::size_t n = arr.shape(0);
	mtl::vec::dense_vector<double> v(n);
	const double* data = arr.data();
	for (std::size_t i = 0; i < n; ++i) v[i] = data[i];
	return v;
}

// ---------------------------------------------------------------------------
// Type-conversion helpers.
// ---------------------------------------------------------------------------

template <typename T>
static mtl::vec::dense_vector<T>
cast_double_to_T(const mtl::vec::dense_vector<double>& v) {
	mtl::vec::dense_vector<T> out(v.size());
	for (std::size_t i = 0; i < v.size(); ++i) {
		out[i] = static_cast<T>(v[i]);
	}
	return out;
}

// Cast a dense_vector<complex_for_t<T>> to dense_vector<complex<double>>.
// Uses `using std::real; using std::imag;` inside so both std::complex and
// sw::universal::complex work via ADL — important because narrow dtypes
// (posit, cfloat) use Universal's complex type, not std::complex.
template <typename T>
static mtl::vec::dense_vector<std::complex<double>>
cast_complex_to_double(
		const mtl::vec::dense_vector<sw::dsp::complex_for_t<T>>& v) {
	using std::imag;
	using std::real;
	mtl::vec::dense_vector<std::complex<double>> out(v.size());
	for (std::size_t i = 0; i < v.size(); ++i) {
		out[i] = std::complex<double>(
			static_cast<double>(real(v[i])),
			static_cast<double>(imag(v[i])));
	}
	return out;
}

// Helper: std::complex<double> vector → NumPy real-only array.
static np_array
complex_vec_to_numpy_real(const mtl::vec::dense_vector<std::complex<double>>& v) {
	std::size_t n = v.size();
	auto* data = new double[n];
	for (std::size_t i = 0; i < n; ++i) data[i] = v[i].real();
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	std::size_t shape[1] = { n };
	return np_array(data, 1, shape, owner);
}

static np_array
complex_vec_to_numpy_imag(const mtl::vec::dense_vector<std::complex<double>>& v) {
	std::size_t n = v.size();
	auto* data = new double[n];
	for (std::size_t i = 0; i < n; ++i) data[i] = v[i].imag();
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	std::size_t shape[1] = { n };
	return np_array(data, 1, shape, owner);
}

static np_array
complex_vec_to_magnitude_db(const mtl::vec::dense_vector<std::complex<double>>& v) {
	std::size_t n = v.size();
	auto* data = new double[n];
	for (std::size_t i = 0; i < n; ++i) {
		double mag = std::abs(v[i]);
		data[i] = (mag > 1e-20) ? 20.0 * std::log10(mag) : -400.0;
	}
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	std::size_t shape[1] = { n };
	return np_array(data, 1, shape, owner);
}

// ---------------------------------------------------------------------------
// Typed wrappers + dispatchers for each spectral primitive.
// ---------------------------------------------------------------------------

template <typename T>
static mtl::vec::dense_vector<std::complex<double>>
fft_typed(const mtl::vec::dense_vector<double>& signal) {
	auto typed = cast_double_to_T<T>(signal);
	auto spectrum = sw::dsp::spectral::fft<T>(typed);
	return cast_complex_to_double<T>(spectrum);
}

template <typename T>
static mtl::vec::dense_vector<double>
ifft_typed(const mtl::vec::dense_vector<std::complex<double>>& spectrum) {
	using complex_T = sw::dsp::complex_for_t<T>;
	mtl::vec::dense_vector<complex_T> typed_spectrum(spectrum.size());
	for (std::size_t i = 0; i < spectrum.size(); ++i) {
		typed_spectrum[i] = complex_T(
			static_cast<T>(spectrum[i].real()),
			static_cast<T>(spectrum[i].imag()));
	}
	auto typed_signal = sw::dsp::spectral::ifft_real<T>(typed_spectrum);
	mtl::vec::dense_vector<double> out(typed_signal.size());
	for (std::size_t i = 0; i < typed_signal.size(); ++i) {
		out[i] = static_cast<double>(typed_signal[i]);
	}
	return out;
}

template <typename T>
static mtl::vec::dense_vector<double>
periodogram_typed(const mtl::vec::dense_vector<double>& signal) {
	auto typed = cast_double_to_T<T>(signal);
	return sw::dsp::spectral::periodogram<T>(typed);  // already returns double
}

// Dispatch switches — one per primitive. Each follows the same shape so
// the pattern is easy to extend when Phase 6 (#55) adds new ArithConfig
// enumerators. A missing case here would fail at compile time with the
// compiler's "enumeration value not handled" warning (-Wswitch).

#define DISPATCH_SPECTRAL(fn, ret_type, input)                             \
	switch (config) {                                                      \
	case mpdsp::ArithConfig::reference:                                    \
		return fn<double>(input);                                          \
	case mpdsp::ArithConfig::gpu_baseline:                                 \
		return fn<float>(input);                                           \
	case mpdsp::ArithConfig::ml_hw:                                        \
	case mpdsp::ArithConfig::half_config:                                  \
		return fn<mpdsp::half_>(input);                                    \
	case mpdsp::ArithConfig::posit_full:                                   \
		return fn<mpdsp::p16>(input);                                      \
	case mpdsp::ArithConfig::tiny_posit:                                   \
		return fn<sw::universal::posit<8, 2>>(input);                      \
	case mpdsp::ArithConfig::cf24_config:                                  \
		return fn<mpdsp::cf24>(input);                                     \
	}                                                                      \
	/* Unreachable: exhaustive switch over ArithConfig. */                 \
	return ret_type{};

static mtl::vec::dense_vector<std::complex<double>>
fft_dispatch(const mtl::vec::dense_vector<double>& signal,
              mpdsp::ArithConfig config) {
	DISPATCH_SPECTRAL(fft_typed, mtl::vec::dense_vector<std::complex<double>>,
	                  signal);
}

static mtl::vec::dense_vector<double>
ifft_dispatch(const mtl::vec::dense_vector<std::complex<double>>& spectrum,
               mpdsp::ArithConfig config) {
	DISPATCH_SPECTRAL(ifft_typed, mtl::vec::dense_vector<double>, spectrum);
}

static mtl::vec::dense_vector<double>
periodogram_dispatch(const mtl::vec::dense_vector<double>& signal,
                      mpdsp::ArithConfig config) {
	DISPATCH_SPECTRAL(periodogram_typed, mtl::vec::dense_vector<double>,
	                  signal);
}

// Spectrogram needs its own dispatcher because of the extra
// window / hop_size parameters and the more complex return shape.
template <typename T>
static nb::tuple
spectrogram_typed(const mtl::vec::dense_vector<double>& signal,
                   double sample_rate, std::size_t window_size,
                   std::size_t hop_size) {
	using std::imag;
	using std::real;

	auto typed_signal = cast_double_to_T<T>(signal);
	auto typed_window = sw::dsp::hanning_window<T>(window_size);
	auto stft = sw::dsp::spectral::spectrogram<T>(typed_signal, typed_window, hop_size);

	std::size_t n_frames = stft.frames.size();
	std::size_t n_freqs = (n_frames > 0) ? stft.frames[0].size() / 2 + 1 : 0;
	auto* mag_data = new double[n_frames * n_freqs];

	for (std::size_t f = 0; f < n_frames; ++f) {
		for (std::size_t k = 0; k < n_freqs; ++k) {
			auto z = stft.frames[f][k];
			double re = static_cast<double>(real(z));
			double im = static_cast<double>(imag(z));
			double mag = std::hypot(re, im);
			mag_data[f * n_freqs + k] =
				(mag > 1e-20) ? 20.0 * std::log10(mag) : -120.0;
		}
	}

	nb::capsule mag_owner(mag_data, [](void* p) noexcept {
		delete[] static_cast<double*>(p);
	});
	std::size_t mag_shape[2] = { n_frames, n_freqs };
	auto magnitudes = nb::ndarray<nb::numpy, double>(
		mag_data, 2, mag_shape, mag_owner);

	auto* time_data = new double[n_frames];
	for (std::size_t f = 0; f < n_frames; ++f) {
		time_data[f] = (static_cast<double>(f) * static_cast<double>(hop_size)
		               + static_cast<double>(window_size) * 0.5) / sample_rate;
	}
	nb::capsule time_owner(time_data, [](void* p) noexcept {
		delete[] static_cast<double*>(p);
	});
	std::size_t time_shape[1] = { n_frames };
	auto times = nb::ndarray<nb::numpy, double>(
		time_data, 1, time_shape, time_owner);

	auto* freq_data = new double[n_freqs];
	for (std::size_t k = 0; k < n_freqs; ++k) {
		freq_data[k] = static_cast<double>(k) * sample_rate
		             / static_cast<double>(stft.fft_size);
	}
	nb::capsule freq_owner(freq_data, [](void* p) noexcept {
		delete[] static_cast<double*>(p);
	});
	std::size_t freq_shape[1] = { n_freqs };
	auto freqs = nb::ndarray<nb::numpy, double>(
		freq_data, 1, freq_shape, freq_owner);

	return nb::make_tuple(times, freqs, magnitudes);
}

static nb::tuple
spectrogram_dispatch(const mtl::vec::dense_vector<double>& signal,
                      double sample_rate, std::size_t window_size,
                      std::size_t hop_size, mpdsp::ArithConfig config) {
	switch (config) {
	case mpdsp::ArithConfig::reference:
		return spectrogram_typed<double>(signal, sample_rate, window_size, hop_size);
	case mpdsp::ArithConfig::gpu_baseline:
		return spectrogram_typed<float>(signal, sample_rate, window_size, hop_size);
	case mpdsp::ArithConfig::ml_hw:
	case mpdsp::ArithConfig::half_config:
		return spectrogram_typed<mpdsp::half_>(signal, sample_rate, window_size, hop_size);
	case mpdsp::ArithConfig::posit_full:
		return spectrogram_typed<mpdsp::p16>(signal, sample_rate, window_size, hop_size);
	case mpdsp::ArithConfig::tiny_posit:
		return spectrogram_typed<sw::universal::posit<8, 2>>(
			signal, sample_rate, window_size, hop_size);
	case mpdsp::ArithConfig::cf24_config:
		return spectrogram_typed<mpdsp::cf24>(
			signal, sample_rate, window_size, hop_size);
	}
	return spectrogram_typed<double>(signal, sample_rate, window_size, hop_size);
}

// ---------------------------------------------------------------------------
// Python bindings.
// ---------------------------------------------------------------------------

void bind_spectral(nb::module_& m) {
	m.def("fft", [](np_array_ro signal, const std::string& dtype) {
		auto v = numpy_to_vec(signal);
		auto config = mpdsp::parse_config(dtype);
		auto spectrum = fft_dispatch(v, config);
		return nb::make_tuple(
			complex_vec_to_numpy_real(spectrum),
			complex_vec_to_numpy_imag(spectrum));
	}, nb::arg("signal"), nb::arg("dtype") = "reference",
	   "Compute FFT of a real signal. Returns (real, imag) tuple of NumPy "
	   "arrays. `dtype` selects the internal arithmetic (see "
	   "`mpdsp.available_dtypes()`).");

	m.def("fft_magnitude_db",
		[](np_array_ro signal, const std::string& dtype) {
			auto v = numpy_to_vec(signal);
			auto config = mpdsp::parse_config(dtype);
			auto spectrum = fft_dispatch(v, config);
			return complex_vec_to_magnitude_db(spectrum);
		},
		nb::arg("signal"), nb::arg("dtype") = "reference",
		"Compute FFT magnitude spectrum in dB. `dtype` selects the internal "
		"arithmetic (see `mpdsp.available_dtypes()`).");

	m.def("ifft",
		[](np_array_ro real_part, np_array_ro imag_part,
		   const std::string& dtype) {
			std::size_t n = real_part.shape(0);
			if (n != imag_part.shape(0)) {
				throw std::invalid_argument(
					"ifft: real and imag arrays must have same length");
			}
			mtl::vec::dense_vector<std::complex<double>> spectrum(n);
			const double* re = real_part.data();
			const double* im = imag_part.data();
			for (std::size_t i = 0; i < n; ++i) {
				spectrum[i] = std::complex<double>(re[i], im[i]);
			}
			auto config = mpdsp::parse_config(dtype);
			return vec_to_numpy(ifft_dispatch(spectrum, config));
		},
		nb::arg("real"), nb::arg("imag"), nb::arg("dtype") = "reference",
		"Compute inverse FFT from (real, imag) arrays. Returns real signal. "
		"`dtype` selects the internal arithmetic (see "
		"`mpdsp.available_dtypes()`).");

	m.def("periodogram",
		[](np_array_ro signal, const std::string& dtype) {
			auto v = numpy_to_vec(signal);
			auto config = mpdsp::parse_config(dtype);
			return vec_to_numpy(periodogram_dispatch(v, config));
		},
		nb::arg("signal"), nb::arg("dtype") = "reference",
		"Compute periodogram power spectral density estimate. `dtype` "
		"selects the internal arithmetic (see `mpdsp.available_dtypes()`).");

	m.def("psd",
		[](np_array_ro signal, double sample_rate, const std::string& dtype) {
			auto v = numpy_to_vec(signal);
			auto config = mpdsp::parse_config(dtype);
			auto power = periodogram_dispatch(v, config);
			std::size_t n_freqs = power.size();
			auto* freq_data = new double[n_freqs];
			for (std::size_t i = 0; i < n_freqs; ++i) {
				freq_data[i] = static_cast<double>(i) * sample_rate
				             / static_cast<double>((n_freqs - 1) * 2);
			}
			nb::capsule fowner(freq_data, [](void* p) noexcept {
				delete[] static_cast<double*>(p);
			});
			std::size_t shape[1] = { n_freqs };
			auto freqs = np_array(freq_data, 1, shape, fowner);
			return nb::make_tuple(freqs, vec_to_numpy(power));
		},
		nb::arg("signal"), nb::arg("sample_rate"),
		nb::arg("dtype") = "reference",
		"Compute PSD with frequency axis. Returns (freqs_hz, power) tuple. "
		"`dtype` selects the internal arithmetic (see "
		"`mpdsp.available_dtypes()`).");

	m.def("spectrogram",
		[](np_array_ro signal, double sample_rate,
		   std::size_t window_size, std::size_t hop_size,
		   const std::string& dtype) {
			auto v = numpy_to_vec(signal);
			auto config = mpdsp::parse_config(dtype);
			return spectrogram_dispatch(v, sample_rate, window_size,
			                             hop_size, config);
		},
		nb::arg("signal"), nb::arg("sample_rate"),
		nb::arg("window_size") = static_cast<std::size_t>(1024),
		nb::arg("hop_size") = static_cast<std::size_t>(256),
		nb::arg("dtype") = "reference",
		"Compute spectrogram. Returns (times, freqs, magnitude_db) tuple. "
		"magnitude_db is a 2D array [n_frames x n_freqs]. `dtype` selects "
		"the internal arithmetic (see `mpdsp.available_dtypes()`).");
}
