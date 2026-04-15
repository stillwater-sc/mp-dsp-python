// spectral_bindings.cpp: FFT, PSD, spectrogram → NumPy

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <sw/dsp/spectral/spectral.hpp>
#include <sw/dsp/windows/windows.hpp>

#include <cmath>
#include <complex>
#include <cstddef>

namespace nb = nanobind;

using np_array = nb::ndarray<nb::numpy, double>;
using np_array_ro = nb::ndarray<nb::numpy, const double, nb::ndim<1>>;

// Helper: dense_vector<double> → NumPy 1D
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

// Helper: dense_vector<complex<double>> → two NumPy arrays (real, imag)
// or a single interleaved complex128 array
static nb::ndarray<nb::numpy, double>
complex_vec_to_numpy_real(const mtl::vec::dense_vector<std::complex<double>>& v) {
	std::size_t n = v.size();
	auto* data = new double[n];
	for (std::size_t i = 0; i < n; ++i) data[i] = v[i].real();
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	std::size_t shape[1] = { n };
	return nb::ndarray<nb::numpy, double>(data, 1, shape, owner);
}

static nb::ndarray<nb::numpy, double>
complex_vec_to_numpy_imag(const mtl::vec::dense_vector<std::complex<double>>& v) {
	std::size_t n = v.size();
	auto* data = new double[n];
	for (std::size_t i = 0; i < n; ++i) data[i] = v[i].imag();
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	std::size_t shape[1] = { n };
	return nb::ndarray<nb::numpy, double>(data, 1, shape, owner);
}

// Helper: magnitude spectrum in dB from complex vector
static nb::ndarray<nb::numpy, double>
complex_vec_to_magnitude_db(const mtl::vec::dense_vector<std::complex<double>>& v) {
	std::size_t n = v.size();
	auto* data = new double[n];
	for (std::size_t i = 0; i < n; ++i) {
		double mag = std::abs(v[i]);
		data[i] = (mag > 1e-20) ? 20.0 * std::log10(mag) : -400.0;
	}
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	std::size_t shape[1] = { n };
	return nb::ndarray<nb::numpy, double>(data, 1, shape, owner);
}

void bind_spectral(nb::module_& m) {
	using namespace sw::dsp;
	using namespace sw::dsp::spectral;

	// FFT: real signal → (real_part, imag_part) of complex spectrum
	m.def("fft", [](np_array_ro signal) {
		auto v = numpy_to_vec(signal);
		auto spectrum = fft<double>(v);
		return nb::make_tuple(
			complex_vec_to_numpy_real(spectrum),
			complex_vec_to_numpy_imag(spectrum)
		);
	}, nb::arg("signal"),
	   "Compute FFT of a real signal. Returns (real, imag) tuple of NumPy arrays.");

	// FFT magnitude in dB
	m.def("fft_magnitude_db", [](np_array_ro signal) {
		auto v = numpy_to_vec(signal);
		auto spectrum = fft<double>(v);
		return complex_vec_to_magnitude_db(spectrum);
	}, nb::arg("signal"),
	   "Compute FFT magnitude spectrum in dB.");

	// IFFT: (real, imag) → real signal
	m.def("ifft", [](np_array_ro real_part, np_array_ro imag_part) {
		std::size_t n = real_part.shape(0);
		if (n != imag_part.shape(0))
			throw std::invalid_argument("ifft: real and imag arrays must have same length");
		mtl::vec::dense_vector<std::complex<double>> spectrum(n);
		const double* re = real_part.data();
		const double* im = imag_part.data();
		for (std::size_t i = 0; i < n; ++i) {
			spectrum[i] = std::complex<double>(re[i], im[i]);
		}
		auto result = ifft_real<double>(spectrum);
		return vec_to_numpy(result);
	}, nb::arg("real"), nb::arg("imag"),
	   "Compute inverse FFT from (real, imag) arrays. Returns real signal.");

	// Periodogram PSD
	m.def("periodogram", [](np_array_ro signal) {
		auto v = numpy_to_vec(signal);
		auto psd = periodogram<double>(v);
		return vec_to_numpy(psd);
	}, nb::arg("signal"),
	   "Compute periodogram power spectral density estimate.");

	// PSD with frequency axis
	m.def("psd", [](np_array_ro signal, double sample_rate) {
		auto v = numpy_to_vec(signal);
		auto power = periodogram<double>(v);
		// Build frequency axis
		std::size_t n_freqs = power.size();
		auto* freq_data = new double[n_freqs];
		for (std::size_t i = 0; i < n_freqs; ++i) {
			freq_data[i] = static_cast<double>(i) * sample_rate
			             / static_cast<double>((n_freqs - 1) * 2);
		}
		nb::capsule fowner(freq_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
		std::size_t shape[1] = { n_freqs };
		auto freqs = nb::ndarray<nb::numpy, double>(freq_data, 1, shape, fowner);
		return nb::make_tuple(freqs, vec_to_numpy(power));
	}, nb::arg("signal"), nb::arg("sample_rate"),
	   "Compute PSD with frequency axis. Returns (freqs_hz, power) tuple.");

	// Spectrogram
	m.def("spectrogram", [](np_array_ro signal, double sample_rate,
	                         std::size_t window_size, std::size_t hop_size) {
		auto v = numpy_to_vec(signal);
		auto win = hanning_window<double>(window_size);
		auto stft = spectrogram<double>(v, win, hop_size);

		// Convert to 2D magnitude dB array [n_frames x n_freqs]
		std::size_t n_frames = stft.frames.size();
		std::size_t n_freqs = (n_frames > 0) ? stft.frames[0].size() / 2 + 1 : 0;
		auto* mag_data = new double[n_frames * n_freqs];

		for (std::size_t f = 0; f < n_frames; ++f) {
			for (std::size_t k = 0; k < n_freqs; ++k) {
				double mag = std::abs(stft.frames[f][k]);
				mag_data[f * n_freqs + k] = (mag > 1e-20)
					? 20.0 * std::log10(mag) : -120.0;
			}
		}

		nb::capsule mag_owner(mag_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
		std::size_t mag_shape[2] = { n_frames, n_freqs };
		auto magnitudes = nb::ndarray<nb::numpy, double>(mag_data, 2, mag_shape, mag_owner);

		// Time axis (center of each frame)
		auto* time_data = new double[n_frames];
		for (std::size_t f = 0; f < n_frames; ++f) {
			time_data[f] = (static_cast<double>(f) * static_cast<double>(hop_size)
			              + static_cast<double>(window_size) * 0.5) / sample_rate;
		}
		nb::capsule time_owner(time_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
		std::size_t time_shape[1] = { n_frames };
		auto times = nb::ndarray<nb::numpy, double>(time_data, 1, time_shape, time_owner);

		// Frequency axis
		auto* freq_data = new double[n_freqs];
		for (std::size_t k = 0; k < n_freqs; ++k) {
			freq_data[k] = static_cast<double>(k) * sample_rate
			             / static_cast<double>(stft.fft_size);
		}
		nb::capsule freq_owner(freq_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
		std::size_t freq_shape[1] = { n_freqs };
		auto freqs = nb::ndarray<nb::numpy, double>(freq_data, 1, freq_shape, freq_owner);

		return nb::make_tuple(times, freqs, magnitudes);
	}, nb::arg("signal"), nb::arg("sample_rate"),
	   nb::arg("window_size") = static_cast<std::size_t>(1024),
	   nb::arg("hop_size") = static_cast<std::size_t>(256),
	   "Compute spectrogram. Returns (times, freqs, magnitude_db) tuple. "
	   "magnitude_db is a 2D array [n_frames x n_freqs].");
}
