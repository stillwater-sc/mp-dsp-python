// quantization_bindings.cpp: ADC/DAC, SQNR, dither → NumPy

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sw/dsp/quantization/quantization.hpp>
#include <sw/dsp/signals/generators.hpp>

#include "types.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace nb = nanobind;

// NumPy array type aliases
using np_array = nb::ndarray<nb::numpy, double>;
using np_array_ro = nb::ndarray<nb::numpy, const double, nb::ndim<1>>;

// Helper: read a 1D NumPy array into dense_vector<double>
static mtl::vec::dense_vector<double> numpy_to_vec(np_array_ro arr) {
	std::size_t n = arr.shape(0);
	mtl::vec::dense_vector<double> v(n);
	const double* data = arr.data();
	for (std::size_t i = 0; i < n; ++i) v[i] = data[i];
	return v;
}

// Helper: dense_vector<double> to NumPy
static np_array vec_to_numpy(const mtl::vec::dense_vector<double>& v) {
	std::size_t n = v.size();
	auto* data = new double[n];
	for (std::size_t i = 0; i < n; ++i) data[i] = v[i];
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	std::size_t shape[1] = { n };
	return np_array(data, 1, shape, owner);
}

// ADC: quantize a double signal through a target type and back
template <typename T>
static mtl::vec::dense_vector<double>
adc_typed(const mtl::vec::dense_vector<double>& signal) {
	sw::dsp::ADC<double, T> adc;
	auto quantized = adc.convert(signal);
	mtl::vec::dense_vector<double> result(signal.size());
	for (std::size_t i = 0; i < signal.size(); ++i) {
		result[i] = static_cast<double>(quantized[i]);
	}
	return result;
}

static mtl::vec::dense_vector<double>
adc_dispatch(const mtl::vec::dense_vector<double>& signal, mpdsp::ArithConfig config) {
	switch (config) {
	case mpdsp::ArithConfig::reference:     return adc_typed<double>(signal);
	case mpdsp::ArithConfig::gpu_baseline:  return adc_typed<float>(signal);
	case mpdsp::ArithConfig::cf24_config:   return adc_typed<mpdsp::cf24>(signal);
	case mpdsp::ArithConfig::half_config:   return adc_typed<mpdsp::half_>(signal);
	case mpdsp::ArithConfig::ml_hw:         return adc_typed<mpdsp::half_>(signal);
	case mpdsp::ArithConfig::posit_full:    return adc_typed<mpdsp::p16>(signal);
	case mpdsp::ArithConfig::tiny_posit:    return adc_typed<sw::universal::posit<8,2>>(signal);
	}
	return signal;
}

static double sqnr_impl(const mtl::vec::dense_vector<double>& ref,
                         const mtl::vec::dense_vector<double>& test) {
	if (ref.size() != test.size() || ref.size() == 0)
		throw std::invalid_argument("sqnr_db: vectors must have same non-zero length");
	double signal_power = 0, noise_power = 0;
	for (std::size_t i = 0; i < ref.size(); ++i) {
		signal_power += ref[i] * ref[i];
		double err = ref[i] - test[i];
		noise_power += err * err;
	}
	if (noise_power < 1e-300) return 300.0;
	return 10.0 * std::log10(signal_power / noise_power);
}

void bind_quantization(nb::module_& m) {

	m.def("adc", [](np_array_ro signal, const std::string& dtype) {
		auto v = numpy_to_vec(signal);
		auto config = mpdsp::parse_config(dtype);
		return vec_to_numpy(adc_dispatch(v, config));
	}, nb::arg("signal"), nb::arg("dtype") = "reference",
	   "Quantize signal through target type (double -> T -> double).");

	m.def("sqnr_db", [](np_array_ro ref, np_array_ro test) {
		return sqnr_impl(numpy_to_vec(ref), numpy_to_vec(test));
	}, nb::arg("reference"), nb::arg("quantized"),
	   "Compute SQNR (dB) between reference and quantized signals.");

	m.def("measure_sqnr_db", [](np_array_ro signal, const std::string& dtype) {
		auto v = numpy_to_vec(signal);
		auto config = mpdsp::parse_config(dtype);
		auto quantized = adc_dispatch(v, config);
		return sqnr_impl(v, quantized);
	}, nb::arg("signal"), nb::arg("dtype"),
	   "Measure SQNR of a signal after ADC round-trip through target type.");

	m.def("max_absolute_error", [](np_array_ro ref, np_array_ro test) {
		auto vr = numpy_to_vec(ref);
		auto vt = numpy_to_vec(test);
		if (vr.size() != vt.size() || vr.size() == 0)
			throw std::invalid_argument("max_absolute_error: vectors must have same non-zero length");
		double max_err = 0;
		for (std::size_t i = 0; i < vr.size(); ++i) {
			double err = std::abs(vr[i] - vt[i]);
			if (err > max_err) max_err = err;
		}
		return max_err;
	}, nb::arg("reference"), nb::arg("test"),
	   "Maximum absolute error between two signals.");

	m.def("max_relative_error", [](np_array_ro ref, np_array_ro test) {
		auto vr = numpy_to_vec(ref);
		auto vt = numpy_to_vec(test);
		if (vr.size() != vt.size() || vr.size() == 0)
			throw std::invalid_argument("max_relative_error: vectors must have same non-zero length");
		double max_ref = 0, max_err = 0;
		for (std::size_t i = 0; i < vr.size(); ++i) {
			if (std::abs(vr[i]) > max_ref) max_ref = std::abs(vr[i]);
			double err = std::abs(vr[i] - vt[i]);
			if (err > max_err) max_err = err;
		}
		return (max_ref > 0) ? max_err / max_ref : 0.0;
	}, nb::arg("reference"), nb::arg("test"),
	   "Maximum relative error between two signals.");

	m.def("available_dtypes", &mpdsp::available_configs,
	   "List available arithmetic configuration names.");
}
