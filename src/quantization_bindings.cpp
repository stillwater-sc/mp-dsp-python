// quantization_bindings.cpp: ADC/DAC, SQNR, dither, noise shaping → NumPy

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sw/dsp/quantization/quantization.hpp>
#include <sw/dsp/signals/generators.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

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

// ADC: quantize a double signal through a target type and back.
//
// For float-like T (float, posit, cfloat, fixpnt), `static_cast<T>(x)` is a
// faithful narrowing because those types can represent fractional values —
// upstream sw::dsp::ADC handles this path directly.
//
// For integer<N>, `static_cast` truncates any |x| < 1 to zero — that's not a
// quantization model, it's signal annihilation. Physical N-bit ADCs instead
// map the full-scale analog range [-1, 1] onto the integer's representable
// range [-(2^(N-1)-1), 2^(N-1)-1], quantize, then scale back to [-1, 1] for
// downstream math. We detect integer<N> via a concept check and route through
// a scale-quantize-unscale pipeline so sensor_* configs produce the
// quantization noise users expect instead of all zeros.
template <typename T>
static mtl::vec::dense_vector<double>
adc_typed(const mtl::vec::dense_vector<double>& signal) {
	if constexpr (sw::universal::is_integer<T>) {
		// Full-scale magnitude for an N-bit signed integer (N = T::nbits).
		// Subtract 1 so we don't hit the asymmetric -2^(N-1) value, keeping
		// the mapping symmetric around zero. Reasonable for an audio-like
		// signal normalized to [-1, 1]; asymmetric real-world ADCs are out
		// of scope for this binding-layer ADC model.
		constexpr double fs = static_cast<double>((1LL << (T::nbits - 1)) - 1);
		mtl::vec::dense_vector<double> result(signal.size());
		for (std::size_t i = 0; i < signal.size(); ++i) {
			T q = static_cast<T>(signal[i] * fs);
			result[i] = static_cast<double>(q) / fs;
		}
		return result;
	} else {
		sw::dsp::ADC<double, T> adc;
		auto quantized = adc.convert(signal);
		mtl::vec::dense_vector<double> result(signal.size());
		for (std::size_t i = 0; i < signal.size(); ++i) {
			result[i] = static_cast<double>(quantized[i]);
		}
		return result;
	}
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
	case mpdsp::ArithConfig::sensor_8bit:   return adc_typed<mpdsp::int8_sample_t>(signal);
	case mpdsp::ArithConfig::sensor_6bit:   return adc_typed<mpdsp::int6_sample_t>(signal);
	case mpdsp::ArithConfig::fpga_fixed:    return adc_typed<mpdsp::fx1612_t>(signal);
	// Posit taxonomy grid (#81) — posit<N,es> as the sample scalar. posit_8_2
	// also covers the tiny_posit alias (same enum value).
	case mpdsp::ArithConfig::posit_8_0:     return adc_typed<mpdsp::p8_0>(signal);
	case mpdsp::ArithConfig::posit_8_1:     return adc_typed<mpdsp::p8_1>(signal);
	case mpdsp::ArithConfig::posit_8_2:     return adc_typed<mpdsp::p8_2>(signal);
	case mpdsp::ArithConfig::posit_16_0:    return adc_typed<mpdsp::p16_0>(signal);
	case mpdsp::ArithConfig::posit_16_1:    return adc_typed<mpdsp::p16_1>(signal);
	case mpdsp::ArithConfig::posit_16_2:    return adc_typed<mpdsp::p16_2>(signal);
	case mpdsp::ArithConfig::posit_32_0:    return adc_typed<mpdsp::p32_0>(signal);
	case mpdsp::ArithConfig::posit_32_1:    return adc_typed<mpdsp::p32_1>(signal);
	case mpdsp::ArithConfig::posit_32_2:    return adc_typed<mpdsp::p32_2>(signal);
	}
	return signal;
}

// DAC: reconstruct a "digital" signal back to double. In Python both input
// and output are float64, so mechanically `dac` is identical to `adc` — both
// do double → narrow → double. The difference is semantic intent:
// `adc` models quantization on the way in; `dac` models reconstruction on
// the way out. We expose both so callers can write pipeline code that reads
// in the natural order without choosing between confusing names.
template <typename T>
static mtl::vec::dense_vector<double>
dac_typed(const mtl::vec::dense_vector<double>& quantized) {
	// Mirror adc_typed's shape: cast input to the narrow type T once,
	// then use the vector-level convert() overload upstream provides.
	// This keeps dac_typed / adc_typed structurally symmetric — important
	// for a "companion" binding that readers will compare side by side.
	//
	// For integer<N> we apply the same scale-quantize-unscale loop as
	// adc_typed so adc(signal, "sensor_8bit") -> dac stays an idempotent
	// round-trip rather than collapsing to zero.
	if constexpr (sw::universal::is_integer<T>) {
		constexpr double fs = static_cast<double>((1LL << (T::nbits - 1)) - 1);
		mtl::vec::dense_vector<double> result(quantized.size());
		for (std::size_t i = 0; i < quantized.size(); ++i) {
			T q = static_cast<T>(quantized[i] * fs);
			result[i] = static_cast<double>(q) / fs;
		}
		return result;
	} else {
		mtl::vec::dense_vector<T> narrow(quantized.size());
		for (std::size_t i = 0; i < quantized.size(); ++i) {
			narrow[i] = static_cast<T>(quantized[i]);
		}
		sw::dsp::DAC<T, double> dac;
		return dac.convert(narrow);
	}
}

static mtl::vec::dense_vector<double>
dac_dispatch(const mtl::vec::dense_vector<double>& quantized,
              mpdsp::ArithConfig config) {
	switch (config) {
	case mpdsp::ArithConfig::reference:     return dac_typed<double>(quantized);
	case mpdsp::ArithConfig::gpu_baseline:  return dac_typed<float>(quantized);
	case mpdsp::ArithConfig::cf24_config:   return dac_typed<mpdsp::cf24>(quantized);
	case mpdsp::ArithConfig::half_config:   return dac_typed<mpdsp::half_>(quantized);
	case mpdsp::ArithConfig::ml_hw:         return dac_typed<mpdsp::half_>(quantized);
	case mpdsp::ArithConfig::posit_full:    return dac_typed<mpdsp::p16>(quantized);
	case mpdsp::ArithConfig::sensor_8bit:   return dac_typed<mpdsp::int8_sample_t>(quantized);
	case mpdsp::ArithConfig::sensor_6bit:   return dac_typed<mpdsp::int6_sample_t>(quantized);
	case mpdsp::ArithConfig::fpga_fixed:    return dac_typed<mpdsp::fx1612_t>(quantized);
	// Posit taxonomy grid (#81).
	case mpdsp::ArithConfig::posit_8_0:     return dac_typed<mpdsp::p8_0>(quantized);
	case mpdsp::ArithConfig::posit_8_1:     return dac_typed<mpdsp::p8_1>(quantized);
	case mpdsp::ArithConfig::posit_8_2:     return dac_typed<mpdsp::p8_2>(quantized);
	case mpdsp::ArithConfig::posit_16_0:    return dac_typed<mpdsp::p16_0>(quantized);
	case mpdsp::ArithConfig::posit_16_1:    return dac_typed<mpdsp::p16_1>(quantized);
	case mpdsp::ArithConfig::posit_16_2:    return dac_typed<mpdsp::p16_2>(quantized);
	case mpdsp::ArithConfig::posit_32_0:    return dac_typed<mpdsp::p32_0>(quantized);
	case mpdsp::ArithConfig::posit_32_1:    return dac_typed<mpdsp::p32_1>(quantized);
	case mpdsp::ArithConfig::posit_32_2:    return dac_typed<mpdsp::p32_2>(quantized);
	}
	// Unreachable: switch is exhaustive over mpdsp::ArithConfig.
	return quantized;
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

// ---------------------------------------------------------------------------
// Dither and noise-shaper classes: stateful, dtype-dispatched at construction.
// Mirror the IImpl + Impl<T> + make_impl_for_dtype pattern used in
// conditioning_bindings.cpp — see BINDING_PATTERNS.md for rv_policy notes
// (dither/shaper don't expose ndarray properties, so no take_ownership needed).
// ---------------------------------------------------------------------------

namespace {

using mpdsp::bindings::make_impl_for_dtype;

// --- Dither interfaces (shared by RPDF and TPDF) -------------------------

struct IDitherImpl {
	virtual ~IDitherImpl() = default;
	virtual double sample() = 0;
	virtual mtl::vec::dense_vector<double>
		apply(const mtl::vec::dense_vector<double>& signal) = 0;
};

template <template <typename> class DitherT, typename T>
struct DitherImpl : IDitherImpl {
	DitherT<T> inner;
	DitherImpl(double amplitude, unsigned seed)
		: inner(static_cast<T>(amplitude), seed) {}
	double sample() override {
		return static_cast<double>(inner());
	}
	mtl::vec::dense_vector<double>
	apply(const mtl::vec::dense_vector<double>& signal) override {
		// Draw each dither sample in type T (dtype selects the dither's
		// precision), promote to double, add to the original double signal.
		// Critically, the signal itself is NOT cast to T — that would
		// quantize the signal before dither is added, defeating the point
		// of dithering upstream of an explicit ADC stage. A user calling
		//   RPDFDither(amplitude=1e-3, dtype="half").apply(signal)
		// means "add half-precision dither of amplitude 1e-3 to this
		// high-precision signal", not "quantize this signal to half first".
		mtl::vec::dense_vector<double> out(signal.size());
		for (std::size_t i = 0; i < signal.size(); ++i) {
			T noise_t = inner();
			out[i] = signal[i] + static_cast<double>(noise_t);
		}
		return out;
	}
};

// Template templates don't thread cleanly through make_impl_for_dtype (which
// expects a class-template-of-one-type), so spell out the per-dither-class
// factory explicitly. It's short, and the duplication stops at two cases.
template <typename T> using RPDFImpl =
	DitherImpl<sw::dsp::RPDFDither, T>;
template <typename T> using TPDFImpl =
	DitherImpl<sw::dsp::TPDFDither, T>;

static std::unique_ptr<IDitherImpl>
make_rpdf_impl(mpdsp::ArithConfig config, double amplitude, unsigned seed) {
	return make_impl_for_dtype<RPDFImpl, IDitherImpl>(
		config, "RPDFDither", amplitude, seed);
}

static std::unique_ptr<IDitherImpl>
make_tpdf_impl(mpdsp::ArithConfig config, double amplitude, unsigned seed) {
	return make_impl_for_dtype<TPDFImpl, IDitherImpl>(
		config, "TPDFDither", amplitude, seed);
}

// --- Noise shaper interface -------------------------------------------------

struct INoiseShaperImpl {
	virtual ~INoiseShaperImpl() = default;
	virtual double process_sample(double input) = 0;
	virtual mtl::vec::dense_vector<double>
		process_block(const mtl::vec::dense_vector<double>& input) = 0;
	virtual void reset() = 0;
};

template <typename T>
struct NoiseShaperImpl : INoiseShaperImpl {
	// HighPrecT = double (always), LowPrecT = T (dispatched).
	sw::dsp::FirstOrderNoiseShaper<double, T> inner;
	NoiseShaperImpl() = default;
	double process_sample(double input) override {
		return static_cast<double>(inner.process(input));
	}
	mtl::vec::dense_vector<double>
	process_block(const mtl::vec::dense_vector<double>& input) override {
		auto shaped = inner.process(input);  // returns dense_vector<T>
		mtl::vec::dense_vector<double> out(shaped.size());
		for (std::size_t i = 0; i < shaped.size(); ++i) {
			out[i] = static_cast<double>(shaped[i]);
		}
		return out;
	}
	void reset() override { inner.reset(); }
};

static std::unique_ptr<INoiseShaperImpl>
make_noise_shaper_impl(mpdsp::ArithConfig config) {
	return make_impl_for_dtype<NoiseShaperImpl, INoiseShaperImpl>(
		config, "FirstOrderNoiseShaper");
}

}  // namespace

// --- Py-facing wrapper classes ----------------------------------------------

class PyRPDFDither {
public:
	PyRPDFDither(double amplitude, const std::string& dtype, unsigned seed) {
		if (!(amplitude >= 0.0)) {
			throw std::invalid_argument(
				"RPDFDither: amplitude must be non-negative");
		}
		impl_ = make_rpdf_impl(mpdsp::parse_config(dtype), amplitude, seed);
		dtype_ = dtype;
		amplitude_ = amplitude;
	}

	double sample_once() { return impl_->sample(); }

	np_array apply(np_array_ro signal) {
		auto in = numpy_to_vec(signal);
		return vec_to_numpy(impl_->apply(in));
	}

	double amplitude() const { return amplitude_; }
	const std::string& dtype() const { return dtype_; }

private:
	std::unique_ptr<IDitherImpl> impl_;
	std::string dtype_;
	double amplitude_{};
};

class PyTPDFDither {
public:
	PyTPDFDither(double amplitude, const std::string& dtype, unsigned seed) {
		if (!(amplitude >= 0.0)) {
			throw std::invalid_argument(
				"TPDFDither: amplitude must be non-negative");
		}
		impl_ = make_tpdf_impl(mpdsp::parse_config(dtype), amplitude, seed);
		dtype_ = dtype;
		amplitude_ = amplitude;
	}

	double sample_once() { return impl_->sample(); }

	np_array apply(np_array_ro signal) {
		auto in = numpy_to_vec(signal);
		return vec_to_numpy(impl_->apply(in));
	}

	double amplitude() const { return amplitude_; }
	const std::string& dtype() const { return dtype_; }

private:
	std::unique_ptr<IDitherImpl> impl_;
	std::string dtype_;
	double amplitude_{};
};

class PyFirstOrderNoiseShaper {
public:
	explicit PyFirstOrderNoiseShaper(const std::string& dtype) {
		impl_ = make_noise_shaper_impl(mpdsp::parse_config(dtype));
		dtype_ = dtype;
	}

	double process(double input) { return impl_->process_sample(input); }

	np_array process_block(np_array_ro signal) {
		auto in = numpy_to_vec(signal);
		return vec_to_numpy(impl_->process_block(in));
	}

	void reset() { impl_->reset(); }

	const std::string& dtype() const { return dtype_; }

private:
	std::unique_ptr<INoiseShaperImpl> impl_;
	std::string dtype_;
};


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

	m.def("dac", [](np_array_ro quantized, const std::string& dtype) {
		auto v = numpy_to_vec(quantized);
		auto config = mpdsp::parse_config(dtype);
		return vec_to_numpy(dac_dispatch(v, config));
	}, nb::arg("quantized"), nb::arg("dtype") = "reference",
	   "Reconstruct a quantized signal through target type (T -> double). "
	   "Companion to adc(): in Python both sides are float64 so the call is "
	   "mechanically symmetric to adc, but dac models the DAC reconstruction "
	   "step of a full ADC->DAC pipeline explicitly.");

	m.def("available_dtypes", &mpdsp::available_configs,
	   "List available arithmetic configuration names.");

	m.def("bits_of", &mpdsp::bits_of, nb::arg("dtype"),
	   "Return the sample-scalar bit width for `dtype`. Use this to label "
	   "a precision-vs-cost axis instead of hardcoding the mapping. "
	   "Raises ValueError for unknown dtype strings.");

	// RPDFDither — uniform [-amplitude, +amplitude] additive noise.
	nb::class_<PyRPDFDither>(m, "RPDFDither",
			"Rectangular-PDF dither generator.\n\n"
			"Adds uniform noise in [-amplitude, +amplitude] to a signal before\n"
			"quantization, decorrelating quantization error from the signal.\n"
			"Returns new arrays — upstream mutates in-place, but that's an\n"
			"unusual idiom in NumPy so the Python wrapper allocates fresh.")
		.def(nb::init<double, const std::string&, unsigned>(),
		     nb::arg("amplitude"),
		     nb::arg("dtype") = "reference",
		     nb::arg("seed") = 0u,
		     "Construct a dither generator. seed=0 draws from std::random_device.")
		.def("sample", &PyRPDFDither::sample_once,
		      "Draw a single dither sample as a Python float.")
		.def("apply", &PyRPDFDither::apply, nb::arg("signal"),
		      "Dither `signal` (float64 ndarray). Returns a new float64 ndarray.")
		.def_prop_ro("amplitude", &PyRPDFDither::amplitude)
		.def_prop_ro("dtype", &PyRPDFDither::dtype,
		              "Arithmetic configuration selected at construction.");

	// TPDFDither — triangular (sum of two uniform) additive noise.
	nb::class_<PyTPDFDither>(m, "TPDFDither",
			"Triangular-PDF dither generator.\n\n"
			"TPDF is the sum of two RPDF draws — cumulative noise distribution\n"
			"is triangular rather than rectangular. Eliminates the noise-\n"
			"modulation artifact that RPDF leaves on low-level signals, at\n"
			"the cost of +3 dB of noise power.")
		.def(nb::init<double, const std::string&, unsigned>(),
		     nb::arg("amplitude"),
		     nb::arg("dtype") = "reference",
		     nb::arg("seed") = 0u,
		     "Construct a dither generator. seed=0 draws from std::random_device.")
		.def("sample", &PyTPDFDither::sample_once,
		      "Draw a single dither sample as a Python float.")
		.def("apply", &PyTPDFDither::apply, nb::arg("signal"),
		      "Dither `signal` (float64 ndarray). Returns a new float64 ndarray.")
		.def_prop_ro("amplitude", &PyTPDFDither::amplitude)
		.def_prop_ro("dtype", &PyTPDFDither::dtype,
		              "Arithmetic configuration selected at construction.");

	// FirstOrderNoiseShaper — error-feedback quantization shaping.
	nb::class_<PyFirstOrderNoiseShaper>(m, "FirstOrderNoiseShaper",
			"First-order error-feedback noise shaper.\n\n"
			"Quantizes double->`dtype`->double while feeding each sample's\n"
			"quantization error back (negated) onto the next input. This\n"
			"pushes noise energy to higher frequencies (first-order high-\n"
			"pass shaping of the noise floor) — useful upstream of a lowpass\n"
			"reconstruction that will reject the shifted noise.")
		.def(nb::init<const std::string&>(),
		     nb::arg("dtype") = "reference",
		     "Construct a noise shaper. The shaper is deterministic (no seed).")
		.def("process", &PyFirstOrderNoiseShaper::process, nb::arg("input"),
		      "Process a single sample. Returns the shaped+quantized output.")
		.def("process_block", &PyFirstOrderNoiseShaper::process_block,
		      nb::arg("signal"),
		      "Process a float64 ndarray signal. Returns a new float64 ndarray "
		      "with the shaped+quantized output.")
		.def("reset", &PyFirstOrderNoiseShaper::reset,
		      "Clear the error-feedback state to zero.")
		.def_prop_ro("dtype", &PyFirstOrderNoiseShaper::dtype,
		              "Arithmetic configuration selected at construction.");
}
