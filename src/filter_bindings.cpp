// filter_bindings.cpp: IIR filter bindings
//
// Exposes a PyIIRFilter class wrapping sw::dsp::Cascade<double, MaxStages>
// and design functions that return instances of it. Processing is
// type-dispatched: coefficients stay in double; state and sample scalars
// vary per the dtype key (shared with quantization bindings).
//
// Families covered:
//   - Butterworth, Chebyshev I/II, Bessel, Legendre, Elliptic: LP/HP/BP/BS
//   - RBJ: LP/HP/BP/BS/allpass/lowshelf/highshelf (single biquad each)

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <sw/dsp/analysis/condition.hpp>
#include <sw/dsp/analysis/sensitivity.hpp>
#include <sw/dsp/analysis/stability.hpp>
#include <sw/dsp/filter/biquad/biquad.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/biquad/state.hpp>
#include <sw/dsp/filter/fir/fir_design.hpp>
#include <sw/dsp/filter/fir/fir_filter.hpp>
#include <sw/dsp/filter/iir/bessel.hpp>
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/iir/chebyshev1.hpp>
#include <sw/dsp/filter/iir/chebyshev2.hpp>
#include <sw/dsp/filter/iir/elliptic.hpp>
#include <sw/dsp/filter/iir/legendre.hpp>
#include <sw/dsp/filter/iir/rbj.hpp>
#include <sw/dsp/windows/windows.hpp>

#include "types.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

// sw::universal::is_integer trait used by the sample quantization helpers
// below — #include <universal/traits/integer_traits.hpp> is pulled in
// transitively via types.hpp's integer.hpp include.

namespace nb = nanobind;

// Max biquad stages exposed to Python. For Butterworth/Cheby/Bessel/Legendre:
//  - LP/HP templates instantiated with MaxOrder=16 → max_stages = 8
//  - BP/BS templates instantiated with MaxOrder=8  → max_stages = 8 (order doubles)
// All design cascades therefore share type Cascade<double, 8>.
static constexpr int kMaxStages    = 8;
static constexpr int kMaxOrderLPHP = 16;
static constexpr int kMaxOrderBPBS = 8;

using CascadeD = sw::dsp::Cascade<double, kMaxStages>;

namespace {

// ---------------------------------------------------------------------------
// Sample quantization helpers.
//
// For float-like SampleScalar (float, posit, cfloat, fixpnt) a plain cast is
// a faithful narrowing — the type represents fractional values natively.
// For integer<N>, a plain static_cast truncates |x|<1 to zero, annihilating
// any audio-range signal. Mirror the scale-quantize-unscale pipeline from
// adc_typed / project_typed: map the full-scale [-1, 1] input range onto the
// integer's representable range, quantize, then scale back. Keeps the
// filter's sample-path quantization semantics consistent with what the ADC
// binding exposes, so sensor_8bit FIR/IIR output isn't silently zero.
// ---------------------------------------------------------------------------

template <typename SampleScalar>
static inline SampleScalar quantize_sample_in(double x) {
	if constexpr (sw::universal::is_integer<SampleScalar>) {
		constexpr double fs =
			static_cast<double>((1LL << (SampleScalar::nbits - 1)) - 1);
		return static_cast<SampleScalar>(x * fs);
	} else {
		return static_cast<SampleScalar>(x);
	}
}

template <typename SampleScalar>
static inline double quantize_sample_out(SampleScalar y) {
	if constexpr (sw::universal::is_integer<SampleScalar>) {
		constexpr double fs =
			static_cast<double>((1LL << (SampleScalar::nbits - 1)) - 1);
		return static_cast<double>(y) / fs;
	} else {
		return static_cast<double>(y);
	}
}

// ---------------------------------------------------------------------------
// Type-dispatched per-sample processing.
// ---------------------------------------------------------------------------

template <typename StateScalar, typename SampleScalar>
static void process_typed(const CascadeD& cascade,
                          const double* in, double* out, std::size_t n) {
	std::array<sw::dsp::DirectFormI<StateScalar>, kMaxStages> state{};
	for (std::size_t i = 0; i < n; ++i) {
		SampleScalar x = quantize_sample_in<SampleScalar>(in[i]);
		SampleScalar y = cascade.template process<sw::dsp::DirectFormI<StateScalar>,
		                                          SampleScalar>(x, state);
		out[i] = quantize_sample_out<SampleScalar>(y);
	}
}

static void process_dispatch(const CascadeD& cascade,
                             const double* in, double* out, std::size_t n,
                             mpdsp::ArithConfig config) {
	using mpdsp::ArithConfig;
	using mpdsp::cf24;
	using mpdsp::fx1612_t;
	using mpdsp::fx3224_t;
	using mpdsp::half_;
	using mpdsp::int6_sample_t;
	using mpdsp::int8_sample_t;
	using mpdsp::p16;
	using mpdsp::p32;
	using tiny_posit_t = sw::universal::posit<8, 2>;
	switch (config) {
	case ArithConfig::reference:
		process_typed<double, double>(cascade, in, out, n); break;
	case ArithConfig::gpu_baseline:
		process_typed<float, float>(cascade, in, out, n); break;
	case ArithConfig::ml_hw:
		process_typed<float, half_>(cascade, in, out, n); break;
	case ArithConfig::cf24_config:
		process_typed<cf24, cf24>(cascade, in, out, n); break;
	case ArithConfig::half_config:
		process_typed<half_, half_>(cascade, in, out, n); break;
	case ArithConfig::posit_full:
		process_typed<p32, p16>(cascade, in, out, n); break;
	case ArithConfig::tiny_posit:
		process_typed<tiny_posit_t, tiny_posit_t>(cascade, in, out, n); break;
	// Sensor configs: coefficient/state in double, sample quantized through
	// integer<N>. integer<N> is ADL-castable from double via static_cast, so
	// process_typed<double, int8_sample_t> models "signal arrives on an 8-bit
	// ADC, filter state stays wide" — matches issue #55's sensor semantics.
	case ArithConfig::sensor_8bit:
		process_typed<double, int8_sample_t>(cascade, in, out, n); break;
	case ArithConfig::sensor_6bit:
		process_typed<double, int6_sample_t>(cascade, in, out, n); break;
	case ArithConfig::fpga_fixed:
		process_typed<fx3224_t, fx1612_t>(cascade, in, out, n); break;
	}
}

// ---------------------------------------------------------------------------
// NumPy helpers.
// ---------------------------------------------------------------------------

using np_f64     = nb::ndarray<nb::numpy, double>;
using np_f64_ro  = nb::ndarray<nb::numpy, const double, nb::ndim<1>>;
using np_c128    = nb::ndarray<nb::numpy, std::complex<double>>;

static np_f64 make_f64_array(std::size_t n, double*& out_ptr) {
	auto* data = new double[n];
	out_ptr = data;
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	std::size_t shape[1] = { n };
	return np_f64(data, 1, shape, owner);
}

static np_c128 make_c128_array(std::size_t n, std::complex<double>*& out_ptr) {
	auto* data = new std::complex<double>[n];
	out_ptr = data;
	nb::capsule owner(data, [](void* p) noexcept {
		delete[] static_cast<std::complex<double>*>(p);
	});
	std::size_t shape[1] = { n };
	return np_c128(data, 1, shape, owner);
}

// ---------------------------------------------------------------------------
// Shared parameter validation.
// ---------------------------------------------------------------------------

static void check_sample_rate(double sr, const char* name) {
	if (!(sr > 0.0)) {
		throw std::invalid_argument(std::string(name) +
			": sample_rate must be positive");
	}
}

static void check_frequency(double f, double sr, const char* name,
                            const char* freq_name) {
	if (!(f > 0.0) || f >= 0.5 * sr) {
		throw std::invalid_argument(std::string(name) + ": " + freq_name +
			" must be in (0, sample_rate/2)");
	}
}

static void check_order(int order, int max_order, const char* name) {
	if (order < 1 || order > max_order) {
		throw std::invalid_argument(std::string(name) +
			": order must be in [1, " + std::to_string(max_order) + "]");
	}
}

static void check_positive(double v, const char* name, const char* field) {
	if (!(v > 0.0)) {
		throw std::invalid_argument(std::string(name) + ": " + field +
			" must be positive");
	}
}

static void check_bp_band(double center, double width, double sr,
                          const char* name) {
	if (!(center > 0.0) || !(width > 0.0)) {
		throw std::invalid_argument(std::string(name) +
			": center_freq and width_freq must be positive");
	}
	double half = 0.5 * width;
	if (center - half <= 0.0 || center + half >= 0.5 * sr) {
		throw std::invalid_argument(std::string(name) +
			": passband [center - width/2, center + width/2] must fit within (0, sample_rate/2)");
	}
}

} // namespace

// ---------------------------------------------------------------------------
// PyIIRFilter: opaque handle wrapping a double-precision biquad cascade.
// ---------------------------------------------------------------------------

class PyIIRFilter {
public:
	CascadeD cascade;

	int num_stages() const { return cascade.num_stages(); }

	std::vector<std::tuple<double, double, double, double, double>>
	coefficients() const {
		std::vector<std::tuple<double, double, double, double, double>> out;
		out.reserve(static_cast<std::size_t>(cascade.num_stages()));
		for (int i = 0; i < cascade.num_stages(); ++i) {
			const auto& s = cascade.stage(i);
			out.emplace_back(s.b0, s.b1, s.b2, s.a1, s.a2);
		}
		return out;
	}

	std::vector<std::complex<double>> poles() const {
		std::vector<std::complex<double>> out;
		out.reserve(static_cast<std::size_t>(cascade.num_stages()) * 2);
		for (int i = 0; i < cascade.num_stages(); ++i) {
			sw::dsp::BiquadPoleState<double> pz(cascade.stage(i));
			out.push_back(pz.poles.first);
			const auto& second = pz.poles.second;
			if (second != std::complex<double>{}) {
				out.push_back(second);
			}
		}
		return out;
	}

	np_f64 process(np_f64_ro signal, const std::string& dtype) const {
		std::size_t n = signal.shape(0);
		double* out_ptr = nullptr;
		auto arr = make_f64_array(n, out_ptr);
		auto config = mpdsp::parse_config(dtype);
		process_dispatch(cascade, signal.data(), out_ptr, n, config);
		return arr;
	}

	np_c128 frequency_response(np_f64_ro normalized_freqs) const {
		std::size_t n = normalized_freqs.shape(0);
		std::complex<double>* out_ptr = nullptr;
		auto arr = make_c128_array(n, out_ptr);
		const double* f = normalized_freqs.data();
		for (std::size_t i = 0; i < n; ++i) {
			out_ptr[i] = cascade.response(f[i]);
		}
		return arr;
	}

	// --- Extended diagnostics --------------------------------------------

	double stability_margin() const {
		return sw::dsp::stability_margin(cascade);
	}

	double condition_number(int num_freqs) const {
		return sw::dsp::cascade_condition_number(cascade, num_freqs);
	}

	double worst_case_sensitivity(double epsilon) const {
		return sw::dsp::worst_case_sensitivity(cascade, epsilon);
	}

	// Pole displacement: quantize each coefficient through the target dtype
	// (double -> T -> double) and measure how far the resulting poles move.
	// This captures the dominant quantization effect (coefficient precision);
	// pole extraction is done in double on both cascades.
	double pole_displacement(const std::string& dtype) const;
};

// ---------------------------------------------------------------------------
// PyFIRFilter: opaque handle wrapping a double-precision tap vector.
// ---------------------------------------------------------------------------

class PyFIRFilter {
public:
	mtl::vec::dense_vector<double> taps;

	int num_taps() const { return static_cast<int>(taps.size()); }

	// Taps as a NumPy float64 array (copied).
	np_f64 coefficients() const {
		std::size_t n = taps.size();
		double* out_ptr = nullptr;
		auto arr = make_f64_array(n, out_ptr);
		for (std::size_t i = 0; i < n; ++i) out_ptr[i] = taps[i];
		return arr;
	}

	// Impulse response of length `length` — the taps padded (or truncated).
	np_f64 impulse_response(int length) const {
		if (length <= 0) {
			throw std::invalid_argument(
				"impulse_response: length must be positive");
		}
		std::size_t n = static_cast<std::size_t>(length);
		double* out_ptr = nullptr;
		auto arr = make_f64_array(n, out_ptr);
		std::size_t copy_n = std::min(n, taps.size());
		for (std::size_t i = 0; i < copy_n; ++i) out_ptr[i] = taps[i];
		for (std::size_t i = copy_n; i < n; ++i) out_ptr[i] = 0.0;
		return arr;
	}

	// H(e^{j2*pi*f}) = sum_n taps[n] * exp(-j * 2*pi*f * n).
	np_c128 frequency_response(np_f64_ro normalized_freqs) const {
		std::size_t n = normalized_freqs.shape(0);
		std::complex<double>* out_ptr = nullptr;
		auto arr = make_c128_array(n, out_ptr);
		const double* f = normalized_freqs.data();
		std::size_t N = taps.size();
		for (std::size_t k = 0; k < n; ++k) {
			std::complex<double> acc{};
			double w = 2.0 * 3.14159265358979323846 * f[k];
			for (std::size_t i = 0; i < N; ++i) {
				acc += taps[i] * std::exp(std::complex<double>(0.0, -w * static_cast<double>(i)));
			}
			out_ptr[k] = acc;
		}
		return arr;
	}

	np_f64 process(np_f64_ro signal, const std::string& dtype) const;
};

namespace {

// ---------------------------------------------------------------------------
// FIR type-dispatched processing. Each call spins up a fresh FIRFilter with
// taps cast to StateScalar (== CoeffScalar here). This matches the IIR
// pattern: the Python-facing filter object is stateless across process() calls.
// ---------------------------------------------------------------------------

template <typename StateScalar, typename SampleScalar>
static void fir_process_typed(const mtl::vec::dense_vector<double>& taps_d,
                              const double* in, double* out, std::size_t n) {
	mtl::vec::dense_vector<StateScalar> taps(taps_d.size());
	for (std::size_t i = 0; i < taps_d.size(); ++i) {
		taps[i] = static_cast<StateScalar>(taps_d[i]);
	}
	sw::dsp::FIRFilter<StateScalar, StateScalar, SampleScalar> filt(taps);
	for (std::size_t i = 0; i < n; ++i) {
		SampleScalar x = quantize_sample_in<SampleScalar>(in[i]);
		out[i] = quantize_sample_out<SampleScalar>(filt.process(x));
	}
}

static void fir_process_dispatch(const mtl::vec::dense_vector<double>& taps_d,
                                 const double* in, double* out, std::size_t n,
                                 mpdsp::ArithConfig config) {
	using mpdsp::ArithConfig;
	using mpdsp::cf24;
	using mpdsp::fx1612_t;
	using mpdsp::fx3224_t;
	using mpdsp::half_;
	using mpdsp::int6_sample_t;
	using mpdsp::int8_sample_t;
	using mpdsp::p16;
	using mpdsp::p32;
	using tiny_posit_t = sw::universal::posit<8, 2>;
	switch (config) {
	case ArithConfig::reference:
		fir_process_typed<double, double>(taps_d, in, out, n); break;
	case ArithConfig::gpu_baseline:
		fir_process_typed<float, float>(taps_d, in, out, n); break;
	case ArithConfig::ml_hw:
		fir_process_typed<float, half_>(taps_d, in, out, n); break;
	case ArithConfig::cf24_config:
		fir_process_typed<cf24, cf24>(taps_d, in, out, n); break;
	case ArithConfig::half_config:
		fir_process_typed<half_, half_>(taps_d, in, out, n); break;
	case ArithConfig::posit_full:
		fir_process_typed<p32, p16>(taps_d, in, out, n); break;
	case ArithConfig::tiny_posit:
		fir_process_typed<tiny_posit_t, tiny_posit_t>(taps_d, in, out, n); break;
	case ArithConfig::sensor_8bit:
		fir_process_typed<double, int8_sample_t>(taps_d, in, out, n); break;
	case ArithConfig::sensor_6bit:
		fir_process_typed<double, int6_sample_t>(taps_d, in, out, n); break;
	case ArithConfig::fpga_fixed:
		fir_process_typed<fx3224_t, fx1612_t>(taps_d, in, out, n); break;
	}
}

// Build a window of the requested kind (name matches signal_bindings.cpp).
static mtl::vec::dense_vector<double>
make_window(const std::string& name, std::size_t N, double kaiser_beta) {
	using namespace sw::dsp;
	if (name == "hamming")     return hamming_window<double>(N);
	if (name == "hanning")     return hanning_window<double>(N);
	if (name == "blackman")    return blackman_window<double>(N);
	if (name == "rectangular") return rectangular_window<double>(N);
	if (name == "flat_top")    return flat_top_window<double>(N);
	if (name == "kaiser")      return kaiser_window<double>(N, kaiser_beta);
	throw std::invalid_argument("Unknown window: " + name +
		" (expected hamming, hanning, blackman, rectangular, flat_top, kaiser)");
}

// Common FIR parameter validation shared by the design functions below.
static void check_num_taps(int n, const char* name) {
	if (n < 1) {
		throw std::invalid_argument(std::string(name) +
			": num_taps must be positive");
	}
}

} // namespace

np_f64 PyFIRFilter::process(np_f64_ro signal, const std::string& dtype) const {
	std::size_t n = signal.shape(0);
	double* out_ptr = nullptr;
	auto arr = make_f64_array(n, out_ptr);
	auto config = mpdsp::parse_config(dtype);
	fir_process_dispatch(taps, signal.data(), out_ptr, n, config);
	return arr;
}

namespace {

// ---------------------------------------------------------------------------
// Coefficient quantization for pole-displacement analysis.
// Round-trips each coefficient through target type T (double -> T -> double)
// so pole extraction can use the existing double-precision cascade machinery.
// ---------------------------------------------------------------------------

template <typename T>
static double round_trip(double v) {
	return static_cast<double>(static_cast<T>(v));
}

template <typename T>
static CascadeD quantize_cascade(const CascadeD& src) {
	CascadeD dst;
	dst.set_num_stages(src.num_stages());
	for (int i = 0; i < src.num_stages(); ++i) {
		const auto& s = src.stage(i);
		auto& d = dst.stage(i);
		d.b0 = round_trip<T>(s.b0);
		d.b1 = round_trip<T>(s.b1);
		d.b2 = round_trip<T>(s.b2);
		d.a1 = round_trip<T>(s.a1);
		d.a2 = round_trip<T>(s.a2);
	}
	return dst;
}

static double pole_displacement_dispatch(const CascadeD& src,
                                         mpdsp::ArithConfig config) {
	using mpdsp::ArithConfig;
	using mpdsp::cf24;
	using mpdsp::fx3224_t;
	using mpdsp::half_;
	using mpdsp::p16;
	using mpdsp::p32;
	using tiny_posit_t = sw::universal::posit<8, 2>;
	CascadeD quantized;
	switch (config) {
	case ArithConfig::reference:    return 0.0;  // no quantization
	case ArithConfig::gpu_baseline: quantized = quantize_cascade<float>(src); break;
	case ArithConfig::ml_hw:        quantized = quantize_cascade<half_>(src); break;
	case ArithConfig::cf24_config:  quantized = quantize_cascade<cf24>(src); break;
	case ArithConfig::half_config:  quantized = quantize_cascade<half_>(src); break;
	case ArithConfig::posit_full:   quantized = quantize_cascade<p32>(src); break;
	case ArithConfig::tiny_posit:   quantized = quantize_cascade<tiny_posit_t>(src); break;
	// sensor_* keep coefficients at double (only the sample path quantizes),
	// so coefficient-level pole displacement is zero for them.
	case ArithConfig::sensor_8bit:
	case ArithConfig::sensor_6bit:
		return 0.0;
	case ArithConfig::fpga_fixed:
		quantized = quantize_cascade<fx3224_t>(src); break;
	}
	return sw::dsp::pole_displacement(src, quantized);
}

// ---------------------------------------------------------------------------
// Design factories. DesignT is a class with .setup(...) and .cascade().
// For LP/HP: DesignT instantiated with MaxOrder = kMaxOrderLPHP (16).
// For BP/BS: DesignT instantiated with MaxOrder = kMaxOrderBPBS (8).
// Both produce Cascade<double, kMaxStages = 8>.
// ---------------------------------------------------------------------------

template <typename DesignT, typename... SetupArgs>
static PyIIRFilter make_from_design(int order, SetupArgs... args) {
	DesignT design;
	design.setup(order, args...);
	PyIIRFilter filt;
	filt.cascade = design.cascade();
	return filt;
}

// RBJ designs have no 'order' and their cascade is size-1. Copy the single
// biquad into stage 0 of our size-kMaxStages cascade.
template <typename DesignT, typename... SetupArgs>
static PyIIRFilter make_from_rbj(SetupArgs... args) {
	DesignT design;
	design.setup(args...);
	PyIIRFilter filt;
	filt.cascade.set_num_stages(1);
	filt.cascade.stage(0) = design.cascade().stage(0);
	return filt;
}

} // namespace

double PyIIRFilter::pole_displacement(const std::string& dtype) const {
	auto config = mpdsp::parse_config(dtype);
	return pole_displacement_dispatch(cascade, config);
}

// ---------------------------------------------------------------------------
// Module registration.
// ---------------------------------------------------------------------------

void bind_filters(nb::module_& m) {
	nb::class_<PyIIRFilter>(m, "IIRFilter",
		"Cascade-of-biquads IIR filter.\n\n"
		"Construct via one of the design functions. Coefficients are stored "
		"in double precision; process() dispatches state/sample arithmetic "
		"on the dtype argument.")
		.def("num_stages", &PyIIRFilter::num_stages,
		     "Number of active biquad sections.")
		.def("coefficients", &PyIIRFilter::coefficients,
		     "List of (b0, b1, b2, a1, a2) tuples, one per stage.")
		.def("poles", &PyIIRFilter::poles,
		     "List of complex pole locations in the z-plane.")
		.def("process", &PyIIRFilter::process,
		     nb::arg("signal"), nb::arg("dtype") = "reference",
		     "Filter a signal. dtype selects arithmetic for state and samples "
		     "(see available_dtypes()). Returns NumPy float64.")
		.def("frequency_response", &PyIIRFilter::frequency_response,
		     nb::arg("normalized_freqs"),
		     "Evaluate H(e^{j2*pi*f}) at each normalized frequency (f/fs). "
		     "Returns complex128.")
		.def("stability_margin", &PyIIRFilter::stability_margin,
		     "1 - max(|pole|). Positive = stable, 0 = marginal, < 0 = unstable.")
		.def("condition_number", &PyIIRFilter::condition_number,
		     nb::arg("num_freqs") = 256,
		     "Worst-case relative change in |H| per coefficient perturbation "
		     "across stages. Higher = more sensitive to coefficient quantization.")
		.def("worst_case_sensitivity", &PyIIRFilter::worst_case_sensitivity,
		     nb::arg("epsilon") = 1e-8,
		     "Worst-case |d(max_pole_radius)/d(coeff)| across stages, "
		     "computed by finite differences.")
		.def("pole_displacement", &PyIIRFilter::pole_displacement,
		     nb::arg("dtype"),
		     "Max pole displacement when coefficients are quantized through "
		     "the target dtype (see available_dtypes). Returns 0 for 'reference'.");

	namespace iir = sw::dsp::iir;
	namespace rbj = sw::dsp::iir::rbj;

	// Common argument names for the LP/HP/BP/BS registration blocks below.
	constexpr const char* A_ORDER = "order";
	constexpr const char* A_SR    = "sample_rate";
	constexpr const char* A_CUT   = "cutoff";
	constexpr const char* A_CTR   = "center_freq";
	constexpr const char* A_WID   = "width_freq";

	// =======================================================================
	// Butterworth — no extra parameters.
	// =======================================================================

	m.def("butterworth_lowpass",
		[](int order, double sr, double cutoff) {
			const char* n = "butterworth_lowpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			return make_from_design<iir::ButterworthLowPass<kMaxOrderLPHP>>(
				order, sr, cutoff);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT),
		"Design a Butterworth lowpass filter. order in [1, 16].");

	m.def("butterworth_highpass",
		[](int order, double sr, double cutoff) {
			const char* n = "butterworth_highpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			return make_from_design<iir::ButterworthHighPass<kMaxOrderLPHP>>(
				order, sr, cutoff);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT),
		"Design a Butterworth highpass filter. order in [1, 16].");

	m.def("butterworth_bandpass",
		[](int order, double sr, double center_freq, double width_freq) {
			const char* n = "butterworth_bandpass";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			return make_from_design<iir::ButterworthBandPass<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		"Design a Butterworth bandpass filter. order in [1, 8] "
		"(the bandpass transform doubles the internal order).");

	m.def("butterworth_bandstop",
		[](int order, double sr, double center_freq, double width_freq) {
			const char* n = "butterworth_bandstop";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			return make_from_design<iir::ButterworthBandStop<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		"Design a Butterworth bandstop filter. order in [1, 8].");

	// =======================================================================
	// Chebyshev I — equiripple passband, extra ripple_db parameter.
	// =======================================================================

	m.def("chebyshev1_lowpass",
		[](int order, double sr, double cutoff, double ripple_db) {
			const char* n = "chebyshev1_lowpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(ripple_db, n, "ripple_db");
			return make_from_design<iir::ChebyshevILowPass<kMaxOrderLPHP>>(
				order, sr, cutoff, ripple_db);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT), nb::arg("ripple_db"),
		"Design a Chebyshev Type I lowpass filter with equiripple passband.");

	m.def("chebyshev1_highpass",
		[](int order, double sr, double cutoff, double ripple_db) {
			const char* n = "chebyshev1_highpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(ripple_db, n, "ripple_db");
			return make_from_design<iir::ChebyshevIHighPass<kMaxOrderLPHP>>(
				order, sr, cutoff, ripple_db);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT), nb::arg("ripple_db"),
		"Design a Chebyshev Type I highpass filter with equiripple passband.");

	m.def("chebyshev1_bandpass",
		[](int order, double sr, double center_freq, double width_freq, double ripple_db) {
			const char* n = "chebyshev1_bandpass";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			check_positive(ripple_db, n, "ripple_db");
			return make_from_design<iir::ChebyshevIBandPass<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq, ripple_db);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		   nb::arg("ripple_db"),
		"Design a Chebyshev Type I bandpass filter.");

	m.def("chebyshev1_bandstop",
		[](int order, double sr, double center_freq, double width_freq, double ripple_db) {
			const char* n = "chebyshev1_bandstop";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			check_positive(ripple_db, n, "ripple_db");
			return make_from_design<iir::ChebyshevIBandStop<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq, ripple_db);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		   nb::arg("ripple_db"),
		"Design a Chebyshev Type I bandstop filter.");

	// =======================================================================
	// Chebyshev II — monotonic passband, equiripple stopband, stopband_db param.
	// =======================================================================

	m.def("chebyshev2_lowpass",
		[](int order, double sr, double cutoff, double stopband_db) {
			const char* n = "chebyshev2_lowpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(stopband_db, n, "stopband_db");
			return make_from_design<iir::ChebyshevIILowPass<kMaxOrderLPHP>>(
				order, sr, cutoff, stopband_db);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT), nb::arg("stopband_db"),
		"Design an inverse Chebyshev (Type II) lowpass filter with "
		"equiripple stopband.");

	m.def("chebyshev2_highpass",
		[](int order, double sr, double cutoff, double stopband_db) {
			const char* n = "chebyshev2_highpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(stopband_db, n, "stopband_db");
			return make_from_design<iir::ChebyshevIIHighPass<kMaxOrderLPHP>>(
				order, sr, cutoff, stopband_db);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT), nb::arg("stopband_db"),
		"Design an inverse Chebyshev (Type II) highpass filter.");

	m.def("chebyshev2_bandpass",
		[](int order, double sr, double center_freq, double width_freq, double stopband_db) {
			const char* n = "chebyshev2_bandpass";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			check_positive(stopband_db, n, "stopband_db");
			return make_from_design<iir::ChebyshevIIBandPass<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq, stopband_db);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		   nb::arg("stopband_db"),
		"Design an inverse Chebyshev (Type II) bandpass filter.");

	m.def("chebyshev2_bandstop",
		[](int order, double sr, double center_freq, double width_freq, double stopband_db) {
			const char* n = "chebyshev2_bandstop";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			check_positive(stopband_db, n, "stopband_db");
			return make_from_design<iir::ChebyshevIIBandStop<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq, stopband_db);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		   nb::arg("stopband_db"),
		"Design an inverse Chebyshev (Type II) bandstop filter.");

	// =======================================================================
	// Bessel — maximally flat group delay. Same signatures as Butterworth.
	// =======================================================================

	m.def("bessel_lowpass",
		[](int order, double sr, double cutoff) {
			const char* n = "bessel_lowpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			return make_from_design<iir::BesselLowPass<kMaxOrderLPHP>>(
				order, sr, cutoff);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT),
		"Design a Bessel (Thomson) lowpass filter — maximally flat group delay.");

	m.def("bessel_highpass",
		[](int order, double sr, double cutoff) {
			const char* n = "bessel_highpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			return make_from_design<iir::BesselHighPass<kMaxOrderLPHP>>(
				order, sr, cutoff);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT),
		"Design a Bessel highpass filter.");

	m.def("bessel_bandpass",
		[](int order, double sr, double center_freq, double width_freq) {
			const char* n = "bessel_bandpass";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			return make_from_design<iir::BesselBandPass<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		"Design a Bessel bandpass filter.");

	m.def("bessel_bandstop",
		[](int order, double sr, double center_freq, double width_freq) {
			const char* n = "bessel_bandstop";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			return make_from_design<iir::BesselBandStop<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		"Design a Bessel bandstop filter.");

	// =======================================================================
	// Legendre — steepest monotonic transition. Same signatures as Butterworth.
	// =======================================================================

	m.def("legendre_lowpass",
		[](int order, double sr, double cutoff) {
			const char* n = "legendre_lowpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			return make_from_design<iir::LegendreLowPass<kMaxOrderLPHP>>(
				order, sr, cutoff);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT),
		"Design a Legendre (Papoulis) lowpass filter — steepest monotonic "
		"passband response.");

	m.def("legendre_highpass",
		[](int order, double sr, double cutoff) {
			const char* n = "legendre_highpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			return make_from_design<iir::LegendreHighPass<kMaxOrderLPHP>>(
				order, sr, cutoff);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT),
		"Design a Legendre highpass filter.");

	m.def("legendre_bandpass",
		[](int order, double sr, double center_freq, double width_freq) {
			const char* n = "legendre_bandpass";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			return make_from_design<iir::LegendreBandPass<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		"Design a Legendre bandpass filter.");

	m.def("legendre_bandstop",
		[](int order, double sr, double center_freq, double width_freq) {
			const char* n = "legendre_bandstop";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			return make_from_design<iir::LegendreBandStop<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		"Design a Legendre bandstop filter.");

	// =======================================================================
	// Elliptic (Cauer) — equiripple passband and stopband. Takes ripple_db
	// and a selectivity parameter 'rolloff' in [0.1, 5.0]. Upstream validates
	// both; higher rolloff gives a steeper transition with more stopband ripple.
	// =======================================================================

	m.def("elliptic_lowpass",
		[](int order, double sr, double cutoff, double ripple_db, double rolloff) {
			const char* n = "elliptic_lowpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			return make_from_design<iir::EllipticLowPass<kMaxOrderLPHP>>(
				order, sr, cutoff, ripple_db, rolloff);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT),
		   nb::arg("ripple_db"), nb::arg("rolloff") = 1.0,
		"Design an Elliptic (Cauer) lowpass filter — equiripple in both "
		"passband and stopband. rolloff in [0.1, 5.0] controls transition "
		"selectivity (higher = steeper).");

	m.def("elliptic_highpass",
		[](int order, double sr, double cutoff, double ripple_db, double rolloff) {
			const char* n = "elliptic_highpass";
			check_order(order, kMaxOrderLPHP, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			return make_from_design<iir::EllipticHighPass<kMaxOrderLPHP>>(
				order, sr, cutoff, ripple_db, rolloff);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CUT),
		   nb::arg("ripple_db"), nb::arg("rolloff") = 1.0,
		"Design an Elliptic highpass filter. rolloff in [0.1, 5.0].");

	m.def("elliptic_bandpass",
		[](int order, double sr, double center_freq, double width_freq,
		   double ripple_db, double rolloff) {
			const char* n = "elliptic_bandpass";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			return make_from_design<iir::EllipticBandPass<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq, ripple_db, rolloff);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		   nb::arg("ripple_db"), nb::arg("rolloff") = 1.0,
		"Design an Elliptic bandpass filter.");

	m.def("elliptic_bandstop",
		[](int order, double sr, double center_freq, double width_freq,
		   double ripple_db, double rolloff) {
			const char* n = "elliptic_bandstop";
			check_order(order, kMaxOrderBPBS, n);
			check_sample_rate(sr, n);
			check_bp_band(center_freq, width_freq, sr, n);
			return make_from_design<iir::EllipticBandStop<kMaxOrderBPBS>>(
				order, sr, center_freq, width_freq, ripple_db, rolloff);
		}, nb::arg(A_ORDER), nb::arg(A_SR), nb::arg(A_CTR), nb::arg(A_WID),
		   nb::arg("ripple_db"), nb::arg("rolloff") = 1.0,
		"Design an Elliptic bandstop filter.");

	// =======================================================================
	// RBJ Audio EQ Cookbook — single biquad per variant, no 'order' parameter.
	// =======================================================================

	m.def("rbj_lowpass",
		[](double sr, double cutoff, double q) {
			const char* n = "rbj_lowpass";
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(q, n, "q");
			return make_from_rbj<rbj::LowPass<>>(sr, cutoff, q);
		}, nb::arg(A_SR), nb::arg(A_CUT), nb::arg("q") = 0.7071,
		"RBJ biquad lowpass. q ~ 0.7071 gives a Butterworth-like response.");

	m.def("rbj_highpass",
		[](double sr, double cutoff, double q) {
			const char* n = "rbj_highpass";
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(q, n, "q");
			return make_from_rbj<rbj::HighPass<>>(sr, cutoff, q);
		}, nb::arg(A_SR), nb::arg(A_CUT), nb::arg("q") = 0.7071,
		"RBJ biquad highpass.");

	m.def("rbj_bandpass",
		[](double sr, double center_freq, double bandwidth) {
			const char* n = "rbj_bandpass";
			check_sample_rate(sr, n);
			check_frequency(center_freq, sr, n, "center_freq");
			check_positive(bandwidth, n, "bandwidth");
			return make_from_rbj<rbj::BandPass<>>(sr, center_freq, bandwidth);
		}, nb::arg(A_SR), nb::arg(A_CTR), nb::arg("bandwidth") = 1.0,
		"RBJ biquad bandpass. bandwidth is in octaves.");

	m.def("rbj_bandstop",
		[](double sr, double center_freq, double bandwidth) {
			const char* n = "rbj_bandstop";
			check_sample_rate(sr, n);
			check_frequency(center_freq, sr, n, "center_freq");
			check_positive(bandwidth, n, "bandwidth");
			return make_from_rbj<rbj::BandStop<>>(sr, center_freq, bandwidth);
		}, nb::arg(A_SR), nb::arg(A_CTR), nb::arg("bandwidth") = 1.0,
		"RBJ biquad bandstop (notch). bandwidth is in octaves.");

	m.def("rbj_allpass",
		[](double sr, double center_freq, double q) {
			const char* n = "rbj_allpass";
			check_sample_rate(sr, n);
			check_frequency(center_freq, sr, n, "center_freq");
			check_positive(q, n, "q");
			return make_from_rbj<rbj::AllPass<>>(sr, center_freq, q);
		}, nb::arg(A_SR), nb::arg(A_CTR), nb::arg("q") = 0.7071,
		"RBJ biquad allpass — unit magnitude, phase shift only.");

	// Shelf filters: gain_db is intentionally not validated — any real value
	// is meaningful (0 dB is a legal unity shelf, negative values cut).
	m.def("rbj_lowshelf",
		[](double sr, double cutoff, double gain_db, double slope) {
			const char* n = "rbj_lowshelf";
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(slope, n, "slope");
			return make_from_rbj<rbj::LowShelf<>>(sr, cutoff, gain_db, slope);
		}, nb::arg(A_SR), nb::arg(A_CUT), nb::arg("gain_db"),
		   nb::arg("slope") = 1.0,
		"RBJ biquad low shelf. gain_db is the low-frequency shelf gain.");

	m.def("rbj_highshelf",
		[](double sr, double cutoff, double gain_db, double slope) {
			const char* n = "rbj_highshelf";
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(slope, n, "slope");
			return make_from_rbj<rbj::HighShelf<>>(sr, cutoff, gain_db, slope);
		}, nb::arg(A_SR), nb::arg(A_CUT), nb::arg("gain_db"),
		   nb::arg("slope") = 1.0,
		"RBJ biquad high shelf. gain_db is the high-frequency shelf gain.");

	// =======================================================================
	// FIR filters.
	// =======================================================================

	nb::class_<PyFIRFilter>(m, "FIRFilter",
		"Finite-impulse-response filter with a double-precision tap vector.\n\n"
		"Construct via fir_lowpass / fir_highpass / fir_bandpass / fir_bandstop, "
		"or from explicit coefficients via fir_filter(taps). "
		"process() dispatches state/sample arithmetic on the dtype argument.")
		.def("num_taps", &PyFIRFilter::num_taps,
		     "Number of tap coefficients.")
		.def("coefficients", &PyFIRFilter::coefficients,
		     "Taps as a NumPy float64 array.")
		.def("impulse_response", &PyFIRFilter::impulse_response,
		     nb::arg("length"),
		     "Impulse response — the taps, padded or truncated to `length`.")
		.def("frequency_response", &PyFIRFilter::frequency_response,
		     nb::arg("normalized_freqs"),
		     "Evaluate H(e^{j2*pi*f}) at each normalized frequency (f/fs). "
		     "Returns complex128.")
		.def("process", &PyFIRFilter::process,
		     nb::arg("signal"), nb::arg("dtype") = "reference",
		     "Filter a signal. dtype selects arithmetic for taps, state, and "
		     "samples (see available_dtypes()). Returns NumPy float64.");

	m.def("fir_filter",
		[](np_f64_ro coeffs) {
			std::size_t n = coeffs.shape(0);
			check_num_taps(static_cast<int>(n), "fir_filter");
			PyFIRFilter f;
			f.taps = mtl::vec::dense_vector<double>(n);
			const double* src = coeffs.data();
			for (std::size_t i = 0; i < n; ++i) f.taps[i] = src[i];
			return f;
		}, nb::arg("coefficients"),
		"Construct an FIR filter from explicit tap coefficients.");

	m.def("fir_lowpass",
		[](int num_taps, double sr, double cutoff,
		   const std::string& window, double kaiser_beta) {
			const char* n = "fir_lowpass";
			check_num_taps(num_taps, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			auto w = make_window(window, static_cast<std::size_t>(num_taps), kaiser_beta);
			PyFIRFilter f;
			f.taps = sw::dsp::design_fir_lowpass<double>(
				static_cast<std::size_t>(num_taps), cutoff / sr, w);
			return f;
		}, nb::arg("num_taps"), nb::arg(A_SR), nb::arg(A_CUT),
		   nb::arg("window") = "hamming", nb::arg("kaiser_beta") = 8.6,
		"Design an FIR lowpass filter via the window method.");

	m.def("fir_highpass",
		[](int num_taps, double sr, double cutoff,
		   const std::string& window, double kaiser_beta) {
			const char* n = "fir_highpass";
			check_num_taps(num_taps, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			auto w = make_window(window, static_cast<std::size_t>(num_taps), kaiser_beta);
			PyFIRFilter f;
			f.taps = sw::dsp::design_fir_highpass<double>(
				static_cast<std::size_t>(num_taps), cutoff / sr, w);
			return f;
		}, nb::arg("num_taps"), nb::arg(A_SR), nb::arg(A_CUT),
		   nb::arg("window") = "hamming", nb::arg("kaiser_beta") = 8.6,
		"Design an FIR highpass filter via spectral inversion of a lowpass.");

	m.def("fir_bandpass",
		[](int num_taps, double sr, double f_low, double f_high,
		   const std::string& window, double kaiser_beta) {
			const char* n = "fir_bandpass";
			check_num_taps(num_taps, n);
			check_sample_rate(sr, n);
			check_frequency(f_low, sr, n, "f_low");
			check_frequency(f_high, sr, n, "f_high");
			if (!(f_high > f_low)) {
				throw std::invalid_argument(
					"fir_bandpass: f_high must be greater than f_low");
			}
			auto w = make_window(window, static_cast<std::size_t>(num_taps), kaiser_beta);
			PyFIRFilter f;
			f.taps = sw::dsp::design_fir_bandpass<double>(
				static_cast<std::size_t>(num_taps), f_low / sr, f_high / sr, w);
			return f;
		}, nb::arg("num_taps"), nb::arg(A_SR), nb::arg("f_low"), nb::arg("f_high"),
		   nb::arg("window") = "hamming", nb::arg("kaiser_beta") = 8.6,
		"Design an FIR bandpass filter.");

	m.def("fir_bandstop",
		[](int num_taps, double sr, double f_low, double f_high,
		   const std::string& window, double kaiser_beta) {
			const char* n = "fir_bandstop";
			check_num_taps(num_taps, n);
			check_sample_rate(sr, n);
			check_frequency(f_low, sr, n, "f_low");
			check_frequency(f_high, sr, n, "f_high");
			if (!(f_high > f_low)) {
				throw std::invalid_argument(
					"fir_bandstop: f_high must be greater than f_low");
			}
			auto w = make_window(window, static_cast<std::size_t>(num_taps), kaiser_beta);
			// Bandstop via spectral inversion of a bandpass:
			// bs[n] = delta[n - M/2] - bp[n]
			auto bp = sw::dsp::design_fir_bandpass<double>(
				static_cast<std::size_t>(num_taps), f_low / sr, f_high / sr, w);
			PyFIRFilter f;
			f.taps = mtl::vec::dense_vector<double>(bp.size());
			for (std::size_t i = 0; i < bp.size(); ++i) f.taps[i] = -bp[i];
			f.taps[(bp.size() - 1) / 2] += 1.0;
			return f;
		}, nb::arg("num_taps"), nb::arg(A_SR), nb::arg("f_low"), nb::arg("f_high"),
		   nb::arg("window") = "hamming", nb::arg("kaiser_beta") = 8.6,
		"Design an FIR bandstop (notch) filter via spectral inversion.");
}
