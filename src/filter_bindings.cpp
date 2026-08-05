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
#include <sw/dsp/filter/filtfilt.hpp>
#include <sw/dsp/filter/fir/fir_design.hpp>
#include <sw/dsp/filter/fir/fir_filter.hpp>
#include <sw/dsp/filter/fir/overlap.hpp>
#include <sw/dsp/filter/fir/remez.hpp>
#include <sw/dsp/filter/iir/bessel.hpp>
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/iir/chebyshev1.hpp>
#include <sw/dsp/filter/iir/chebyshev2.hpp>
#include <sw/dsp/filter/iir/elliptic.hpp>
#include <sw/dsp/filter/iir/legendre.hpp>
#include <sw/dsp/filter/iir/rbj.hpp>
#include <sw/dsp/transfer_function/bode.hpp>
#include <sw/dsp/windows/windows.hpp>

#include "types.hpp"
#include "_binding_helpers.hpp"

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
	using mpdsp::p8_0;
	using mpdsp::p8_1;
	using mpdsp::p8_2;
	using mpdsp::p16_0;
	using mpdsp::p16_1;
	using mpdsp::p16_2;
	using mpdsp::p32_0;
	using mpdsp::p32_1;
	using mpdsp::p32_2;
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
	// Posit taxonomy grid (#81) — every cell uses a single posit type for
	// both state and sample. posit_8_2 also covers the tiny_posit alias.
	case ArithConfig::posit_8_0:
		process_typed<p8_0, p8_0>(cascade, in, out, n); break;
	case ArithConfig::posit_8_1:
		process_typed<p8_1, p8_1>(cascade, in, out, n); break;
	case ArithConfig::posit_8_2:
		process_typed<p8_2, p8_2>(cascade, in, out, n); break;
	case ArithConfig::posit_16_0:
		process_typed<p16_0, p16_0>(cascade, in, out, n); break;
	case ArithConfig::posit_16_1:
		process_typed<p16_1, p16_1>(cascade, in, out, n); break;
	case ArithConfig::posit_16_2:
		process_typed<p16_2, p16_2>(cascade, in, out, n); break;
	case ArithConfig::posit_32_0:
		process_typed<p32_0, p32_0>(cascade, in, out, n); break;
	case ArithConfig::posit_32_1:
		process_typed<p32_1, p32_1>(cascade, in, out, n); break;
	case ArithConfig::posit_32_2:
		process_typed<p32_2, p32_2>(cascade, in, out, n); break;
	}
}

// ---------------------------------------------------------------------------
// Type-dispatched zero-phase forward-backward processing (filtfilt).
//
// Marshals the double input into std::vector<SampleScalar>, invokes the
// upstream sw::dsp::filtfilt (which handles edge reflection + forward pass
// + backward pass internally), then quantizes results back to double.
// StateForm is fixed to DirectFormII<StateScalar> — the default the C++
// convenience overload picks, and the numerically robust choice for
// floating-point coefficients.
// ---------------------------------------------------------------------------

template <typename StateScalar, typename SampleScalar>
static void filtfilt_typed(const CascadeD& cascade,
                           const double* in, double* out, std::size_t n) {
	std::vector<SampleScalar> input(n);
	for (std::size_t i = 0; i < n; ++i) {
		input[i] = quantize_sample_in<SampleScalar>(in[i]);
	}
	auto result = sw::dsp::filtfilt<sw::dsp::DirectFormII<StateScalar>,
	                                double, kMaxStages,
	                                SampleScalar>(cascade, input);
	for (std::size_t i = 0; i < n; ++i) {
		out[i] = quantize_sample_out<SampleScalar>(result[i]);
	}
}

static void filtfilt_dispatch(const CascadeD& cascade,
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
	using mpdsp::p8_0;
	using mpdsp::p8_1;
	using mpdsp::p8_2;
	using mpdsp::p16_0;
	using mpdsp::p16_1;
	using mpdsp::p16_2;
	using mpdsp::p32_0;
	using mpdsp::p32_1;
	using mpdsp::p32_2;
	switch (config) {
	case ArithConfig::reference:
		filtfilt_typed<double, double>(cascade, in, out, n); break;
	case ArithConfig::gpu_baseline:
		filtfilt_typed<float, float>(cascade, in, out, n); break;
	case ArithConfig::ml_hw:
		filtfilt_typed<float, half_>(cascade, in, out, n); break;
	case ArithConfig::cf24_config:
		filtfilt_typed<cf24, cf24>(cascade, in, out, n); break;
	case ArithConfig::half_config:
		filtfilt_typed<half_, half_>(cascade, in, out, n); break;
	case ArithConfig::posit_full:
		filtfilt_typed<p32, p16>(cascade, in, out, n); break;
	case ArithConfig::sensor_8bit:
		filtfilt_typed<double, int8_sample_t>(cascade, in, out, n); break;
	case ArithConfig::sensor_6bit:
		filtfilt_typed<double, int6_sample_t>(cascade, in, out, n); break;
	case ArithConfig::fpga_fixed:
		filtfilt_typed<fx3224_t, fx1612_t>(cascade, in, out, n); break;
	case ArithConfig::posit_8_0:
		filtfilt_typed<p8_0, p8_0>(cascade, in, out, n); break;
	case ArithConfig::posit_8_1:
		filtfilt_typed<p8_1, p8_1>(cascade, in, out, n); break;
	case ArithConfig::posit_8_2:
		filtfilt_typed<p8_2, p8_2>(cascade, in, out, n); break;
	case ArithConfig::posit_16_0:
		filtfilt_typed<p16_0, p16_0>(cascade, in, out, n); break;
	case ArithConfig::posit_16_1:
		filtfilt_typed<p16_1, p16_1>(cascade, in, out, n); break;
	case ArithConfig::posit_16_2:
		filtfilt_typed<p16_2, p16_2>(cascade, in, out, n); break;
	case ArithConfig::posit_32_0:
		filtfilt_typed<p32_0, p32_0>(cascade, in, out, n); break;
	case ArithConfig::posit_32_1:
		filtfilt_typed<p32_1, p32_1>(cascade, in, out, n); break;
	case ArithConfig::posit_32_2:
		filtfilt_typed<p32_2, p32_2>(cascade, in, out, n); break;
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

	// Construct from a list of BiquadCoefficients — the "from raw
	// coefficients" workflow that was previously unreachable from Python
	// (design functions were the only way to make an IIRFilter). Bound as
	// a def_static classmethod: IIRFilter.from_coefficients(list). Length
	// must be in [1, kMaxStages]; each element is copied into a cascade
	// stage in order.
	static PyIIRFilter from_coefficients(
			const std::vector<sw::dsp::BiquadCoefficients<double>>& biquads) {
		if (biquads.empty()) {
			throw std::invalid_argument(
				"IIRFilter.from_coefficients: at least one biquad required");
		}
		if (biquads.size() > static_cast<std::size_t>(kMaxStages)) {
			throw std::invalid_argument(
				"IIRFilter.from_coefficients: too many biquads (max "
				+ std::to_string(kMaxStages) + ", got "
				+ std::to_string(biquads.size()) + ")");
		}
		PyIIRFilter filt;
		filt.cascade.set_num_stages(static_cast<int>(biquads.size()));
		for (std::size_t i = 0; i < biquads.size(); ++i) {
			filt.cascade.stage(static_cast<int>(i)) = biquads[i];
		}
		return filt;
	}

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

	// Extract zeros from the numerator of each biquad stage. `BiquadPoleState`
	// solves the numerator quadratic upstream (see biquad/biquad.hpp); we were
	// throwing that data away when building `poles()`, so the dashboard's
	// pole-zero plot was missing half its content. Mirrors `poles()` exactly,
	// including the second-slot guard — for first-order sections (b2 = 0)
	// BiquadPoleState leaves `zeros.second` default-constructed, same pattern
	// as the pole path.
	std::vector<std::complex<double>> zeros() const {
		std::vector<std::complex<double>> out;
		out.reserve(static_cast<std::size_t>(cascade.num_stages()) * 2);
		for (int i = 0; i < cascade.num_stages(); ++i) {
			sw::dsp::BiquadPoleState<double> pz(cascade.stage(i));
			out.push_back(pz.zeros.first);
			const auto& second = pz.zeros.second;
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

	// Frequency response with the coefficients quantized through `dtype`
	// first. Defined out of line because the quantization dispatcher lives
	// below, next to pole_displacement's.
	np_c128 frequency_response_quantized(np_f64_ro normalized_freqs,
	                                     const std::string& dtype) const;
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
	using mpdsp::p8_0;
	using mpdsp::p8_1;
	using mpdsp::p8_2;
	using mpdsp::p16_0;
	using mpdsp::p16_1;
	using mpdsp::p16_2;
	using mpdsp::p32_0;
	using mpdsp::p32_1;
	using mpdsp::p32_2;
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
	case ArithConfig::sensor_8bit:
		fir_process_typed<double, int8_sample_t>(taps_d, in, out, n); break;
	case ArithConfig::sensor_6bit:
		fir_process_typed<double, int6_sample_t>(taps_d, in, out, n); break;
	case ArithConfig::fpga_fixed:
		fir_process_typed<fx3224_t, fx1612_t>(taps_d, in, out, n); break;
	// Posit taxonomy grid (#81).
	case ArithConfig::posit_8_0:
		fir_process_typed<p8_0, p8_0>(taps_d, in, out, n); break;
	case ArithConfig::posit_8_1:
		fir_process_typed<p8_1, p8_1>(taps_d, in, out, n); break;
	case ArithConfig::posit_8_2:
		fir_process_typed<p8_2, p8_2>(taps_d, in, out, n); break;
	case ArithConfig::posit_16_0:
		fir_process_typed<p16_0, p16_0>(taps_d, in, out, n); break;
	case ArithConfig::posit_16_1:
		fir_process_typed<p16_1, p16_1>(taps_d, in, out, n); break;
	case ArithConfig::posit_16_2:
		fir_process_typed<p16_2, p16_2>(taps_d, in, out, n); break;
	case ArithConfig::posit_32_0:
		fir_process_typed<p32_0, p32_0>(taps_d, in, out, n); break;
	case ArithConfig::posit_32_1:
		fir_process_typed<p32_1, p32_1>(taps_d, in, out, n); break;
	case ArithConfig::posit_32_2:
		fir_process_typed<p32_2, p32_2>(taps_d, in, out, n); break;
	}
}

// Cast a dense_vector<T> to dense_vector<double> for storage in the
// double-typed PyFIRFilter. Element-wise static_cast.
template <typename T>
static mtl::vec::dense_vector<double>
to_double_vec(const mtl::vec::dense_vector<T>& v) {
	mtl::vec::dense_vector<double> out(v.size());
	for (std::size_t i = 0; i < v.size(); ++i)
		out[i] = static_cast<double>(v[i]);
	return out;
}

// make_window_T() lives in _binding_helpers.hpp — shared with spectral_bindings
// so Welch's PSD can build windows the same way FIR design does.
using mpdsp::bindings::make_window_T;

// Used by both the RBJ biquad designers and the FIR designers below.
using mpdsp::bindings::dispatch_dtype_fn;

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

// std::vector<double> -> owning float64 ndarray. BodeResult carries three of
// them; the mtl-based vec_to_numpy in _binding_helpers.hpp takes a
// dense_vector, so this is the std::vector counterpart.
static np_f64 to_np(const std::vector<double>& v) {
	double* out = nullptr;
	auto arr = mpdsp::bindings::make_f64_array(v.size(), out);
	for (std::size_t i = 0; i < v.size(); ++i) out[i] = v[i];
	return arr;
}

// ---------------------------------------------------------------------------
// Bode-sweep adapters (Issue #115).
//
// Upstream sweep_bode drives an LTI block with a settled sine at each
// frequency and correlates the output, so it needs a *stateful* block
// exposing sample_scalar, reset(), and per-sample process(). Neither
// PyIIRFilter nor PyFIRFilter is that: both are stateless holders that
// build their state fresh inside process_typed / fir_process_typed.
//
// These adapters supply the missing shape while routing through exactly the
// same quantize_sample_in / quantize_sample_out path that process() uses, so
// a swept measurement reports the response the filter actually realizes at
// that dtype — including the integer sample-path scaling that a plain cast
// would flatten to zero.
//
// sample_scalar is deliberately `double`: upstream casts its cosine drive to
// sample_scalar before handing it over, and letting that cast be the
// quantization step would bypass the scale-quantize-unscale that integer
// sample types need. Quantizing inside process() instead keeps one code path.
// ---------------------------------------------------------------------------

template <typename StateScalar, typename SampleScalar>
class BodeIIRBlock {
public:
	using sample_scalar = double;

	explicit BodeIIRBlock(const CascadeD& cascade) : cascade_(cascade) {}

	void reset() { state_ = {}; }

	double process(double x) {
		SampleScalar xs = quantize_sample_in<SampleScalar>(x);
		SampleScalar y = cascade_.template process<
			sw::dsp::DirectFormI<StateScalar>, SampleScalar>(xs, state_);
		return quantize_sample_out<SampleScalar>(y);
	}

private:
	const CascadeD& cascade_;
	std::array<sw::dsp::DirectFormI<StateScalar>, kMaxStages> state_{};
};

template <typename StateScalar, typename SampleScalar>
class BodeFIRBlock {
public:
	using sample_scalar = double;

	explicit BodeFIRBlock(const mtl::vec::dense_vector<double>& taps_d)
		: taps_(taps_d.size()) {
		for (std::size_t i = 0; i < taps_d.size(); ++i)
			taps_[i] = static_cast<StateScalar>(taps_d[i]);
		filt_ = sw::dsp::FIRFilter<StateScalar, StateScalar, SampleScalar>(taps_);
	}

	void reset() {
		filt_ = sw::dsp::FIRFilter<StateScalar, StateScalar, SampleScalar>(taps_);
	}

	double process(double x) {
		SampleScalar xs = quantize_sample_in<SampleScalar>(x);
		return quantize_sample_out<SampleScalar>(filt_.process(xs));
	}

private:
	mtl::vec::dense_vector<StateScalar> taps_;
	sw::dsp::FIRFilter<StateScalar, StateScalar, SampleScalar> filt_{
		mtl::vec::dense_vector<StateScalar>(1)};
};

struct BodeParams {
	double      sample_rate_hz;
	double      freq_min_hz;
	double      freq_max_hz;
	std::size_t num_points;
	std::size_t settle_samples;
	double      target_cycles;
	std::size_t max_measure_samples;
};

template <class Block>
static sw::dsp::transfer_function::BodeResult
run_bode(Block& block, const BodeParams& p) {
	return sw::dsp::transfer_function::sweep_bode(
		block, p.sample_rate_hz, p.freq_min_hz, p.freq_max_hz,
		p.num_points, p.settle_samples, p.target_cycles,
		p.max_measure_samples);
}

// Same (StateScalar, SampleScalar) pairing as process_dispatch, so a swept
// Bode measurement and a process() call at the same dtype exercise identical
// arithmetic.
static sw::dsp::transfer_function::BodeResult
bode_iir_dispatch(const CascadeD& src, const BodeParams& p,
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
	using mpdsp::p8_0;
	using mpdsp::p8_1;
	using mpdsp::p8_2;
	using mpdsp::p16_0;
	using mpdsp::p16_1;
	using mpdsp::p16_2;
	using mpdsp::p32_0;
	using mpdsp::p32_1;
	using mpdsp::p32_2;
	switch (config) {
	case ArithConfig::reference: { BodeIIRBlock<double, double> b(src); return run_bode(b, p); }
	case ArithConfig::gpu_baseline: { BodeIIRBlock<float, float> b(src); return run_bode(b, p); }
	case ArithConfig::ml_hw: { BodeIIRBlock<float, half_> b(src); return run_bode(b, p); }
	case ArithConfig::cf24_config: { BodeIIRBlock<cf24, cf24> b(src); return run_bode(b, p); }
	case ArithConfig::half_config: { BodeIIRBlock<half_, half_> b(src); return run_bode(b, p); }
	case ArithConfig::posit_full: { BodeIIRBlock<p32, p16> b(src); return run_bode(b, p); }
	case ArithConfig::sensor_8bit: { BodeIIRBlock<double, int8_sample_t> b(src); return run_bode(b, p); }
	case ArithConfig::sensor_6bit: { BodeIIRBlock<double, int6_sample_t> b(src); return run_bode(b, p); }
	case ArithConfig::fpga_fixed: { BodeIIRBlock<fx3224_t, fx1612_t> b(src); return run_bode(b, p); }
	case ArithConfig::posit_8_0: { BodeIIRBlock<p8_0, p8_0> b(src); return run_bode(b, p); }
	case ArithConfig::posit_8_1: { BodeIIRBlock<p8_1, p8_1> b(src); return run_bode(b, p); }
	case ArithConfig::posit_8_2: { BodeIIRBlock<p8_2, p8_2> b(src); return run_bode(b, p); }
	case ArithConfig::posit_16_0: { BodeIIRBlock<p16_0, p16_0> b(src); return run_bode(b, p); }
	case ArithConfig::posit_16_1: { BodeIIRBlock<p16_1, p16_1> b(src); return run_bode(b, p); }
	case ArithConfig::posit_16_2: { BodeIIRBlock<p16_2, p16_2> b(src); return run_bode(b, p); }
	case ArithConfig::posit_32_0: { BodeIIRBlock<p32_0, p32_0> b(src); return run_bode(b, p); }
	case ArithConfig::posit_32_1: { BodeIIRBlock<p32_1, p32_1> b(src); return run_bode(b, p); }
	case ArithConfig::posit_32_2: { BodeIIRBlock<p32_2, p32_2> b(src); return run_bode(b, p); }
	}
	throw std::invalid_argument("sweep_bode: unsupported ArithConfig");
}

// Same pairing as fir_process_dispatch.
static sw::dsp::transfer_function::BodeResult
bode_fir_dispatch(const mtl::vec::dense_vector<double>& src,
                  const BodeParams& p, mpdsp::ArithConfig config) {
	using mpdsp::ArithConfig;
	using mpdsp::cf24;
	using mpdsp::fx1612_t;
	using mpdsp::fx3224_t;
	using mpdsp::half_;
	using mpdsp::int6_sample_t;
	using mpdsp::int8_sample_t;
	using mpdsp::p16;
	using mpdsp::p32;
	using mpdsp::p8_0;
	using mpdsp::p8_1;
	using mpdsp::p8_2;
	using mpdsp::p16_0;
	using mpdsp::p16_1;
	using mpdsp::p16_2;
	using mpdsp::p32_0;
	using mpdsp::p32_1;
	using mpdsp::p32_2;
	switch (config) {
	case ArithConfig::reference: { BodeFIRBlock<double, double> b(src); return run_bode(b, p); }
	case ArithConfig::gpu_baseline: { BodeFIRBlock<float, float> b(src); return run_bode(b, p); }
	case ArithConfig::ml_hw: { BodeFIRBlock<float, half_> b(src); return run_bode(b, p); }
	case ArithConfig::cf24_config: { BodeFIRBlock<cf24, cf24> b(src); return run_bode(b, p); }
	case ArithConfig::half_config: { BodeFIRBlock<half_, half_> b(src); return run_bode(b, p); }
	case ArithConfig::posit_full: { BodeFIRBlock<p32, p16> b(src); return run_bode(b, p); }
	case ArithConfig::sensor_8bit: { BodeFIRBlock<double, int8_sample_t> b(src); return run_bode(b, p); }
	case ArithConfig::sensor_6bit: { BodeFIRBlock<double, int6_sample_t> b(src); return run_bode(b, p); }
	case ArithConfig::fpga_fixed: { BodeFIRBlock<fx3224_t, fx1612_t> b(src); return run_bode(b, p); }
	case ArithConfig::posit_8_0: { BodeFIRBlock<p8_0, p8_0> b(src); return run_bode(b, p); }
	case ArithConfig::posit_8_1: { BodeFIRBlock<p8_1, p8_1> b(src); return run_bode(b, p); }
	case ArithConfig::posit_8_2: { BodeFIRBlock<p8_2, p8_2> b(src); return run_bode(b, p); }
	case ArithConfig::posit_16_0: { BodeFIRBlock<p16_0, p16_0> b(src); return run_bode(b, p); }
	case ArithConfig::posit_16_1: { BodeFIRBlock<p16_1, p16_1> b(src); return run_bode(b, p); }
	case ArithConfig::posit_16_2: { BodeFIRBlock<p16_2, p16_2> b(src); return run_bode(b, p); }
	case ArithConfig::posit_32_0: { BodeFIRBlock<p32_0, p32_0> b(src); return run_bode(b, p); }
	case ArithConfig::posit_32_1: { BodeFIRBlock<p32_1, p32_1> b(src); return run_bode(b, p); }
	case ArithConfig::posit_32_2: { BodeFIRBlock<p32_2, p32_2> b(src); return run_bode(b, p); }
	}
	throw std::invalid_argument("sweep_bode: unsupported ArithConfig");
}

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

// Round-trip every coefficient through `config`'s scalar type and hand back
// the resulting double cascade. Shared by pole_displacement (which measures
// how far the poles moved) and frequency_response(dtype=) (which evaluates
// what the quantized cascade actually does) — the same quantization seen
// through two different questions, so it must be exactly one table.
static CascadeD quantize_cascade_dispatch(const CascadeD& src,
                                          mpdsp::ArithConfig config) {
	using mpdsp::ArithConfig;
	using mpdsp::cf24;
	using mpdsp::fx3224_t;
	using mpdsp::half_;
	using mpdsp::p16;
	using mpdsp::p32;
	using mpdsp::p8_0;
	using mpdsp::p8_1;
	using mpdsp::p8_2;
	using mpdsp::p16_0;
	using mpdsp::p16_1;
	using mpdsp::p16_2;
	using mpdsp::p32_0;
	using mpdsp::p32_1;
	using mpdsp::p32_2;
	CascadeD quantized;
	switch (config) {
	case ArithConfig::reference:    return src;  // no quantization
	case ArithConfig::gpu_baseline: quantized = quantize_cascade<float>(src); break;
	case ArithConfig::ml_hw:        quantized = quantize_cascade<half_>(src); break;
	case ArithConfig::cf24_config:  quantized = quantize_cascade<cf24>(src); break;
	case ArithConfig::half_config:  quantized = quantize_cascade<half_>(src); break;
	case ArithConfig::posit_full:   quantized = quantize_cascade<p32>(src); break;
	// sensor_* keep coefficients at double (only the sample path quantizes),
	// so there is nothing to quantize here: pole displacement is zero and
	// the frequency response is identical to reference. That is a true
	// answer to the coefficient question, not a missing case.
	case ArithConfig::sensor_8bit:
	case ArithConfig::sensor_6bit:
		return src;
	case ArithConfig::fpga_fixed:
		quantized = quantize_cascade<fx3224_t>(src); break;
	// Posit taxonomy grid (#81) — coefficient-level quantization through
	// the grid's posit type. posit_8_2 also covers the tiny_posit alias.
	case ArithConfig::posit_8_0:  quantized = quantize_cascade<p8_0>(src); break;
	case ArithConfig::posit_8_1:  quantized = quantize_cascade<p8_1>(src); break;
	case ArithConfig::posit_8_2:  quantized = quantize_cascade<p8_2>(src); break;
	case ArithConfig::posit_16_0: quantized = quantize_cascade<p16_0>(src); break;
	case ArithConfig::posit_16_1: quantized = quantize_cascade<p16_1>(src); break;
	case ArithConfig::posit_16_2: quantized = quantize_cascade<p16_2>(src); break;
	case ArithConfig::posit_32_0: quantized = quantize_cascade<p32_0>(src); break;
	case ArithConfig::posit_32_1: quantized = quantize_cascade<p32_1>(src); break;
	case ArithConfig::posit_32_2: quantized = quantize_cascade<p32_2>(src); break;
	}
	return quantized;
}

static double pole_displacement_dispatch(const CascadeD& src,
                                         mpdsp::ArithConfig config) {
	if (config == mpdsp::ArithConfig::reference) return 0.0;
	return sw::dsp::pole_displacement(
		src, quantize_cascade_dispatch(src, config));
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

// ---------------------------------------------------------------------------
// Coefficient-precision dispatch for the RBJ designers (Issue #94).
//
// Strategy (a) of the two the issue weighed: run the design math in T, then
// narrow the finished biquad back to double for the Cascade<double,
// kMaxStages> that PyIIRFilter stores. Every downstream method — process(),
// poles(), frequency_response(), stability_margin(), pole_displacement() —
// keeps operating on the one cascade shape it already assumes, so this is a
// change to the designers alone.
//
// Narrowing to double is lossless here and not a fudge: an RBJ biquad
// designed in T yields coefficients that are by construction T-representable,
// and every T in the dispatch table is narrower than double, so double stores
// them exactly. What coeff_dtype actually varies is the *arithmetic* used to
// get there — the w0 scaling, cos/sin, the alpha divide, and the a0
// normalization — which is where coefficient error in a biquad design is
// born.
//
// The counterpart to this is IIRFilter.pole_displacement(dtype), which
// quantizes an already-designed double cascade. The two answer different
// questions: pole_displacement asks "what does storing these coefficients in
// T cost?", coeff_dtype asks "what does computing them in T cost?".
// ---------------------------------------------------------------------------

template <typename DesignT, typename... SetupArgs>
static PyIIRFilter rbj_design_in(SetupArgs... args) {
	DesignT design;
	design.setup(args...);
	const auto& src = design.cascade().stage(0);
	PyIIRFilter filt;
	filt.cascade.set_num_stages(1);
	auto& dst = filt.cascade.stage(0);
	dst.b0 = static_cast<double>(src.b0);
	dst.b1 = static_cast<double>(src.b1);
	dst.b2 = static_cast<double>(src.b2);
	dst.a1 = static_cast<double>(src.a1);
	dst.a2 = static_cast<double>(src.a2);
	return filt;
}

// ---------------------------------------------------------------------------
// Remez helpers. The band-type enum is a namespace-scope (not templated)
// enum, so we can just pass it directly across the dispatch boundary.
// ---------------------------------------------------------------------------

static sw::dsp::RemezBandType parse_remez_type(const std::string& s) {
	using T = sw::dsp::RemezBandType;
	if (s == "bandpass")       return T::bandpass;
	if (s == "differentiator") return T::differentiator;
	if (s == "hilbert")        return T::hilbert;
	throw std::invalid_argument(
		"remez: unknown type '" + s + "' (expected 'bandpass', "
		"'differentiator', or 'hilbert')");
}

// ---------------------------------------------------------------------------
// OverlapAdd / OverlapSave impls. Both take a design-time tap vector and a
// block_size; both stream block_size samples in -> block_size samples out
// via FFT convolution. OverlapAdd additionally exposes flush() to retrieve
// the trailing M-1 convolution tail after the final process_block().
// ---------------------------------------------------------------------------

struct IOverlapAddImpl {
	virtual ~IOverlapAddImpl() = default;
	virtual mtl::vec::dense_vector<double>
	process_block(const double* in, std::size_t n) = 0;
	virtual mtl::vec::dense_vector<double> flush() = 0;
	virtual void reset() = 0;
	virtual std::size_t block_size()    const = 0;
	virtual std::size_t fft_size()      const = 0;
	virtual std::size_t filter_length() const = 0;
};

template <typename T>
struct OverlapAddImpl : IOverlapAddImpl {
	sw::dsp::OverlapAddConvolver<T, T, T> inner;

	OverlapAddImpl(const std::vector<double>& taps, std::size_t block_size)
	    : inner(cast_taps(taps), block_size) {}

	mtl::vec::dense_vector<double>
	process_block(const double* in, std::size_t n) override {
		mtl::vec::dense_vector<T> typed_in(n);
		for (std::size_t i = 0; i < n; ++i) typed_in[i] = static_cast<T>(in[i]);
		auto typed_out = inner.process_block(
			std::span<const T>(typed_in.data(), typed_in.size()));
		return to_double_dv(typed_out);
	}

	mtl::vec::dense_vector<double> flush() override {
		return to_double_dv(inner.flush());
	}

	void reset()                       override { inner.reset(); }
	std::size_t block_size()    const override { return inner.block_size(); }
	std::size_t fft_size()      const override { return inner.fft_size(); }
	std::size_t filter_length() const override { return inner.filter_length(); }

private:
	static mtl::vec::dense_vector<T>
	cast_taps(const std::vector<double>& src) {
		mtl::vec::dense_vector<T> out(src.size());
		for (std::size_t i = 0; i < src.size(); ++i) out[i] = static_cast<T>(src[i]);
		return out;
	}
	static mtl::vec::dense_vector<double>
	to_double_dv(const mtl::vec::dense_vector<T>& src) {
		mtl::vec::dense_vector<double> out(src.size());
		for (std::size_t i = 0; i < src.size(); ++i)
			out[i] = static_cast<double>(src[i]);
		return out;
	}
};

struct IOverlapSaveImpl {
	virtual ~IOverlapSaveImpl() = default;
	virtual mtl::vec::dense_vector<double>
	process_block(const double* in, std::size_t n) = 0;
	virtual void reset() = 0;
	virtual std::size_t block_size()    const = 0;
	virtual std::size_t fft_size()      const = 0;
	virtual std::size_t filter_length() const = 0;
};

template <typename T>
struct OverlapSaveImpl : IOverlapSaveImpl {
	sw::dsp::OverlapSaveConvolver<T, T, T> inner;

	OverlapSaveImpl(const std::vector<double>& taps, std::size_t block_size)
	    : inner(cast_taps(taps), block_size) {}

	mtl::vec::dense_vector<double>
	process_block(const double* in, std::size_t n) override {
		mtl::vec::dense_vector<T> typed_in(n);
		for (std::size_t i = 0; i < n; ++i) typed_in[i] = static_cast<T>(in[i]);
		auto typed_out = inner.process_block(
			std::span<const T>(typed_in.data(), typed_in.size()));
		return to_double_dv(typed_out);
	}

	void reset()                       override { inner.reset(); }
	std::size_t block_size()    const override { return inner.block_size(); }
	std::size_t fft_size()      const override { return inner.fft_size(); }
	std::size_t filter_length() const override { return inner.filter_length(); }

private:
	static mtl::vec::dense_vector<T>
	cast_taps(const std::vector<double>& src) {
		mtl::vec::dense_vector<T> out(src.size());
		for (std::size_t i = 0; i < src.size(); ++i) out[i] = static_cast<T>(src[i]);
		return out;
	}
	static mtl::vec::dense_vector<double>
	to_double_dv(const mtl::vec::dense_vector<T>& src) {
		mtl::vec::dense_vector<double> out(src.size());
		for (std::size_t i = 0; i < src.size(); ++i)
			out[i] = static_cast<double>(src[i]);
		return out;
	}
};

} // namespace

// PyOverlapAddConvolver / PyOverlapSaveConvolver: block-FFT convolution.
class PyOverlapAddConvolver {
public:
	PyOverlapAddConvolver(np_f64_ro taps, std::size_t block_size,
	                      const std::string& dtype)
	    : dtype_(dtype) {
		if (taps.shape(0) == 0) {
			throw std::invalid_argument(
				"OverlapAddConvolver: taps must not be empty");
		}
		if (block_size == 0) {
			throw std::invalid_argument(
				"OverlapAddConvolver: block_size must be > 0");
		}
		std::vector<double> tap_vec(taps.shape(0));
		for (std::size_t i = 0; i < tap_vec.size(); ++i) tap_vec[i] = taps.data()[i];
		impl_ = mpdsp::bindings::make_impl_for_dtype<
			OverlapAddImpl, IOverlapAddImpl>(
			mpdsp::parse_config(dtype), "OverlapAddConvolver",
			tap_vec, block_size);
	}

	np_f64 process_block(np_f64_ro signal) {
		if (signal.shape(0) != impl_->block_size()) {
			throw std::invalid_argument(
				"OverlapAddConvolver.process_block: signal length must equal "
				"block_size (" + std::to_string(impl_->block_size()) + ")");
		}
		return mpdsp::bindings::vec_to_numpy(
			impl_->process_block(signal.data(), signal.shape(0)));
	}

	np_f64 flush() {
		return mpdsp::bindings::vec_to_numpy(impl_->flush());
	}

	void reset()                       { impl_->reset(); }
	std::size_t block_size()    const  { return impl_->block_size(); }
	std::size_t fft_size()      const  { return impl_->fft_size(); }
	std::size_t filter_length() const  { return impl_->filter_length(); }
	const std::string& dtype()  const  { return dtype_; }

private:
	std::unique_ptr<IOverlapAddImpl> impl_;
	std::string dtype_;
};

class PyOverlapSaveConvolver {
public:
	PyOverlapSaveConvolver(np_f64_ro taps, std::size_t block_size,
	                       const std::string& dtype)
	    : dtype_(dtype) {
		if (taps.shape(0) == 0) {
			throw std::invalid_argument(
				"OverlapSaveConvolver: taps must not be empty");
		}
		if (block_size == 0) {
			throw std::invalid_argument(
				"OverlapSaveConvolver: block_size must be > 0");
		}
		std::vector<double> tap_vec(taps.shape(0));
		for (std::size_t i = 0; i < tap_vec.size(); ++i) tap_vec[i] = taps.data()[i];
		impl_ = mpdsp::bindings::make_impl_for_dtype<
			OverlapSaveImpl, IOverlapSaveImpl>(
			mpdsp::parse_config(dtype), "OverlapSaveConvolver",
			tap_vec, block_size);
	}

	np_f64 process_block(np_f64_ro signal) {
		if (signal.shape(0) != impl_->block_size()) {
			throw std::invalid_argument(
				"OverlapSaveConvolver.process_block: signal length must equal "
				"block_size (" + std::to_string(impl_->block_size()) + ")");
		}
		return mpdsp::bindings::vec_to_numpy(
			impl_->process_block(signal.data(), signal.shape(0)));
	}

	void reset()                       { impl_->reset(); }
	std::size_t block_size()    const  { return impl_->block_size(); }
	std::size_t fft_size()      const  { return impl_->fft_size(); }
	std::size_t filter_length() const  { return impl_->filter_length(); }
	const std::string& dtype()  const  { return dtype_; }

private:
	std::unique_ptr<IOverlapSaveImpl> impl_;
	std::string dtype_;
};

double PyIIRFilter::pole_displacement(const std::string& dtype) const {
	auto config = mpdsp::parse_config(dtype);
	return pole_displacement_dispatch(cascade, config);
}

np_c128 PyIIRFilter::frequency_response_quantized(
		np_f64_ro normalized_freqs, const std::string& dtype) const {
	auto config = mpdsp::parse_config(dtype);
	// Evaluation stays in double; only the coefficients are quantized. That
	// is the deployment question this answers — "what does storing these
	// coefficients in T do to my response" — and it is exactly the dual of
	// pole_displacement, which asks the same thing about the poles.
	//
	// It deliberately does NOT model state or sample-path arithmetic: the
	// filter state stays double throughout. For the response a filter
	// actually realizes at a dtype, including sample-path quantization, use
	// sweep_bode(), which measures it empirically by running samples through.
	const CascadeD quantized = quantize_cascade_dispatch(cascade, config);

	std::size_t n = normalized_freqs.shape(0);
	std::complex<double>* out_ptr = nullptr;
	auto arr = make_c128_array(n, out_ptr);
	const double* f = normalized_freqs.data();
	for (std::size_t i = 0; i < n; ++i) {
		out_ptr[i] = quantized.response(f[i]);
	}
	return arr;
}

// ---------------------------------------------------------------------------
// Module registration.
// ---------------------------------------------------------------------------

// Shared tail for the RBJ designer docstrings — see the RBJ section below.
#define RBJ_COEFF_DTYPE_DOC \
	"coeff_dtype selects the arithmetic used to compute the biquad " \
	"coefficients (w0, cos/sin, alpha, and the a0 normalization); the " \
	"result is stored in double either way. Note that sensor_8bit / " \
	"sensor_6bit dispatch their compute path to double, so they design " \
	"identically to reference."

void bind_filters(nb::module_& m) {
	nb::class_<PyIIRFilter>(m, "IIRFilter",
		"Cascade-of-biquads IIR filter.\n\n"
		"Construct via one of the design functions (butterworth_lowpass, "
		"chebyshev1_bandpass, etc.) or via IIRFilter.from_coefficients() "
		"if you already have the raw biquad coefficients. Coefficients are "
		"stored in double precision; process() dispatches state/sample "
		"arithmetic on the dtype argument.")
		.def_static("from_coefficients", &PyIIRFilter::from_coefficients,
		     nb::arg("biquads"),
		     "Construct an IIRFilter from a list of BiquadCoefficients. "
		     "Length must be in [1, 8] (compile-time cascade bound). Each "
		     "element populates one biquad section in order. Enables "
		     "importing coefficients designed elsewhere (scipy, MATLAB, "
		     "hand-designed cascades) into the mpdsp processing pipeline.")
		.def("num_stages", &PyIIRFilter::num_stages,
		     "Number of active biquad sections.")
		.def("coefficients", &PyIIRFilter::coefficients,
		     "List of (b0, b1, b2, a1, a2) tuples, one per stage.")
		.def("poles", &PyIIRFilter::poles,
		     "List of complex pole locations in the z-plane.")
		.def("zeros", &PyIIRFilter::zeros,
		     "List of complex zero locations in the z-plane. For all-pole "
		     "families (Butterworth / Chebyshev I / Bessel / Legendre), all "
		     "finite zeros map to z = -1, so expect an N-fold cluster there. "
		     "Chebyshev II and Elliptic distribute zeros on the unit circle.")
		.def("process", &PyIIRFilter::process,
		     nb::arg("signal"), nb::arg("dtype") = "reference",
		     "Filter a signal. dtype selects arithmetic for state and samples "
		     "(see available_dtypes()). Returns NumPy float64.")
		.def("frequency_response",
		     &PyIIRFilter::frequency_response_quantized,
		     nb::arg("normalized_freqs"), nb::arg("dtype") = "reference",
		     "Evaluate H(e^{j2*pi*f}) at each normalized frequency (f/fs). "
		     "Returns complex128.\n\n"
		     "dtype quantizes the coefficients through that type before "
		     "evaluating — the deployment question, and the dual of "
		     "pole_displacement(dtype): one asks what quantizing the "
		     "coefficients does to the response, the other what it does to "
		     "the poles. Evaluation itself stays in double, so this does "
		     "not model state or sample-path arithmetic; for the response a "
		     "filter actually realizes at a dtype, use sweep_bode(), which "
		     "measures it by running samples through.")
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
	//
	// Every designer takes coeff_dtype= (#94): the biquad's coefficient math
	// runs in that type, and the finished coefficients are stored as double.
	// See rbj_design_in() for why that storage narrowing is lossless.
	// RBJ_COEFF_DTYPE_DOC is a macro rather than a constant because nanobind
	// docstrings must be const char* — adjacent string-literal concatenation
	// is what lets the shared tail be appended at compile time.
	static constexpr const char* A_COEFF_DTYPE = "coeff_dtype";

	m.def("rbj_lowpass",
		[](double sr, double cutoff, double q, const std::string& coeff_dtype) {
			const char* n = "rbj_lowpass";
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(q, n, "q");
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() {
				return rbj_design_in<rbj::LowPass<T>>(sr, cutoff, q);
			});
		}, nb::arg(A_SR), nb::arg(A_CUT), nb::arg("q") = 0.7071,
		   nb::arg(A_COEFF_DTYPE) = "reference",
		"RBJ biquad lowpass. q ~ 0.7071 gives a Butterworth-like response. "
		RBJ_COEFF_DTYPE_DOC);

	m.def("rbj_highpass",
		[](double sr, double cutoff, double q, const std::string& coeff_dtype) {
			const char* n = "rbj_highpass";
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(q, n, "q");
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() {
				return rbj_design_in<rbj::HighPass<T>>(sr, cutoff, q);
			});
		}, nb::arg(A_SR), nb::arg(A_CUT), nb::arg("q") = 0.7071,
		   nb::arg(A_COEFF_DTYPE) = "reference",
		"RBJ biquad highpass. " RBJ_COEFF_DTYPE_DOC);

	m.def("rbj_bandpass",
		[](double sr, double center_freq, double bandwidth,
		   const std::string& coeff_dtype) {
			const char* n = "rbj_bandpass";
			check_sample_rate(sr, n);
			check_frequency(center_freq, sr, n, "center_freq");
			check_positive(bandwidth, n, "bandwidth");
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() {
				return rbj_design_in<rbj::BandPass<T>>(sr, center_freq, bandwidth);
			});
		}, nb::arg(A_SR), nb::arg(A_CTR), nb::arg("bandwidth") = 1.0,
		   nb::arg(A_COEFF_DTYPE) = "reference",
		"RBJ biquad bandpass. bandwidth is in octaves. "
		RBJ_COEFF_DTYPE_DOC);

	m.def("rbj_bandstop",
		[](double sr, double center_freq, double bandwidth,
		   const std::string& coeff_dtype) {
			const char* n = "rbj_bandstop";
			check_sample_rate(sr, n);
			check_frequency(center_freq, sr, n, "center_freq");
			check_positive(bandwidth, n, "bandwidth");
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() {
				return rbj_design_in<rbj::BandStop<T>>(sr, center_freq, bandwidth);
			});
		}, nb::arg(A_SR), nb::arg(A_CTR), nb::arg("bandwidth") = 1.0,
		   nb::arg(A_COEFF_DTYPE) = "reference",
		"RBJ biquad bandstop (notch). bandwidth is in octaves. "
		RBJ_COEFF_DTYPE_DOC);

	m.def("rbj_allpass",
		[](double sr, double center_freq, double q,
		   const std::string& coeff_dtype) {
			const char* n = "rbj_allpass";
			check_sample_rate(sr, n);
			check_frequency(center_freq, sr, n, "center_freq");
			check_positive(q, n, "q");
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() {
				return rbj_design_in<rbj::AllPass<T>>(sr, center_freq, q);
			});
		}, nb::arg(A_SR), nb::arg(A_CTR), nb::arg("q") = 0.7071,
		   nb::arg(A_COEFF_DTYPE) = "reference",
		"RBJ biquad allpass — unit magnitude, phase shift only. "
		RBJ_COEFF_DTYPE_DOC);

	// Shelf filters: gain_db is intentionally not validated — any real value
	// is meaningful (0 dB is a legal unity shelf, negative values cut).
	m.def("rbj_lowshelf",
		[](double sr, double cutoff, double gain_db, double slope,
		   const std::string& coeff_dtype) {
			const char* n = "rbj_lowshelf";
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(slope, n, "slope");
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() {
				return rbj_design_in<rbj::LowShelf<T>>(sr, cutoff, gain_db, slope);
			});
		}, nb::arg(A_SR), nb::arg(A_CUT), nb::arg("gain_db"),
		   nb::arg("slope") = 1.0, nb::arg(A_COEFF_DTYPE) = "reference",
		"RBJ biquad low shelf. gain_db is the low-frequency shelf gain. "
		RBJ_COEFF_DTYPE_DOC);

	m.def("rbj_highshelf",
		[](double sr, double cutoff, double gain_db, double slope,
		   const std::string& coeff_dtype) {
			const char* n = "rbj_highshelf";
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			check_positive(slope, n, "slope");
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() {
				return rbj_design_in<rbj::HighShelf<T>>(sr, cutoff, gain_db, slope);
			});
		}, nb::arg(A_SR), nb::arg(A_CUT), nb::arg("gain_db"),
		   nb::arg("slope") = 1.0, nb::arg(A_COEFF_DTYPE) = "reference",
		"RBJ biquad high shelf. gain_db is the high-frequency shelf gain. "
		RBJ_COEFF_DTYPE_DOC);

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
		   const std::string& window, double kaiser_beta,
		   const std::string& coeff_dtype) {
			const char* n = "fir_lowpass";
			check_num_taps(num_taps, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			std::size_t N = static_cast<std::size_t>(num_taps);
			double cutoff_norm = cutoff / sr;
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() -> PyFIRFilter {
				auto w = make_window_T<T>(window, N, kaiser_beta);
				auto taps = sw::dsp::design_fir_lowpass<T>(N, T(cutoff_norm), w);
				PyFIRFilter f;
				f.taps = to_double_vec(taps);
				return f;
			});
		}, nb::arg("num_taps"), nb::arg(A_SR), nb::arg(A_CUT),
		   nb::arg("window") = "hamming", nb::arg("kaiser_beta") = 8.6,
		   nb::arg("coeff_dtype") = "reference",
		"Design an FIR lowpass filter via the window method. coeff_dtype "
		"controls the precision of the design-time math; the resulting "
		"taps are stored as float64 in the returned filter.");

	m.def("fir_highpass",
		[](int num_taps, double sr, double cutoff,
		   const std::string& window, double kaiser_beta,
		   const std::string& coeff_dtype) {
			const char* n = "fir_highpass";
			check_num_taps(num_taps, n);
			check_sample_rate(sr, n);
			check_frequency(cutoff, sr, n, "cutoff");
			std::size_t N = static_cast<std::size_t>(num_taps);
			double cutoff_norm = cutoff / sr;
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() -> PyFIRFilter {
				auto w = make_window_T<T>(window, N, kaiser_beta);
				auto taps = sw::dsp::design_fir_highpass<T>(N, T(cutoff_norm), w);
				PyFIRFilter f;
				f.taps = to_double_vec(taps);
				return f;
			});
		}, nb::arg("num_taps"), nb::arg(A_SR), nb::arg(A_CUT),
		   nb::arg("window") = "hamming", nb::arg("kaiser_beta") = 8.6,
		   nb::arg("coeff_dtype") = "reference",
		"Design an FIR highpass filter via spectral inversion of a lowpass.");

	m.def("fir_bandpass",
		[](int num_taps, double sr, double f_low, double f_high,
		   const std::string& window, double kaiser_beta,
		   const std::string& coeff_dtype) {
			const char* n = "fir_bandpass";
			check_num_taps(num_taps, n);
			check_sample_rate(sr, n);
			check_frequency(f_low, sr, n, "f_low");
			check_frequency(f_high, sr, n, "f_high");
			if (!(f_high > f_low)) {
				throw std::invalid_argument(
					"fir_bandpass: f_high must be greater than f_low");
			}
			std::size_t N = static_cast<std::size_t>(num_taps);
			double fl_norm = f_low / sr;
			double fh_norm = f_high / sr;
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() -> PyFIRFilter {
				auto w = make_window_T<T>(window, N, kaiser_beta);
				auto taps = sw::dsp::design_fir_bandpass<T>(
					N, T(fl_norm), T(fh_norm), w);
				PyFIRFilter f;
				f.taps = to_double_vec(taps);
				return f;
			});
		}, nb::arg("num_taps"), nb::arg(A_SR), nb::arg("f_low"), nb::arg("f_high"),
		   nb::arg("window") = "hamming", nb::arg("kaiser_beta") = 8.6,
		   nb::arg("coeff_dtype") = "reference",
		"Design an FIR bandpass filter.");

	m.def("fir_bandstop",
		[](int num_taps, double sr, double f_low, double f_high,
		   const std::string& window, double kaiser_beta,
		   const std::string& coeff_dtype) {
			const char* n = "fir_bandstop";
			check_num_taps(num_taps, n);
			check_sample_rate(sr, n);
			check_frequency(f_low, sr, n, "f_low");
			check_frequency(f_high, sr, n, "f_high");
			if (!(f_high > f_low)) {
				throw std::invalid_argument(
					"fir_bandstop: f_high must be greater than f_low");
			}
			std::size_t N = static_cast<std::size_t>(num_taps);
			double fl_norm = f_low / sr;
			double fh_norm = f_high / sr;
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() -> PyFIRFilter {
				auto w = make_window_T<T>(window, N, kaiser_beta);
				// Bandstop via spectral inversion of a bandpass:
				// bs[n] = delta[n - M/2] - bp[n]
				auto bp = sw::dsp::design_fir_bandpass<T>(
					N, T(fl_norm), T(fh_norm), w);
				// const, not constexpr: posit<N,es>(...) only became
				// constexpr in universal v4.6.10, but the FetchContent
				// pin still ships v4.6.9 (lockstep with DSP v0.5.0).
				const T one = T(1);
				mtl::vec::dense_vector<T> bs(bp.size());
				for (std::size_t i = 0; i < bp.size(); ++i) bs[i] = -bp[i];
				bs[(bp.size() - 1) / 2] = bs[(bp.size() - 1) / 2] + one;
				PyFIRFilter f;
				f.taps = to_double_vec(bs);
				return f;
			});
		}, nb::arg("num_taps"), nb::arg(A_SR), nb::arg("f_low"), nb::arg("f_high"),
		   nb::arg("window") = "hamming", nb::arg("kaiser_beta") = 8.6,
		   nb::arg("coeff_dtype") = "reference",
		"Design an FIR bandstop (notch) filter via spectral inversion.");

	// -----------------------------------------------------------------------
	// filtfilt — zero-phase forward-backward IIR filtering (scipy analogue).
	// -----------------------------------------------------------------------
	m.def("filtfilt",
		[](const PyIIRFilter& filt, np_f64_ro signal, const std::string& dtype) {
			std::size_t n = signal.shape(0);
			double* out_ptr = nullptr;
			auto arr = make_f64_array(n, out_ptr);
			auto config = mpdsp::parse_config(dtype);
			filtfilt_dispatch(filt.cascade, signal.data(), out_ptr, n, config);
			return arr;
		}, nb::arg("iir_filter"), nb::arg("signal"),
		   nb::arg("dtype") = "reference",
		"Zero-phase IIR filtering via forward-backward biquad cascade processing.\n\n"
		"Applies the cascade forward, then backward on the reversed signal.\n"
		"The result has zero phase distortion; the magnitude response is squared\n"
		"relative to a single forward pass. Signal edges are reflected (length =\n"
		"3*(2*num_stages + 1) - 1, clamped to N-1) to suppress transient artifacts.\n"
		"Analogous to scipy.signal.filtfilt. Coefficient precision stays double;\n"
		"state and sample scalars follow the dtype key.");

	// -----------------------------------------------------------------------
	// Remez (Parks-McClellan) equiripple FIR design (Phase 5 / #111).
	// Returns FIRFilter objects (matches the mpdsp fir_lowpass et al.
	// convention); users can extract raw taps via .coefficients().
	// -----------------------------------------------------------------------
	m.def("remez_lowpass",
		[](int num_taps, double sr, double passband_edge_hz,
		   double stopband_edge_hz, double passband_weight,
		   double stopband_weight, const std::string& coeff_dtype) {
			const char* n = "remez_lowpass";
			check_num_taps(num_taps, n);
			check_sample_rate(sr, n);
			check_frequency(passband_edge_hz, sr, n, "passband_edge_hz");
			check_frequency(stopband_edge_hz, sr, n, "stopband_edge_hz");
			if (!(stopband_edge_hz > passband_edge_hz)) {
				throw std::invalid_argument(
					"remez_lowpass: stopband_edge_hz must be > passband_edge_hz");
			}
			check_positive(passband_weight, n, "passband_weight");
			check_positive(stopband_weight, n, "stopband_weight");
			std::size_t N = static_cast<std::size_t>(num_taps);
			double pb_norm = passband_edge_hz / sr;
			double sb_norm = stopband_edge_hz / sr;
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() -> PyFIRFilter {
				auto taps = sw::dsp::design_fir_equiripple_lowpass<T>(
					N, T(pb_norm), T(sb_norm),
					T(passband_weight), T(stopband_weight));
				PyFIRFilter f;
				f.taps = to_double_vec(taps);
				return f;
			});
		}, nb::arg("num_taps"), nb::arg(A_SR),
		   nb::arg("passband_edge_hz"), nb::arg("stopband_edge_hz"),
		   nb::arg("passband_weight") = 1.0,
		   nb::arg("stopband_weight") = 1.0,
		   nb::arg("coeff_dtype") = "reference",
		"Equiripple lowpass FIR via Parks-McClellan (Remez exchange). "
		"passband_edge_hz and stopband_edge_hz define the transition band; "
		"weights control the passband-vs-stopband trade-off (larger stopband "
		"weight -> deeper stopband)."
		);

	m.def("remez_bandpass",
		[](int num_taps, double sr, double stop1_hz, double pass1_hz,
		   double pass2_hz, double stop2_hz,
		   double stopband_weight, double passband_weight,
		   const std::string& coeff_dtype) {
			const char* n = "remez_bandpass";
			check_num_taps(num_taps, n);
			check_sample_rate(sr, n);
			check_frequency(stop1_hz, sr, n, "stop1_hz");
			check_frequency(pass1_hz, sr, n, "pass1_hz");
			check_frequency(pass2_hz, sr, n, "pass2_hz");
			check_frequency(stop2_hz, sr, n, "stop2_hz");
			if (!(stop1_hz < pass1_hz && pass1_hz < pass2_hz && pass2_hz < stop2_hz)) {
				throw std::invalid_argument(
					"remez_bandpass: require stop1 < pass1 < pass2 < stop2");
			}
			check_positive(stopband_weight, n, "stopband_weight");
			check_positive(passband_weight, n, "passband_weight");
			std::size_t N = static_cast<std::size_t>(num_taps);
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() -> PyFIRFilter {
				auto taps = sw::dsp::design_fir_equiripple_bandpass<T>(
					N, T(stop1_hz / sr), T(pass1_hz / sr),
					T(pass2_hz / sr), T(stop2_hz / sr),
					T(stopband_weight), T(passband_weight));
				PyFIRFilter f;
				f.taps = to_double_vec(taps);
				return f;
			});
		}, nb::arg("num_taps"), nb::arg(A_SR),
		   nb::arg("stop1_hz"), nb::arg("pass1_hz"),
		   nb::arg("pass2_hz"), nb::arg("stop2_hz"),
		   nb::arg("stopband_weight") = 1.0,
		   nb::arg("passband_weight") = 1.0,
		   nb::arg("coeff_dtype") = "reference",
		"Equiripple bandpass FIR via Parks-McClellan. Requires "
		"stop1 < pass1 < pass2 < stop2, all in Hz. Symmetric stopband "
		"weights on both sides."
		);

	m.def("remez",
		[](int num_taps, np_f64_ro bands, np_f64_ro desired,
		   np_f64_ro weights, const std::string& type_str,
		   int max_iterations, int grid_density,
		   const std::string& coeff_dtype) {
			const char* n = "remez";
			check_num_taps(num_taps, n);
			if (bands.shape(0) < 2 || (bands.shape(0) & 1) != 0) {
				throw std::invalid_argument(
					"remez: bands must have even number of elements (>= 2)");
			}
			if (desired.shape(0) != bands.shape(0)) {
				throw std::invalid_argument(
					"remez: desired must have same length as bands");
			}
			if (weights.shape(0) * 2 != bands.shape(0)) {
				throw std::invalid_argument(
					"remez: weights must have exactly bands/2 entries");
			}
			auto band_type = parse_remez_type(type_str);
			std::size_t N = static_cast<std::size_t>(num_taps);
			auto config = mpdsp::parse_config(coeff_dtype);
			return dispatch_dtype_fn(config, n, [&]<typename T>() -> PyFIRFilter {
				std::vector<T> b(bands.shape(0));
				std::vector<T> d(desired.shape(0));
				std::vector<T> w(weights.shape(0));
				for (std::size_t i = 0; i < b.size(); ++i) b[i] = T(bands.data()[i]);
				for (std::size_t i = 0; i < d.size(); ++i) d[i] = T(desired.data()[i]);
				for (std::size_t i = 0; i < w.size(); ++i) w[i] = T(weights.data()[i]);
				auto taps = sw::dsp::remez<T>(N, b, d, w, band_type,
				                              max_iterations, grid_density);
				PyFIRFilter f;
				f.taps = to_double_vec(taps);
				return f;
			});
		}, nb::arg("num_taps"), nb::arg("bands"), nb::arg("desired"),
		   nb::arg("weights"),
		   nb::arg("type") = "bandpass",
		   nb::arg("max_iterations") = 40,
		   nb::arg("grid_density") = 16,
		   nb::arg("coeff_dtype") = "reference",
		"General Parks-McClellan equiripple FIR design. bands is a flat "
		"list of band edges in normalized frequency [0, 0.5], length 2N "
		"for N bands; desired has one value per band edge; weights has one "
		"per band. type is 'bandpass' (default; symmetric taps), "
		"'differentiator', or 'hilbert' (both antisymmetric)."
		);

	// -----------------------------------------------------------------------
	// OverlapAdd / OverlapSave block-FFT convolvers (Phase 5 / #111).
	// -----------------------------------------------------------------------
	nb::class_<PyOverlapAddConvolver>(m, "OverlapAddConvolver",
			"Block-based fast FIR convolution via the overlap-add method. "
			"Feed exactly block_size samples per process_block() call; each "
			"call returns block_size output samples. Call flush() once after "
			"the final process_block() to retrieve the trailing M-1 "
			"convolution tail (needed to recover the complete linear "
			"convolution).\n\n"
			"Complexity vs. a direct FIR: OA is O((L+M) log(L+M)) per L-sample "
			"block, so it's faster than direct O(L*M) FIR when M is large.")
		.def(nb::init<np_f64_ro, std::size_t, const std::string&>(),
		     nb::arg("taps"), nb::arg("block_size"),
		     nb::arg("dtype") = "reference",
		     "Construct with a 1D tap array and block size (both > 0). "
		     "Internal FFT size = next_pow2(block_size + len(taps) - 1).")
		.def("process_block", &PyOverlapAddConvolver::process_block,
		     nb::arg("signal"),
		     "Process exactly block_size samples; return block_size samples.")
		.def("flush", &PyOverlapAddConvolver::flush,
		     "Emit the trailing M-1 convolution tail. Call once after the "
		     "final process_block(); returns an empty array if no tail "
		     "remains.")
		.def("reset", &PyOverlapAddConvolver::reset,
		     "Clear the internal tail state. Coefficients and sizes are "
		     "preserved.")
		.def_prop_ro("block_size",    &PyOverlapAddConvolver::block_size)
		.def_prop_ro("fft_size",      &PyOverlapAddConvolver::fft_size)
		.def_prop_ro("filter_length", &PyOverlapAddConvolver::filter_length)
		.def_prop_ro("dtype",
		     [](const PyOverlapAddConvolver& self) { return self.dtype(); });

	nb::class_<PyOverlapSaveConvolver>(m, "OverlapSaveConvolver",
			"Block-based fast FIR convolution via the overlap-save method. "
			"Feed exactly block_size samples per process_block() call; each "
			"call returns block_size output samples. No flush() needed — "
			"overlap-save keeps its history in a running buffer and never "
			"emits a tail past the last block.\n\n"
			"The first M-1 output samples of the very first call are the "
			"start-up transient (correctly handled via zero-initial history).")
		.def(nb::init<np_f64_ro, std::size_t, const std::string&>(),
		     nb::arg("taps"), nb::arg("block_size"),
		     nb::arg("dtype") = "reference",
		     "Same argument shape as OverlapAddConvolver.")
		.def("process_block", &PyOverlapSaveConvolver::process_block,
		     nb::arg("signal"),
		     "Process exactly block_size samples; return block_size samples.")
		.def("reset", &PyOverlapSaveConvolver::reset,
		     "Clear the internal history. Coefficients and sizes are preserved.")
		.def_prop_ro("block_size",    &PyOverlapSaveConvolver::block_size)
		.def_prop_ro("fft_size",      &PyOverlapSaveConvolver::fft_size)
		.def_prop_ro("filter_length", &PyOverlapSaveConvolver::filter_length)
		.def_prop_ro("dtype",
		     [](const PyOverlapSaveConvolver& self) { return self.dtype(); });

	// ---- Bode sweep (#115) ---------------------------------------------
	//
	// An *empirical* frequency response: the filter is driven with a settled
	// sine at each frequency and the output correlated against cos/sin. That
	// is the point of it existing alongside frequency_response(), which
	// evaluates H(z) analytically from the coefficients. The analytic form
	// cannot see quantization in the sample path; this can. Overlay the two
	// to read off what a given dtype costs.
	{
		namespace tf = sw::dsp::transfer_function;
		using BR = tf::BodeResult;

		nb::class_<BR>(m, "BodeResult",
			"Result of a swept Bode measurement: one entry per frequency.")
			.def_prop_ro("freqs_hz", [](const BR& r) {
				return to_np(r.freqs_hz);
			}, nb::rv_policy::take_ownership,
			   "Log-spaced sweep frequencies, in Hz.")
			.def_prop_ro("magnitudes_db", [](const BR& r) {
				return to_np(r.magnitudes_dB);
			}, nb::rv_policy::take_ownership,
			   "Measured |H| in dB. Floored at -300 dB.")
			.def_prop_ro("phases_rad", [](const BR& r) {
				return to_np(r.phases_rad);
			}, nb::rv_policy::take_ownership,
			   "Measured phase in radians, wrapped to (-pi, pi].")
			.def("__len__", [](const BR& r) { return r.freqs_hz.size(); })
			.def("__repr__", [](const BR& r) {
				return "BodeResult(" + std::to_string(r.freqs_hz.size())
				     + " points)";
			});

		auto params = [](double sr, double fmin, double fmax,
		                 std::size_t npts, std::size_t settle,
		                 double cycles, std::size_t max_meas) {
			return BodeParams{sr, fmin, fmax, npts, settle, cycles, max_meas};
		};

		const char* doc =
			"Measure the frequency response by driving the filter with a "
			"settled sine at each of `num_points` log-spaced frequencies "
			"and correlating the output against cos/sin (Hann-windowed).\n\n"
			"Unlike frequency_response(), which evaluates H(z) analytically "
			"from the coefficients, this runs actual samples through the "
			"filter at the requested dtype — so it registers quantization "
			"in the sample path that the analytic form cannot see. "
			"Overlaying the two is how you read off the cost of a dtype.\n\n"
			"Requires 0 < freq_min_hz < freq_max_hz < sample_rate/2 and "
			"num_points >= 2. The filter is reset before each frequency, so "
			"the caller's object is left with the last sweep's state; this "
			"does not disturb coefficients. Measurement length adapts to "
			"`target_cycles` periods per frequency, floored at 512 samples "
			"and capped at `max_measure_samples` — the cap trades "
			"low-frequency accuracy for runtime.";

		m.def("sweep_bode",
			[&params](const PyIIRFilter& filt, double sr, double fmin,
			   double fmax, std::size_t npts, std::size_t settle,
			   double cycles, std::size_t max_meas,
			   const std::string& dtype) {
				auto config = mpdsp::parse_config(dtype);
				return bode_iir_dispatch(
					filt.cascade,
					params(sr, fmin, fmax, npts, settle, cycles, max_meas),
					config);
			},
			nb::arg("filt"), nb::arg(A_SR), nb::arg("freq_min_hz"),
			nb::arg("freq_max_hz"), nb::arg("num_points") = 200,
			nb::arg("settle_samples") = 512,
			nb::arg("target_cycles") = 32.0,
			nb::arg("max_measure_samples") = 32768,
			nb::arg("dtype") = "reference",
			doc);

		m.def("sweep_bode",
			[&params](const PyFIRFilter& filt, double sr, double fmin,
			   double fmax, std::size_t npts, std::size_t settle,
			   double cycles, std::size_t max_meas,
			   const std::string& dtype) {
				auto config = mpdsp::parse_config(dtype);
				return bode_fir_dispatch(
					filt.taps,
					params(sr, fmin, fmax, npts, settle, cycles, max_meas),
					config);
			},
			nb::arg("filt"), nb::arg(A_SR), nb::arg("freq_min_hz"),
			nb::arg("freq_max_hz"), nb::arg("num_points") = 200,
			nb::arg("settle_samples") = 512,
			nb::arg("target_cycles") = 32.0,
			nb::arg("max_measure_samples") = 32768,
			nb::arg("dtype") = "reference",
			doc);
	}
}

#undef RBJ_COEFF_DTYPE_DOC
