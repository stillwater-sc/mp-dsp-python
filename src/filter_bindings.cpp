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

#include <sw/dsp/filter/biquad/biquad.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/biquad/state.hpp>
#include <sw/dsp/filter/iir/bessel.hpp>
#include <sw/dsp/filter/iir/butterworth.hpp>
#include <sw/dsp/filter/iir/chebyshev1.hpp>
#include <sw/dsp/filter/iir/chebyshev2.hpp>
#include <sw/dsp/filter/iir/elliptic.hpp>
#include <sw/dsp/filter/iir/legendre.hpp>
#include <sw/dsp/filter/iir/rbj.hpp>

#include "types.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

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
// Type-dispatched per-sample processing.
// ---------------------------------------------------------------------------

template <typename StateScalar, typename SampleScalar>
static void process_typed(const CascadeD& cascade,
                          const double* in, double* out, std::size_t n) {
	std::array<sw::dsp::DirectFormI<StateScalar>, kMaxStages> state{};
	for (std::size_t i = 0; i < n; ++i) {
		SampleScalar x = static_cast<SampleScalar>(in[i]);
		SampleScalar y = cascade.template process<sw::dsp::DirectFormI<StateScalar>,
		                                          SampleScalar>(x, state);
		out[i] = static_cast<double>(y);
	}
}

static void process_dispatch(const CascadeD& cascade,
                             const double* in, double* out, std::size_t n,
                             mpdsp::ArithConfig config) {
	using mpdsp::ArithConfig;
	using mpdsp::cf24;
	using mpdsp::half_;
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
	case ArithConfig::tiny_posit:
		throw std::invalid_argument(
			"posit dtypes for filter.process are not yet enabled");
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
};

namespace {

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
		     "Returns complex128.");

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
}
