// filter_bindings.cpp: IIR filter bindings (Butterworth LP/HP to start)
//
// Exposes a PyIIRFilter class wrapping sw::dsp::Cascade<double, MaxStages>
// and design functions that return instances of it. Processing is
// type-dispatched: coefficients stay in double; state and sample scalars
// vary per the dtype key (shared with quantization bindings).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <sw/dsp/filter/biquad/biquad.hpp>
#include <sw/dsp/filter/biquad/cascade.hpp>
#include <sw/dsp/filter/biquad/state.hpp>
#include <sw/dsp/filter/iir/butterworth.hpp>

#include "types.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

// Upper bound on biquad stages exposed to Python. MaxOrder=16 gives up to
// 8 LP/HP sections. Bandpass variants (future) double this; we'll revisit
// when we add them.
static constexpr int kMaxStages = 8;
static constexpr int kMaxOrder  = kMaxStages * 2;

using CascadeD = sw::dsp::Cascade<double, kMaxStages>;

namespace {

// Type-dispatched per-sample processing. Coefficients are in double;
// state and sample arithmetic are parameterized.
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
	// Posit paths deferred until we confirm DspField conformance in a
	// follow-up PR — reference/float/cfloat paths exercise the pattern.
	case ArithConfig::posit_full:
	case ArithConfig::tiny_posit:
		throw std::invalid_argument(
			"posit dtypes for filter.process are not yet enabled");
	}
}

// NumPy helpers mirroring the pattern in quantization_bindings.cpp.
using np_f64    = nb::ndarray<nb::numpy, double>;
using np_f64_ro = nb::ndarray<nb::numpy, const double, nb::ndim<1>>;
using np_c128   = nb::ndarray<nb::numpy, std::complex<double>>;
using np_c128_ro = nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<1>>;

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

} // namespace

// PyIIRFilter: opaque handle wrapping a double-precision biquad cascade.
// Python sees this as mpdsp.IIRFilter with process/frequency_response/etc.
class PyIIRFilter {
public:
	CascadeD cascade;

	int num_stages() const { return cascade.num_stages(); }

	// Returns list of (b0, b1, b2, a1, a2) tuples — one per active stage.
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

	// Poles extracted per-stage via BiquadPoleState. Skips the spurious
	// zero-valued "second" entry on first-order (odd-order trailing) sections.
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

void bind_filters(nb::module_& m) {
	nb::class_<PyIIRFilter>(m, "IIRFilter",
		"Cascade-of-biquads IIR filter.\n\n"
		"Construct via one of the design functions "
		"(e.g. butterworth_lowpass). Coefficients are stored in "
		"double precision; process() dispatches state/sample arithmetic "
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
		     "Input and output are 1D NumPy arrays; returns complex128.");

	m.def("butterworth_lowpass",
		[](int order, double sample_rate, double cutoff) {
			if (order < 1 || order > kMaxOrder) {
				throw std::invalid_argument(
					"butterworth_lowpass: order must be in [1, 16]");
			}
			if (!(sample_rate > 0.0)) {
				throw std::invalid_argument(
					"butterworth_lowpass: sample_rate must be positive");
			}
			if (!(cutoff > 0.0) || cutoff >= 0.5 * sample_rate) {
				throw std::invalid_argument(
					"butterworth_lowpass: cutoff must be in (0, sample_rate/2)");
			}
			sw::dsp::iir::ButterworthLowPass<kMaxOrder> design;
			design.setup(order, sample_rate, cutoff);
			PyIIRFilter filt;
			filt.cascade = design.cascade();
			return filt;
		},
		nb::arg("order"), nb::arg("sample_rate"), nb::arg("cutoff"),
		"Design a Butterworth lowpass filter. "
		"order in [1, 16]; cutoff in Hz, must be < sample_rate/2.");

	m.def("butterworth_highpass",
		[](int order, double sample_rate, double cutoff) {
			if (order < 1 || order > kMaxOrder) {
				throw std::invalid_argument(
					"butterworth_highpass: order must be in [1, 16]");
			}
			if (!(sample_rate > 0.0)) {
				throw std::invalid_argument(
					"butterworth_highpass: sample_rate must be positive");
			}
			if (!(cutoff > 0.0) || cutoff >= 0.5 * sample_rate) {
				throw std::invalid_argument(
					"butterworth_highpass: cutoff must be in (0, sample_rate/2)");
			}
			sw::dsp::iir::ButterworthHighPass<kMaxOrder> design;
			design.setup(order, sample_rate, cutoff);
			PyIIRFilter filt;
			filt.cascade = design.cascade();
			return filt;
		},
		nb::arg("order"), nb::arg("sample_rate"), nb::arg("cutoff"),
		"Design a Butterworth highpass filter. "
		"order in [1, 16]; cutoff in Hz, must be < sample_rate/2.");
}
