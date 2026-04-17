// conditioning_bindings.cpp: stateful signal-conditioning bindings.
//
// Establishes the stateful-object pattern used by Phase 5:
//   - dtype is a construction-time parameter (not per-call)
//   - internal type-erased IEnvelope interface holds a concrete
//     sw::dsp::PeakEnvelope<T> for the chosen dtype
//   - Python sees one class (mpdsp.PeakEnvelope); NumPy I/O stays in double
//
// Starts with PeakEnvelope. RMSEnvelope, Compressor, AGC follow the same
// template.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <sw/dsp/conditioning/envelope.hpp>

#include "types.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

using np_f64    = nb::ndarray<nb::numpy, double>;
using np_f64_ro = nb::ndarray<nb::numpy, const double, nb::ndim<1>>;

static np_f64 make_f64_array(std::size_t n, double*& out_ptr) {
	auto* data = new double[n];
	out_ptr = data;
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	std::size_t shape[1] = { n };
	return np_f64(data, 1, shape, owner);
}

// Type-erased interface. Python always sees double-precision I/O; the
// internal arithmetic happens in whatever T the concrete impl chose.
struct IPeakEnvelopeImpl {
	virtual ~IPeakEnvelopeImpl() = default;
	virtual void setup(double sample_rate, double attack_ms, double release_ms) = 0;
	virtual double process(double input) = 0;
	virtual void process_block(const double* in, double* out, std::size_t n) = 0;
	virtual double value() const = 0;
	virtual void reset() = 0;
};

template <typename T>
struct PeakEnvelopeImpl : IPeakEnvelopeImpl {
	sw::dsp::PeakEnvelope<T> inner;
	void setup(double sr, double a, double r) override { inner.setup(sr, a, r); }
	double process(double input) override {
		return static_cast<double>(inner.process(static_cast<T>(input)));
	}
	void process_block(const double* in, double* out, std::size_t n) override {
		for (std::size_t i = 0; i < n; ++i) {
			out[i] = static_cast<double>(inner.process(static_cast<T>(in[i])));
		}
	}
	double value() const override { return static_cast<double>(inner.value()); }
	void reset() override { inner.reset(); }
};

static std::unique_ptr<IPeakEnvelopeImpl>
make_peak_envelope_impl(mpdsp::ArithConfig config) {
	using mpdsp::ArithConfig;
	using mpdsp::cf24;
	using mpdsp::half_;
	using mpdsp::p16;
	using mpdsp::p32;
	using tiny_posit_t = sw::universal::posit<8, 2>;
	switch (config) {
	case ArithConfig::reference:    return std::make_unique<PeakEnvelopeImpl<double>>();
	case ArithConfig::gpu_baseline: return std::make_unique<PeakEnvelopeImpl<float>>();
	case ArithConfig::ml_hw:        return std::make_unique<PeakEnvelopeImpl<half_>>();
	case ArithConfig::cf24_config:  return std::make_unique<PeakEnvelopeImpl<cf24>>();
	case ArithConfig::half_config:  return std::make_unique<PeakEnvelopeImpl<half_>>();
	case ArithConfig::posit_full:   return std::make_unique<PeakEnvelopeImpl<p32>>();
	case ArithConfig::tiny_posit:   return std::make_unique<PeakEnvelopeImpl<tiny_posit_t>>();
	}
	return std::make_unique<PeakEnvelopeImpl<double>>();
}

} // namespace

// PyPeakEnvelope: stateful envelope follower with construction-time dtype.
class PyPeakEnvelope {
public:
	PyPeakEnvelope(double sample_rate, double attack_ms, double release_ms,
	               const std::string& dtype) {
		if (!(sample_rate > 0.0)) {
			throw std::invalid_argument(
				"PeakEnvelope: sample_rate must be positive");
		}
		if (!(attack_ms > 0.0) || !(release_ms > 0.0)) {
			throw std::invalid_argument(
				"PeakEnvelope: attack_ms and release_ms must be positive");
		}
		impl_ = make_peak_envelope_impl(mpdsp::parse_config(dtype));
		impl_->setup(sample_rate, attack_ms, release_ms);
		dtype_ = dtype;
	}

	double process(double input) { return impl_->process(input); }

	np_f64 process_block(np_f64_ro signal) {
		std::size_t n = signal.shape(0);
		double* out_ptr = nullptr;
		auto arr = make_f64_array(n, out_ptr);
		impl_->process_block(signal.data(), out_ptr, n);
		return arr;
	}

	double value() const { return impl_->value(); }

	void reset() { impl_->reset(); }

	const std::string& dtype() const { return dtype_; }

private:
	std::unique_ptr<IPeakEnvelopeImpl> impl_;
	std::string dtype_;
};

void bind_conditioning(nb::module_& m) {
	nb::class_<PyPeakEnvelope>(m, "PeakEnvelope",
		"Peak envelope follower with exponential attack and release.\n\n"
		"Tracks |x[n]| through a one-pole filter whose time constant switches "
		"between attack_ms (when rising) and release_ms (when falling). State "
		"is kept across process() calls; use reset() to clear.")
		.def(nb::init<double, double, double, const std::string&>(),
		     nb::arg("sample_rate"), nb::arg("attack_ms"), nb::arg("release_ms"),
		     nb::arg("dtype") = "reference",
		     "Construct an envelope follower. dtype selects the arithmetic "
		     "used internally (see available_dtypes).")
		.def("process", &PyPeakEnvelope::process,
		     nb::arg("input"),
		     "Process a single sample. Returns the updated envelope value.")
		.def("process_block", &PyPeakEnvelope::process_block,
		     nb::arg("signal"),
		     "Process a 1D NumPy float64 signal. Returns the envelope trace "
		     "(same length as the input).")
		.def("value", &PyPeakEnvelope::value,
		     "Current envelope value without consuming a sample.")
		.def("reset", &PyPeakEnvelope::reset,
		     "Clear the internal envelope state to zero.")
		.def_prop_ro("dtype", &PyPeakEnvelope::dtype,
		     "The arithmetic configuration selected at construction.");
}
