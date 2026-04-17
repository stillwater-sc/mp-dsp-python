// conditioning_bindings.cpp: stateful signal-conditioning bindings.
//
// Stateful-object pattern used by Phase 5:
//   - dtype is a construction-time parameter (not per-call)
//   - internal type-erased interface holds a concrete
//     sw::dsp::<class><T> for the chosen dtype
//   - Python sees one concrete class per conditioning stage; NumPy I/O
//     stays in double
//
// Classes exposed: PeakEnvelope, RMSEnvelope, Compressor, AGC.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <sw/dsp/conditioning/agc.hpp>
#include <sw/dsp/conditioning/compressor.hpp>
#include <sw/dsp/conditioning/envelope.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

// Pull shared NumPy typedefs and helpers into this TU's namespace.
using mpdsp::bindings::np_f64;
using mpdsp::bindings::np_f64_ro;
using mpdsp::bindings::make_f64_array;
using mpdsp::bindings::make_impl_for_dtype;

namespace {

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
	return make_impl_for_dtype<PeakEnvelopeImpl, IPeakEnvelopeImpl>(
		config, "PeakEnvelope");
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
		const double* in_ptr = signal.data();
		// The processing loop touches no Python state, so drop the GIL
		// across the hot path. We can't use nb::call_guard at the .def()
		// site because make_f64_array creates Python objects (capsule +
		// ndarray) — those require the GIL held.
		{
			nb::gil_scoped_release release;
			impl_->process_block(in_ptr, out_ptr, n);
		}
		return arr;
	}

	double value() const { return impl_->value(); }

	void reset() { impl_->reset(); }

	const std::string& dtype() const { return dtype_; }

private:
	std::unique_ptr<IPeakEnvelopeImpl> impl_;
	std::string dtype_;
};

// ===========================================================================
// RMSEnvelope
// ===========================================================================

namespace {

struct IRMSEnvelopeImpl {
	virtual ~IRMSEnvelopeImpl() = default;
	virtual void setup(double sample_rate, double window_ms) = 0;
	virtual double process(double input) = 0;
	virtual void process_block(const double* in, double* out, std::size_t n) = 0;
	virtual double value() const = 0;
	virtual void reset() = 0;
};

template <typename T>
struct RMSEnvelopeImpl : IRMSEnvelopeImpl {
	sw::dsp::RMSEnvelope<T> inner;
	void setup(double sr, double w) override { inner.setup(sr, w); }
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

static std::unique_ptr<IRMSEnvelopeImpl>
make_rms_envelope_impl(mpdsp::ArithConfig config) {
	return make_impl_for_dtype<RMSEnvelopeImpl, IRMSEnvelopeImpl>(
		config, "RMSEnvelope");
}

} // namespace

class PyRMSEnvelope {
public:
	PyRMSEnvelope(double sample_rate, double window_ms,
	              const std::string& dtype) {
		if (!(sample_rate > 0.0)) {
			throw std::invalid_argument(
				"RMSEnvelope: sample_rate must be positive");
		}
		if (!(window_ms > 0.0)) {
			throw std::invalid_argument(
				"RMSEnvelope: window_ms must be positive");
		}
		impl_ = make_rms_envelope_impl(mpdsp::parse_config(dtype));
		impl_->setup(sample_rate, window_ms);
		dtype_ = dtype;
	}

	double process(double input) { return impl_->process(input); }

	np_f64 process_block(np_f64_ro signal) {
		std::size_t n = signal.shape(0);
		double* out_ptr = nullptr;
		auto arr = make_f64_array(n, out_ptr);
		const double* in_ptr = signal.data();
		{
			nb::gil_scoped_release release;
			impl_->process_block(in_ptr, out_ptr, n);
		}
		return arr;
	}

	double value() const { return impl_->value(); }
	void reset() { impl_->reset(); }
	const std::string& dtype() const { return dtype_; }

private:
	std::unique_ptr<IRMSEnvelopeImpl> impl_;
	std::string dtype_;
};

// ===========================================================================
// Compressor
// ===========================================================================

namespace {

struct ICompressorImpl {
	virtual ~ICompressorImpl() = default;
	virtual void setup(double sample_rate, double threshold_db, double ratio,
	                   double attack_ms, double release_ms,
	                   double makeup_db, double knee_db) = 0;
	virtual double process(double input) = 0;
	virtual void process_block(const double* in, double* out, std::size_t n) = 0;
	virtual void reset() = 0;
};

template <typename T>
struct CompressorImpl : ICompressorImpl {
	sw::dsp::Compressor<T> inner;
	void setup(double sr, double thr_db, double ratio,
	           double attack_ms, double release_ms,
	           double makeup_db, double knee_db) override {
		inner.setup(sr, thr_db, ratio, attack_ms, release_ms, makeup_db, knee_db);
	}
	double process(double input) override {
		return static_cast<double>(inner.process(static_cast<T>(input)));
	}
	void process_block(const double* in, double* out, std::size_t n) override {
		for (std::size_t i = 0; i < n; ++i) {
			out[i] = static_cast<double>(inner.process(static_cast<T>(in[i])));
		}
	}
	void reset() override { inner.reset(); }
};

static std::unique_ptr<ICompressorImpl>
make_compressor_impl(mpdsp::ArithConfig config) {
	return make_impl_for_dtype<CompressorImpl, ICompressorImpl>(
		config, "Compressor");
}

} // namespace

class PyCompressor {
public:
	PyCompressor(double sample_rate, double threshold_db, double ratio,
	             double attack_ms, double release_ms,
	             double makeup_db, double knee_db,
	             const std::string& dtype) {
		if (!(sample_rate > 0.0)) {
			throw std::invalid_argument(
				"Compressor: sample_rate must be positive");
		}
		if (!(attack_ms > 0.0) || !(release_ms > 0.0)) {
			throw std::invalid_argument(
				"Compressor: attack_ms and release_ms must be positive");
		}
		if (ratio < 1.0) {
			throw std::invalid_argument(
				"Compressor: ratio must be >= 1.0");
		}
		if (knee_db < 0.0) {
			throw std::invalid_argument(
				"Compressor: knee_db must be non-negative");
		}
		impl_ = make_compressor_impl(mpdsp::parse_config(dtype));
		impl_->setup(sample_rate, threshold_db, ratio, attack_ms, release_ms,
		             makeup_db, knee_db);
		dtype_ = dtype;
	}

	double process(double input) { return impl_->process(input); }

	np_f64 process_block(np_f64_ro signal) {
		std::size_t n = signal.shape(0);
		double* out_ptr = nullptr;
		auto arr = make_f64_array(n, out_ptr);
		const double* in_ptr = signal.data();
		{
			nb::gil_scoped_release release;
			impl_->process_block(in_ptr, out_ptr, n);
		}
		return arr;
	}

	void reset() { impl_->reset(); }
	const std::string& dtype() const { return dtype_; }

private:
	std::unique_ptr<ICompressorImpl> impl_;
	std::string dtype_;
};

// ===========================================================================
// AGC — Automatic Gain Control.
// Note: sw::dsp::AGC requires DspOrderedField (DspField + std::totally_ordered)
// so its dtype dispatch is narrower than PeakEnvelope's. Universal posit,
// cfloat, and standard float/double all satisfy totally_ordered via their
// comparison operators.
// ===========================================================================

namespace {

struct IAGCImpl {
	virtual ~IAGCImpl() = default;
	virtual void setup(double sample_rate, double target_level,
	                   double window_ms, double max_gain) = 0;
	virtual double process(double input) = 0;
	virtual void process_block(const double* in, double* out, std::size_t n) = 0;
	virtual void reset() = 0;
};

template <typename T>
struct AGCImpl : IAGCImpl {
	sw::dsp::AGC<T> inner;
	void setup(double sr, double target, double window_ms, double max_gain) override {
		inner.setup(sr, target, window_ms, max_gain);
	}
	double process(double input) override {
		return static_cast<double>(inner.process(static_cast<T>(input)));
	}
	void process_block(const double* in, double* out, std::size_t n) override {
		for (std::size_t i = 0; i < n; ++i) {
			out[i] = static_cast<double>(inner.process(static_cast<T>(in[i])));
		}
	}
	void reset() override { inner.reset(); }
};

static std::unique_ptr<IAGCImpl>
make_agc_impl(mpdsp::ArithConfig config) {
	// Note: AGC's underlying concept is DspOrderedField (narrower than
	// DspField). All 7 pre-instantiated ArithConfig dtypes satisfy it
	// today. If a future dtype is added that's DspField-only, this call
	// must drop back to a hand-written switch that omits the offending
	// case, keeping the shared dispatcher for the other three wrappers.
	return make_impl_for_dtype<AGCImpl, IAGCImpl>(config, "AGC");
}

} // namespace

class PyAGC {
public:
	PyAGC(double sample_rate, double target_level,
	      double window_ms, double max_gain,
	      const std::string& dtype) {
		if (!(sample_rate > 0.0)) {
			throw std::invalid_argument(
				"AGC: sample_rate must be positive");
		}
		if (!(target_level > 0.0)) {
			throw std::invalid_argument(
				"AGC: target_level must be positive");
		}
		if (!(window_ms > 0.0)) {
			throw std::invalid_argument(
				"AGC: window_ms must be positive");
		}
		if (!(max_gain > 0.0)) {
			throw std::invalid_argument(
				"AGC: max_gain must be positive");
		}
		impl_ = make_agc_impl(mpdsp::parse_config(dtype));
		impl_->setup(sample_rate, target_level, window_ms, max_gain);
		dtype_ = dtype;
	}

	double process(double input) { return impl_->process(input); }

	np_f64 process_block(np_f64_ro signal) {
		std::size_t n = signal.shape(0);
		double* out_ptr = nullptr;
		auto arr = make_f64_array(n, out_ptr);
		const double* in_ptr = signal.data();
		{
			nb::gil_scoped_release release;
			impl_->process_block(in_ptr, out_ptr, n);
		}
		return arr;
	}

	void reset() { impl_->reset(); }
	const std::string& dtype() const { return dtype_; }

private:
	std::unique_ptr<IAGCImpl> impl_;
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
		     "(same length as the input). The per-sample loop releases the "
		     "GIL internally so other Python threads can run.")
		.def("value", &PyPeakEnvelope::value,
		     "Current envelope value without consuming a sample.")
		.def("reset", &PyPeakEnvelope::reset,
		     "Clear the internal envelope state to zero.")
		.def_prop_ro("dtype", &PyPeakEnvelope::dtype,
		     "The arithmetic configuration selected at construction.");

	nb::class_<PyRMSEnvelope>(m, "RMSEnvelope",
		"RMS envelope follower.\n\n"
		"Tracks the root-mean-square level using a one-pole lowpass on x[n]^2. "
		"The `window_ms` parameter sets the averaging time constant.")
		.def(nb::init<double, double, const std::string&>(),
		     nb::arg("sample_rate"), nb::arg("window_ms"),
		     nb::arg("dtype") = "reference",
		     "Construct an RMS envelope follower.")
		.def("process", &PyRMSEnvelope::process,
		     nb::arg("input"),
		     "Process a single sample. Returns the updated RMS level.")
		.def("process_block", &PyRMSEnvelope::process_block,
		     nb::arg("signal"),
		     "Process a 1D NumPy float64 signal. Returns the RMS envelope trace "
		     "(same length as the input). The per-sample loop releases the GIL.")
		.def("value", &PyRMSEnvelope::value,
		     "Current RMS value without consuming a sample.")
		.def("reset", &PyRMSEnvelope::reset,
		     "Clear the internal mean-square state to zero.")
		.def_prop_ro("dtype", &PyRMSEnvelope::dtype,
		     "The arithmetic configuration selected at construction.");

	nb::class_<PyCompressor>(m, "Compressor",
		"Dynamic-range compressor with soft-knee option.\n\n"
		"Detects the signal level via a peak envelope follower and applies a "
		"gain reduction when the level exceeds threshold_db. `ratio` must be "
		">= 1.0 (use 1.0 for no compression). `makeup_db` adds constant output "
		"gain; `knee_db` specifies the soft-knee width (0 = hard knee).")
		.def(nb::init<double, double, double, double, double, double, double,
		              const std::string&>(),
		     nb::arg("sample_rate"), nb::arg("threshold_db"), nb::arg("ratio"),
		     nb::arg("attack_ms"), nb::arg("release_ms"),
		     nb::arg("makeup_db") = 0.0, nb::arg("knee_db") = 0.0,
		     nb::arg("dtype") = "reference",
		     "Construct a dynamic-range compressor.")
		.def("process", &PyCompressor::process,
		     nb::arg("input"),
		     "Process a single sample. Returns the compressed output.")
		.def("process_block", &PyCompressor::process_block,
		     nb::arg("signal"),
		     "Process a 1D NumPy float64 signal. Returns the compressed signal "
		     "(same length as the input). The per-sample loop releases the GIL.")
		.def("reset", &PyCompressor::reset,
		     "Clear the internal envelope state.")
		.def_prop_ro("dtype", &PyCompressor::dtype,
		     "The arithmetic configuration selected at construction.");

	nb::class_<PyAGC>(m, "AGC",
		"Automatic Gain Control.\n\n"
		"Measures the RMS level with a configurable window and applies a gain "
		"so the output RMS approaches target_level. `max_gain` caps the gain "
		"to prevent amplifying silence or noise floors.")
		.def(nb::init<double, double, double, double, const std::string&>(),
		     nb::arg("sample_rate"), nb::arg("target_level"),
		     nb::arg("window_ms") = 100.0, nb::arg("max_gain") = 100.0,
		     nb::arg("dtype") = "reference",
		     "Construct an AGC. target_level is in linear units "
		     "(e.g. 0.5 for -6 dBFS).")
		.def("process", &PyAGC::process,
		     nb::arg("input"),
		     "Process a single sample. Returns the gain-adjusted output.")
		.def("process_block", &PyAGC::process_block,
		     nb::arg("signal"),
		     "Process a 1D NumPy float64 signal. Returns the gain-adjusted "
		     "signal (same length as the input). The per-sample loop releases "
		     "the GIL.")
		.def("reset", &PyAGC::reset,
		     "Clear the internal RMS envelope state.")
		.def_prop_ro("dtype", &PyAGC::dtype,
		     "The arithmetic configuration selected at construction.");
}
