// instrument_bindings.cpp: oscilloscope-style measurement primitives.
//
// Phase 2 of the bindings-gap roadmap (docs/gap_analysis_2026-08-01.md,
// epic stillwater-sc/mp-dsp-python#100). Wraps the seven stateless
// measurement free functions from sw::dsp::instrument. Stateful capture
// primitives (PeakDetectDecimator, TriggerRingBuffer) land in follow-up
// PRs against the same file.
//
// Dispatch pattern: each measurement returns double regardless of dtype;
// the dtype key only controls the sample-container scalar the signal is
// cast into before measurement. Sensor configs run in double per the
// project convention (see _binding_helpers.hpp:dispatch_dtype_fn).
//
// Naming note: `mean` and `rms` are prefixed with `instrument_` in Python
// to avoid shadowing numpy.mean / numpy.rms when a user imports from
// mpdsp with `from mpdsp import *`. The other five keep their C++ names
// (with the `_samples` / `_hz` suffixes stripped since Python returns
// plain floats — the unit is documented, not encoded).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <sw/dsp/instrument/measurements.hpp>
#include <sw/dsp/instrument/peak_detect.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace nb = nanobind;

namespace {

// Cast a NumPy float64 signal into a dense_vector<T>. Element-wise
// static_cast — narrow types lose precision but don't scale, matching
// the convention used by cast_double_to_T in spectral_bindings.cpp.
template <typename T>
static mtl::vec::dense_vector<T>
cast_signal(mpdsp::bindings::np_f64_ro signal) {
	std::size_t n = signal.shape(0);
	mtl::vec::dense_vector<T> v(n);
	const double* src = signal.data();
	for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<T>(src[i]);
	return v;
}

// Every measurement rejects an empty input the same way; centralise.
static void require_nonempty(mpdsp::bindings::np_f64_ro signal, const char* fn) {
	if (signal.shape(0) == 0) {
		throw std::invalid_argument(std::string(fn) + ": signal must be non-empty");
	}
}

// ---------------------------------------------------------------------------
// PeakDetectDecimator impl. Type-erased interface with per-dtype concrete
// impls, held behind std::unique_ptr in PyPeakDetectDecimator. Same shape as
// the conditioning / acquisition classes.
//
// SampleScalar is constrained on DspOrderedField upstream (< / > required),
// which every dtype instantiated by make_impl_for_dtype satisfies. Sensor
// configs dispatch to double per the project convention.
// ---------------------------------------------------------------------------

struct IPeakDetectDecimatorImpl {
	virtual ~IPeakDetectDecimatorImpl() = default;
	virtual std::optional<std::pair<double, double>> process(double x) = 0;
	virtual std::pair<mtl::vec::dense_vector<double>,
	                  mtl::vec::dense_vector<double>>
	process_block(const double* in, std::size_t n) = 0;
	virtual mtl::vec::dense_vector<double>
	process_block_min(const double* in, std::size_t n) = 0;
	virtual mtl::vec::dense_vector<double>
	process_block_max(const double* in, std::size_t n) = 0;
	virtual void reset() = 0;
	virtual std::size_t decimation_factor() const = 0;
	virtual std::size_t samples_in_window() const = 0;
};

template <typename T>
struct PeakDetectDecimatorImpl : IPeakDetectDecimatorImpl {
	sw::dsp::instrument::PeakDetectDecimator<T> inner;
	explicit PeakDetectDecimatorImpl(std::size_t R) : inner(R) {}

	std::optional<std::pair<double, double>> process(double x) override {
		auto r = inner.process(static_cast<T>(x));
		if (!r.has_value()) return std::nullopt;
		return std::make_pair(static_cast<double>(r->first),
		                      static_cast<double>(r->second));
	}

	std::pair<mtl::vec::dense_vector<double>, mtl::vec::dense_vector<double>>
	process_block(const double* in, std::size_t n) override {
		mtl::vec::dense_vector<T> typed_in(n);
		for (std::size_t i = 0; i < n; ++i) typed_in[i] = static_cast<T>(in[i]);
		auto env = inner.process_block(
			std::span<const T>(typed_in.data(), typed_in.size()));
		mtl::vec::dense_vector<double> mins(env.mins.size());
		mtl::vec::dense_vector<double> maxs(env.maxs.size());
		for (std::size_t i = 0; i < env.mins.size(); ++i) {
			mins[i] = static_cast<double>(env.mins[i]);
			maxs[i] = static_cast<double>(env.maxs[i]);
		}
		return {std::move(mins), std::move(maxs)};
	}

	mtl::vec::dense_vector<double>
	process_block_min(const double* in, std::size_t n) override {
		mtl::vec::dense_vector<T> typed_in(n);
		for (std::size_t i = 0; i < n; ++i) typed_in[i] = static_cast<T>(in[i]);
		auto out = inner.process_block_min(
			std::span<const T>(typed_in.data(), typed_in.size()));
		mtl::vec::dense_vector<double> result(out.size());
		for (std::size_t i = 0; i < out.size(); ++i)
			result[i] = static_cast<double>(out[i]);
		return result;
	}

	mtl::vec::dense_vector<double>
	process_block_max(const double* in, std::size_t n) override {
		mtl::vec::dense_vector<T> typed_in(n);
		for (std::size_t i = 0; i < n; ++i) typed_in[i] = static_cast<T>(in[i]);
		auto out = inner.process_block_max(
			std::span<const T>(typed_in.data(), typed_in.size()));
		mtl::vec::dense_vector<double> result(out.size());
		for (std::size_t i = 0; i < out.size(); ++i)
			result[i] = static_cast<double>(out[i]);
		return result;
	}

	void reset() override                        { inner.reset(); }
	std::size_t decimation_factor() const override { return inner.decimation_factor(); }
	std::size_t samples_in_window() const override { return inner.samples_in_window(); }
};

} // namespace

// PyPeakDetectDecimator: Python-facing class holding the type-erased impl.
class PyPeakDetectDecimator {
public:
	PyPeakDetectDecimator(std::size_t decimation_factor, const std::string& dtype)
	    : dtype_(dtype) {
		if (decimation_factor == 0) {
			throw std::invalid_argument(
				"PeakDetectDecimator: decimation_factor must be >= 1");
		}
		impl_ = mpdsp::bindings::make_impl_for_dtype<
			PeakDetectDecimatorImpl, IPeakDetectDecimatorImpl>(
			mpdsp::parse_config(dtype), "PeakDetectDecimator",
			decimation_factor);
	}

	std::optional<std::pair<double, double>> process(double x) {
		return impl_->process(x);
	}

	nb::tuple process_block(mpdsp::bindings::np_f64_ro signal) {
		std::size_t n = signal.shape(0);
		auto [mins, maxs] = impl_->process_block(signal.data(), n);
		return nb::make_tuple(mpdsp::bindings::vec_to_numpy(mins),
		                     mpdsp::bindings::vec_to_numpy(maxs));
	}

	mpdsp::bindings::np_f64 process_block_min(mpdsp::bindings::np_f64_ro signal) {
		auto out = impl_->process_block_min(signal.data(), signal.shape(0));
		return mpdsp::bindings::vec_to_numpy(out);
	}

	mpdsp::bindings::np_f64 process_block_max(mpdsp::bindings::np_f64_ro signal) {
		auto out = impl_->process_block_max(signal.data(), signal.shape(0));
		return mpdsp::bindings::vec_to_numpy(out);
	}

	void reset()                            { impl_->reset(); }
	std::size_t decimation_factor() const   { return impl_->decimation_factor(); }
	std::size_t samples_in_window() const   { return impl_->samples_in_window(); }
	const std::string& dtype() const        { return dtype_; }

private:
	std::unique_ptr<IPeakDetectDecimatorImpl> impl_;
	std::string dtype_;
};

void bind_instrument(nb::module_& m) {
	using mpdsp::bindings::dispatch_dtype_fn;
	using mpdsp::bindings::np_f64_ro;

	m.def("peak_to_peak",
		[](np_f64_ro signal, const std::string& dtype) {
			require_nonempty(signal, "peak_to_peak");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "peak_to_peak", [&]<typename T>() {
				auto v = cast_signal<T>(signal);
				return sw::dsp::instrument::peak_to_peak<T>(
					std::span<const T>(v.data(), v.size()));
			});
		},
		nb::arg("signal"), nb::arg("dtype") = "reference",
		"Peak-to-peak amplitude of the segment: max(signal) - min(signal). "
		"For a unit-amplitude sine returns 2.0.");

	m.def("instrument_mean",
		[](np_f64_ro signal, const std::string& dtype) {
			require_nonempty(signal, "instrument_mean");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "instrument_mean", [&]<typename T>() {
				auto v = cast_signal<T>(signal);
				return sw::dsp::instrument::mean<T>(
					std::span<const T>(v.data(), v.size()));
			});
		},
		nb::arg("signal"), nb::arg("dtype") = "reference",
		"Arithmetic mean (DC level) of the segment. Sum is accumulated in "
		"double regardless of dtype. Prefixed to avoid shadowing numpy.mean "
		"when users do `from mpdsp import *`.");

	m.def("instrument_rms",
		[](np_f64_ro signal, const std::string& dtype) {
			require_nonempty(signal, "instrument_rms");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "instrument_rms", [&]<typename T>() {
				auto v = cast_signal<T>(signal);
				return sw::dsp::instrument::rms<T>(
					std::span<const T>(v.data(), v.size()));
			});
		},
		nb::arg("signal"), nb::arg("dtype") = "reference",
		"Root-mean-square of the segment. For a unit-amplitude sine returns "
		"1/sqrt(2). Sum-of-squares is accumulated in double.");

	m.def("rise_time",
		[](np_f64_ro signal, double low_pct, double high_pct,
		   const std::string& dtype) {
			require_nonempty(signal, "rise_time");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "rise_time", [&]<typename T>() {
				auto v = cast_signal<T>(signal);
				return sw::dsp::instrument::rise_time_samples<T>(
					std::span<const T>(v.data(), v.size()), low_pct, high_pct);
			});
		},
		nb::arg("signal"), nb::arg("low_pct") = 0.1, nb::arg("high_pct") = 0.9,
		nb::arg("dtype") = "reference",
		"Rise time in SAMPLES between low_pct and high_pct of the segment's "
		"peak-to-peak range, on the first rising transition. Returns NaN if "
		"no transition spans both thresholds. Divide by sample_rate for "
		"seconds. Sub-sample crossings via linear interpolation.");

	m.def("fall_time",
		[](np_f64_ro signal, double low_pct, double high_pct,
		   const std::string& dtype) {
			require_nonempty(signal, "fall_time");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "fall_time", [&]<typename T>() {
				auto v = cast_signal<T>(signal);
				return sw::dsp::instrument::fall_time_samples<T>(
					std::span<const T>(v.data(), v.size()), low_pct, high_pct);
			});
		},
		nb::arg("signal"), nb::arg("low_pct") = 0.1, nb::arg("high_pct") = 0.9,
		nb::arg("dtype") = "reference",
		"Fall time in SAMPLES: mirror of rise_time for the first falling "
		"transition from high_pct down to low_pct. Returns NaN if no such "
		"transition. Divide by sample_rate for seconds.");

	m.def("period",
		[](np_f64_ro signal, double threshold, const std::string& dtype) {
			require_nonempty(signal, "period");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "period", [&]<typename T>() {
				auto v = cast_signal<T>(signal);
				return sw::dsp::instrument::period_samples<T>(
					std::span<const T>(v.data(), v.size()),
					static_cast<T>(threshold));
			});
		},
		nb::arg("signal"), nb::arg("threshold") = 0.0,
		nb::arg("dtype") = "reference",
		"Period in SAMPLES: average distance between consecutive rising "
		"threshold-crossings. Threshold defaults to 0 (zero-crossing, "
		"appropriate for AC-coupled signals). Returns NaN if fewer than two "
		"rising crossings occur.");

	m.def("frequency",
		[](np_f64_ro signal, double sample_rate, double threshold,
		   const std::string& dtype) {
			require_nonempty(signal, "frequency");
			if (!(sample_rate > 0.0)) {
				throw std::invalid_argument(
					"frequency: sample_rate must be positive");
			}
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "frequency", [&]<typename T>() {
				auto v = cast_signal<T>(signal);
				return sw::dsp::instrument::frequency_hz<T>(
					std::span<const T>(v.data(), v.size()),
					sample_rate, static_cast<T>(threshold));
			});
		},
		nb::arg("signal"), nb::arg("sample_rate"), nb::arg("threshold") = 0.0,
		nb::arg("dtype") = "reference",
		"Fundamental frequency in Hz: sample_rate / period_samples. Returns "
		"NaN if the period cannot be measured (see `period`).");

	nb::class_<PyPeakDetectDecimator>(m, "PeakDetectDecimator",
			"Scope-style decimator that emits one (min, max) pair per R input "
			"samples. Unlike a generic averaging decimator, a glitch shorter "
			"than the decimation interval still shows up in the output because "
			"both extremes are preserved.")
		.def(nb::init<std::size_t, const std::string&>(),
		     nb::arg("decimation_factor"), nb::arg("dtype") = "reference",
		     "Construct a decimator with the given decimation factor R (>= 1). "
		     "dtype selects the internal sample scalar; the Python I/O is "
		     "always float64.")
		.def("process", &PyPeakDetectDecimator::process, nb::arg("sample"),
		     "Push one sample. Returns None while accumulating within a "
		     "decimation window; returns (min, max) as a tuple of floats on "
		     "the sample that completes the current window.")
		.def("process_block", &PyPeakDetectDecimator::process_block,
		     nb::arg("signal"),
		     "Push a block of samples. Returns (mins, maxs) as a pair of "
		     "NumPy arrays. Length of each output = (samples_in_window + "
		     "len(signal)) // decimation_factor. Partial trailing windows "
		     "carry over as internal state; call process() or another "
		     "process_block() to keep going.")
		.def("process_block_min", &PyPeakDetectDecimator::process_block_min,
		     nb::arg("signal"),
		     "Same as process_block() but returns only the lower envelope "
		     "(the mins array). Convenience for callers building single-"
		     "envelope views.")
		.def("process_block_max", &PyPeakDetectDecimator::process_block_max,
		     nb::arg("signal"),
		     "Same as process_block() but returns only the upper envelope "
		     "(the maxs array).")
		.def("reset", &PyPeakDetectDecimator::reset,
		     "Drop any partial window in progress and re-arm the decimator.")
		.def_prop_ro("decimation_factor",
		     &PyPeakDetectDecimator::decimation_factor,
		     "Decimation factor R (samples per output pair). Read-only.")
		.def_prop_ro("samples_in_window",
		     &PyPeakDetectDecimator::samples_in_window,
		     "Number of samples pushed into the current incomplete window. "
		     "Reaches decimation_factor - 1 just before the next output pair "
		     "is emitted, then wraps back to 0.")
		.def_prop_ro("dtype",
		     [](const PyPeakDetectDecimator& self) { return self.dtype(); },
		     "Sample-scalar dtype fixed at construction. Read-only.");
}
