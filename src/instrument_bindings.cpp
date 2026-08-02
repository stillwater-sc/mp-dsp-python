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
#include <nanobind/stl/string.h>

#include <sw/dsp/instrument/measurements.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

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

} // namespace

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
}
