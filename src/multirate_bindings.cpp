// multirate_bindings.cpp: bindings for sw::dsp::multirate.
//
// Surface bound here:
//   - Channelizer        (multirate/channelizer.hpp)
//   - FractionalDelay    (multirate/fractional_delay.hpp)
//   - channelizer_prototype_bank (Channelizer::prototype_bank, free function)
//
// Both are upstream v0.7.0 additions that had no Python entry points. They
// follow the same type-erased IImpl / Impl<T> / Py-wrapper pattern as
// acquisition_bindings.cpp, and the same single-T dispatch: T fills
// CoeffScalar, StateScalar and SampleScalar together.
//
// Complex output is returned as a (real, imag) tuple of float64 arrays, which
// is what every other complex-valued binding in this package does — see
// complex_split_to_numpy in _binding_helpers.hpp.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sw/dsp/multirate/channelizer.hpp>
#include <sw/dsp/multirate/fractional_delay.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

using mpdsp::bindings::complex_split_to_numpy;
using mpdsp::bindings::dispatch_dtype_fn;
using mpdsp::bindings::make_f64_2d_array;
using mpdsp::bindings::np_f64;
using mpdsp::bindings::np_f64_ro;
using mpdsp::bindings::numpy_to_vec_fresh;
using mpdsp::bindings::vec_to_numpy;

namespace {

// ===========================================================================
// Channelizer
// ===========================================================================

struct IChannelizerImpl {
	virtual ~IChannelizerImpl() = default;
	virtual nb::tuple process(np_f64_ro block) = 0;
	virtual nb::tuple process_block(np_f64_ro signal) = 0;
	virtual std::size_t num_channels() const = 0;
	virtual std::size_t taps_per_phase() const = 0;
	virtual std::size_t num_taps() const = 0;
	virtual void reset() = 0;
};

template <typename T>
class ChannelizerImpl : public IChannelizerImpl {
public:
	ChannelizerImpl(std::size_t channels, std::size_t taps_per_phase,
	                double kaiser_beta)
		: ch_(channels, taps_per_phase, kaiser_beta) {}

	nb::tuple process(np_f64_ro block) override {
		auto in = numpy_to_vec_fresh<T>(block);
		auto out = ch_.process(std::span<const T>(in.data(), in.size()));
		return complex_split_to_numpy(out);
	}

	// Consume floor(N / M) whole blocks and return one row of channel
	// outputs per block. Looping in Python would work but costs a binding
	// round trip per M samples, which for a 64-channel bank over a second of
	// audio is tens of thousands of crossings.
	nb::tuple process_block(np_f64_ro signal) override {
		const std::size_t channels = ch_.M();
		auto in = numpy_to_vec_fresh<T>(signal);
		const std::size_t blocks = in.size() / channels;

		double* re_ptr = nullptr;
		double* im_ptr = nullptr;
		auto re_arr = make_f64_2d_array(blocks, channels, re_ptr);
		auto im_arr = make_f64_2d_array(blocks, channels, im_ptr);

		for (std::size_t b = 0; b < blocks; ++b) {
			auto out = ch_.process(
				std::span<const T>(in.data() + b * channels, channels));
			for (std::size_t c = 0; c < channels; ++c) {
				re_ptr[b * channels + c] = static_cast<double>(out[c].real());
				im_ptr[b * channels + c] = static_cast<double>(out[c].imag());
			}
		}
		return nb::make_tuple(re_arr, im_arr);
	}

	std::size_t num_channels() const override   { return ch_.M(); }
	std::size_t taps_per_phase() const override { return ch_.taps_per_phase(); }
	std::size_t num_taps() const override       { return ch_.num_taps(); }
	void reset() override                       { ch_.reset(); }

private:
	sw::dsp::multirate::Channelizer<T> ch_;
};

// ===========================================================================
// FractionalDelay
// ===========================================================================

struct IFractionalDelayImpl {
	virtual ~IFractionalDelayImpl() = default;
	virtual double delay(double in, double offset_samples) = 0;
	virtual np_f64 delay_block(np_f64_ro signal, double offset_samples) = 0;
	virtual double base_group_delay_samples() const = 0;
	virtual std::size_t num_phases() const = 0;
	virtual std::size_t taps_per_phase() const = 0;
	virtual std::size_t num_taps() const = 0;
	virtual std::size_t max_int_delay() const = 0;
	virtual void reset() = 0;
};

template <typename T>
class FractionalDelayImpl : public IFractionalDelayImpl {
public:
	FractionalDelayImpl(std::size_t phases, std::size_t taps_per_phase,
	                    std::size_t max_int_delay, double kaiser_beta)
		: fd_(phases, taps_per_phase, max_int_delay, kaiser_beta) {}

	double delay(double in, double offset_samples) override {
		return static_cast<double>(
			fd_.delay(static_cast<T>(in), offset_samples));
	}

	np_f64 delay_block(np_f64_ro signal, double offset_samples) override {
		auto in = numpy_to_vec_fresh<T>(signal);
		mtl::vec::dense_vector<T> out(in.size());
		for (std::size_t i = 0; i < in.size(); ++i)
			out[i] = fd_.delay(in[i], offset_samples);
		return vec_to_numpy(out);
	}

	double base_group_delay_samples() const override {
		return fd_.base_group_delay_samples();
	}
	std::size_t num_phases() const override     { return fd_.L(); }
	std::size_t taps_per_phase() const override { return fd_.taps_per_phase(); }
	std::size_t num_taps() const override       { return fd_.num_taps(); }
	std::size_t max_int_delay() const override  { return fd_.max_int_delay(); }
	void reset() override                       { fd_.reset(); }

private:
	sw::dsp::multirate::FractionalDelay<T> fd_;
};

} // anonymous namespace

// ===========================================================================
// Py-wrappers
// ===========================================================================

class PyChannelizer {
public:
	PyChannelizer(std::size_t num_channels, std::size_t taps_per_phase,
	              double kaiser_beta, const std::string& dtype) {
		// Validated here as well as upstream so the message names the Python
		// argument (`num_channels`) rather than the C++ template parameter M.
		if (num_channels == 0 ||
		    (num_channels & (num_channels - 1)) != 0)
			throw std::invalid_argument(
				"Channelizer: num_channels must be a nonzero power of two "
				"(the library FFT requires it)");
		if (taps_per_phase == 0)
			throw std::invalid_argument(
				"Channelizer: taps_per_phase must be > 0");
		auto config = mpdsp::parse_config(dtype);
		impl_ = dispatch_dtype_fn(config, "Channelizer",
			[&]<typename T>() -> std::unique_ptr<IChannelizerImpl> {
				return std::make_unique<ChannelizerImpl<T>>(
					num_channels, taps_per_phase, kaiser_beta);
			});
	}

	nb::tuple process(np_f64_ro block) {
		if (block.shape(0) != impl_->num_channels())
			throw std::invalid_argument(
				"Channelizer.process: block length must equal num_channels ("
				+ std::to_string(impl_->num_channels()) + ", got "
				+ std::to_string(block.shape(0)) + "). Use process_block() "
				"for a longer signal.");
		return impl_->process(block);
	}
	nb::tuple process_block(np_f64_ro signal) {
		return impl_->process_block(signal);
	}
	std::size_t num_channels() const   { return impl_->num_channels(); }
	std::size_t taps_per_phase() const { return impl_->taps_per_phase(); }
	std::size_t num_taps() const       { return impl_->num_taps(); }
	void reset()                       { impl_->reset(); }

private:
	std::unique_ptr<IChannelizerImpl> impl_;
};

class PyFractionalDelay {
public:
	PyFractionalDelay(std::size_t num_phases, std::size_t taps_per_phase,
	                  std::size_t max_int_delay, double kaiser_beta,
	                  const std::string& dtype) {
		if (num_phases == 0)
			throw std::invalid_argument(
				"FractionalDelay: num_phases must be > 0");
		// Upstream requires odd taps (the group delay is (K-1)/2, and odd K
		// keeps that floor an integer). Checked here so the error names the
		// Python argument and says what to do, rather than surfacing from
		// inside the bank designer.
		if (taps_per_phase < 3 || (taps_per_phase % 2) == 0)
			throw std::invalid_argument(
				"FractionalDelay: taps_per_phase must be odd and >= 3 (got "
				+ std::to_string(taps_per_phase) + "). The group delay is "
				"(taps_per_phase - 1) / 2, which an odd length keeps an "
				"integer.");
		auto config = mpdsp::parse_config(dtype);
		impl_ = dispatch_dtype_fn(config, "FractionalDelay",
			[&]<typename T>() -> std::unique_ptr<IFractionalDelayImpl> {
				return std::make_unique<FractionalDelayImpl<T>>(
					num_phases, taps_per_phase, max_int_delay, kaiser_beta);
			});
	}

	double delay(double in, double offset_samples) {
		return impl_->delay(in, offset_samples);
	}
	np_f64 delay_block(np_f64_ro signal, double offset_samples) {
		return impl_->delay_block(signal, offset_samples);
	}
	double base_group_delay_samples() const {
		return impl_->base_group_delay_samples();
	}
	std::size_t num_phases() const     { return impl_->num_phases(); }
	std::size_t taps_per_phase() const { return impl_->taps_per_phase(); }
	std::size_t num_taps() const       { return impl_->num_taps(); }
	std::size_t max_int_delay() const  { return impl_->max_int_delay(); }
	void reset()                       { impl_->reset(); }

private:
	std::unique_ptr<IFractionalDelayImpl> impl_;
};

// ===========================================================================
// bind_multirate
// ===========================================================================

void bind_multirate(nb::module_& m) {
	m.def("channelizer_prototype_bank",
		[](std::size_t num_channels, std::size_t taps_per_phase,
		   double kaiser_beta, const std::string& dtype) {
			if (num_channels == 0 ||
			    (num_channels & (num_channels - 1)) != 0)
				throw std::invalid_argument(
					"channelizer_prototype_bank: num_channels must be a "
					"nonzero power of two");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "channelizer_prototype_bank",
				[&]<typename T>() -> std::vector<np_f64> {
					auto bank = sw::dsp::multirate::Channelizer<T>
						::prototype_bank(num_channels, taps_per_phase,
						                 kaiser_beta);
					std::vector<np_f64> out;
					out.reserve(bank.size());
					for (auto& sub : bank) out.push_back(vec_to_numpy(sub));
					return out;
				});
		}, nb::arg("num_channels"), nb::arg("taps_per_phase") = 16,
		   nb::arg("kaiser_beta") = 8.0, nb::arg("dtype") = "reference",
		"The polyphase decomposition of a Channelizer's shared prototype, as "
		"a list of num_channels sub-filter tap arrays.\\n\\n"
		"Exposed so an analysis bank and a synthesis bank can be built on the "
		"*same* prototype. Mismatching the two halves of such a pair destroys "
		"reconstruction rather than merely degrading it, so there is "
		"deliberately no second copy of this design to drift out of step.");

	// ---- Channelizer ---------------------------------------------------
	nb::class_<PyChannelizer>(m, "Channelizer",
		"Bellanger polyphase channelizer: splits a wideband input into "
		"num_channels uniformly-spaced complex baseband channels at "
		"1/num_channels of the input rate.\\n\\n"
		"Each block of num_channels input samples advances the polyphase "
		"sub-filters once and is then inverse-transformed, so the whole bank "
		"costs about one prototype-filter evaluation per input sample rather "
		"than one per channel — which is the entire reason to build a "
		"channelizer instead of num_channels independent down-converters.\\n\\n"
		"num_channels must be a power of two (the library FFT requires it). "
		"Longer taps_per_phase sharpens the channel edges and deepens "
		"adjacent-channel rejection, at the cost of compute and "
		"(taps_per_phase-1)/2 samples of group delay. kaiser_beta sets the "
		"prototype window: 8 gives roughly -58 dB, 12 roughly -115 dB.")
		.def(nb::init<std::size_t, std::size_t, double, const std::string&>(),
		     nb::arg("num_channels"), nb::arg("taps_per_phase") = 16,
		     nb::arg("kaiser_beta") = 8.0, nb::arg("dtype") = "reference")
		.def("process", &PyChannelizer::process, nb::arg("block"),
		     "Push exactly num_channels input samples through the bank. "
		     "Returns a (real, imag) tuple of length-num_channels float64 "
		     "arrays — one complex sample per channel.")
		.def("process_block", &PyChannelizer::process_block, nb::arg("signal"),
		     "Consume floor(len(signal) / num_channels) whole blocks. "
		     "Returns a (real, imag) tuple of (num_blocks, num_channels) "
		     "float64 arrays, so row b column c is channel c at output-rate "
		     "sample b. Combine with `real + 1j*imag`. A trailing partial "
		     "block is left unconsumed rather than zero-padded, since padding "
		     "would inject a transient the caller did not ask for.")
		.def_prop_ro("num_channels", &PyChannelizer::num_channels)
		.def_prop_ro("taps_per_phase", &PyChannelizer::taps_per_phase)
		.def_prop_ro("num_taps", &PyChannelizer::num_taps,
		     "Total prototype length, num_channels * taps_per_phase.")
		.def("reset", &PyChannelizer::reset,
		     "Clear every sub-filter delay line.");

	// ---- FractionalDelay -----------------------------------------------
	nb::class_<PyFractionalDelay>(m, "FractionalDelay",
		"Polyphase fractional-sample delay line: resamples the input at an "
		"arbitrary sub-sample offset, with resolution 1/num_phases.\\n\\n"
		"The filter has an intrinsic group delay of "
		"(taps_per_phase-1)/2 input samples, and that is the *smallest* "
		"offset it can serve — a filter cannot reconstruct samples from the "
		"future, so requests below the floor round up to it rather than "
		"failing silently. Requests beyond group_delay + max_int_delay raise, "
		"because the ring buffer no longer holds the history they need.\\n\\n"
		"Higher num_phases buys finer offset resolution at the cost of "
		"coefficient memory; longer taps_per_phase buys in-band flatness and "
		"stopband depth. It must be odd and >= 3, which keeps the group "
		"delay an integer.\n\n"
		"taps_per_phase defaults to 11, not to upstream's 12: that default "
		"is even and its own validator rejects it, so constructing the C++ "
		"class with documented defaults throws "
		"(mixed-precision-dsp#208). 11 is the nearest usable value.")
		.def(nb::init<std::size_t, std::size_t, std::size_t, double,
		              const std::string&>(),
		     nb::arg("num_phases"), nb::arg("taps_per_phase") = 11,
		     nb::arg("max_int_delay") = 32, nb::arg("kaiser_beta") = 8.0,
		     nb::arg("dtype") = "reference")
		.def("delay", &PyFractionalDelay::delay,
		     nb::arg("input"), nb::arg("offset_samples"),
		     "Push one sample and return the interpolated output at "
		     "offset_samples measured back from it.")
		.def("delay_block", &PyFractionalDelay::delay_block,
		     nb::arg("signal"), nb::arg("offset_samples"),
		     "Apply a constant offset across a whole signal. Equivalent to "
		     "calling delay() per sample, without the per-sample binding "
		     "crossing.")
		.def_prop_ro("base_group_delay_samples",
		     &PyFractionalDelay::base_group_delay_samples,
		     "(taps_per_phase - 1) / 2 — the smallest offset this filter can "
		     "serve. Smaller requests round up to it.")
		.def_prop_ro("num_phases", &PyFractionalDelay::num_phases,
		     "Offset resolution is 1 / num_phases samples.")
		.def_prop_ro("taps_per_phase", &PyFractionalDelay::taps_per_phase)
		.def_prop_ro("num_taps", &PyFractionalDelay::num_taps,
		     "Total prototype length, num_phases * taps_per_phase.")
		.def_prop_ro("max_int_delay", &PyFractionalDelay::max_int_delay)
		.def("reset", &PyFractionalDelay::reset,
		     "Clear the delay line.");
}
