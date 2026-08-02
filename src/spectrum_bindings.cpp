// spectrum_bindings.cpp: spectrum-analyzer primitives.
//
// Phase 3 of the bindings-gap roadmap (docs/gap_analysis_2026-08-01.md,
// epic stillwater-sc/mp-dsp-python#100). First landed piece is the
// RealtimeSpectrum streaming FFT engine plus the five stateless
// detector reducers and the runtime-dispatch detect(mode) entry. RBW/
// VBW filters, swept-LO, trace averaging, waterfall, and markers land
// in subsequent PRs against this same file (#105-#107).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <sw/dsp/instrument/calibration.hpp>  // CalibrationProfile
#include <sw/dsp/spectrum/detectors.hpp>
#include <sw/dsp/spectrum/front_end_corrector.hpp>
#include <sw/dsp/spectrum/markers.hpp>
#include <sw/dsp/spectrum/rbw_filter.hpp>
#include <sw/dsp/spectrum/realtime_spectrum.hpp>
#include <sw/dsp/spectrum/swept_lo.hpp>
#include <sw/dsp/spectrum/trace_averaging.hpp>
#include <sw/dsp/spectrum/vbw_filter.hpp>
#include <sw/dsp/spectrum/waterfall_buffer.hpp>

#include <nanobind/stl/vector.h>

#include "_binding_helpers.hpp"
#include "types.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

// Element-wise static_cast of a NumPy float64 signal into dense_vector<T>.
// Same shape as the instrument_bindings helper.
template <typename T>
static mtl::vec::dense_vector<T>
cast_signal(mpdsp::bindings::np_f64_ro signal) {
	std::size_t n = signal.shape(0);
	mtl::vec::dense_vector<T> v(n);
	const double* src = signal.data();
	for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<T>(src[i]);
	return v;
}

static void require_nonempty(mpdsp::bindings::np_f64_ro bin, const char* fn) {
	if (bin.shape(0) == 0) {
		throw std::invalid_argument(std::string(fn) + ": bin must be non-empty");
	}
}

// String -> DetectorMode for the runtime-dispatch detect() binding.
static sw::dsp::spectrum::DetectorMode parse_detector_mode(const std::string& s) {
	using sw::dsp::spectrum::DetectorMode;
	if (s == "peak")          return DetectorMode::Peak;
	if (s == "sample")        return DetectorMode::Sample;
	if (s == "average")       return DetectorMode::Average;
	if (s == "rms")           return DetectorMode::RMS;
	if (s == "negative_peak") return DetectorMode::NegativePeak;
	throw std::invalid_argument(
		"detect: unknown mode '" + s + "' (expected peak, sample, "
		"average, rms, negative_peak)");
}

// ---------------------------------------------------------------------------
// RealtimeSpectrum impl. Type-erased interface; concrete impl instantiates
// sw::dsp::spectrum::RealtimeSpectrum with all four template parameters set
// to the same T (coeff = state = sample = window), matching how other
// stateful classes in this project dispatch dtype.
//
// Complex FFT output is unpacked to two double vectors (real, imag) to
// keep the Python interface tuple-based, matching the existing fft() /
// ifft() bindings in spectral_bindings.cpp.
// ---------------------------------------------------------------------------

struct IRealtimeSpectrumImpl {
	virtual ~IRealtimeSpectrumImpl() = default;
	virtual std::size_t push(const double* in, std::size_t n) = 0;
	virtual std::pair<mtl::vec::dense_vector<double>,
	                  mtl::vec::dense_vector<double>>
	latest_complex() const = 0;
	virtual mtl::vec::dense_vector<double> latest_magnitude_db() const = 0;
	virtual void reset() = 0;
	virtual std::size_t fft_size()       const = 0;
	virtual std::size_t hop_size()       const = 0;
	virtual std::size_t total_ffts()     const = 0;
	virtual bool        first_fft_ready() const = 0;
};

template <typename T>
struct RealtimeSpectrumImpl : IRealtimeSpectrumImpl {
	// Declaration order matters: window_storage must precede inner so it's
	// fully constructed before inner's ctor reads a span into it. The
	// upstream RealtimeSpectrum ctor copies from the span, so we could
	// technically drop the storage after — keeping it makes future
	// window-swap methods (if added) trivial.
	mtl::vec::dense_vector<T> window_storage;
	sw::dsp::spectrum::RealtimeSpectrum<T, T, T, T> inner;

	RealtimeSpectrumImpl(std::size_t fft_size, std::size_t hop_size,
	                     const double* window_data, std::size_t window_size)
	    : window_storage(make_window(window_data, window_size)),
	      inner(fft_size, hop_size,
	            std::span<const T>(window_storage.data(),
	                               window_storage.size())) {}

	std::size_t push(const double* in, std::size_t n) override {
		// Cast input into typed samples and feed via a span. Allocating
		// a per-call buffer is fine here — RealtimeSpectrum's own copy
		// into the ring dominates.
		mtl::vec::dense_vector<T> typed(n);
		for (std::size_t i = 0; i < n; ++i) typed[i] = static_cast<T>(in[i]);
		return inner.push(std::span<const T>(typed.data(), typed.size()));
	}

	std::pair<mtl::vec::dense_vector<double>, mtl::vec::dense_vector<double>>
	latest_complex() const override {
		auto cplx = inner.latest_complex();
		mtl::vec::dense_vector<double> re(cplx.size());
		mtl::vec::dense_vector<double> im(cplx.size());
		using std::real;
		using std::imag;
		for (std::size_t i = 0; i < cplx.size(); ++i) {
			re[i] = static_cast<double>(real(cplx[i]));
			im[i] = static_cast<double>(imag(cplx[i]));
		}
		return {std::move(re), std::move(im)};
	}

	mtl::vec::dense_vector<double> latest_magnitude_db() const override {
		auto mag = inner.latest_magnitude_db();
		mtl::vec::dense_vector<double> out(mag.size());
		for (std::size_t i = 0; i < mag.size(); ++i) out[i] = mag[i];
		return out;
	}

	void reset()                    override { inner.reset(); }
	std::size_t fft_size()          const override { return inner.fft_size(); }
	std::size_t hop_size()          const override { return inner.hop_size(); }
	std::size_t total_ffts()        const override { return inner.total_ffts(); }
	bool        first_fft_ready()   const override { return inner.first_fft_ready(); }

private:
	static mtl::vec::dense_vector<T>
	make_window(const double* data, std::size_t n) {
		mtl::vec::dense_vector<T> w(n);
		for (std::size_t i = 0; i < n; ++i) w[i] = static_cast<T>(data[i]);
		return w;
	}
};

// ---------------------------------------------------------------------------
// RBWFilter impl. Same type-erased pattern. process_block returns a new
// NumPy array rather than binding the in-place / two-span overloads —
// Python callers create new buffers rather than mutating in place.
// ---------------------------------------------------------------------------

struct IRBWFilterImpl {
	virtual ~IRBWFilterImpl() = default;
	virtual double process(double x) = 0;
	virtual mtl::vec::dense_vector<double>
	process_block(const double* in, std::size_t n) = 0;
	virtual void retune(double center_freq_hz, double bandwidth_hz) = 0;
	virtual double shape_factor() const = 0;
	virtual void reset() = 0;
	virtual double      center_freq_hz() const = 0;
	virtual double      bandwidth_hz()   const = 0;
	virtual double      sample_rate_hz() const = 0;
	virtual std::size_t order()          const = 0;
};

template <typename T>
struct RBWFilterImpl : IRBWFilterImpl {
	sw::dsp::spectrum::RBWFilter<T, T, T> inner;

	RBWFilterImpl(double center_freq_hz, double bandwidth_hz,
	              double sample_rate_hz, std::size_t order)
	    : inner(center_freq_hz, bandwidth_hz, sample_rate_hz, order) {}

	double process(double x) override {
		return static_cast<double>(inner.process(static_cast<T>(x)));
	}

	mtl::vec::dense_vector<double>
	process_block(const double* in, std::size_t n) override {
		mtl::vec::dense_vector<double> out(n);
		for (std::size_t i = 0; i < n; ++i) {
			out[i] = static_cast<double>(inner.process(static_cast<T>(in[i])));
		}
		return out;
	}

	void retune(double c, double b) override { inner.retune(c, b); }
	double shape_factor() const   override { return inner.shape_factor(); }
	void reset()                  override { inner.reset(); }
	double      center_freq_hz() const override { return inner.center_freq_hz(); }
	double      bandwidth_hz()   const override { return inner.bandwidth_hz(); }
	double      sample_rate_hz() const override { return inner.sample_rate_hz(); }
	std::size_t order()          const override { return inner.order(); }
};

// ---------------------------------------------------------------------------
// VBWFilter impl. Same pattern; simpler class (one-pole LPF, single cutoff).
// ---------------------------------------------------------------------------

struct IVBWFilterImpl {
	virtual ~IVBWFilterImpl() = default;
	virtual double process(double x) = 0;
	virtual mtl::vec::dense_vector<double>
	process_block(const double* in, std::size_t n) = 0;
	virtual void set_cutoff(double cutoff_hz) = 0;
	virtual void reset() = 0;
	virtual double cutoff_hz()      const = 0;
	virtual double sample_rate_hz() const = 0;
};

template <typename T>
struct VBWFilterImpl : IVBWFilterImpl {
	sw::dsp::spectrum::VBWFilter<T, T, T> inner;

	VBWFilterImpl(double cutoff_hz, double sample_rate_hz)
	    : inner(cutoff_hz, sample_rate_hz) {}

	double process(double x) override {
		return static_cast<double>(inner.process(static_cast<T>(x)));
	}

	mtl::vec::dense_vector<double>
	process_block(const double* in, std::size_t n) override {
		mtl::vec::dense_vector<double> out(n);
		for (std::size_t i = 0; i < n; ++i) {
			out[i] = static_cast<double>(inner.process(static_cast<T>(in[i])));
		}
		return out;
	}

	void set_cutoff(double cutoff_hz) override { inner.set_cutoff(cutoff_hz); }
	void reset()                      override { inner.reset(); }
	double cutoff_hz()      const override { return inner.cutoff_hz(); }
	double sample_rate_hz() const override { return inner.sample_rate_hz(); }
};

// ---------------------------------------------------------------------------
// SweptLO impl. Same type-erased pattern. The mode enum is scoped inside
// SweptLO<T, T, T> — each template instantiation has its own type-distinct
// `Sweep` even though the values are the same integers. Pass a plain bool
// across the type-erasure boundary and rebuild the correct-typed enum
// inside each SweptLOImpl<T> ctor.
// ---------------------------------------------------------------------------

static bool parse_sweep_mode_is_log(const std::string& s) {
	if (s == "linear")      return false;
	if (s == "log")         return true;
	if (s == "logarithmic") return true;
	throw std::invalid_argument(
		"SweptLO: unknown mode '" + s + "' (expected 'linear' or 'logarithmic')");
}

struct ISweptLOImpl {
	virtual ~ISweptLOImpl() = default;
	virtual std::pair<double, double> process() = 0;
	virtual std::pair<mtl::vec::dense_vector<double>,
	                  mtl::vec::dense_vector<double>>
	generate_block(std::size_t n) = 0;
	virtual void reset() = 0;
	virtual double      current_frequency_hz() const = 0;
	virtual bool        sweep_complete()   const = 0;
	virtual std::size_t total_sweeps()     const = 0;
	virtual double      f_start_hz()       const = 0;
	virtual double      f_stop_hz()        const = 0;
	virtual double      sweep_duration_s() const = 0;
	virtual double      sample_rate_hz()   const = 0;
	virtual std::string mode()             const = 0;
	virtual std::size_t num_sweep_samples() const = 0;
};

template <typename T>
struct SweptLOImpl : ISweptLOImpl {
	sw::dsp::spectrum::SweptLO<T, T, T> inner;

	SweptLOImpl(double f_start_hz, double f_stop_hz,
	            double sweep_duration_s, double sample_rate_hz,
	            bool is_log)
	    : inner(f_start_hz, f_stop_hz, sweep_duration_s, sample_rate_hz,
	            is_log
	                ? sw::dsp::spectrum::SweptLO<T, T, T>::Sweep::Logarithmic
	                : sw::dsp::spectrum::SweptLO<T, T, T>::Sweep::Linear) {}

	std::pair<double, double> process() override {
		auto [c, s] = inner.process();
		return {static_cast<double>(c), static_cast<double>(s)};
	}

	std::pair<mtl::vec::dense_vector<double>, mtl::vec::dense_vector<double>>
	generate_block(std::size_t n) override {
		mtl::vec::dense_vector<double> cos_out(n);
		mtl::vec::dense_vector<double> sin_out(n);
		for (std::size_t i = 0; i < n; ++i) {
			auto [c, s] = inner.process();
			cos_out[i] = static_cast<double>(c);
			sin_out[i] = static_cast<double>(s);
		}
		return {std::move(cos_out), std::move(sin_out)};
	}

	void reset()                            override { inner.reset(); }
	double current_frequency_hz() const     override { return inner.current_frequency_hz(); }
	bool   sweep_complete()       const     override { return inner.sweep_complete(); }
	std::size_t total_sweeps()    const     override { return inner.total_sweeps(); }
	double f_start_hz()           const     override { return inner.f_start_hz(); }
	double f_stop_hz()            const     override { return inner.f_stop_hz(); }
	double sweep_duration_s()     const     override { return inner.sweep_duration_s(); }
	double sample_rate_hz()       const     override { return inner.sample_rate_hz(); }
	std::string mode()            const     override {
		using Mode = typename sw::dsp::spectrum::SweptLO<T, T, T>::Sweep;
		return inner.mode() == Mode::Linear ? "linear" : "logarithmic";
	}
	std::size_t num_sweep_samples() const   override { return inner.num_sweep_samples(); }
};

// ---------------------------------------------------------------------------
// FrontEndCorrector impl. FrontEndCorrector is a using-alias for
// EqualizerFilter — same underlying template, same streaming interface.
// The design math runs once in the ctor (frequency-sampling design +
// Hamming window + inverse DFT), then the class is a plain FIR.
// ---------------------------------------------------------------------------

struct IFrontEndCorrectorImpl {
	virtual ~IFrontEndCorrectorImpl() = default;
	virtual double process(double x) = 0;
	virtual mtl::vec::dense_vector<double>
	process_block(const double* in, std::size_t n) = 0;
	virtual std::size_t num_taps() const = 0;
};

template <typename T>
struct FrontEndCorrectorImpl : IFrontEndCorrectorImpl {
	sw::dsp::spectrum::FrontEndCorrector<T, T, T> inner;

	FrontEndCorrectorImpl(const sw::dsp::instrument::CalibrationProfile& profile,
	                      std::size_t num_taps, double sample_rate_hz,
	                      double max_gain_dB)
	    : inner(profile, num_taps, sample_rate_hz, max_gain_dB) {}

	double process(double x) override {
		return static_cast<double>(inner.process(static_cast<T>(x)));
	}

	mtl::vec::dense_vector<double>
	process_block(const double* in, std::size_t n) override {
		mtl::vec::dense_vector<double> out(n);
		for (std::size_t i = 0; i < n; ++i) {
			out[i] = static_cast<double>(inner.process(static_cast<T>(in[i])));
		}
		return out;
	}

	std::size_t num_taps() const override { return inner.num_taps(); }
};

// ---------------------------------------------------------------------------
// TraceAverager impl. Mode is a class-template-scoped enum, so cross the
// type-erasure boundary as an int mode_code (same integer values across
// all instantiations, cast per-Impl to the right enum type — same
// technique as SweptLO's is_log bool).
//
// Mode codes match the enum member order in trace_averaging.hpp:
//   0: Linear, 1: Exponential, 2: MaxHold, 3: MinHold, 4: MaxHoldN.
// ---------------------------------------------------------------------------

static int parse_averager_mode_code(const std::string& s) {
	if (s == "linear")      return 0;
	if (s == "exponential") return 1;
	if (s == "max_hold")    return 2;
	if (s == "min_hold")    return 3;
	if (s == "max_hold_n")  return 4;
	throw std::invalid_argument(
		"TraceAverager: unknown mode '" + s + "' (expected 'linear', "
		"'exponential', 'max_hold', 'min_hold', 'max_hold_n')");
}

static std::string averager_mode_to_string(int mode_code) {
	switch (mode_code) {
		case 0: return "linear";
		case 1: return "exponential";
		case 2: return "max_hold";
		case 3: return "min_hold";
		case 4: return "max_hold_n";
	}
	return "unknown";
}

struct ITraceAveragerImpl {
	virtual ~ITraceAveragerImpl() = default;
	virtual void accept_sweep(const double* trace, std::size_t n) = 0;
	virtual mtl::vec::dense_vector<double> current_trace() const = 0;
	virtual void reset() = 0;
	virtual std::size_t sweeps_accumulated() const = 0;
	virtual std::size_t trace_length()       const = 0;
	virtual int         mode_code()          const = 0;
};

template <typename T>
struct TraceAveragerImpl : ITraceAveragerImpl {
	using ModeEnum = typename sw::dsp::spectrum::TraceAverager<T>::Mode;

	static ModeEnum from_code(int code) {
		switch (code) {
			case 0: return ModeEnum::Linear;
			case 1: return ModeEnum::Exponential;
			case 2: return ModeEnum::MaxHold;
			case 3: return ModeEnum::MinHold;
			case 4: return ModeEnum::MaxHoldN;
		}
		throw std::invalid_argument("TraceAverager: bad mode code");
	}

	sw::dsp::spectrum::TraceAverager<T> inner;
	int mode_code_;

	TraceAveragerImpl(std::size_t trace_length, int mode_code, double config)
	    : inner(trace_length, from_code(mode_code), config),
	      mode_code_(mode_code) {}

	void accept_sweep(const double* trace, std::size_t n) override {
		mtl::vec::dense_vector<T> typed(n);
		for (std::size_t i = 0; i < n; ++i) typed[i] = static_cast<T>(trace[i]);
		inner.accept_sweep(std::span<const T>(typed.data(), typed.size()));
	}

	mtl::vec::dense_vector<double> current_trace() const override {
		auto span = inner.current_trace();
		mtl::vec::dense_vector<double> out(span.size());
		for (std::size_t i = 0; i < span.size(); ++i) {
			out[i] = static_cast<double>(span[i]);
		}
		return out;
	}

	void reset()                              override { inner.reset(); }
	std::size_t sweeps_accumulated() const    override { return inner.sweeps_accumulated(); }
	std::size_t trace_length()       const    override { return inner.trace_length(); }
	int         mode_code()          const    override { return mode_code_; }
};

// ---------------------------------------------------------------------------
// WaterfallBuffer impl. Pure storage; no arithmetic. Reads copy out to
// fresh dense_vector<double> — the upstream zero-copy span into the ring
// is invalidated by subsequent push_frame() calls, so we can't safely
// expose a NumPy view without holding a Python reference to the buffer.
// The `last_frames(count)` method returns a flat vector of length
// count*num_bins; the Py wrapper reshapes to 2D on the way out.
// ---------------------------------------------------------------------------

struct IWaterfallBufferImpl {
	virtual ~IWaterfallBufferImpl() = default;
	virtual void push_frame(const double* magnitude, std::size_t n) = 0;
	virtual mtl::vec::dense_vector<double>
	frame_at(std::size_t idx_from_oldest) const = 0;
	virtual mtl::vec::dense_vector<double>
	last_frames(std::size_t count) = 0;
	virtual void clear() = 0;
	virtual std::size_t num_bins()            const = 0;
	virtual std::size_t num_frames_capacity() const = 0;
	virtual std::size_t num_frames_filled()   const = 0;
};

template <typename T>
struct WaterfallBufferImpl : IWaterfallBufferImpl {
	sw::dsp::spectrum::WaterfallBuffer<T> inner;

	WaterfallBufferImpl(std::size_t num_bins, std::size_t num_frames)
	    : inner(num_bins, num_frames) {}

	void push_frame(const double* magnitude, std::size_t n) override {
		mtl::vec::dense_vector<T> typed(n);
		for (std::size_t i = 0; i < n; ++i) typed[i] = static_cast<T>(magnitude[i]);
		inner.push_frame(std::span<const T>(typed.data(), typed.size()));
	}

	mtl::vec::dense_vector<double>
	frame_at(std::size_t idx_from_oldest) const override {
		auto span = inner.frame_at(idx_from_oldest);
		mtl::vec::dense_vector<double> out(span.size());
		for (std::size_t i = 0; i < span.size(); ++i) {
			out[i] = static_cast<double>(span[i]);
		}
		return out;
	}

	mtl::vec::dense_vector<double>
	last_frames(std::size_t count) override {
		auto span = inner.last_frames(count);
		mtl::vec::dense_vector<double> out(span.size());
		for (std::size_t i = 0; i < span.size(); ++i) {
			out[i] = static_cast<double>(span[i]);
		}
		return out;
	}

	void clear() override { inner.clear(); }
	std::size_t num_bins()            const override { return inner.num_bins(); }
	std::size_t num_frames_capacity() const override { return inner.num_frames_capacity(); }
	std::size_t num_frames_filled()   const override { return inner.num_frames_filled(); }
};

} // namespace

// PyRealtimeSpectrum: streaming FFT engine.
class PyRealtimeSpectrum {
public:
	PyRealtimeSpectrum(std::size_t fft_size, std::size_t hop_size,
	                   mpdsp::bindings::np_f64_ro window,
	                   const std::string& dtype)
	    : dtype_(dtype) {
		if (window.shape(0) != fft_size) {
			throw std::invalid_argument(
				"RealtimeSpectrum: window length must equal fft_size");
		}
		impl_ = mpdsp::bindings::make_impl_for_dtype<
			RealtimeSpectrumImpl, IRealtimeSpectrumImpl>(
			mpdsp::parse_config(dtype), "RealtimeSpectrum",
			fft_size, hop_size, window.data(), window.shape(0));
	}

	std::size_t push(mpdsp::bindings::np_f64_ro signal) {
		return impl_->push(signal.data(), signal.shape(0));
	}

	nb::tuple latest_complex() const {
		auto [re, im] = impl_->latest_complex();
		return nb::make_tuple(mpdsp::bindings::vec_to_numpy(re),
		                      mpdsp::bindings::vec_to_numpy(im));
	}

	mpdsp::bindings::np_f64 latest_magnitude_db() const {
		return mpdsp::bindings::vec_to_numpy(impl_->latest_magnitude_db());
	}

	void reset()                        { impl_->reset(); }
	std::size_t fft_size()      const   { return impl_->fft_size(); }
	std::size_t hop_size()      const   { return impl_->hop_size(); }
	std::size_t total_ffts()    const   { return impl_->total_ffts(); }
	bool        first_fft_ready() const { return impl_->first_fft_ready(); }
	const std::string& dtype()  const   { return dtype_; }

private:
	std::unique_ptr<IRealtimeSpectrumImpl> impl_;
	std::string dtype_;
};

// PyRBWFilter: N-stage synchronously-tuned bandpass cascade around a
// tuned center frequency. Bumpless retune preserves biquad state.
class PyRBWFilter {
public:
	PyRBWFilter(double center_freq_hz, double bandwidth_hz,
	            double sample_rate_hz, std::size_t order,
	            const std::string& dtype)
	    : dtype_(dtype) {
		impl_ = mpdsp::bindings::make_impl_for_dtype<
			RBWFilterImpl, IRBWFilterImpl>(
			mpdsp::parse_config(dtype), "RBWFilter",
			center_freq_hz, bandwidth_hz, sample_rate_hz, order);
	}

	double process(double x) { return impl_->process(x); }

	mpdsp::bindings::np_f64 process_block(mpdsp::bindings::np_f64_ro signal) {
		return mpdsp::bindings::vec_to_numpy(
			impl_->process_block(signal.data(), signal.shape(0)));
	}

	void retune(double center_freq_hz, double bandwidth_hz) {
		impl_->retune(center_freq_hz, bandwidth_hz);
	}

	double      shape_factor()   const { return impl_->shape_factor(); }
	void        reset()                { impl_->reset(); }
	double      center_freq_hz() const { return impl_->center_freq_hz(); }
	double      bandwidth_hz()   const { return impl_->bandwidth_hz(); }
	double      sample_rate_hz() const { return impl_->sample_rate_hz(); }
	std::size_t order()          const { return impl_->order(); }
	const std::string& dtype()   const { return dtype_; }

private:
	std::unique_ptr<IRBWFilterImpl> impl_;
	std::string dtype_;
};

// PyVBWFilter: single-pole leaky-integrator LPF for post-detector smoothing.
// Bumpless set_cutoff() preserves y_prev.
class PyVBWFilter {
public:
	PyVBWFilter(double cutoff_hz, double sample_rate_hz,
	            const std::string& dtype)
	    : dtype_(dtype) {
		impl_ = mpdsp::bindings::make_impl_for_dtype<
			VBWFilterImpl, IVBWFilterImpl>(
			mpdsp::parse_config(dtype), "VBWFilter",
			cutoff_hz, sample_rate_hz);
	}

	double process(double x) { return impl_->process(x); }

	mpdsp::bindings::np_f64 process_block(mpdsp::bindings::np_f64_ro signal) {
		return mpdsp::bindings::vec_to_numpy(
			impl_->process_block(signal.data(), signal.shape(0)));
	}

	void   set_cutoff(double cutoff_hz) { impl_->set_cutoff(cutoff_hz); }
	void   reset()                      { impl_->reset(); }
	double cutoff_hz()      const       { return impl_->cutoff_hz(); }
	double sample_rate_hz() const       { return impl_->sample_rate_hz(); }
	const std::string& dtype() const    { return dtype_; }

private:
	std::unique_ptr<IVBWFilterImpl> impl_;
	std::string dtype_;
};

// PySweptLO: phase-coherent chirp generator.
class PySweptLO {
public:
	PySweptLO(double f_start_hz, double f_stop_hz,
	          double sweep_duration_s, double sample_rate_hz,
	          const std::string& mode, const std::string& dtype)
	    : dtype_(dtype) {
		bool is_log = parse_sweep_mode_is_log(mode);
		impl_ = mpdsp::bindings::make_impl_for_dtype<
			SweptLOImpl, ISweptLOImpl>(
			mpdsp::parse_config(dtype), "SweptLO",
			f_start_hz, f_stop_hz, sweep_duration_s, sample_rate_hz,
			is_log);
	}

	std::pair<double, double> process() { return impl_->process(); }

	nb::tuple generate_block(std::size_t n) {
		auto [c, s] = impl_->generate_block(n);
		return nb::make_tuple(mpdsp::bindings::vec_to_numpy(c),
		                      mpdsp::bindings::vec_to_numpy(s));
	}

	void reset()                                { impl_->reset(); }
	double      current_frequency_hz() const    { return impl_->current_frequency_hz(); }
	bool        sweep_complete()       const    { return impl_->sweep_complete(); }
	std::size_t total_sweeps()         const    { return impl_->total_sweeps(); }
	double      f_start_hz()           const    { return impl_->f_start_hz(); }
	double      f_stop_hz()            const    { return impl_->f_stop_hz(); }
	double      sweep_duration_s()     const    { return impl_->sweep_duration_s(); }
	double      sample_rate_hz()       const    { return impl_->sample_rate_hz(); }
	std::string mode()                 const    { return impl_->mode(); }
	std::size_t num_sweep_samples()    const    { return impl_->num_sweep_samples(); }
	const std::string& dtype()         const    { return dtype_; }

private:
	std::unique_ptr<ISweptLOImpl> impl_;
	std::string dtype_;
};

// PyFrontEndCorrector: FIR equalizer that inverts a CalibrationProfile.
class PyFrontEndCorrector {
public:
	PyFrontEndCorrector(const sw::dsp::instrument::CalibrationProfile& profile,
	                    std::size_t num_taps, double sample_rate_hz,
	                    double max_gain_dB, const std::string& dtype)
	    : dtype_(dtype) {
		impl_ = mpdsp::bindings::make_impl_for_dtype<
			FrontEndCorrectorImpl, IFrontEndCorrectorImpl>(
			mpdsp::parse_config(dtype), "FrontEndCorrector",
			profile, num_taps, sample_rate_hz, max_gain_dB);
	}

	double process(double x) { return impl_->process(x); }

	mpdsp::bindings::np_f64 process_block(mpdsp::bindings::np_f64_ro signal) {
		return mpdsp::bindings::vec_to_numpy(
			impl_->process_block(signal.data(), signal.shape(0)));
	}

	std::size_t num_taps() const           { return impl_->num_taps(); }
	const std::string& dtype() const       { return dtype_; }

private:
	std::unique_ptr<IFrontEndCorrectorImpl> impl_;
	std::string dtype_;
};

// PyTraceAverager: cross-sweep trace accumulator (5 modes).
class PyTraceAverager {
public:
	PyTraceAverager(std::size_t trace_length, const std::string& mode,
	                double config, const std::string& dtype)
	    : dtype_(dtype) {
		int code = parse_averager_mode_code(mode);
		impl_ = mpdsp::bindings::make_impl_for_dtype<
			TraceAveragerImpl, ITraceAveragerImpl>(
			mpdsp::parse_config(dtype), "TraceAverager",
			trace_length, code, config);
	}

	void accept_sweep(mpdsp::bindings::np_f64_ro trace) {
		impl_->accept_sweep(trace.data(), trace.shape(0));
	}

	mpdsp::bindings::np_f64 current_trace() const {
		return mpdsp::bindings::vec_to_numpy(impl_->current_trace());
	}

	void reset()                              { impl_->reset(); }
	std::size_t sweeps_accumulated() const    { return impl_->sweeps_accumulated(); }
	std::size_t trace_length()       const    { return impl_->trace_length(); }
	std::string mode()               const    { return averager_mode_to_string(impl_->mode_code()); }
	const std::string& dtype()       const    { return dtype_; }

private:
	std::unique_ptr<ITraceAveragerImpl> impl_;
	std::string dtype_;
};

// PyWaterfallBuffer: 2D ring of the last num_frames FFT magnitude frames.
class PyWaterfallBuffer {
public:
	PyWaterfallBuffer(std::size_t num_bins, std::size_t num_frames,
	                  const std::string& dtype)
	    : dtype_(dtype) {
		impl_ = mpdsp::bindings::make_impl_for_dtype<
			WaterfallBufferImpl, IWaterfallBufferImpl>(
			mpdsp::parse_config(dtype), "WaterfallBuffer",
			num_bins, num_frames);
	}

	void push_frame(mpdsp::bindings::np_f64_ro magnitude) {
		impl_->push_frame(magnitude.data(), magnitude.shape(0));
	}

	mpdsp::bindings::np_f64 frame_at(std::size_t idx_from_oldest) const {
		return mpdsp::bindings::vec_to_numpy(impl_->frame_at(idx_from_oldest));
	}

	// last_frames returns a 2D NumPy array shape (available, num_bins).
	// The Impl produces a flat vector; we reshape by building the ndarray
	// with two-dim shape directly.
	nb::ndarray<nb::numpy, double>
	last_frames(std::size_t count) {
		auto flat = impl_->last_frames(count);
		std::size_t nb = impl_->num_bins();
		std::size_t nf = nb == 0 ? 0 : flat.size() / nb;
		double* out_ptr = nullptr;
		double* raw = new double[nf * nb];
		out_ptr = raw;
		nb::capsule owner(raw, [](void* p) noexcept {
			delete[] static_cast<double*>(p);
		});
		for (std::size_t i = 0; i < flat.size(); ++i) out_ptr[i] = flat[i];
		std::size_t shape[2] = { nf, nb };
		return nb::ndarray<nb::numpy, double>(raw, 2, shape, owner);
	}

	void clear() { impl_->clear(); }
	std::size_t num_bins()            const { return impl_->num_bins(); }
	std::size_t num_frames_capacity() const { return impl_->num_frames_capacity(); }
	std::size_t num_frames_filled()   const { return impl_->num_frames_filled(); }
	const std::string& dtype()        const { return dtype_; }

private:
	std::unique_ptr<IWaterfallBufferImpl> impl_;
	std::string dtype_;
};

void bind_spectrum(nb::module_& m) {
	using mpdsp::bindings::dispatch_dtype_fn;
	using mpdsp::bindings::np_f64_ro;

	// -----------------------------------------------------------------------
	// Detector free functions. All take a NumPy float64 bin (a window of
	// post-RBW samples in a swept analyzer, or per-bin samples in an
	// FFT-based one) and return a scalar float.
	// -----------------------------------------------------------------------

	m.def("detect_peak",
		[](np_f64_ro bin, const std::string& dtype) {
			require_nonempty(bin, "detect_peak");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "detect_peak", [&]<typename T>() {
				auto v = cast_signal<T>(bin);
				return sw::dsp::spectrum::detect_peak<T>(
					std::span<const T>(v.data(), v.size()));
			});
		},
		nb::arg("bin"), nb::arg("dtype") = "reference",
		"Peak detector: max(bin). The standard scope/analyzer 'peak' mode.");

	m.def("detect_negative_peak",
		[](np_f64_ro bin, const std::string& dtype) {
			require_nonempty(bin, "detect_negative_peak");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "detect_negative_peak",
				[&]<typename T>() {
					auto v = cast_signal<T>(bin);
					return sw::dsp::spectrum::detect_negative_peak<T>(
						std::span<const T>(v.data(), v.size()));
				});
		},
		nb::arg("bin"), nb::arg("dtype") = "reference",
		"Negative-peak detector: min(bin). Finds the deepest notch or the "
		"noise floor.");

	m.def("detect_sample",
		[](np_f64_ro bin, const std::string& dtype) {
			require_nonempty(bin, "detect_sample");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "detect_sample", [&]<typename T>() {
				auto v = cast_signal<T>(bin);
				return sw::dsp::spectrum::detect_sample<T>(
					std::span<const T>(v.data(), v.size()));
			});
		},
		nb::arg("bin"), nb::arg("dtype") = "reference",
		"Sample detector: returns the FIRST sample in the bin. Conceptually "
		"a 'no-detector' mode — picks one representative time instant per "
		"bin, matching the CISPR/Keysight sample-detector convention.");

	m.def("detect_average",
		[](np_f64_ro bin, const std::string& dtype) {
			require_nonempty(bin, "detect_average");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "detect_average",
				[&]<typename T>() {
					auto v = cast_signal<T>(bin);
					return sw::dsp::spectrum::detect_average<T>(
						std::span<const T>(v.data(), v.size()));
				});
		},
		nb::arg("bin"), nb::arg("dtype") = "reference",
		"Average detector: arithmetic mean of the bin samples (linear). "
		"Sum accumulated in double regardless of dtype.");

	m.def("detect_rms",
		[](np_f64_ro bin, const std::string& dtype) {
			require_nonempty(bin, "detect_rms");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "detect_rms", [&]<typename T>() {
				auto v = cast_signal<T>(bin);
				return sw::dsp::spectrum::detect_rms<T>(
					std::span<const T>(v.data(), v.size()));
			});
		},
		nb::arg("bin"), nb::arg("dtype") = "reference",
		"RMS (energy) detector: sqrt(mean(bin**2)). For a unit-amplitude "
		"sine returns 1/sqrt(2).");

	m.def("detect",
		[](const std::string& mode, np_f64_ro bin, const std::string& dtype) {
			require_nonempty(bin, "detect");
			auto detector_mode = parse_detector_mode(mode);
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "detect", [&]<typename T>() {
				auto v = cast_signal<T>(bin);
				return sw::dsp::spectrum::detect<T>(
					std::span<const T>(v.data(), v.size()), detector_mode);
			});
		},
		nb::arg("mode"), nb::arg("bin"), nb::arg("dtype") = "reference",
		"Runtime-dispatch detector. mode is one of: 'peak', 'sample', "
		"'average', 'rms', 'negative_peak'. For a compile-time-known mode "
		"prefer the named detect_* functions (one less string parse and "
		"switch branch).");

	// -----------------------------------------------------------------------
	// RealtimeSpectrum — streaming FFT engine.
	// -----------------------------------------------------------------------
	nb::class_<PyRealtimeSpectrum>(m, "RealtimeSpectrum",
			"Streaming FFT engine that maintains a circular sample ring and "
			"produces an FFT every `hop_size` input samples once the initial "
			"`fft_size` samples have accumulated. Non-overlapping analysis "
			"uses hop_size == fft_size; the conventional 50%-overlap Hann "
			"analysis uses hop_size == fft_size // 2.\n\n"
			"Latest results are readable via `latest_complex()` (as a "
			"(real, imag) tuple of NumPy arrays) and `latest_magnitude_db()` "
			"(NumPy array with a -200 dB floor). Both return empty arrays "
			"until `first_fft_ready` becomes True.")
		.def(nb::init<std::size_t, std::size_t,
		              mpdsp::bindings::np_f64_ro, const std::string&>(),
		     nb::arg("fft_size"), nb::arg("hop_size"), nb::arg("window"),
		     nb::arg("dtype") = "reference",
		     "fft_size must be a power of 2. hop_size in [1, fft_size]. "
		     "window is a NumPy float64 array of length fft_size (build via "
		     "mpdsp.hanning(fft_size) or similar).")
		.def("push", &PyRealtimeSpectrum::push, nb::arg("signal"),
		     "Feed a block of samples. Returns the number of complete FFTs "
		     "produced by this call (0 while still accumulating the initial "
		     "fft_size samples, then one per hop_size samples).")
		.def("latest_complex", &PyRealtimeSpectrum::latest_complex,
		     "Return the most recent FFT as (real, imag) — two NumPy float64 "
		     "arrays of length fft_size. Both arrays are empty until "
		     "`first_fft_ready` is True.")
		.def("latest_magnitude_db", &PyRealtimeSpectrum::latest_magnitude_db,
		     "Return the most recent magnitude spectrum in dB with a -200 dB "
		     "floor. Empty array until `first_fft_ready` is True.")
		.def("reset", &PyRealtimeSpectrum::reset,
		     "Clear the sample ring and counters; configuration (fft_size, "
		     "hop_size, window) is preserved. Use between independent stream "
		     "segments.")
		.def_prop_ro("fft_size",   &PyRealtimeSpectrum::fft_size,
		     "Configured FFT length. Read-only.")
		.def_prop_ro("hop_size",   &PyRealtimeSpectrum::hop_size,
		     "Configured hop size. Read-only.")
		.def_prop_ro("total_ffts", &PyRealtimeSpectrum::total_ffts,
		     "Number of FFTs produced since construction or last reset(). "
		     "Read-only.")
		.def_prop_ro("first_fft_ready", &PyRealtimeSpectrum::first_fft_ready,
		     "True once at least one FFT has been produced (equivalent to "
		     "`total_ffts > 0`). Read-only.")
		.def_prop_ro("dtype",
		     [](const PyRealtimeSpectrum& self) { return self.dtype(); },
		     "Scalar dtype fixed at construction. Read-only.");

	// -----------------------------------------------------------------------
	// RBWFilter — resolution-bandwidth filter (pre-detection).
	// -----------------------------------------------------------------------
	nb::class_<PyRBWFilter>(m, "RBWFilter",
			"Resolution-bandwidth filter for a spectrum analyzer: an N-stage "
			"synchronously-tuned cascade of RBJ-style bandpass biquads. Sits "
			"between the mixer and the detector; selects a narrow window "
			"around a center frequency. Higher order tightens the shape "
			"factor (60 dB / 3 dB bandwidth ratio) — order=5 gives ~10x "
			"shape factor, comparable to a Gaussian for analyzer use.\n\n"
			"retune() is bumpless — biquad state is preserved across the "
			"coefficient redesign, so the displayed trace stays continuous "
			"when the user slides the RBW knob.")
		.def(nb::init<double, double, double, std::size_t, const std::string&>(),
		     nb::arg("center_freq_hz"), nb::arg("bandwidth_hz"),
		     nb::arg("sample_rate_hz"),
		     nb::arg("order") = static_cast<std::size_t>(5),
		     nb::arg("dtype") = "reference",
		     "Design an RBW filter. center_freq_hz in (0, sample_rate/2); "
		     "bandwidth_hz must keep both -3 dB shoulders inside the sampled "
		     "band; order in [1, 8].")
		.def("process", &PyRBWFilter::process, nb::arg("sample"),
		     "Filter one sample; returns the filtered scalar.")
		.def("process_block", &PyRBWFilter::process_block, nb::arg("signal"),
		     "Filter a block of samples; returns a new NumPy array of the "
		     "same length.")
		.def("retune", &PyRBWFilter::retune,
		     nb::arg("center_freq_hz"), nb::arg("bandwidth_hz"),
		     "Redesign coefficients around the new (center, bandwidth). "
		     "State is preserved (bumpless).")
		.def("reset", &PyRBWFilter::reset,
		     "Clear biquad delay-line state; coefficients and order retained.")
		.def_prop_ro("shape_factor", &PyRBWFilter::shape_factor,
		     "Closed-form analytical 60 dB / 3 dB bandwidth ratio for the "
		     "current order. Doesn't depend on tuning. Read-only.")
		.def_prop_ro("center_freq_hz", &PyRBWFilter::center_freq_hz,
		     "Currently tuned center frequency. Read-only.")
		.def_prop_ro("bandwidth_hz", &PyRBWFilter::bandwidth_hz,
		     "Currently configured -3 dB bandwidth. Read-only.")
		.def_prop_ro("sample_rate_hz", &PyRBWFilter::sample_rate_hz,
		     "Streaming sample rate. Read-only.")
		.def_prop_ro("order", &PyRBWFilter::order,
		     "Number of biquad stages (fixed at construction). Read-only.")
		.def_prop_ro("dtype",
		     [](const PyRBWFilter& self) { return self.dtype(); },
		     "Scalar dtype fixed at construction. Read-only.");

	// -----------------------------------------------------------------------
	// VBWFilter — video-bandwidth filter (post-detection).
	// -----------------------------------------------------------------------
	nb::class_<PyVBWFilter>(m, "VBWFilter",
			"Video-bandwidth filter: a single-pole leaky-integrator LPF that "
			"smooths detector output before the trace memory. Lower cutoff = "
			"more averaging = lower noise floor at the cost of slower "
			"response; higher cutoff = faster response but noisier trace. "
			"The standard analyzer noise-vs-speed knob.\n\n"
			"set_cutoff() is bumpless — the running y_prev is preserved, so "
			"the displayed trace stays continuous when the user slides the "
			"VBW knob.")
		.def(nb::init<double, double, const std::string&>(),
		     nb::arg("cutoff_hz"), nb::arg("sample_rate_hz"),
		     nb::arg("dtype") = "reference",
		     "Design a VBW filter. cutoff_hz in (0, sample_rate/2].")
		.def("process", &PyVBWFilter::process, nb::arg("sample"),
		     "Filter one sample; returns the filtered scalar.")
		.def("process_block", &PyVBWFilter::process_block, nb::arg("signal"),
		     "Filter a block of samples; returns a new NumPy array of the "
		     "same length.")
		.def("set_cutoff", &PyVBWFilter::set_cutoff, nb::arg("cutoff_hz"),
		     "Redesign alpha for the new cutoff. y_prev is preserved "
		     "(bumpless).")
		.def("reset", &PyVBWFilter::reset,
		     "Clear the running state y_prev to zero; cutoff and sample "
		     "rate are preserved.")
		.def_prop_ro("cutoff_hz", &PyVBWFilter::cutoff_hz,
		     "Currently configured -3 dB cutoff. Read-only.")
		.def_prop_ro("sample_rate_hz", &PyVBWFilter::sample_rate_hz,
		     "Streaming sample rate. Read-only.")
		.def_prop_ro("dtype",
		     [](const PyVBWFilter& self) { return self.dtype(); },
		     "Scalar dtype fixed at construction. Read-only.");

	// -----------------------------------------------------------------------
	// SweptLO — phase-coherent chirp generator (analyzer local oscillator).
	// -----------------------------------------------------------------------
	nb::class_<PySweptLO>(m, "SweptLO",
			"Phase-coherent chirp generator that walks a frequency schedule "
			"from f_start to f_stop over a configurable duration, then "
			"restarts. The phase accumulator is continuous across the sweep "
			"boundary — no glitch at restart. Linear and logarithmic "
			"schedules are supported.\n\n"
			"Produces (cos, sin) pairs per sample; use `generate_block(n)` "
			"for batch generation into two ndarrays.")
		.def(nb::init<double, double, double, double,
		              const std::string&, const std::string&>(),
		     nb::arg("f_start_hz"), nb::arg("f_stop_hz"),
		     nb::arg("sweep_duration_s"), nb::arg("sample_rate_hz"),
		     nb::arg("mode") = "linear",
		     nb::arg("dtype") = "reference",
		     "Design a swept LO. mode is 'linear' or 'logarithmic' "
		     "('log' also accepted). sweep_duration_s * sample_rate_hz "
		     "must yield at least 2 samples.")
		.def("process", &PySweptLO::process,
		     "Advance one sample; returns (cos, sin) as a tuple of floats.")
		.def("generate_block", &PySweptLO::generate_block, nb::arg("n"),
		     "Advance n samples; returns (cos_array, sin_array) as a tuple "
		     "of NumPy float64 arrays.")
		.def("reset", &PySweptLO::reset,
		     "Restart the sweep at f_start with phase = 0. Coefficients "
		     "(delta_inc / ratio_inc) are preserved.")
		.def_prop_ro("current_frequency_hz",
		     &PySweptLO::current_frequency_hz,
		     "Instantaneous frequency in Hz, derived from the current "
		     "phase increment. Read-only.")
		.def_prop_ro("sweep_complete", &PySweptLO::sweep_complete,
		     "True iff the MOST RECENT process() call wrapped a sweep "
		     "boundary. One-shot per sweep — the next process() clears it. "
		     "Read-only.")
		.def_prop_ro("total_sweeps", &PySweptLO::total_sweeps,
		     "Monotone count of sweep boundaries crossed since construction "
		     "or the last reset(). Read-only.")
		.def_prop_ro("f_start_hz",       &PySweptLO::f_start_hz)
		.def_prop_ro("f_stop_hz",        &PySweptLO::f_stop_hz)
		.def_prop_ro("sweep_duration_s", &PySweptLO::sweep_duration_s)
		.def_prop_ro("sample_rate_hz",   &PySweptLO::sample_rate_hz)
		.def_prop_ro("mode",             &PySweptLO::mode,
		     "'linear' or 'logarithmic'. Fixed at construction. Read-only.")
		.def_prop_ro("num_sweep_samples", &PySweptLO::num_sweep_samples,
		     "Samples per sweep = floor(sweep_duration_s * sample_rate_hz). "
		     "Read-only.")
		.def_prop_ro("dtype",
		     [](const PySweptLO& self) { return self.dtype(); },
		     "Scalar dtype fixed at construction. Read-only.");

	// -----------------------------------------------------------------------
	// CalibrationProfile — non-templated value type that FrontEndCorrector
	// consumes. Bound at the module level (not per-dtype) since the
	// calibration measurement itself is double-precision-native.
	// -----------------------------------------------------------------------
	nb::class_<sw::dsp::instrument::CalibrationProfile>(m, "CalibrationProfile",
			"Tabulated frequency-response correction for a spectrum-analyzer "
			"or scope front end. Stores (frequency_hz, gain_dB, phase_rad) "
			"triples; the interpolants linearly interpolate between tabulated "
			"points and clamp outside the calibrated band. Fed to "
			"FrontEndCorrector to design an inverse-response equalizer.")
		.def(nb::init<std::vector<double>, std::vector<double>,
		              std::vector<double>>(),
		     nb::arg("frequencies"), nb::arg("gain_dB"), nb::arg("phase_rad"),
		     "All three inputs must have the same length (>= 2). "
		     "`frequencies` must be strictly monotonically increasing.")
		.def_static("from_csv",
		     &sw::dsp::instrument::CalibrationProfile::from_csv,
		     nb::arg("path"),
		     "Load a profile from CSV. Format: one row per frequency, "
		     "columns freq_hz, gain_dB, phase_rad. Header row is optional; "
		     "lines starting with '#' are treated as comments.")
		.def("gain_dB",
		     &sw::dsp::instrument::CalibrationProfile::gain_dB,
		     nb::arg("freq_hz"),
		     "Interpolated gain (dB) at the query frequency. Clamps to the "
		     "endpoint values below freq_min / above freq_max.")
		.def("phase_rad",
		     &sw::dsp::instrument::CalibrationProfile::phase_rad,
		     nb::arg("freq_hz"),
		     "Interpolated phase (radians) at the query frequency.")
		.def_prop_ro("size",     &sw::dsp::instrument::CalibrationProfile::size)
		.def_prop_ro("freq_min", &sw::dsp::instrument::CalibrationProfile::freq_min)
		.def_prop_ro("freq_max", &sw::dsp::instrument::CalibrationProfile::freq_max);

	// -----------------------------------------------------------------------
	// FrontEndCorrector — FIR equalizer inverting a CalibrationProfile.
	// -----------------------------------------------------------------------
	nb::class_<PyFrontEndCorrector>(m, "FrontEndCorrector",
			"Front-end equalizer for the analyzer input path: an FIR filter "
			"whose magnitude/phase response cancels a CalibrationProfile. "
			"Design uses frequency-sampling with a Hamming window; the "
			"inverse magnitude is clamped to `max_gain_dB` to avoid "
			"amplifying noise where the profile has deep nulls.\n\n"
			"Alias for sw::dsp::instrument::EqualizerFilter — same math, "
			"exposed in the spectrum module under an analyzer-facing name.")
		.def(nb::init<const sw::dsp::instrument::CalibrationProfile&,
		              std::size_t, double, double, const std::string&>(),
		     nb::arg("profile"), nb::arg("num_taps"),
		     nb::arg("sample_rate_hz"),
		     nb::arg("max_gain_dB") = 60.0,
		     nb::arg("dtype") = "reference",
		     "Design an equalizer from the given profile. num_taps >= 3; "
		     "max_gain_dB caps the inverse magnitude to prevent noise "
		     "amplification at profile nulls.")
		.def("process", &PyFrontEndCorrector::process, nb::arg("sample"),
		     "Filter one sample; returns the equalized scalar.")
		.def("process_block", &PyFrontEndCorrector::process_block,
		     nb::arg("signal"),
		     "Filter a block of samples; returns a new NumPy array of the "
		     "same length.")
		.def_prop_ro("num_taps", &PyFrontEndCorrector::num_taps,
		     "Length of the designed FIR (fixed at construction). Read-only.")
		.def_prop_ro("dtype",
		     [](const PyFrontEndCorrector& self) { return self.dtype(); },
		     "Scalar dtype fixed at construction. Read-only.");

	// -----------------------------------------------------------------------
	// TraceAverager — cross-sweep trace accumulation (5 modes).
	// -----------------------------------------------------------------------
	nb::class_<PyTraceAverager>(m, "TraceAverager",
			"Cross-sweep trace averaging with five commercial-analyzer modes:\n"
			"  linear      — cumulative unweighted mean of all sweeps.\n"
			"  exponential — single-pole IIR y = alpha*x + (1-alpha)*y_prev.\n"
			"                config is alpha in (0, 1].\n"
			"  max_hold    — element-wise max across all sweeps.\n"
			"  min_hold    — element-wise min across all sweeps.\n"
			"  max_hold_n  — element-wise max over the last N sweeps.\n"
			"                config is the window N >= 1 (integer-valued).\n\n"
			"Distinct from within-bin detector reduction (see mpdsp.detect_*): "
			"trace averaging reduces ACROSS sweeps, detectors reduce WITHIN a "
			"bin.")
		.def(nb::init<std::size_t, const std::string&, double, const std::string&>(),
		     nb::arg("trace_length"), nb::arg("mode"),
		     nb::arg("config") = 0.0,
		     nb::arg("dtype") = "reference",
		     "Construct with a fixed trace_length (>= 1). mode selects the "
		     "reduction; config is mode-specific (alpha for exponential, N "
		     "for max_hold_n, ignored otherwise).")
		.def("accept_sweep", &PyTraceAverager::accept_sweep, nb::arg("trace"),
		     "Push a new sweep. Length must equal trace_length.")
		.def("current_trace", &PyTraceAverager::current_trace,
		     "Return the current accumulated trace as a NumPy array. Value "
		     "is meaningful only after at least one accept_sweep().")
		.def("reset", &PyTraceAverager::reset,
		     "Discard accumulated state; mode and config are preserved.")
		.def_prop_ro("sweeps_accumulated",
		     &PyTraceAverager::sweeps_accumulated,
		     "Number of sweeps accepted since construction or last reset(). "
		     "Read-only.")
		.def_prop_ro("trace_length", &PyTraceAverager::trace_length,
		     "Fixed bin count per sweep. Read-only.")
		.def_prop_ro("mode", &PyTraceAverager::mode,
		     "'linear' / 'exponential' / 'max_hold' / 'min_hold' / "
		     "'max_hold_n'. Read-only.")
		.def_prop_ro("dtype",
		     [](const PyTraceAverager& self) { return self.dtype(); },
		     "Scalar dtype fixed at construction. Read-only.");

	// -----------------------------------------------------------------------
	// WaterfallBuffer — 2D ring for the most recent N FFT magnitude frames.
	// -----------------------------------------------------------------------
	nb::class_<PyWaterfallBuffer>(m, "WaterfallBuffer",
			"Circular buffer storing the last num_frames FFT magnitude "
			"frames from a streaming spectrum processor. Each frame has "
			"num_bins samples. When the ring is full, push_frame overwrites "
			"the oldest frame.\n\n"
			"Distinct from spectral.Spectrogram (batch STFT) — this class is "
			"the analyzer-side display memory that fills incrementally as "
			"FFTs land.")
		.def(nb::init<std::size_t, std::size_t, const std::string&>(),
		     nb::arg("num_bins"), nb::arg("num_frames"),
		     nb::arg("dtype") = "reference",
		     "Construct with num_bins per frame and num_frames capacity. "
		     "Both must be > 0; num_bins * num_frames must fit in size_t.")
		.def("push_frame", &PyWaterfallBuffer::push_frame,
		     nb::arg("magnitude"),
		     "Append one frame. Length must equal num_bins.")
		.def("frame_at", &PyWaterfallBuffer::frame_at,
		     nb::arg("idx_from_oldest"),
		     "Return the chronologically-indexed frame (0 = oldest, "
		     "num_frames_filled - 1 = newest) as a NumPy 1D array. Fresh "
		     "copy — safe to hold across further push_frame calls.")
		.def("last_frames", &PyWaterfallBuffer::last_frames, nb::arg("count"),
		     "Return the most recent `count` frames as a 2D NumPy array "
		     "shape (available, num_bins), oldest first. count is clamped "
		     "to num_frames_filled — fewer-than-requested frames are "
		     "returned when the buffer hasn't filled yet.")
		.def("clear", &PyWaterfallBuffer::clear,
		     "Discard all stored frames; capacity preserved.")
		.def_prop_ro("num_bins", &PyWaterfallBuffer::num_bins)
		.def_prop_ro("num_frames_capacity",
		     &PyWaterfallBuffer::num_frames_capacity)
		.def_prop_ro("num_frames_filled",
		     &PyWaterfallBuffer::num_frames_filled)
		.def_prop_ro("dtype",
		     [](const PyWaterfallBuffer& self) { return self.dtype(); });

	// -----------------------------------------------------------------------
	// Marker + DeltaMarker structs. Simple data types — expose fields as
	// read-only properties so Python users can inspect but not mutate the
	// results of find_peaks / harmonic_markers.
	// -----------------------------------------------------------------------
	nb::class_<sw::dsp::spectrum::Marker>(m, "Marker",
			"A single marker on a spectrum trace: bin index, sub-bin-"
			"interpolated frequency (Hz), and amplitude. Returned by "
			"find_peaks() and harmonic_markers(); consumed by "
			"make_delta_marker().")
		.def(nb::init<>())
		.def_rw("bin_index",    &sw::dsp::spectrum::Marker::bin_index)
		.def_rw("frequency_hz", &sw::dsp::spectrum::Marker::frequency_hz)
		.def_rw("amplitude",    &sw::dsp::spectrum::Marker::amplitude)
		.def("__repr__", [](const sw::dsp::spectrum::Marker& m) {
			return "Marker(bin_index=" + std::to_string(m.bin_index)
			     + ", frequency_hz=" + std::to_string(m.frequency_hz)
			     + ", amplitude="    + std::to_string(m.amplitude) + ")";
		});

	nb::class_<sw::dsp::spectrum::DeltaMarker>(m, "DeltaMarker",
			"Two-marker delta measurement. delta_freq_hz and "
			"delta_amplitude are `b` minus `a`, matching the convention "
			"of every commercial analyzer's delta-marker mode.")
		.def(nb::init<>())
		.def_rw("a",               &sw::dsp::spectrum::DeltaMarker::a)
		.def_rw("b",               &sw::dsp::spectrum::DeltaMarker::b)
		.def_rw("delta_freq_hz",   &sw::dsp::spectrum::DeltaMarker::delta_freq_hz)
		.def_rw("delta_amplitude", &sw::dsp::spectrum::DeltaMarker::delta_amplitude);

	// -----------------------------------------------------------------------
	// Marker free functions. find_peaks / harmonic_markers dispatch on the
	// dtype used to compare amplitudes; make_delta_marker is pure double
	// arithmetic on two Markers (no dispatch needed).
	// -----------------------------------------------------------------------
	m.def("find_peaks",
		[](np_f64_ro trace, double bin_freq_step_hz,
		   std::size_t top_n, std::size_t min_separation_bins,
		   const std::string& dtype) {
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "find_peaks", [&]<typename T>() {
				auto v = cast_signal<T>(trace);
				return sw::dsp::spectrum::find_peaks<T>(
					std::span<const T>(v.data(), v.size()),
					bin_freq_step_hz, top_n, min_separation_bins);
			});
		},
		nb::arg("trace"), nb::arg("bin_freq_step_hz"), nb::arg("top_n"),
		nb::arg("min_separation_bins") = static_cast<std::size_t>(3),
		nb::arg("dtype") = "reference",
		"Find the top-N strongest peaks in a trace with a minimum-"
		"separation greedy selection. Returns a list of Marker objects in "
		"descending amplitude order. Sub-bin frequency position is "
		"recovered via parabolic interpolation across the three bins around "
		"each peak; edge bins skip interpolation.");

	m.def("harmonic_markers",
		[](np_f64_ro trace, double bin_freq_step_hz,
		   double fundamental_hz, std::size_t harmonics,
		   const std::string& dtype) {
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "harmonic_markers",
				[&]<typename T>() {
					auto v = cast_signal<T>(trace);
					return sw::dsp::spectrum::harmonic_markers<T>(
						std::span<const T>(v.data(), v.size()),
						bin_freq_step_hz, fundamental_hz, harmonics);
				});
		},
		nb::arg("trace"), nb::arg("bin_freq_step_hz"),
		nb::arg("fundamental_hz"), nb::arg("harmonics"),
		nb::arg("dtype") = "reference",
		"Markers at bins nearest k * fundamental_hz for k = 2..harmonics+1. "
		"Returns a list of Marker objects; harmonics past the trace's "
		"frequency range are silently omitted. Combine with find_peaks() "
		"and a small neighborhood search to peak-snap each harmonic.");

	m.def("make_delta_marker",
		&sw::dsp::spectrum::make_delta_marker,
		nb::arg("a"), nb::arg("b"),
		"Compute a DeltaMarker from two Markers: delta_freq_hz and "
		"delta_amplitude are b - a.");
}
