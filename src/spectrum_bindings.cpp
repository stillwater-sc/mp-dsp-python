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
#include <sw/dsp/spectrum/rbw_filter.hpp>
#include <sw/dsp/spectrum/realtime_spectrum.hpp>
#include <sw/dsp/spectrum/swept_lo.hpp>
#include <sw/dsp/spectrum/vbw_filter.hpp>

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
}
