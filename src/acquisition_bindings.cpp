// acquisition_bindings.cpp: bindings for the high-rate data-acquisition
// pipeline primitives (Phase 3 of the v0.6 acquisition epic, Issue #86).
//
// Surface bound here:
//   - NCO                         (nco.hpp)
//   - CICDecimator, CICInterpolator   (cic.hpp)
//   - HalfBandFilter              (halfband.hpp)
//   - PolyphaseDecimator, PolyphaseInterpolator (filter/fir/polyphase.hpp,
//                                  re-exported by acquisition/polyphase_decimator.hpp)
//   - design_halfband             (halfband.hpp free function)
//   - polyphase_decompose         (filter/fir/polyphase.hpp free function)
//
// Binding pattern follows conditioning_bindings.cpp: type-erased virtual
// IImpl interface + concrete templated Impl<T> + Py wrapper holding a
// unique_ptr<IImpl>. NumPy I/O is always float64; the chosen dtype only
// controls the precision of the internal arithmetic.
//
// All primitives use single-T dispatch — T fills CoeffScalar, StateScalar,
// and SampleScalar simultaneously. Per-scalar dispatch is a separate
// follow-up if precision-research workflows need it.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sw/dsp/acquisition/nco.hpp>
#include <sw/dsp/acquisition/cic.hpp>
#include <sw/dsp/acquisition/halfband.hpp>
#include <sw/dsp/acquisition/polyphase_decimator.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

using mpdsp::bindings::dispatch_dtype_fn;
using mpdsp::bindings::make_impl_for_dtype;
using mpdsp::bindings::np_f64;
using mpdsp::bindings::np_f64_ro;
using mpdsp::bindings::numpy_to_vec_fresh;
using mpdsp::bindings::vec_to_numpy;

namespace {

// ---------------------------------------------------------------------------
// Helpers for complex_T -> (real_ndarray, imag_ndarray) tuple. Templated on
// the complex type itself rather than the underlying scalar, so the same
// helpers work for both std::complex<T> (when T is float/double) and
// sw::universal::complex<T> (when T is posit/cfloat/etc.) — the upstream
// `complex_for_t<T>` metafunction picks between those at instantiation.
// ---------------------------------------------------------------------------

template <typename CT>
np_f64 complex_real_to_numpy(const mtl::vec::dense_vector<CT>& v) {
	mtl::vec::dense_vector<double> out(v.size());
	for (std::size_t i = 0; i < v.size(); ++i)
		out[i] = static_cast<double>(v[i].real());
	return vec_to_numpy(out);
}

template <typename CT>
np_f64 complex_imag_to_numpy(const mtl::vec::dense_vector<CT>& v) {
	mtl::vec::dense_vector<double> out(v.size());
	for (std::size_t i = 0; i < v.size(); ++i)
		out[i] = static_cast<double>(v[i].imag());
	return vec_to_numpy(out);
}

// ===========================================================================
// NCO
// ===========================================================================

struct INCOImpl {
	virtual ~INCOImpl() = default;
	virtual void set_frequency(double frequency, double sample_rate) = 0;
	virtual void set_phase_offset(double offset) = 0;
	virtual double phase() const = 0;
	virtual double phase_increment() const = 0;
	virtual std::pair<double, double> generate_sample() = 0;
	virtual double generate_real() = 0;
	virtual nb::tuple generate_block(std::size_t length) = 0;
	virtual np_f64 generate_block_real(std::size_t length) = 0;
	virtual nb::tuple mix_down(np_f64_ro input) = 0;
	virtual void reset() = 0;
};

template <typename T>
class NCOImpl : public INCOImpl {
public:
	NCOImpl(double frequency, double sample_rate)
		: nco_(static_cast<T>(frequency), static_cast<T>(sample_rate)) {}

	void set_frequency(double f, double sr) override {
		if (!(sr > 0.0))
			throw std::invalid_argument("NCO: sample_rate must be positive");
		nco_.set_frequency(static_cast<T>(f), static_cast<T>(sr));
	}
	void set_phase_offset(double offset) override {
		nco_.set_phase_offset(static_cast<T>(offset));
	}
	double phase() const override {
		return static_cast<double>(nco_.phase());
	}
	double phase_increment() const override {
		return static_cast<double>(nco_.phase_increment());
	}
	std::pair<double, double> generate_sample() override {
		auto z = nco_.generate_sample();
		return {static_cast<double>(z.real()), static_cast<double>(z.imag())};
	}
	double generate_real() override {
		return static_cast<double>(nco_.generate_real());
	}
	nb::tuple generate_block(std::size_t length) override {
		auto block = nco_.generate_block(length);
		return nb::make_tuple(complex_real_to_numpy(block),
		                      complex_imag_to_numpy(block));
	}
	np_f64 generate_block_real(std::size_t length) override {
		auto block = nco_.generate_block_real(length);
		return vec_to_numpy(block);
	}
	nb::tuple mix_down(np_f64_ro input) override {
		auto in = numpy_to_vec_fresh<typename decltype(nco_)::sample_scalar>(input);
		auto out = nco_.mix_down(in);
		return nb::make_tuple(complex_real_to_numpy(out),
		                      complex_imag_to_numpy(out));
	}
	void reset() override { nco_.reset(); }

private:
	sw::dsp::NCO<T> nco_;
};

// ===========================================================================
// CIC Decimator
// ===========================================================================

struct ICICDecimatorImpl {
	virtual ~ICICDecimatorImpl() = default;
	virtual std::pair<bool, double> push(double in) = 0;
	virtual double output() const = 0;
	virtual np_f64 process_block(np_f64_ro input) = 0;
	virtual int decimation_ratio() const = 0;
	virtual int num_stages() const = 0;
	virtual int differential_delay() const = 0;
	virtual void reset() = 0;
};

template <typename T>
class CICDecimatorImpl : public ICICDecimatorImpl {
public:
	CICDecimatorImpl(int ratio, int stages, int delay)
		: cic_(ratio, stages, delay) {}

	std::pair<bool, double> push(double in) override {
		bool emit = cic_.push(static_cast<T>(in));
		double out = emit ? static_cast<double>(cic_.output()) : 0.0;
		return {emit, out};
	}
	double output() const override {
		return static_cast<double>(cic_.output());
	}
	np_f64 process_block(np_f64_ro input) override {
		auto in = numpy_to_vec_fresh<T>(input);
		auto out = cic_.process_block(in);
		return vec_to_numpy(out);
	}
	int decimation_ratio() const override { return cic_.decimation_ratio(); }
	int num_stages() const override { return cic_.num_stages(); }
	int differential_delay() const override { return cic_.differential_delay(); }
	void reset() override { cic_.reset(); }

private:
	sw::dsp::CICDecimator<T> cic_;
};

// ===========================================================================
// CIC Interpolator
// ===========================================================================

struct ICICInterpolatorImpl {
	virtual ~ICICInterpolatorImpl() = default;
	virtual void push(double in) = 0;
	virtual double output() = 0;
	virtual np_f64 process_block(np_f64_ro input) = 0;
	virtual int interpolation_ratio() const = 0;
	virtual int num_stages() const = 0;
	virtual int differential_delay() const = 0;
	virtual void reset() = 0;
};

template <typename T>
class CICInterpolatorImpl : public ICICInterpolatorImpl {
public:
	CICInterpolatorImpl(int ratio, int stages, int delay)
		: cic_(ratio, stages, delay) {}

	void push(double in) override { cic_.push(static_cast<T>(in)); }
	double output() override {
		return static_cast<double>(cic_.output());
	}
	np_f64 process_block(np_f64_ro input) override {
		auto in_v = numpy_to_vec_fresh<T>(input);
		std::vector<T> in_buf(in_v.size());
		for (std::size_t i = 0; i < in_v.size(); ++i) in_buf[i] = in_v[i];
		std::vector<T> out_buf;
		out_buf.reserve(in_buf.size() * static_cast<std::size_t>(cic_.interpolation_ratio()));
		cic_.process_block(std::span<const T>(in_buf), out_buf);
		mtl::vec::dense_vector<T> out_v(out_buf.size());
		for (std::size_t i = 0; i < out_buf.size(); ++i) out_v[i] = out_buf[i];
		return vec_to_numpy(out_v);
	}
	int interpolation_ratio() const override { return cic_.interpolation_ratio(); }
	int num_stages() const override { return cic_.num_stages(); }
	int differential_delay() const override { return cic_.differential_delay(); }
	void reset() override { cic_.reset(); }

private:
	sw::dsp::CICInterpolator<T> cic_;
};

// ===========================================================================
// HalfBandFilter
// ===========================================================================

struct IHalfBandImpl {
	virtual ~IHalfBandImpl() = default;
	virtual double process(double in) = 0;
	virtual np_f64 process_block(np_f64_ro input) = 0;
	virtual std::pair<bool, double> process_decimate(double in) = 0;
	virtual np_f64 process_block_decimate(np_f64_ro input) = 0;
	virtual std::size_t num_taps() const = 0;
	virtual std::size_t num_nonzero_taps() const = 0;
	virtual void reset() = 0;
};

template <typename T>
class HalfBandImpl : public IHalfBandImpl {
public:
	HalfBandImpl(const mtl::vec::dense_vector<T>& taps) : hb_(taps) {}

	double process(double in) override {
		return static_cast<double>(hb_.process(static_cast<T>(in)));
	}
	np_f64 process_block(np_f64_ro input) override {
		auto in = numpy_to_vec_fresh<T>(input);
		auto out = hb_.process_block(in);
		return vec_to_numpy(out);
	}
	std::pair<bool, double> process_decimate(double in) override {
		auto p = hb_.process_decimate(static_cast<T>(in));
		return {p.first, static_cast<double>(p.second)};
	}
	np_f64 process_block_decimate(np_f64_ro input) override {
		auto in = numpy_to_vec_fresh<T>(input);
		std::vector<T> in_buf(in.size());
		for (std::size_t i = 0; i < in.size(); ++i) in_buf[i] = in[i];
		auto out = hb_.process_block_decimate(std::span<const T>(in_buf));
		return vec_to_numpy(out);
	}
	std::size_t num_taps() const override { return hb_.num_taps(); }
	std::size_t num_nonzero_taps() const override { return hb_.num_nonzero_taps(); }
	void reset() override { hb_.reset(); }

private:
	sw::dsp::HalfBandFilter<T> hb_;
};

// ===========================================================================
// PolyphaseDecimator
// ===========================================================================

struct IPolyphaseDecimatorImpl {
	virtual ~IPolyphaseDecimatorImpl() = default;
	virtual std::pair<bool, double> process(double in) = 0;
	virtual np_f64 process_block(np_f64_ro input) = 0;
	virtual std::size_t factor() const = 0;
	virtual void reset() = 0;
};

template <typename T>
class PolyphaseDecimatorImpl : public IPolyphaseDecimatorImpl {
public:
	PolyphaseDecimatorImpl(const mtl::vec::dense_vector<T>& taps,
	                        std::size_t factor)
		: pd_(taps, factor) {}

	std::pair<bool, double> process(double in) override {
		auto p = pd_.process(static_cast<T>(in));
		return {p.first, static_cast<double>(p.second)};
	}
	np_f64 process_block(np_f64_ro input) override {
		auto in_v = numpy_to_vec_fresh<T>(input);
		std::vector<T> in_buf(in_v.size());
		for (std::size_t i = 0; i < in_v.size(); ++i) in_buf[i] = in_v[i];
		auto out = pd_.process_block(std::span<const T>(in_buf));
		return vec_to_numpy(out);
	}
	std::size_t factor() const override { return pd_.factor(); }
	void reset() override { pd_.reset(); }

private:
	sw::dsp::PolyphaseDecimator<T> pd_;
};

// ===========================================================================
// PolyphaseInterpolator
// ===========================================================================

struct IPolyphaseInterpolatorImpl {
	virtual ~IPolyphaseInterpolatorImpl() = default;
	virtual np_f64 process(double in) = 0;
	virtual np_f64 process_block(np_f64_ro input) = 0;
	virtual std::size_t factor() const = 0;
	virtual void reset() = 0;
};

template <typename T>
class PolyphaseInterpolatorImpl : public IPolyphaseInterpolatorImpl {
public:
	PolyphaseInterpolatorImpl(const mtl::vec::dense_vector<T>& taps,
	                           std::size_t factor)
		: pi_(taps, factor) {}

	np_f64 process(double in) override {
		auto out = pi_.process(static_cast<T>(in));
		return vec_to_numpy(out);
	}
	np_f64 process_block(np_f64_ro input) override {
		auto in_v = numpy_to_vec_fresh<T>(input);
		std::vector<T> in_buf(in_v.size());
		for (std::size_t i = 0; i < in_v.size(); ++i) in_buf[i] = in_v[i];
		auto out = pi_.process_block(std::span<const T>(in_buf));
		return vec_to_numpy(out);
	}
	std::size_t factor() const override { return pi_.factor(); }
	void reset() override { pi_.reset(); }

private:
	sw::dsp::PolyphaseInterpolator<T> pi_;
};

} // anonymous namespace

// ===========================================================================
// Py-wrappers (visible names — held by std::unique_ptr<I*Impl> + dispatch)
// ===========================================================================

class PyNCO {
public:
	PyNCO(double frequency, double sample_rate, const std::string& dtype) {
		if (!(sample_rate > 0.0))
			throw std::invalid_argument("NCO: sample_rate must be positive");
		auto config = mpdsp::parse_config(dtype);
		impl_ = make_impl_for_dtype<NCOImpl, INCOImpl>(
			config, "NCO", frequency, sample_rate);
	}

	void set_frequency(double f, double sr) { impl_->set_frequency(f, sr); }
	void set_phase_offset(double off)        { impl_->set_phase_offset(off); }
	double phase() const                     { return impl_->phase(); }
	double phase_increment() const           { return impl_->phase_increment(); }
	std::pair<double, double> generate_sample() { return impl_->generate_sample(); }
	double generate_real()                   { return impl_->generate_real(); }
	nb::tuple generate_block(std::size_t n)  { return impl_->generate_block(n); }
	np_f64 generate_block_real(std::size_t n){ return impl_->generate_block_real(n); }
	nb::tuple mix_down(np_f64_ro input)      { return impl_->mix_down(input); }
	void reset()                             { impl_->reset(); }

private:
	std::unique_ptr<INCOImpl> impl_;
};

class PyCICDecimator {
public:
	PyCICDecimator(int decimation_ratio, int num_stages, int differential_delay,
	               const std::string& dtype) {
		auto config = mpdsp::parse_config(dtype);
		impl_ = make_impl_for_dtype<CICDecimatorImpl, ICICDecimatorImpl>(
			config, "CICDecimator",
			decimation_ratio, num_stages, differential_delay);
	}

	std::pair<bool, double> push(double in) { return impl_->push(in); }
	double output() const                   { return impl_->output(); }
	np_f64 process_block(np_f64_ro input)   { return impl_->process_block(input); }
	int decimation_ratio() const            { return impl_->decimation_ratio(); }
	int num_stages() const                  { return impl_->num_stages(); }
	int differential_delay() const          { return impl_->differential_delay(); }
	void reset()                            { impl_->reset(); }

private:
	std::unique_ptr<ICICDecimatorImpl> impl_;
};

class PyCICInterpolator {
public:
	PyCICInterpolator(int interpolation_ratio, int num_stages, int differential_delay,
	                   const std::string& dtype) {
		auto config = mpdsp::parse_config(dtype);
		impl_ = make_impl_for_dtype<CICInterpolatorImpl, ICICInterpolatorImpl>(
			config, "CICInterpolator",
			interpolation_ratio, num_stages, differential_delay);
	}

	void push(double in)                    { impl_->push(in); }
	double output()                         { return impl_->output(); }
	np_f64 process_block(np_f64_ro input)   { return impl_->process_block(input); }
	int interpolation_ratio() const         { return impl_->interpolation_ratio(); }
	int num_stages() const                  { return impl_->num_stages(); }
	int differential_delay() const          { return impl_->differential_delay(); }
	void reset()                            { impl_->reset(); }

private:
	std::unique_ptr<ICICInterpolatorImpl> impl_;
};

class PyHalfBandFilter {
public:
	PyHalfBandFilter(np_f64_ro taps, const std::string& dtype) {
		auto config = mpdsp::parse_config(dtype);
		// We have to materialize the taps in the chosen T before constructing
		// the impl; dispatch on config to pick the right <T>.
		impl_ = dispatch_dtype_fn(config, "HalfBandFilter",
			[&]<typename T>() -> std::unique_ptr<IHalfBandImpl> {
				auto t = numpy_to_vec_fresh<T>(taps);
				return std::make_unique<HalfBandImpl<T>>(t);
			});
	}

	double process(double in)               { return impl_->process(in); }
	np_f64 process_block(np_f64_ro input)   { return impl_->process_block(input); }
	std::pair<bool, double> process_decimate(double in) {
		return impl_->process_decimate(in);
	}
	np_f64 process_block_decimate(np_f64_ro input) {
		return impl_->process_block_decimate(input);
	}
	std::size_t num_taps() const            { return impl_->num_taps(); }
	std::size_t num_nonzero_taps() const    { return impl_->num_nonzero_taps(); }
	void reset()                            { impl_->reset(); }

private:
	std::unique_ptr<IHalfBandImpl> impl_;
};

class PyPolyphaseDecimator {
public:
	PyPolyphaseDecimator(np_f64_ro taps, std::size_t factor,
	                     const std::string& dtype) {
		if (factor == 0)
			throw std::invalid_argument(
				"PolyphaseDecimator: factor must be > 0");
		auto config = mpdsp::parse_config(dtype);
		impl_ = dispatch_dtype_fn(config, "PolyphaseDecimator",
			[&]<typename T>() -> std::unique_ptr<IPolyphaseDecimatorImpl> {
				auto t = numpy_to_vec_fresh<T>(taps);
				return std::make_unique<PolyphaseDecimatorImpl<T>>(t, factor);
			});
	}

	std::pair<bool, double> process(double in) { return impl_->process(in); }
	np_f64 process_block(np_f64_ro input)      { return impl_->process_block(input); }
	std::size_t factor() const                 { return impl_->factor(); }
	void reset()                               { impl_->reset(); }

private:
	std::unique_ptr<IPolyphaseDecimatorImpl> impl_;
};

class PyPolyphaseInterpolator {
public:
	PyPolyphaseInterpolator(np_f64_ro taps, std::size_t factor,
	                         const std::string& dtype) {
		if (factor == 0)
			throw std::invalid_argument(
				"PolyphaseInterpolator: factor must be > 0");
		auto config = mpdsp::parse_config(dtype);
		impl_ = dispatch_dtype_fn(config, "PolyphaseInterpolator",
			[&]<typename T>() -> std::unique_ptr<IPolyphaseInterpolatorImpl> {
				auto t = numpy_to_vec_fresh<T>(taps);
				return std::make_unique<PolyphaseInterpolatorImpl<T>>(t, factor);
			});
	}

	np_f64 process(double in)                  { return impl_->process(in); }
	np_f64 process_block(np_f64_ro input)      { return impl_->process_block(input); }
	std::size_t factor() const                 { return impl_->factor(); }
	void reset()                               { impl_->reset(); }

private:
	std::unique_ptr<IPolyphaseInterpolatorImpl> impl_;
};

// ===========================================================================
// bind_acquisition: wires the Py classes + free helpers into the module.
// ===========================================================================

void bind_acquisition(nb::module_& m) {
	// ---- Free design helpers -------------------------------------------
	m.def("design_halfband",
		[](std::size_t num_taps, double transition_width,
		   const std::string& dtype) {
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "design_halfband", [&]<typename T>() {
				auto taps = sw::dsp::design_halfband<T>(
					num_taps, T(transition_width));
				return vec_to_numpy(taps);
			});
		}, nb::arg("num_taps"), nb::arg("transition_width") = 0.1,
		   nb::arg("dtype") = "reference",
		"Design an equiripple half-band lowpass filter via Remez exchange. "
		"num_taps must be of the form 4K+3 (e.g., 7, 11, 15, 19, ...). "
		"Returns NumPy float64 taps; dtype controls internal design precision.");

	m.def("polyphase_decompose",
		[](np_f64_ro taps, std::size_t factor, const std::string& dtype) {
			if (factor == 0)
				throw std::invalid_argument("polyphase_decompose: factor must be > 0");
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "polyphase_decompose",
				[&]<typename T>() -> std::vector<np_f64> {
					auto t = numpy_to_vec_fresh<T>(taps);
					auto sub = sw::dsp::polyphase_decompose(t, factor);
					std::vector<np_f64> out;
					out.reserve(sub.size());
					for (auto& s : sub) out.push_back(vec_to_numpy(s));
					return out;
				});
		}, nb::arg("taps"), nb::arg("factor"), nb::arg("dtype") = "reference",
		"Decompose an FIR prototype into `factor` polyphase sub-filters. "
		"Returns a list of NumPy float64 arrays of length ceil(N/factor).");

	// ---- NCO -----------------------------------------------------------
	nb::class_<PyNCO>(m, "NCO",
		"Numerically Controlled Oscillator. Generates complex sinusoids "
		"(I/Q) for digital mixing. Phase accumulator precision determines SFDR.")
		.def(nb::init<double, double, const std::string&>(),
		     nb::arg("frequency"), nb::arg("sample_rate"),
		     nb::arg("dtype") = "reference")
		.def("set_frequency", &PyNCO::set_frequency,
		     nb::arg("frequency"), nb::arg("sample_rate"))
		.def("set_phase_offset", &PyNCO::set_phase_offset, nb::arg("offset"))
		.def_prop_ro("phase", &PyNCO::phase)
		.def_prop_ro("phase_increment", &PyNCO::phase_increment)
		.def("generate_sample", &PyNCO::generate_sample,
		     "Generate one (real, imag) I/Q sample and advance the phase.")
		.def("generate_real", &PyNCO::generate_real,
		     "Generate one real-valued sample (cos only) and advance the phase.")
		.def("generate_block", &PyNCO::generate_block, nb::arg("length"),
		     "Generate a block of complex samples. Returns (real, imag) tuple.")
		.def("generate_block_real", &PyNCO::generate_block_real, nb::arg("length"),
		     "Generate a block of real-valued samples (cos).")
		.def("mix_down", &PyNCO::mix_down, nb::arg("input"),
		     "Multiply real input by conj(NCO output). Returns (real, imag) tuple "
		     "of the resulting complex baseband signal.")
		.def("reset", &PyNCO::reset);

	// ---- CICDecimator --------------------------------------------------
	nb::class_<PyCICDecimator>(m, "CICDecimator",
		"Cascaded Integrator-Comb decimation filter. Multiplier-free; "
		"ideal for the first decimation stage after a high-rate ADC.")
		.def(nb::init<int, int, int, const std::string&>(),
		     nb::arg("decimation_ratio"), nb::arg("num_stages"),
		     nb::arg("differential_delay") = 1,
		     nb::arg("dtype") = "reference")
		.def("push", &PyCICDecimator::push, nb::arg("input"),
		     "Feed one input sample. Returns (emit, output) — emit is True "
		     "when the decimated output is valid this call.")
		.def_prop_ro("output", &PyCICDecimator::output,
		     "Most recent decimated output (valid after push() emits).")
		.def("process_block", &PyCICDecimator::process_block, nb::arg("input"),
		     "Decimate a block; returns the decimated outputs.")
		.def_prop_ro("decimation_ratio", &PyCICDecimator::decimation_ratio)
		.def_prop_ro("num_stages", &PyCICDecimator::num_stages)
		.def_prop_ro("differential_delay", &PyCICDecimator::differential_delay)
		.def("reset", &PyCICDecimator::reset);

	// ---- CICInterpolator -----------------------------------------------
	nb::class_<PyCICInterpolator>(m, "CICInterpolator",
		"Cascaded Integrator-Comb interpolation filter (the dual of "
		"CICDecimator). Multiplier-free upsampling.")
		.def(nb::init<int, int, int, const std::string&>(),
		     nb::arg("interpolation_ratio"), nb::arg("num_stages"),
		     nb::arg("differential_delay") = 1,
		     nb::arg("dtype") = "reference")
		.def("push", &PyCICInterpolator::push, nb::arg("input"))
		.def("output", &PyCICInterpolator::output)
		.def("process_block", &PyCICInterpolator::process_block, nb::arg("input"),
		     "Interpolate a block; returns ratio*N output samples.")
		.def_prop_ro("interpolation_ratio", &PyCICInterpolator::interpolation_ratio)
		.def_prop_ro("num_stages", &PyCICInterpolator::num_stages)
		.def_prop_ro("differential_delay", &PyCICInterpolator::differential_delay)
		.def("reset", &PyCICInterpolator::reset);

	// ---- HalfBandFilter ------------------------------------------------
	nb::class_<PyHalfBandFilter>(m, "HalfBandFilter",
		"Half-band FIR filter. Use process_decimate() / process_block_decimate() "
		"for efficient 2x decimation that skips zero-valued tap multiplies.")
		.def(nb::init<np_f64_ro, const std::string&>(),
		     nb::arg("taps"), nb::arg("dtype") = "reference")
		.def("process", &PyHalfBandFilter::process, nb::arg("input"),
		     "Full-rate process: one input -> one output.")
		.def("process_block", &PyHalfBandFilter::process_block, nb::arg("input"))
		.def("process_decimate", &PyHalfBandFilter::process_decimate, nb::arg("input"),
		     "2x decimation: feed one input, returns (emit, output) where "
		     "emit alternates True/False.")
		.def("process_block_decimate", &PyHalfBandFilter::process_block_decimate,
		     nb::arg("input"),
		     "Decimate a block; returns floor(N/2) output samples.")
		.def_prop_ro("num_taps", &PyHalfBandFilter::num_taps)
		.def_prop_ro("num_nonzero_taps", &PyHalfBandFilter::num_nonzero_taps)
		.def("reset", &PyHalfBandFilter::reset);

	// ---- PolyphaseDecimator --------------------------------------------
	nb::class_<PyPolyphaseDecimator>(m, "PolyphaseDecimator",
		"M-factor polyphase FIR decimator. Decomposes the prototype into "
		"M sub-filters; each advances once per output sample, so the cost "
		"is ~N mults per output instead of ~N*M for naive filter+downsample.")
		.def(nb::init<np_f64_ro, std::size_t, const std::string&>(),
		     nb::arg("taps"), nb::arg("factor"),
		     nb::arg("dtype") = "reference")
		.def("process", &PyPolyphaseDecimator::process, nb::arg("input"),
		     "Feed one input. Returns (emit, output).")
		.def("process_block", &PyPolyphaseDecimator::process_block, nb::arg("input"))
		.def_prop_ro("factor", &PyPolyphaseDecimator::factor)
		.def("reset", &PyPolyphaseDecimator::reset);

	// ---- PolyphaseInterpolator -----------------------------------------
	nb::class_<PyPolyphaseInterpolator>(m, "PolyphaseInterpolator",
		"L-factor polyphase FIR interpolator. Each input produces L outputs.")
		.def(nb::init<np_f64_ro, std::size_t, const std::string&>(),
		     nb::arg("taps"), nb::arg("factor"),
		     nb::arg("dtype") = "reference")
		.def("process", &PyPolyphaseInterpolator::process, nb::arg("input"),
		     "Feed one input, returns array of `factor` upsampled outputs.")
		.def("process_block", &PyPolyphaseInterpolator::process_block, nb::arg("input"))
		.def_prop_ro("factor", &PyPolyphaseInterpolator::factor)
		.def("reset", &PyPolyphaseInterpolator::reset);
}
