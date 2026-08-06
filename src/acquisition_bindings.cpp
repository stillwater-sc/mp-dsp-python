// acquisition_bindings.cpp: bindings for the high-rate data-acquisition
// pipeline primitives (Phase 3 of the v0.6 acquisition epic, Issue #86).
//
// Surface bound here:
//   - NCO                         (nco.hpp)
//   - CICDecimator, CICInterpolator   (cic.hpp)
//   - HalfBandFilter              (halfband.hpp)
//   - PolyphaseDecimator, PolyphaseInterpolator (filter/fir/polyphase.hpp,
//                                  re-exported by acquisition/polyphase_decimator.hpp)
//   - DDC                         (ddc.hpp, Issue #87)
//   - DecimationChain             (decimation_chain.hpp, Issue #88)
//   - design_halfband             (halfband.hpp free function)
//   - design_cic_compensator      (decimation_chain.hpp free function, #88)
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
#include <nanobind/stl/complex.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sw/dsp/acquisition/nco.hpp>
#include <sw/dsp/acquisition/cic.hpp>
#include <sw/dsp/acquisition/ddc.hpp>
#include <sw/dsp/acquisition/decimation_chain.hpp>
#include <sw/dsp/acquisition/halfband.hpp>
#include <sw/dsp/acquisition/polyphase_decimator.hpp>
#include <sw/dsp/analysis/acquisition_precision.hpp>
#include <sw/dsp/filter/fir/polyphase.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

using mpdsp::bindings::complex_split_to_numpy;
using mpdsp::bindings::dispatch_dtype_fn;
using mpdsp::bindings::make_f64_array;
using mpdsp::bindings::make_impl_for_dtype;
using mpdsp::bindings::np_f64;
using mpdsp::bindings::np_f64_ro;
using mpdsp::bindings::numpy_to_vec_fresh;
using mpdsp::bindings::vec_to_numpy;

namespace {

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
	virtual double measure_sfdr_db(std::size_t fft_size,
	                                std::size_t guard_bins) = 0;
	virtual void reset() = 0;
};

template <typename T>
class NCOImpl : public INCOImpl {
public:
	// Pass the rates through as double. Upstream takes them templated and
	// forms frequency/sample_rate in double *before* converting the ratio
	// (issue #207), so casting to T here would overflow a narrow state type
	// at the argument boundary and defeat the fix — which is exactly what
	// this binding used to do.
	NCOImpl(double frequency, double sample_rate)
		: nco_(frequency, sample_rate) {}

	void set_frequency(double f, double sr) override {
		if (!(sr > 0.0))
			throw std::invalid_argument("NCO: sample_rate must be positive");
		nco_.set_frequency(f, sr);       // double in, ratio converted — see ctor
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
		return complex_split_to_numpy(block);
	}
	np_f64 generate_block_real(std::size_t length) override {
		auto block = nco_.generate_block_real(length);
		return vec_to_numpy(block);
	}
	nb::tuple mix_down(np_f64_ro input) override {
		auto in = numpy_to_vec_fresh<typename decltype(nco_)::sample_scalar>(input);
		auto out = nco_.mix_down(in);
		return complex_split_to_numpy(out);
	}
	double measure_sfdr_db(std::size_t fft_size,
	                       std::size_t guard_bins) override {
		return sw::dsp::analysis::measure_nco_sfdr_db(
			nco_, fft_size, guard_bins);
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
	virtual sw::dsp::analysis::CICBitGrowthReport
	check_bit_growth(np_f64_ro input) = 0;
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
	sw::dsp::analysis::CICBitGrowthReport
	check_bit_growth(np_f64_ro input) override {
		// Cast the input to the CIC's sample scalar type, then hand it to
		// the analysis helper as a span. The upstream free function
		// mutates the CIC (pushes samples through it) and records the
		// output peak — same side-effect the caller sees from a normal
		// process_block() run, but with per-sample .output() sampling.
		auto typed_in = numpy_to_vec_fresh<T>(input);
		return sw::dsp::analysis::check_cic_bit_growth(
			cic_, std::span<const T>(typed_in.data(), typed_in.size()));
	}
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
		std::vector<T> out_buf;
		out_buf.reserve(in_v.size() * static_cast<std::size_t>(cic_.interpolation_ratio()));
		cic_.process_block(std::span<const T>(in_v.data(), in_v.size()), out_buf);
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
		auto out = hb_.process_block_decimate(
			std::span<const T>(in.data(), in.size()));
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
		auto out = pd_.process_block(
			std::span<const T>(in_v.data(), in_v.size()));
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
		auto out = pi_.process_block(
			std::span<const T>(in_v.data(), in_v.size()));
		return vec_to_numpy(out);
	}
	std::size_t factor() const override { return pi_.factor(); }
	void reset() override { pi_.reset(); }

private:
	sw::dsp::PolyphaseInterpolator<T> pi_;
};

// ===========================================================================
// DDC (digital down-converter): NCO -> mixer -> I/Q decimation
// ===========================================================================
//
// The upstream DDC takes the decimator as a template parameter, accepting
// anything that steps via process() / process_decimate() / push()+output().
// This binding fixes it to PolyphaseDecimator, per the Phase 4 scope note
// (#87): a Python-side decimator choice would multiply the instantiation
// count across every ArithConfig for no immediate research payoff. Richer
// composition is the job of DecimationChain (#88).
//
// The decimator prototype is built here from (taps, factor) rather than
// accepting a PyPolyphaseDecimator: that class holds a type-erased
// unique_ptr<IPolyphaseDecimatorImpl>, so there is no way to recover the
// concrete PolyphaseDecimator<T> the DDC constructor needs to copy.

struct IDDCImpl {
	virtual ~IDDCImpl() = default;
	virtual std::pair<bool, std::complex<double>> process(double in) = 0;
	virtual nb::tuple process_block(np_f64_ro input) = 0;
	virtual void set_center_frequency(double frequency) = 0;
	virtual double center_frequency() const = 0;
	virtual double sample_rate() const = 0;
	virtual std::size_t decimation_factor() const = 0;
	virtual double nco_phase() const = 0;
	virtual double nco_phase_increment() const = 0;
	virtual void reset() = 0;
};

template <typename T>
class DDCImpl : public IDDCImpl {
public:
	DDCImpl(double center_frequency, double sample_rate,
	        const mtl::vec::dense_vector<T>& taps, std::size_t factor)
		// Rates as double, same reasoning as NCOImpl — upstream divides
		// before converting.
		: ddc_(center_frequency, sample_rate,
		       sw::dsp::PolyphaseDecimator<T>(taps, factor)),
		  factor_(factor) {}

	std::pair<bool, std::complex<double>> process(double in) override {
		auto [ready, z] = ddc_.process(static_cast<T>(in));
		// complex_for_t<T> is std::complex<T> for float/double and
		// sw::universal::complex<T> otherwise; both expose real()/imag().
		return {ready, std::complex<double>(static_cast<double>(z.real()),
		                                    static_cast<double>(z.imag()))};
	}
	nb::tuple process_block(np_f64_ro input) override {
		auto in = numpy_to_vec_fresh<T>(input);
		auto out = ddc_.process_block(in);
		return complex_split_to_numpy(out);
	}
	void set_center_frequency(double frequency) override {
		ddc_.set_center_frequency(frequency);
	}
	double center_frequency() const override {
		return static_cast<double>(ddc_.center_frequency());
	}
	double sample_rate() const override {
		return static_cast<double>(ddc_.sample_rate());
	}
	std::size_t decimation_factor() const override { return factor_; }
	double nco_phase() const override {
		return static_cast<double>(ddc_.nco().phase());
	}
	double nco_phase_increment() const override {
		return static_cast<double>(ddc_.nco().phase_increment());
	}
	void reset() override { ddc_.reset(); }

private:
	sw::dsp::DDC<T> ddc_;
	std::size_t     factor_;
};

// ===========================================================================
// DecimationChain (multi-stage cascade)
// ===========================================================================
//
// Upstream DecimationChain<Sample, Stages...> is variadic and holds its stages
// in a std::tuple, so the stage count is a compile-time property. Python needs
// a runtime-length list. Bridging that without reimplementing the chain (rate
// bookkeeping, short-circuit threading, ratio products all live upstream and
// should stay there) takes two pieces:
//
//   1. ErasedStage<T> — a move-only value type satisfying the upstream stage
//      contract (process() -> pair<bool,T>, factor(), reset()) that forwards
//      through a virtual interface. Every stage in a chain then has the SAME
//      static type, so a chain of N stages is DecimationChain<T,
//      ErasedStage<T> x N>.
//
//   2. A runtime switch over N, instantiating that alias for each supported
//      stage count. kMaxChainStages caps the instantiation count: each dtype
//      pays for every arity, so this is 18 * kMaxChainStages instantiations
//      of DecimationChain. Six covers the deep SDR cascades (CIC -> HB -> HB
//      -> HB -> FIR) that motivate the class; raising it costs compile time,
//      nothing else.
//
// Stages are rebuilt at the chain's dtype from the prototype objects' design
// parameters rather than adopted wholesale — same reasoning as DDC's
// decimator, and additionally required here because the chain's inter-stage
// sample type is T, not double.

constexpr std::size_t kMaxChainStages = 6;

// Parameters extracted from a Python prototype stage, enough to rebuild the
// concrete decimator at any T.
struct StageSpec {
	enum class Kind { cic, halfband, polyphase };
	Kind                kind;
	int                 cic_ratio = 0;
	int                 cic_stages = 0;
	int                 cic_delay = 1;
	std::vector<double> taps;
	std::size_t         factor = 1;
};

template <typename T>
struct IChainStage {
	virtual ~IChainStage() = default;
	virtual std::pair<bool, T> process(T in) = 0;
	virtual std::size_t factor() const = 0;
	virtual void reset() = 0;
};

// Wraps one concrete upstream decimator. step_decimator / decimation_ratio_of
// are the same detail helpers DecimationChain itself uses to talk to a stage,
// so this adapter stays honest if that contract ever grows a new shape.
template <typename T, class Decim>
class ChainStage : public IChainStage<T> {
public:
	explicit ChainStage(Decim d) : d_(std::move(d)) {}
	std::pair<bool, T> process(T in) override {
		return sw::dsp::detail::step_decimator(d_, in);
	}
	std::size_t factor() const override {
		return sw::dsp::detail::decimation_ratio_of(d_);
	}
	void reset() override { d_.reset(); }

private:
	Decim d_;
};

template <typename T>
class ErasedStage {
public:
	using sample_scalar = T;

	explicit ErasedStage(std::unique_ptr<IChainStage<T>> impl)
		: impl_(std::move(impl)) {}
	ErasedStage(ErasedStage&&) noexcept = default;
	ErasedStage& operator=(ErasedStage&&) noexcept = default;

	std::pair<bool, T> process(T in) { return impl_->process(in); }
	std::size_t factor() const       { return impl_->factor(); }
	void reset()                     { impl_->reset(); }

private:
	std::unique_ptr<IChainStage<T>> impl_;
};

template <typename T>
std::unique_ptr<IChainStage<T>> make_chain_stage(const StageSpec& spec) {
	switch (spec.kind) {
	case StageSpec::Kind::cic:
		return std::make_unique<ChainStage<T, sw::dsp::CICDecimator<T>>>(
			sw::dsp::CICDecimator<T>(spec.cic_ratio, spec.cic_stages,
			                          spec.cic_delay));
	case StageSpec::Kind::halfband: {
		mtl::vec::dense_vector<T> taps(spec.taps.size());
		for (std::size_t i = 0; i < spec.taps.size(); ++i)
			taps[i] = static_cast<T>(spec.taps[i]);
		return std::make_unique<ChainStage<T, sw::dsp::HalfBandFilter<T>>>(
			sw::dsp::HalfBandFilter<T>(taps));
	}
	case StageSpec::Kind::polyphase: {
		mtl::vec::dense_vector<T> taps(spec.taps.size());
		for (std::size_t i = 0; i < spec.taps.size(); ++i)
			taps[i] = static_cast<T>(spec.taps[i]);
		return std::make_unique<ChainStage<T, sw::dsp::PolyphaseDecimator<T>>>(
			sw::dsp::PolyphaseDecimator<T>(taps, spec.factor));
	}
	}
	throw std::invalid_argument("DecimationChain: unsupported stage kind");
}

// DecimationChain<T, ErasedStage<T> repeated N times>.
template <class T, std::size_t>
using ErasedStageAt = ErasedStage<T>;

template <class T, class Seq>
struct ChainTypeFor;
template <class T, std::size_t... Is>
struct ChainTypeFor<T, std::index_sequence<Is...>> {
	using type = sw::dsp::DecimationChain<T, ErasedStageAt<T, Is>...>;
};
template <class T, std::size_t N>
using ChainOf = typename ChainTypeFor<T, std::make_index_sequence<N>>::type;

template <class T, std::size_t N, std::size_t... Is>
ChainOf<T, N> build_chain(T rate, std::vector<ErasedStage<T>>& stages,
                          std::index_sequence<Is...>) {
	return ChainOf<T, N>(rate, std::move(stages[Is])...);
}

struct IDecimationChainImpl {
	virtual ~IDecimationChainImpl() = default;
	virtual std::pair<bool, double> process(double in) = 0;
	virtual np_f64 process_block(np_f64_ro input) = 0;
	virtual double input_rate() const = 0;
	virtual double output_rate() const = 0;
	virtual std::size_t total_decimation() const = 0;
	virtual std::vector<std::size_t> stage_ratios() const = 0;
	virtual std::vector<double> stage_rates() const = 0;
	virtual std::size_t num_stages() const = 0;
	virtual void reset() = 0;
};

template <typename T, std::size_t N>
class DecimationChainImpl : public IDecimationChainImpl {
public:
	DecimationChainImpl(double input_rate, std::vector<ErasedStage<T>>& stages)
		: chain_(build_chain<T, N>(static_cast<T>(input_rate), stages,
		                            std::make_index_sequence<N>{})) {}

	std::pair<bool, double> process(double in) override {
		auto [ready, y] = chain_.process(static_cast<T>(in));
		return {ready, static_cast<double>(y)};
	}
	np_f64 process_block(np_f64_ro input) override {
		auto in = numpy_to_vec_fresh<T>(input);
		auto out = chain_.process_block(in);
		return vec_to_numpy(out);
	}
	double input_rate() const override {
		return static_cast<double>(chain_.input_rate());
	}
	double output_rate() const override {
		return static_cast<double>(chain_.output_rate());
	}
	std::size_t total_decimation() const override {
		return chain_.total_decimation();
	}
	std::vector<std::size_t> stage_ratios() const override {
		auto arr = chain_.stage_ratios();
		return std::vector<std::size_t>(arr.begin(), arr.end());
	}
	std::vector<double> stage_rates() const override {
		auto arr = chain_.stage_rates();
		std::vector<double> out;
		out.reserve(arr.size());
		for (auto r : arr) out.push_back(static_cast<double>(r));
		return out;
	}
	std::size_t num_stages() const override { return N; }
	void reset() override { chain_.reset(); }

private:
	ChainOf<T, N> chain_;
};

// Runtime stage-count -> compile-time arity. Recursion over the arity range
// keeps this to one line per supported N without hand-writing the switch.
template <typename T, std::size_t N = 1>
std::unique_ptr<IDecimationChainImpl>
make_chain_impl(double input_rate, std::vector<ErasedStage<T>>& stages) {
	if (stages.size() == N)
		return std::make_unique<DecimationChainImpl<T, N>>(input_rate, stages);
	if constexpr (N < kMaxChainStages)
		return make_chain_impl<T, N + 1>(input_rate, stages);
	else
		throw std::invalid_argument(
			"DecimationChain: stage count " + std::to_string(stages.size()) +
			" exceeds the supported maximum of " +
			std::to_string(kMaxChainStages));
}

} // anonymous namespace

// ===========================================================================
// Py-wrappers (visible names — held by std::unique_ptr<I*Impl> + dispatch)
// ===========================================================================

// ---------------------------------------------------------------------------
// Phase-accumulator sanity check (Issue #117).
//
// NCO holds `frequency` and `sample_rate` at the configuration's state scalar
// and divides only afterwards (`phase_inc = frequency / sample_rate`), so
// absolute rates at RF scale overflow narrow state types before the division
// can bring the ratio back into range. fixpnt trips upstream's own
// "sample_rate must be positive" check, but the cfloat types (`cf24`, `half`)
// quietly produce a NaN phase increment and then emit NaN for every sample
// thereafter, with nothing to point at the cause.
//
// Turning that into an error at construction is the whole difference between
// a one-line fix and an afternoon of bisecting a pipeline. The message names
// the workaround because it is not guessable from the symptom.
//
// Upstream #207 fixed the cause: NCO now forms frequency/sample_rate in
// double before converting, so absolute Hz work for every state type — and
// this binding had to stop casting to T at the boundary for that to take
// effect, since the premature cast overflowed before upstream saw the values.
// The check stays as a backstop: it catches any non-finite accumulator, not
// only that cause, and it costs one comparison per construction.
// ---------------------------------------------------------------------------

static void require_finite_phase(double increment, const char* cls,
                                 const std::string& dtype) {
	if (std::isfinite(increment)) return;
	throw std::invalid_argument(
		std::string(cls) + ": phase increment is not finite for dtype '"
		+ dtype + "'. frequency and sample_rate are held at that dtype's "
		"state precision, so absolute rates can overflow it before "
		"frequency/sample_rate is evaluated. Pass normalized rates instead "
		"(sample_rate=1.0, frequency as a fraction of it) — an oscillator "
		"only ever uses the ratio.");
}

class PyNCO {
public:
	PyNCO(double frequency, double sample_rate, const std::string& dtype) {
		if (!(sample_rate > 0.0))
			throw std::invalid_argument("NCO: sample_rate must be positive");
		auto config = mpdsp::parse_config(dtype);
		impl_ = make_impl_for_dtype<NCOImpl, INCOImpl>(
			config, "NCO", frequency, sample_rate);
		dtype_ = dtype;
		require_finite_phase(impl_->phase_increment(), "NCO", dtype_);
	}

	void set_frequency(double f, double sr) {
		impl_->set_frequency(f, sr);
		require_finite_phase(impl_->phase_increment(), "NCO", dtype_);
	}
	void set_phase_offset(double off)        { impl_->set_phase_offset(off); }
	double phase() const                     { return impl_->phase(); }
	double phase_increment() const           { return impl_->phase_increment(); }
	std::pair<double, double> generate_sample() { return impl_->generate_sample(); }
	double generate_real()                   { return impl_->generate_real(); }
	nb::tuple generate_block(std::size_t n)  { return impl_->generate_block(n); }
	np_f64 generate_block_real(std::size_t n){ return impl_->generate_block_real(n); }
	nb::tuple mix_down(np_f64_ro input)      { return impl_->mix_down(input); }
	double measure_sfdr_db(std::size_t fft_size, std::size_t guard_bins) {
		return impl_->measure_sfdr_db(fft_size, guard_bins);
	}
	void reset()                             { impl_->reset(); }

private:
	std::unique_ptr<INCOImpl> impl_;
	std::string               dtype_;
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
	sw::dsp::analysis::CICBitGrowthReport
	check_bit_growth(np_f64_ro input)       { return impl_->check_bit_growth(input); }
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
	PyHalfBandFilter(np_f64_ro taps, const std::string& dtype)
		: taps_(taps.data(), taps.data() + taps.shape(0)) {
		auto config = mpdsp::parse_config(dtype);
		// We have to materialize the taps in the chosen T before constructing
		// the impl; dispatch on config to pick the right <T>.
		impl_ = dispatch_dtype_fn(config, "HalfBandFilter",
			[&]<typename T>() -> std::unique_ptr<IHalfBandImpl> {
				auto t = numpy_to_vec_fresh<T>(taps);
				return std::make_unique<HalfBandImpl<T>>(t);
			});
	}

	// Design parameters retained for DecimationChain, which has to rebuild a
	// concrete stage at the chain's own dtype: upstream HalfBandFilter keeps
	// no retrievable copy of its taps.
	const std::vector<double>& taps_ref() const { return taps_; }
	np_f64 taps() const {
		double* out = nullptr;
		auto arr = make_f64_array(taps_.size(), out);
		for (std::size_t i = 0; i < taps_.size(); ++i) out[i] = taps_[i];
		return arr;
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
	std::vector<double>            taps_;
};

class PyPolyphaseDecimator {
public:
	PyPolyphaseDecimator(np_f64_ro taps, std::size_t factor,
	                     const std::string& dtype)
		: taps_(taps.data(), taps.data() + taps.shape(0)) {
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

	// See PyHalfBandFilter::taps_ref — upstream PolyphaseDecimator decomposes
	// the prototype into sub-filters and keeps no copy of the original taps.
	const std::vector<double>& taps_ref() const { return taps_; }
	np_f64 taps() const {
		double* out = nullptr;
		auto arr = make_f64_array(taps_.size(), out);
		for (std::size_t i = 0; i < taps_.size(); ++i) out[i] = taps_[i];
		return arr;
	}

private:
	std::unique_ptr<IPolyphaseDecimatorImpl> impl_;
	std::vector<double>                      taps_;
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

class PyDDC {
public:
	PyDDC(double center_frequency, double sample_rate, np_f64_ro taps,
	      std::size_t decimation_factor, const std::string& dtype) {
		// Validate here rather than letting the NCO constructor throw: the
		// upstream message names the NCO, which is confusing when the caller
		// only ever mentioned a DDC.
		if (!(sample_rate > 0.0))
			throw std::invalid_argument("DDC: sample_rate must be positive");
		if (decimation_factor == 0)
			throw std::invalid_argument("DDC: decimation_factor must be > 0");
		if (taps.shape(0) == 0)
			throw std::invalid_argument("DDC: taps must be non-empty");
		auto config = mpdsp::parse_config(dtype);
		impl_ = dispatch_dtype_fn(config, "DDC",
			[&]<typename T>() -> std::unique_ptr<IDDCImpl> {
				auto t = numpy_to_vec_fresh<T>(taps);
				return std::make_unique<DDCImpl<T>>(
					center_frequency, sample_rate, t, decimation_factor);
			});
		dtype_ = dtype;
		require_finite_phase(impl_->nco_phase_increment(), "DDC", dtype_);
	}

	std::pair<bool, std::complex<double>> process(double in) {
		return impl_->process(in);
	}
	nb::tuple process_block(np_f64_ro input) { return impl_->process_block(input); }
	void set_center_frequency(double f) {
		impl_->set_center_frequency(f);
		require_finite_phase(impl_->nco_phase_increment(), "DDC", dtype_);
	}
	double center_frequency() const          { return impl_->center_frequency(); }
	double sample_rate() const               { return impl_->sample_rate(); }
	std::size_t decimation_factor() const    { return impl_->decimation_factor(); }
	double nco_phase() const                 { return impl_->nco_phase(); }
	double nco_phase_increment() const       { return impl_->nco_phase_increment(); }
	void reset()                             { impl_->reset(); }

private:
	std::unique_ptr<IDDCImpl> impl_;
	std::string               dtype_;
};

class PyDecimationChain {
public:
	PyDecimationChain(double input_rate, nb::sequence stages,
	                  const std::string& dtype) {
		if (!(input_rate > 0.0))
			throw std::invalid_argument(
				"DecimationChain: input_rate must be positive");

		std::vector<StageSpec> specs;
		for (nb::handle h : stages) specs.push_back(spec_from(h));
		if (specs.empty())
			throw std::invalid_argument(
				"DecimationChain: needs at least one stage");
		if (specs.size() > kMaxChainStages)
			throw std::invalid_argument(
				"DecimationChain: stage count " + std::to_string(specs.size()) +
				" exceeds the supported maximum of " +
				std::to_string(kMaxChainStages));

		auto config = mpdsp::parse_config(dtype);
		impl_ = dispatch_dtype_fn(config, "DecimationChain",
			[&]<typename T>() -> std::unique_ptr<IDecimationChainImpl> {
				std::vector<ErasedStage<T>> built;
				built.reserve(specs.size());
				for (const auto& s : specs)
					built.emplace_back(make_chain_stage<T>(s));
				return make_chain_impl<T>(input_rate, built);
			});
	}

	std::pair<bool, double> process(double in) { return impl_->process(in); }
	np_f64 process_block(np_f64_ro input)      { return impl_->process_block(input); }
	double input_rate() const                  { return impl_->input_rate(); }
	double output_rate() const                 { return impl_->output_rate(); }
	std::size_t total_decimation() const       { return impl_->total_decimation(); }
	std::vector<std::size_t> stage_ratios() const { return impl_->stage_ratios(); }
	std::vector<double> stage_rates() const    { return impl_->stage_rates(); }
	std::size_t num_stages() const             { return impl_->num_stages(); }
	void reset()                               { impl_->reset(); }

private:
	// Read design parameters off a prototype stage. The prototype's own dtype
	// is ignored: every stage in a chain runs at the chain's dtype, because
	// upstream threads a single Sample type between stages.
	static StageSpec spec_from(nb::handle h) {
		StageSpec spec{};
		if (nb::isinstance<PyCICDecimator>(h)) {
			const auto& c = nb::cast<const PyCICDecimator&>(h);
			spec.kind       = StageSpec::Kind::cic;
			spec.cic_ratio  = c.decimation_ratio();
			spec.cic_stages = c.num_stages();
			spec.cic_delay  = c.differential_delay();
			return spec;
		}
		if (nb::isinstance<PyHalfBandFilter>(h)) {
			const auto& f = nb::cast<const PyHalfBandFilter&>(h);
			spec.kind = StageSpec::Kind::halfband;
			spec.taps = f.taps_ref();
			return spec;
		}
		if (nb::isinstance<PyPolyphaseDecimator>(h)) {
			const auto& p = nb::cast<const PyPolyphaseDecimator&>(h);
			spec.kind   = StageSpec::Kind::polyphase;
			spec.taps   = p.taps_ref();
			spec.factor = p.factor();
			return spec;
		}
		throw std::invalid_argument(
			"DecimationChain: stages must be CICDecimator, HalfBandFilter, or "
			"PolyphaseDecimator instances (got " +
			nb::cast<std::string>(nb::str(h.type())) + ")");
	}

	std::unique_ptr<IDecimationChainImpl> impl_;
};

// ===========================================================================
// bind_acquisition: wires the Py classes + free helpers into the module.
// ===========================================================================

void bind_acquisition(nb::module_& m) {
	// ---- Free design helpers -------------------------------------------
	m.def("design_halfband",
		[](std::size_t num_taps, double transition_width,
		   bool exact_dc_gain, const std::string& dtype) {
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "design_halfband", [&]<typename T>() {
				auto taps = sw::dsp::design_halfband<T>(
					num_taps, T(transition_width), exact_dc_gain);
				return vec_to_numpy(taps);
			});
		}, nb::arg("num_taps"), nb::arg("transition_width") = 0.1,
		   nb::arg("exact_dc_gain") = false,
		   nb::arg("dtype") = "reference",
		"Design an equiripple half-band lowpass filter via Remez exchange. "
		"num_taps must be of the form 4K+3 (e.g., 7, 11, 15, 19, ...). "
		"Returns NumPy float64 taps; dtype controls internal design "
		"precision.\n\n"
		"exact_dc_gain trades the two properties a half-band cannot have at "
		"once. A(0) + A(0.5) = 1 holds identically for a half-band and "
		"A(0.5) is a stopband extremum, so the DC gain is 1 -/+ delta for "
		"the equiripple design. Leave it false (the default) for the "
		"deeper stopband — the DC error is bounded by the same ripple the "
		"passband already accepts. Pass true when unity DC gain through "
		"cascaded stages matters more, at a cost of about 6 dB of "
		"attenuation.");

	m.def("design_cic_compensator",
		[](std::size_t num_taps, int cic_stages, int cic_ratio,
		   double passband, int differential_delay, const std::string& dtype) {
			auto config = mpdsp::parse_config(dtype);
			return dispatch_dtype_fn(config, "design_cic_compensator",
				[&]<typename T>() {
					auto taps = sw::dsp::design_cic_compensator<T>(
						num_taps, cic_stages, cic_ratio,
						static_cast<T>(passband), differential_delay);
					return vec_to_numpy(taps);
				});
		}, nb::arg("num_taps"), nb::arg("cic_stages"), nb::arg("cic_ratio"),
		   nb::arg("passband"), nb::arg("differential_delay") = 1,
		   nb::arg("dtype") = "reference",
		"Design an FIR that inverts a CIC decimator's passband droop, to be "
		"run at the CIC's output rate. Frequency-sampling design: samples "
		"1/|H_cic(f)| across [0, passband], rolls off smoothly to Nyquist, "
		"IDFTs, applies a Hamming window, and normalizes to unit DC gain.\n\n"
		"`passband` is normalized to the CIC *output* rate and must lie in "
		"(0, 0.5). num_taps >= 3 (odd gives a linear-phase centered tap), "
		"cic_stages >= 1, cic_ratio >= 2. Returns NumPy float64 taps; dtype "
		"controls the precision of the design-time arithmetic, which matters "
		"when the compensator is designed on the target.");

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
		.def("measure_sfdr_db", &PyNCO::measure_sfdr_db,
		     nb::arg("fft_size"), nb::arg("guard_bins") = static_cast<std::size_t>(2),
		     "Measure spurious-free dynamic range: generate fft_size samples, "
		     "FFT them (zero-padded to next power of 2), find the largest "
		     "spur outside `guard_bins` around the tuned peak, and return "
		     "20*log10(peak / spur) in dB. **Mutates the NCO's phase** — "
		     "call reset() before/after for a reproducible measurement. "
		     "Returns +300 dB for a spur-free signal (bit-exact NCO).")
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
		.def("check_bit_growth", &PyCICDecimator::check_bit_growth,
		     nb::arg("input"),
		     "Run `input` through the CIC and record the peak absolute "
		     "output. Returns a CICBitGrowthReport comparing observed vs. "
		     "theoretical (Hogenauer M*ceil(log2(R*D))) bit growth. "
		     "**Mutates the CIC state** (same as calling process_block); "
		     "reset() before/after if you need a clean run.")
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
		// take_ownership: the getter builds a fresh capsule-owned ndarray on
		// every call, so the default reference_internal policy throws at
		// runtime. See src/BINDING_PATTERNS.md.
		.def_prop_ro("taps", &PyHalfBandFilter::taps,
		     nb::rv_policy::take_ownership,
		     "The design taps this filter was constructed with, as float64.")
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
		// take_ownership — see the HalfBandFilter.taps note above.
		.def_prop_ro("taps", &PyPolyphaseDecimator::taps,
		     nb::rv_policy::take_ownership,
		     "The full-rate prototype taps this decimator was constructed "
		     "with, as float64 (not the decomposed sub-filters — use "
		     "polyphase_decompose for those).")
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

	// ---- DDC -----------------------------------------------------------
	nb::class_<PyDDC>(m, "DDC",
		"Digital Down-Converter: mixes a real input band down to complex "
		"baseband with an NCO, then decimates the I and Q streams through "
		"matched polyphase FIR decimators.\n\n"
		"The decimator is fixed to PolyphaseDecimator and built from the "
		"`taps` / `decimation_factor` arguments; two independent copies run "
		"in lockstep on I and Q. Design `taps` as a lowpass with cutoff "
		"below 0.5/decimation_factor (normalized to the input rate) to "
		"suppress aliasing — `mpdsp.fir_lowpass` is the usual source.")
		.def(nb::init<double, double, np_f64_ro, std::size_t, const std::string&>(),
		     nb::arg("center_frequency"), nb::arg("sample_rate"),
		     nb::arg("taps"), nb::arg("decimation_factor"),
		     nb::arg("dtype") = "reference")
		.def("process", &PyDDC::process, nb::arg("input"),
		     "Feed one real input sample. Returns (emit, value) where `value` "
		     "is the complex baseband sample, valid only when emit is True "
		     "(once per decimation_factor inputs). On non-emit cycles the "
		     "value is 0j.")
		.def("process_block", &PyDDC::process_block, nb::arg("input"),
		     "Down-convert a block of real samples. Returns a (real, imag) "
		     "tuple of float64 arrays holding the ~len(input)/decimation_factor "
		     "complex baseband samples produced during the block — matching "
		     "the convention used by NCO.mix_down() and NCO.generate_block(). "
		     "Combine with `real + 1j*imag` for a complex128 array.")
		.def("set_center_frequency", &PyDDC::set_center_frequency,
		     nb::arg("frequency"),
		     "Retune the local oscillator. The decimator state is left "
		     "untouched; call reset() first for a clean retune.")
		.def_prop_ro("center_frequency", &PyDDC::center_frequency)
		.def_prop_ro("sample_rate", &PyDDC::sample_rate)
		.def_prop_ro("decimation_factor", &PyDDC::decimation_factor)
		.def_prop_ro("nco_phase", &PyDDC::nco_phase,
		     "Current phase of the internal NCO, in normalized cycles in "
		     "[0, 1) — multiply by 2*pi for radians. Exposed as a read-only "
		     "scalar rather than an NCO handle: the DDC owns its oscillator, "
		     "and handing out a live reference through the type-erased impl "
		     "would outlive-alias it.")
		.def_prop_ro("nco_phase_increment", &PyDDC::nco_phase_increment,
		     "Per-sample phase step of the internal NCO, in normalized "
		     "cycles — equal to center_frequency / sample_rate.")
		.def("reset", &PyDDC::reset,
		     "Clear the NCO phase and both decimator delay lines.");

	// ---- DecimationChain -----------------------------------------------
	nb::class_<PyDecimationChain>(m, "DecimationChain",
		"Multi-stage decimation cascade: ADC -> CIC -> half-band -> ... -> "
		"baseband. Large decimation ratios are cheapest as a cascade of "
		"small ones, each stage running at the (progressively lower) rate "
		"its predecessor emits.\n\n"
		"`stages` is a list of CICDecimator / HalfBandFilter / "
		"PolyphaseDecimator instances used as **prototypes**: the chain "
		"reads their design parameters and rebuilds equivalent stages at the "
		"chain's own dtype. The prototypes are neither mutated nor aliased, "
		"and their individual dtypes are ignored — upstream threads a single "
		"sample type between stages, so the chain's dtype governs "
		"throughout.\n\n"
		"At most 6 stages; each additional arity is a separate template "
		"instantiation per dtype.")
		.def(nb::init<double, nb::sequence, const std::string&>(),
		     nb::arg("input_rate"), nb::arg("stages"),
		     nb::arg("dtype") = "reference")
		.def("process", &PyDecimationChain::process, nb::arg("input"),
		     "Feed one input sample. Returns (emit, output); emit is True "
		     "only on the cycle where the *final* stage produces a sample, "
		     "i.e. once per total_decimation inputs.")
		.def("process_block", &PyDecimationChain::process_block, nb::arg("input"),
		     "Decimate a block; returns the ~len(input)/total_decimation "
		     "samples emitted by the final stage.")
		.def_prop_ro("input_rate", &PyDecimationChain::input_rate)
		.def_prop_ro("output_rate", &PyDecimationChain::output_rate,
		     "input_rate / total_decimation.")
		.def_prop_ro("total_decimation", &PyDecimationChain::total_decimation,
		     "Product of the per-stage decimation ratios.")
		.def_prop_ro("num_stages", &PyDecimationChain::num_stages)
		.def("stage_ratios", &PyDecimationChain::stage_ratios,
		     "Per-stage decimation ratios, in input order. HalfBandFilter "
		     "reports 2 (it is structurally fixed at 2:1).")
		.def("stage_rates", &PyDecimationChain::stage_rates,
		     "Sample rate at the *output* of each stage, in input order. The "
		     "last element equals output_rate.")
		.def("reset", &PyDecimationChain::reset,
		     "Reset every stage's internal state.");
}
