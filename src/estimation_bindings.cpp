// estimation_bindings.cpp: state-estimation bindings (Kalman, LMS, RLS).
//
// Phase 5 stateful pattern (per #21 / #22): dtype fixed at construction,
// internal type-erased interface per class, NumPy float64 I/O at the Python
// boundary. New for this file: 2D NumPy matrix marshalling for the Kalman
// system matrices F, H, Q, R, P, B.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <sw/dsp/estimation/kalman.hpp>
#include <sw/dsp/estimation/lms.hpp>
#include <sw/dsp/estimation/rls.hpp>

#include "_binding_helpers.hpp"
#include "types.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

namespace nb = nanobind;

// Pull shared NumPy typedefs + helpers into this TU's namespace.
using mpdsp::bindings::np_f64;
using mpdsp::bindings::np_f64_ro;
using mpdsp::bindings::np_f64_2d;
using mpdsp::bindings::np_f64_2d_ro;
using mpdsp::bindings::make_f64_array;
using mpdsp::bindings::make_f64_2d_array;
using mpdsp::bindings::mat_to_numpy;
using mpdsp::bindings::numpy_to_mat;
using mpdsp::bindings::vec_to_numpy;
using mpdsp::bindings::numpy_to_vec;
using mpdsp::bindings::make_impl_for_dtype;

namespace {

// ===========================================================================
// KalmanFilter
// ===========================================================================

struct IKalmanImpl {
	virtual ~IKalmanImpl() = default;

	virtual std::size_t state_dim() const = 0;
	virtual std::size_t meas_dim() const = 0;
	virtual std::size_t ctrl_dim() const = 0;

	virtual np_f64_2d get_F() const = 0;
	virtual np_f64_2d get_H() const = 0;
	virtual np_f64_2d get_Q() const = 0;
	virtual np_f64_2d get_R() const = 0;
	virtual np_f64_2d get_P() const = 0;
	virtual np_f64_2d get_B() const = 0;
	virtual np_f64    get_state() const = 0;

	virtual void set_F(np_f64_2d_ro a) = 0;
	virtual void set_H(np_f64_2d_ro a) = 0;
	virtual void set_Q(np_f64_2d_ro a) = 0;
	virtual void set_R(np_f64_2d_ro a) = 0;
	virtual void set_P(np_f64_2d_ro a) = 0;
	virtual void set_B(np_f64_2d_ro a) = 0;
	virtual void set_state(np_f64_ro v) = 0;

	virtual void predict() = 0;
	virtual void predict_with_control(np_f64_ro u) = 0;
	virtual void update(np_f64_ro z) = 0;
};

template <typename T>
struct KalmanImpl : IKalmanImpl {
	sw::dsp::KalmanFilter<T> inner;

	KalmanImpl(std::size_t s, std::size_t m, std::size_t c) : inner(s, m, c) {}

	std::size_t state_dim() const override { return inner.state_dim(); }
	std::size_t meas_dim() const override { return inner.meas_dim(); }
	std::size_t ctrl_dim() const override { return inner.ctrl_dim(); }

	np_f64_2d get_F() const override { return mat_to_numpy(inner.F()); }
	np_f64_2d get_H() const override { return mat_to_numpy(inner.H()); }
	np_f64_2d get_Q() const override { return mat_to_numpy(inner.Q()); }
	np_f64_2d get_R() const override { return mat_to_numpy(inner.R()); }
	np_f64_2d get_P() const override { return mat_to_numpy(inner.P()); }
	np_f64_2d get_B() const override { return mat_to_numpy(inner.B()); }
	np_f64    get_state() const override { return vec_to_numpy(inner.state()); }

	void set_F(np_f64_2d_ro a) override { numpy_to_mat(a, inner.F(), "F"); }
	void set_H(np_f64_2d_ro a) override { numpy_to_mat(a, inner.H(), "H"); }
	void set_Q(np_f64_2d_ro a) override { numpy_to_mat(a, inner.Q(), "Q"); }
	void set_R(np_f64_2d_ro a) override { numpy_to_mat(a, inner.R(), "R"); }
	void set_P(np_f64_2d_ro a) override { numpy_to_mat(a, inner.P(), "P"); }
	void set_B(np_f64_2d_ro a) override { numpy_to_mat(a, inner.B(), "B"); }
	void set_state(np_f64_ro v) override { numpy_to_vec(v, inner.state(), "state"); }

	void predict() override { inner.predict(); }

	void predict_with_control(np_f64_ro u) override {
		// Size the temporary from the expected dim rather than u.shape(0) so
		// numpy_to_vec's size check is a meaningful backstop to the outer
		// PyKalmanFilter validation rather than a tautology.
		mtl::vec::dense_vector<T> uv(inner.ctrl_dim());
		numpy_to_vec(u, uv, "u");
		inner.predict(uv);
	}

	void update(np_f64_ro z) override {
		mtl::vec::dense_vector<T> zv(inner.meas_dim());
		numpy_to_vec(z, zv, "z");
		inner.update(zv);
	}
};

static std::unique_ptr<IKalmanImpl>
make_kalman_impl(mpdsp::ArithConfig config,
                 std::size_t state_dim, std::size_t meas_dim,
                 std::size_t ctrl_dim) {
	return make_impl_for_dtype<KalmanImpl, IKalmanImpl>(
		config, "KalmanFilter", state_dim, meas_dim, ctrl_dim);
}

} // namespace

class PyKalmanFilter {
public:
	PyKalmanFilter(std::size_t state_dim, std::size_t meas_dim,
	               std::size_t ctrl_dim, const std::string& dtype) {
		if (state_dim == 0) {
			throw std::invalid_argument(
				"KalmanFilter: state_dim must be > 0");
		}
		if (meas_dim == 0) {
			throw std::invalid_argument(
				"KalmanFilter: meas_dim must be > 0");
		}
		impl_ = make_kalman_impl(mpdsp::parse_config(dtype),
		                         state_dim, meas_dim, ctrl_dim);
		dtype_ = dtype;
	}

	std::size_t state_dim() const { return impl_->state_dim(); }
	std::size_t meas_dim() const { return impl_->meas_dim(); }
	std::size_t ctrl_dim() const { return impl_->ctrl_dim(); }

	np_f64_2d get_F() const { return impl_->get_F(); }
	np_f64_2d get_H() const { return impl_->get_H(); }
	np_f64_2d get_Q() const { return impl_->get_Q(); }
	np_f64_2d get_R() const { return impl_->get_R(); }
	np_f64_2d get_P() const { return impl_->get_P(); }
	np_f64_2d get_B() const { return impl_->get_B(); }
	np_f64    get_state() const { return impl_->get_state(); }

	void set_F(np_f64_2d_ro a) { impl_->set_F(a); }
	void set_H(np_f64_2d_ro a) { impl_->set_H(a); }
	void set_Q(np_f64_2d_ro a) { impl_->set_Q(a); }
	void set_R(np_f64_2d_ro a) { impl_->set_R(a); }
	void set_P(np_f64_2d_ro a) { impl_->set_P(a); }
	void set_B(np_f64_2d_ro a) {
		if (ctrl_dim() == 0) {
			throw std::invalid_argument(
				"KalmanFilter.B: filter was constructed with ctrl_dim=0; "
				"reconstruct with ctrl_dim>0 to set a control matrix");
		}
		impl_->set_B(a);
	}
	void set_state(np_f64_ro v) { impl_->set_state(v); }

	void predict() { impl_->predict(); }

	void predict_with_control(np_f64_ro u) {
		if (ctrl_dim() == 0) {
			throw std::invalid_argument(
				"KalmanFilter.predict(u): filter was constructed with "
				"ctrl_dim=0; pass ctrl_dim>0 to use control input");
		}
		if (u.shape(0) != ctrl_dim()) {
			throw std::invalid_argument(
				"KalmanFilter.predict(u): u must have length ctrl_dim");
		}
		impl_->predict_with_control(u);
	}

	void update(np_f64_ro z) {
		if (z.shape(0) != meas_dim()) {
			throw std::invalid_argument(
				"KalmanFilter.update(z): z must have length meas_dim");
		}
		impl_->update(z);
	}

	const std::string& dtype() const { return dtype_; }

private:
	std::unique_ptr<IKalmanImpl> impl_;
	std::string dtype_;
};

// ===========================================================================
// Adaptive filters (LMS / NLMS / RLS)
//
// All three share the same Python-visible shape:
//   .process(input, desired) -> (output, error) tuple
//   .process_block(inputs, desireds) -> (outputs, errors) tuple of arrays
//   .weights -> NumPy float64 array (read-only)
//   .num_taps, .dtype, .last_error (read-only)
//   .reset()
//
// Constructor parameters differ: LMS(num_taps, step_size),
// NLMS(num_taps, step_size, epsilon), RLS(num_taps, lambda, delta).
// ===========================================================================

namespace {

struct IAdaptiveFilter {
	virtual ~IAdaptiveFilter() = default;
	virtual std::size_t num_taps() const = 0;
	virtual double last_error() const = 0;
	virtual void process(double input, double desired,
	                     double& out, double& err) = 0;
	virtual void process_block(const double* xs, const double* ds,
	                           double* out_y, double* out_e,
	                           std::size_t n) = 0;
	virtual np_f64 weights() = 0;
	virtual void reset() = 0;
};

template <template <typename> class Filter, typename T>
struct AdaptiveFilterImpl : IAdaptiveFilter {
	Filter<T> inner;
	template <typename... Args>
	AdaptiveFilterImpl(Args&&... args) : inner(std::forward<Args>(args)...) {}

	std::size_t num_taps() const override { return inner.num_taps(); }
	double last_error() const override {
		return static_cast<double>(inner.last_error());
	}

	void process(double input, double desired,
	             double& out, double& err) override {
		T y = inner.process(static_cast<T>(input), static_cast<T>(desired));
		out = static_cast<double>(y);
		err = static_cast<double>(inner.last_error());
	}

	void process_block(const double* xs, const double* ds,
	                   double* out_y, double* out_e,
	                   std::size_t n) override {
		for (std::size_t i = 0; i < n; ++i) {
			T y = inner.process(static_cast<T>(xs[i]),
			                    static_cast<T>(ds[i]));
			out_y[i] = static_cast<double>(y);
			out_e[i] = static_cast<double>(inner.last_error());
		}
	}

	np_f64 weights() override {
		// Use non-const accessor where available; const otherwise. All three
		// filters expose at least a const weights() accessor.
		const auto& w = inner.weights();
		std::size_t n = w.size();
		double* out_ptr = nullptr;
		auto arr = make_f64_array(n, out_ptr);
		for (std::size_t i = 0; i < n; ++i) {
			out_ptr[i] = static_cast<double>(w[i]);
		}
		return arr;
	}

	void reset() override { inner.reset(); }
};

// Construct an adaptive-filter impl of the requested dtype, forwarding
// constructor arguments (num_taps + filter-specific params) to the T-typed
// inner. Mirrors make_impl_for_dtype but with the double-valued constructor
// arguments cast to T at the call site.
template <template <typename> class Filter, typename... DoubleArgs>
static std::unique_ptr<IAdaptiveFilter>
make_adaptive_impl(mpdsp::ArithConfig config, const char* cls,
                   std::size_t num_taps, DoubleArgs... args) {
	using mpdsp::ArithConfig;
	using mpdsp::cf24;
	using mpdsp::fx3224_t;
	using mpdsp::half_;
	using mpdsp::p32;
	using tiny_posit_t = sw::universal::posit<8, 2>;
	switch (config) {
	case ArithConfig::reference:
		return std::make_unique<AdaptiveFilterImpl<Filter, double>>(
			num_taps, static_cast<double>(args)...);
	case ArithConfig::gpu_baseline:
		return std::make_unique<AdaptiveFilterImpl<Filter, float>>(
			num_taps, static_cast<float>(args)...);
	case ArithConfig::ml_hw:
		return std::make_unique<AdaptiveFilterImpl<Filter, half_>>(
			num_taps, static_cast<half_>(args)...);
	case ArithConfig::cf24_config:
		return std::make_unique<AdaptiveFilterImpl<Filter, cf24>>(
			num_taps, static_cast<cf24>(args)...);
	case ArithConfig::half_config:
		return std::make_unique<AdaptiveFilterImpl<Filter, half_>>(
			num_taps, static_cast<half_>(args)...);
	case ArithConfig::posit_full:
		return std::make_unique<AdaptiveFilterImpl<Filter, p32>>(
			num_taps, static_cast<p32>(args)...);
	case ArithConfig::tiny_posit:
		return std::make_unique<AdaptiveFilterImpl<Filter, tiny_posit_t>>(
			num_taps, static_cast<tiny_posit_t>(args)...);
	// Sensor configs keep the filter state at double; the sample-path
	// quantization lives in project_dispatch / adc_dispatch rather than
	// the adaptive update itself.
	case ArithConfig::sensor_8bit:
	case ArithConfig::sensor_6bit:
		return std::make_unique<AdaptiveFilterImpl<Filter, double>>(
			num_taps, static_cast<double>(args)...);
	case ArithConfig::fpga_fixed:
		return std::make_unique<AdaptiveFilterImpl<Filter, fx3224_t>>(
			num_taps, static_cast<fx3224_t>(args)...);
	}
	throw std::invalid_argument(std::string(cls) + ": unsupported ArithConfig");
}

} // namespace

// Shared Python wrapper logic for all three adaptive filters. Differs only
// in how the underlying impl is constructed; delegate via CRTP-free helper.
class PyAdaptiveFilter {
public:
	std::size_t num_taps() const { return impl_->num_taps(); }
	double last_error() const { return impl_->last_error(); }

	std::tuple<double, double> process(double input, double desired) {
		double y, e;
		impl_->process(input, desired, y, e);
		return {y, e};
	}

	std::tuple<np_f64, np_f64> process_block(np_f64_ro inputs,
	                                         np_f64_ro desireds) {
		std::size_t n = inputs.shape(0);
		if (desireds.shape(0) != n) {
			throw std::invalid_argument(
				"process_block: inputs and desireds must have the same length");
		}
		double* out_y = nullptr;
		double* out_e = nullptr;
		auto y_arr = make_f64_array(n, out_y);
		auto e_arr = make_f64_array(n, out_e);
		const double* xs = inputs.data();
		const double* ds = desireds.data();
		{
			nb::gil_scoped_release release;
			impl_->process_block(xs, ds, out_y, out_e, n);
		}
		return {std::move(y_arr), std::move(e_arr)};
	}

	np_f64 weights() { return impl_->weights(); }
	void reset() { impl_->reset(); }
	const std::string& dtype() const { return dtype_; }

protected:
	std::unique_ptr<IAdaptiveFilter> impl_;
	std::string dtype_;
};

class PyLMSFilter : public PyAdaptiveFilter {
public:
	PyLMSFilter(std::size_t num_taps, double step_size,
	            const std::string& dtype) {
		if (num_taps == 0) {
			throw std::invalid_argument(
				"LMSFilter: num_taps must be > 0");
		}
		// !(x > 0.0) catches non-positive values and NaN uniformly. A
		// non-positive step size either never adapts (0) or silently diverges
		// (negative); NaN poisons the weights on the first sample.
		if (!(step_size > 0.0)) {
			throw std::invalid_argument(
				"LMSFilter: step_size must be positive");
		}
		impl_ = make_adaptive_impl<sw::dsp::LMSFilter>(
			mpdsp::parse_config(dtype), "LMSFilter", num_taps, step_size);
		dtype_ = dtype;
	}
};

class PyNLMSFilter : public PyAdaptiveFilter {
public:
	PyNLMSFilter(std::size_t num_taps, double step_size, double epsilon,
	             const std::string& dtype) {
		if (num_taps == 0) {
			throw std::invalid_argument(
				"NLMSFilter: num_taps must be > 0");
		}
		if (!(step_size > 0.0)) {
			throw std::invalid_argument(
				"NLMSFilter: step_size must be positive");
		}
		if (!(epsilon > 0.0)) {
			throw std::invalid_argument(
				"NLMSFilter: epsilon must be positive");
		}
		impl_ = make_adaptive_impl<sw::dsp::NLMSFilter>(
			mpdsp::parse_config(dtype), "NLMSFilter",
			num_taps, step_size, epsilon);
		dtype_ = dtype;
	}
};

class PyRLSFilter : public PyAdaptiveFilter {
public:
	PyRLSFilter(std::size_t num_taps, double forgetting_factor, double delta,
	            const std::string& dtype) {
		if (num_taps == 0) {
			throw std::invalid_argument(
				"RLSFilter: num_taps must be > 0");
		}
		if (!(forgetting_factor > 0.0) || forgetting_factor > 1.0) {
			throw std::invalid_argument(
				"RLSFilter: forgetting_factor must be in (0, 1]");
		}
		if (!(delta > 0.0)) {
			throw std::invalid_argument(
				"RLSFilter: delta must be positive");
		}
		impl_ = make_adaptive_impl<sw::dsp::RLSFilter>(
			mpdsp::parse_config(dtype), "RLSFilter",
			num_taps, forgetting_factor, delta);
		dtype_ = dtype;
	}
};

void bind_estimation(nb::module_& m) {
	nb::class_<PyKalmanFilter>(m, "KalmanFilter",
		"Linear Kalman filter for state estimation.\n\n"
		"Constructed with state_dim, meas_dim, and optional ctrl_dim. "
		"Initial P, F, Q, R are identity; H and B are zero. Set system "
		"matrices as NumPy 2D float64 arrays via the F, H, Q, R, P, B "
		"properties; read them back the same way. The Python wrapper "
		"always marshals through double; internal arithmetic uses the "
		"dtype chosen at construction.")
		.def(nb::init<std::size_t, std::size_t, std::size_t, const std::string&>(),
		     nb::arg("state_dim"), nb::arg("meas_dim"),
		     nb::arg("ctrl_dim") = std::size_t{0},
		     nb::arg("dtype") = "reference",
		     "Construct a linear Kalman filter.")
		.def_prop_ro("state_dim", &PyKalmanFilter::state_dim)
		.def_prop_ro("meas_dim",  &PyKalmanFilter::meas_dim)
		.def_prop_ro("ctrl_dim",  &PyKalmanFilter::ctrl_dim)
		// Every getter builds a fresh NumPy array with its own capsule, so
		// the default reference_internal policy doesn't apply — the returned
		// ndarray already has an owner. Use take_ownership to hand the buffer
		// off to Python cleanly.
		.def_prop_rw("F", &PyKalmanFilter::get_F, &PyKalmanFilter::set_F,
		             nb::rv_policy::take_ownership,
		             "State transition matrix (state_dim x state_dim).")
		.def_prop_rw("H", &PyKalmanFilter::get_H, &PyKalmanFilter::set_H,
		             nb::rv_policy::take_ownership,
		             "Observation matrix (meas_dim x state_dim).")
		.def_prop_rw("Q", &PyKalmanFilter::get_Q, &PyKalmanFilter::set_Q,
		             nb::rv_policy::take_ownership,
		             "Process-noise covariance (state_dim x state_dim).")
		.def_prop_rw("R", &PyKalmanFilter::get_R, &PyKalmanFilter::set_R,
		             nb::rv_policy::take_ownership,
		             "Measurement-noise covariance (meas_dim x meas_dim).")
		.def_prop_rw("P", &PyKalmanFilter::get_P, &PyKalmanFilter::set_P,
		             nb::rv_policy::take_ownership,
		             "Estimation-error covariance (state_dim x state_dim).")
		.def_prop_rw("B", &PyKalmanFilter::get_B, &PyKalmanFilter::set_B,
		             nb::rv_policy::take_ownership,
		             "Control-input matrix (state_dim x ctrl_dim).")
		.def_prop_rw("state", &PyKalmanFilter::get_state,
		                      &PyKalmanFilter::set_state,
		             nb::rv_policy::take_ownership,
		             "Current state estimate (length state_dim).")
		.def("predict", nb::overload_cast<>(&PyKalmanFilter::predict),
		     "Predict step without control input.")
		.def("predict", &PyKalmanFilter::predict_with_control,
		     nb::arg("u"),
		     "Predict step with a control vector of length ctrl_dim.")
		.def("update", &PyKalmanFilter::update,
		     nb::arg("z"),
		     "Update step with a measurement vector of length meas_dim.")
		.def_prop_ro("dtype", &PyKalmanFilter::dtype,
		             "Arithmetic configuration selected at construction.");

	// Shared docstring fragment for the three adaptive filters.
	constexpr const char* ADAPTIVE_PROCESS_DOC =
		"Process one sample with adaptation. Returns a (output, error) tuple "
		"where output is y[n] = w^T x[n] and error is d[n] - y[n].";
	constexpr const char* ADAPTIVE_BLOCK_DOC =
		"Process two equal-length NumPy float64 signals (input, desired) and "
		"return a (outputs, errors) tuple of float64 arrays. The per-sample "
		"loop releases the GIL.";
	constexpr const char* ADAPTIVE_WEIGHTS_DOC =
		"Current tap weights as a 1D NumPy float64 array (read-only copy).";

	nb::class_<PyLMSFilter>(m, "LMSFilter",
		"Least-mean-squares adaptive FIR filter.\n\n"
		"At each step, runs the FIR cascade w^T x[n], measures error against "
		"the desired signal, and updates weights by mu * error * x[n].")
		.def(nb::init<std::size_t, double, const std::string&>(),
		     nb::arg("num_taps"), nb::arg("step_size"),
		     nb::arg("dtype") = "reference",
		     "Construct an LMS adaptive filter.")
		.def_prop_ro("num_taps", &PyLMSFilter::num_taps)
		.def_prop_ro("last_error", &PyLMSFilter::last_error,
		             "Error residual from the most recent process() call.")
		.def_prop_ro("weights", &PyLMSFilter::weights,
		             nb::rv_policy::take_ownership,
		             ADAPTIVE_WEIGHTS_DOC)
		.def_prop_ro("dtype", &PyLMSFilter::dtype,
		             "Arithmetic configuration selected at construction.")
		.def("process", &PyLMSFilter::process,
		     nb::arg("input"), nb::arg("desired"),
		     ADAPTIVE_PROCESS_DOC)
		.def("process_block", &PyLMSFilter::process_block,
		     nb::arg("inputs"), nb::arg("desireds"),
		     ADAPTIVE_BLOCK_DOC)
		.def("reset", &PyLMSFilter::reset,
		     "Zero the weights and delay line.");

	nb::class_<PyNLMSFilter>(m, "NLMSFilter",
		"Normalized LMS adaptive filter — scales the step size by input "
		"power to stay stable across varying signal levels.")
		.def(nb::init<std::size_t, double, double, const std::string&>(),
		     nb::arg("num_taps"), nb::arg("step_size"),
		     nb::arg("epsilon") = 1e-6,
		     nb::arg("dtype") = "reference",
		     "Construct an NLMS filter. epsilon regularizes the normalization "
		     "when input power is near zero.")
		.def_prop_ro("num_taps", &PyNLMSFilter::num_taps)
		.def_prop_ro("last_error", &PyNLMSFilter::last_error,
		             "Error residual from the most recent process() call.")
		.def_prop_ro("weights", &PyNLMSFilter::weights,
		             nb::rv_policy::take_ownership,
		             ADAPTIVE_WEIGHTS_DOC)
		.def_prop_ro("dtype", &PyNLMSFilter::dtype,
		             "Arithmetic configuration selected at construction.")
		.def("process", &PyNLMSFilter::process,
		     nb::arg("input"), nb::arg("desired"),
		     ADAPTIVE_PROCESS_DOC)
		.def("process_block", &PyNLMSFilter::process_block,
		     nb::arg("inputs"), nb::arg("desireds"),
		     ADAPTIVE_BLOCK_DOC)
		.def("reset", &PyNLMSFilter::reset,
		     "Zero the weights and delay line.");

	nb::class_<PyRLSFilter>(m, "RLSFilter",
		"Recursive least-squares adaptive filter. Faster convergence than "
		"LMS at O(N^2) per sample cost. forgetting_factor in (0, 1] "
		"controls tracking of non-stationary signals (1.0 = no forgetting).")
		.def(nb::init<std::size_t, double, double, const std::string&>(),
		     nb::arg("num_taps"),
		     nb::arg("forgetting_factor") = 0.99,
		     nb::arg("delta") = 1000.0,
		     nb::arg("dtype") = "reference",
		     "Construct an RLS filter. delta is the initial P diagonal "
		     "(larger = faster initial convergence).")
		.def_prop_ro("num_taps", &PyRLSFilter::num_taps)
		.def_prop_ro("last_error", &PyRLSFilter::last_error,
		             "Error residual from the most recent process() call.")
		.def_prop_ro("weights", &PyRLSFilter::weights,
		             nb::rv_policy::take_ownership,
		             ADAPTIVE_WEIGHTS_DOC)
		.def_prop_ro("dtype", &PyRLSFilter::dtype,
		             "Arithmetic configuration selected at construction.")
		.def("process", &PyRLSFilter::process,
		     nb::arg("input"), nb::arg("desired"),
		     ADAPTIVE_PROCESS_DOC)
		.def("process_block", &PyRLSFilter::process_block,
		     nb::arg("inputs"), nb::arg("desireds"),
		     ADAPTIVE_BLOCK_DOC)
		.def("reset", &PyRLSFilter::reset,
		     "Zero the weights, delay line, and reset P to delta*I.");
}
