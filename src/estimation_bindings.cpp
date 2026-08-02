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

#include <sw/dsp/estimation/ekf.hpp>
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
// ExtendedKalmanFilter (EKF)
//
// Nonlinear generalization of KalmanFilter. Python users supply four
// callbacks — f(x), F(x), h(x), H(x) — that return numpy arrays. Each
// per-dtype impl wraps those callbacks in std::function closures that
// marshal dense_vector<T> <-> NumPy at the boundary.
//
// The Python callback objects (nb::callable) live inside the closure via
// capture-by-value; nanobind manages their destruction under the GIL when
// the EKFImpl is torn down (which happens under Python GC, GIL held).
//
// GIL: predict()/update() are bound normally (no gil_scoped_release), so
// the GIL is held when C++ invokes the closures — safe to call Python from
// within.
// ===========================================================================

namespace {

// Convert a Python callback's return value into a dense_vector<T> of the
// expected length. The callback is required to return a 1D float64 ndarray;
// lists / other convertibles are rejected loudly (nb::cast throws).
template <typename T>
static mtl::vec::dense_vector<T>
callback_result_to_vec(nb::handle result, std::size_t expected,
                        const char* name) {
	auto arr = nb::cast<np_f64_ro>(result);
	if (arr.shape(0) != expected) {
		throw std::runtime_error(std::string(name) +
			" returned vector of length " + std::to_string(arr.shape(0)) +
			" but expected " + std::to_string(expected));
	}
	mtl::vec::dense_vector<T> out(expected);
	for (std::size_t i = 0; i < expected; ++i) {
		out[i] = static_cast<T>(arr.data()[i]);
	}
	return out;
}

// Convert a Python callback's return value into a dense2D<T> of the expected
// shape. Callback must return a 2D float64 c-contiguous ndarray.
template <typename T>
static mtl::mat::dense2D<T>
callback_result_to_mat(nb::handle result,
                        std::size_t rows, std::size_t cols,
                        const char* name) {
	auto arr = nb::cast<np_f64_2d_ro>(result);
	if (arr.shape(0) != rows || arr.shape(1) != cols) {
		throw std::runtime_error(std::string(name) +
			" returned matrix of shape " + std::to_string(arr.shape(0)) +
			"x" + std::to_string(arr.shape(1)) +
			" but expected " + std::to_string(rows) + "x" + std::to_string(cols));
	}
	mtl::mat::dense2D<T> out(rows, cols);
	const double* data = arr.data();
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			out(r, c) = static_cast<T>(data[r * cols + c]);
		}
	}
	return out;
}

struct IEKFImpl {
	virtual ~IEKFImpl() = default;

	virtual std::size_t state_dim() const = 0;
	virtual std::size_t meas_dim()  const = 0;

	virtual np_f64_2d get_Q() const = 0;
	virtual np_f64_2d get_R() const = 0;
	virtual np_f64_2d get_P() const = 0;
	virtual np_f64    get_state() const = 0;

	virtual void set_Q(np_f64_2d_ro a) = 0;
	virtual void set_R(np_f64_2d_ro a) = 0;
	virtual void set_P(np_f64_2d_ro a) = 0;
	virtual void set_state(np_f64_ro v) = 0;

	virtual void set_state_function(nb::callable f, nb::callable F) = 0;
	virtual void set_observation_function(nb::callable h, nb::callable H) = 0;
	virtual bool state_func_set() const = 0;
	virtual bool obs_func_set()   const = 0;

	virtual void predict() = 0;
	virtual void update(np_f64_ro z) = 0;
};

template <typename T>
struct EKFImpl : IEKFImpl {
	sw::dsp::ExtendedKalmanFilter<T> inner;
	bool state_func_set_ = false;
	bool obs_func_set_   = false;

	EKFImpl(std::size_t s, std::size_t m) : inner(s, m) {}

	std::size_t state_dim() const override { return inner.state_dim(); }
	std::size_t meas_dim()  const override { return inner.meas_dim();  }

	np_f64_2d get_Q() const override { return mat_to_numpy(inner.Q()); }
	np_f64_2d get_R() const override { return mat_to_numpy(inner.R()); }
	np_f64_2d get_P() const override { return mat_to_numpy(inner.P()); }
	np_f64    get_state() const override { return vec_to_numpy(inner.state()); }

	void set_Q(np_f64_2d_ro a) override { numpy_to_mat(a, inner.Q(), "Q"); }
	void set_R(np_f64_2d_ro a) override { numpy_to_mat(a, inner.R(), "R"); }
	void set_P(np_f64_2d_ro a) override { numpy_to_mat(a, inner.P(), "P"); }
	void set_state(np_f64_ro v) override { numpy_to_vec(v, inner.state(), "state"); }

	void set_state_function(nb::callable f, nb::callable F) override {
		const std::size_t sd = inner.state_dim();
		// Capture callbacks by value — copies keep the underlying Python
		// object alive for the lifetime of the closures.
		inner.set_state_function(
			[f, sd](const mtl::vec::dense_vector<T>& x)
			      -> mtl::vec::dense_vector<T> {
				auto x_arr = vec_to_numpy(x);
				nb::object result = f(x_arr);
				return callback_result_to_vec<T>(result, sd,
				    "ExtendedKalmanFilter state function f(x)");
			},
			[F, sd](const mtl::vec::dense_vector<T>& x)
			      -> mtl::mat::dense2D<T> {
				auto x_arr = vec_to_numpy(x);
				nb::object result = F(x_arr);
				return callback_result_to_mat<T>(result, sd, sd,
				    "ExtendedKalmanFilter state Jacobian F(x)");
			});
		state_func_set_ = true;
	}

	void set_observation_function(nb::callable h, nb::callable H) override {
		const std::size_t sd = inner.state_dim();
		const std::size_t md = inner.meas_dim();
		inner.set_observation_function(
			[h, md](const mtl::vec::dense_vector<T>& x)
			      -> mtl::vec::dense_vector<T> {
				auto x_arr = vec_to_numpy(x);
				nb::object result = h(x_arr);
				return callback_result_to_vec<T>(result, md,
				    "ExtendedKalmanFilter observation function h(x)");
			},
			[H, sd, md](const mtl::vec::dense_vector<T>& x)
			      -> mtl::mat::dense2D<T> {
				auto x_arr = vec_to_numpy(x);
				nb::object result = H(x_arr);
				return callback_result_to_mat<T>(result, md, sd,
				    "ExtendedKalmanFilter observation Jacobian H(x)");
			});
		obs_func_set_ = true;
	}

	bool state_func_set() const override { return state_func_set_; }
	bool obs_func_set()   const override { return obs_func_set_;   }

	void predict() override { inner.predict(); }

	void update(np_f64_ro z) override {
		mtl::vec::dense_vector<T> zv(inner.meas_dim());
		numpy_to_vec(z, zv, "z");
		inner.update(zv);
	}
};

static std::unique_ptr<IEKFImpl>
make_ekf_impl(mpdsp::ArithConfig config,
              std::size_t state_dim, std::size_t meas_dim) {
	return make_impl_for_dtype<EKFImpl, IEKFImpl>(
		config, "ExtendedKalmanFilter", state_dim, meas_dim);
}

} // namespace

class PyExtendedKalmanFilter {
public:
	PyExtendedKalmanFilter(std::size_t state_dim, std::size_t meas_dim,
	                       const std::string& dtype) {
		if (state_dim == 0)
			throw std::invalid_argument(
				"ExtendedKalmanFilter: state_dim must be > 0");
		if (meas_dim == 0)
			throw std::invalid_argument(
				"ExtendedKalmanFilter: meas_dim must be > 0");
		impl_ = make_ekf_impl(mpdsp::parse_config(dtype), state_dim, meas_dim);
		dtype_ = dtype;
	}

	std::size_t state_dim() const { return impl_->state_dim(); }
	std::size_t meas_dim()  const { return impl_->meas_dim(); }

	np_f64_2d get_Q() const { return impl_->get_Q(); }
	np_f64_2d get_R() const { return impl_->get_R(); }
	np_f64_2d get_P() const { return impl_->get_P(); }
	np_f64    get_state() const { return impl_->get_state(); }

	void set_Q(np_f64_2d_ro a) { impl_->set_Q(a); }
	void set_R(np_f64_2d_ro a) { impl_->set_R(a); }
	void set_P(np_f64_2d_ro a) { impl_->set_P(a); }
	void set_state(np_f64_ro v) { impl_->set_state(v); }

	void set_state_function(nb::callable f, nb::callable F) {
		impl_->set_state_function(f, F);
	}
	void set_observation_function(nb::callable h, nb::callable H) {
		impl_->set_observation_function(h, H);
	}

	void predict() {
		if (!impl_->state_func_set()) {
			throw std::runtime_error(
				"ExtendedKalmanFilter.predict: state function not set "
				"— call set_state_function(f, F) first");
		}
		impl_->predict();
	}

	void update(np_f64_ro z) {
		if (!impl_->obs_func_set()) {
			throw std::runtime_error(
				"ExtendedKalmanFilter.update: observation function not set "
				"— call set_observation_function(h, H) first");
		}
		if (z.shape(0) != meas_dim()) {
			throw std::invalid_argument(
				"ExtendedKalmanFilter.update(z): z must have length meas_dim");
		}
		impl_->update(z);
	}

	const std::string& dtype() const { return dtype_; }

private:
	std::unique_ptr<IEKFImpl> impl_;
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
	using mpdsp::p8_0;
	using mpdsp::p8_1;
	using mpdsp::p8_2;
	using mpdsp::p16_0;
	using mpdsp::p16_1;
	using mpdsp::p16_2;
	using mpdsp::p32_0;
	using mpdsp::p32_1;
	using mpdsp::p32_2;
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
	// Posit taxonomy grid (#81) — single-type, so the adaptive update runs
	// entirely at posit<N,es>. No state-vs-sample split to worry about;
	// none of the "reject these" reasoning for sensor_*/fpga_fixed applies.
	// posit_8_2 also covers the tiny_posit alias.
	case ArithConfig::posit_8_0:
		return std::make_unique<AdaptiveFilterImpl<Filter, p8_0>>(
			num_taps, static_cast<p8_0>(args)...);
	case ArithConfig::posit_8_1:
		return std::make_unique<AdaptiveFilterImpl<Filter, p8_1>>(
			num_taps, static_cast<p8_1>(args)...);
	case ArithConfig::posit_8_2:
		return std::make_unique<AdaptiveFilterImpl<Filter, p8_2>>(
			num_taps, static_cast<p8_2>(args)...);
	case ArithConfig::posit_16_0:
		return std::make_unique<AdaptiveFilterImpl<Filter, p16_0>>(
			num_taps, static_cast<p16_0>(args)...);
	case ArithConfig::posit_16_1:
		return std::make_unique<AdaptiveFilterImpl<Filter, p16_1>>(
			num_taps, static_cast<p16_1>(args)...);
	case ArithConfig::posit_16_2:
		return std::make_unique<AdaptiveFilterImpl<Filter, p16_2>>(
			num_taps, static_cast<p16_2>(args)...);
	case ArithConfig::posit_32_0:
		return std::make_unique<AdaptiveFilterImpl<Filter, p32_0>>(
			num_taps, static_cast<p32_0>(args)...);
	case ArithConfig::posit_32_1:
		return std::make_unique<AdaptiveFilterImpl<Filter, p32_1>>(
			num_taps, static_cast<p32_1>(args)...);
	case ArithConfig::posit_32_2:
		return std::make_unique<AdaptiveFilterImpl<Filter, p32_2>>(
			num_taps, static_cast<p32_2>(args)...);
	// Reject the sensor/FPGA configs here. Adaptive filters only accept a
	// single scalar T, so we can't independently honour both a double
	// state and an integer<N> / fixpnt<16,12> sample path — either the
	// sensor_* configs would silently collapse to `reference` (with no
	// quantization visible), or fpga_fixed would claim a 16-bit sample
	// path while actually running the adaptive update at fixpnt<32,24>
	// and breaking dtype-to-dtype comparisons. Surface the limitation
	// instead of fabricating a misleading answer; a future wrapper can
	// add explicit pre/post sample quantization around the update if a
	// real use case appears.
	case ArithConfig::sensor_8bit:
	case ArithConfig::sensor_6bit:
	case ArithConfig::fpga_fixed:
		throw std::invalid_argument(
			std::string(cls) +
			": dtype not supported by adaptive filters — sensor_* and "
			"fpga_fixed mix state and sample precisions, which the "
			"single-T adaptive update can't honour. Use reference, "
			"gpu_baseline, ml_hw, posit_full, cf24, half, or any "
			"posit_N_es grid cell.");
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

	// -----------------------------------------------------------------------
	// ExtendedKalmanFilter — nonlinear Kalman via analytical Jacobians.
	// -----------------------------------------------------------------------
	nb::class_<PyExtendedKalmanFilter>(m, "ExtendedKalmanFilter",
			"Nonlinear Kalman filter that linearizes the state-transition "
			"f(x) and observation h(x) via their Jacobians F(x), H(x) at "
			"each step. Users supply FOUR Python callbacks — f, F, h, H — "
			"that take a 1D NumPy state vector and return either a 1D "
			"vector (for f, h) or a 2D matrix (for F, H).\n\n"
			"predict() advances the state through f and propagates the "
			"covariance with F; update(z) computes the Kalman gain from H "
			"and applies the measurement innovation.\n\n"
			"Q, R, P, and state are read/write NumPy arrays sharing the "
			"KalmanFilter accessor pattern. No F/H/B properties — those "
			"matrices are computed per-step by the user's Jacobian "
			"callbacks.")
		.def(nb::init<std::size_t, std::size_t, const std::string&>(),
		     nb::arg("state_dim"), nb::arg("meas_dim"),
		     nb::arg("dtype") = "reference",
		     "Construct an EKF. Both dimensions must be > 0. Callbacks "
		     "must be set via set_state_function() and "
		     "set_observation_function() before calling predict()/update().")
		.def("set_state_function", &PyExtendedKalmanFilter::set_state_function,
		     nb::arg("f"), nb::arg("F"),
		     "Register the nonlinear state transition f(x) -> vector[state_dim] "
		     "and its Jacobian F(x) -> matrix[state_dim, state_dim]. Both must "
		     "be Python callables returning float64 ndarrays.")
		.def("set_observation_function",
		     &PyExtendedKalmanFilter::set_observation_function,
		     nb::arg("h"), nb::arg("H"),
		     "Register the nonlinear observation h(x) -> vector[meas_dim] and "
		     "its Jacobian H(x) -> matrix[meas_dim, state_dim].")
		.def_prop_ro("state_dim", &PyExtendedKalmanFilter::state_dim)
		.def_prop_ro("meas_dim",  &PyExtendedKalmanFilter::meas_dim)
		.def_prop_rw("Q", &PyExtendedKalmanFilter::get_Q,
		                  &PyExtendedKalmanFilter::set_Q,
		             nb::rv_policy::take_ownership,
		             "Process-noise covariance (state_dim x state_dim).")
		.def_prop_rw("R", &PyExtendedKalmanFilter::get_R,
		                  &PyExtendedKalmanFilter::set_R,
		             nb::rv_policy::take_ownership,
		             "Measurement-noise covariance (meas_dim x meas_dim).")
		.def_prop_rw("P", &PyExtendedKalmanFilter::get_P,
		                  &PyExtendedKalmanFilter::set_P,
		             nb::rv_policy::take_ownership,
		             "State-estimation covariance (state_dim x state_dim). "
		             "Initialized to the identity; overwrite for informative "
		             "priors.")
		.def_prop_rw("state", &PyExtendedKalmanFilter::get_state,
		                      &PyExtendedKalmanFilter::set_state,
		             nb::rv_policy::take_ownership,
		             "Current state estimate (length state_dim).")
		.def("predict", &PyExtendedKalmanFilter::predict,
		     "Propagate the state through f and the covariance through F. "
		     "Raises if the state function pair hasn't been set.")
		.def("update", &PyExtendedKalmanFilter::update, nb::arg("z"),
		     "Apply a measurement of length meas_dim. Raises if the "
		     "observation function pair hasn't been set.")
		.def_prop_ro("dtype", &PyExtendedKalmanFilter::dtype,
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
