// estimation_bindings.cpp: state-estimation bindings (Kalman, LMS, RLS).
//
// Phase 5 stateful pattern (per #21 / #22): dtype fixed at construction,
// internal type-erased interface per class, NumPy float64 I/O at the Python
// boundary. New for this file: 2D NumPy matrix marshalling for the Kalman
// system matrices F, H, Q, R, P, B.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <sw/dsp/estimation/kalman.hpp>

#include "types.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

using np_f64       = nb::ndarray<nb::numpy, double>;
using np_f64_ro    = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
using np_f64_2d    = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
// c_contig on 2D inputs matches MTL's default row-major orientation, so
// numpy_to_mat can walk both buffers with a shared linear index.
using np_f64_2d_ro = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>;

// -- Shared array builders ---------------------------------------------------

static np_f64 make_f64_array(std::size_t n, double*& out_ptr) {
	auto buf = std::unique_ptr<double[]>(new double[n]);
	double* data = buf.get();
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	buf.release();
	out_ptr = data;
	std::size_t shape[1] = { n };
	return np_f64(data, 1, shape, owner);
}

static np_f64_2d make_f64_2d_array(std::size_t rows, std::size_t cols,
                                   double*& out_ptr) {
	std::size_t total = rows * cols;
	auto buf = std::unique_ptr<double[]>(new double[total]);
	double* data = buf.get();
	nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
	buf.release();
	out_ptr = data;
	std::size_t shape[2] = { rows, cols };
	return np_f64_2d(data, 2, shape, owner);
}

// -- dense2D <-> NumPy marshalling -------------------------------------------
//
// MTL's dense2D defaults to row-major orientation (same as NumPy C-order),
// but dense2D exposes only operator()(r, c) — not data() with a guaranteed
// row-major stride — so walk (r, c) element by element. Casts convert
// T<->double at the boundary.

template <typename T>
static np_f64_2d mat_to_numpy(const mtl::mat::dense2D<T>& m) {
	std::size_t rows = m.num_rows();
	std::size_t cols = m.num_cols();
	double* out_ptr = nullptr;
	auto arr = make_f64_2d_array(rows, cols, out_ptr);
	for (std::size_t r = 0; r < rows; ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			out_ptr[r * cols + c] = static_cast<double>(m(r, c));
		}
	}
	return arr;
}

template <typename T>
static void numpy_to_mat(np_f64_2d_ro src, mtl::mat::dense2D<T>& dst,
                         const char* name) {
	if (src.shape(0) != dst.num_rows() || src.shape(1) != dst.num_cols()) {
		throw std::invalid_argument(
			std::string(name) + ": shape mismatch (expected " +
			std::to_string(dst.num_rows()) + "x" +
			std::to_string(dst.num_cols()) + ", got " +
			std::to_string(src.shape(0)) + "x" +
			std::to_string(src.shape(1)) + ")");
	}
	std::size_t cols = dst.num_cols();
	const double* data = src.data();
	for (std::size_t r = 0; r < dst.num_rows(); ++r) {
		for (std::size_t c = 0; c < cols; ++c) {
			dst(r, c) = static_cast<T>(data[r * cols + c]);
		}
	}
}

template <typename T>
static np_f64 vec_to_numpy(const mtl::vec::dense_vector<T>& v) {
	std::size_t n = v.size();
	double* out_ptr = nullptr;
	auto arr = make_f64_array(n, out_ptr);
	for (std::size_t i = 0; i < n; ++i) {
		out_ptr[i] = static_cast<double>(v[i]);
	}
	return arr;
}

template <typename T>
static void numpy_to_vec(np_f64_ro src, mtl::vec::dense_vector<T>& dst,
                         const char* name) {
	if (src.shape(0) != dst.size()) {
		throw std::invalid_argument(
			std::string(name) + ": size mismatch (expected " +
			std::to_string(dst.size()) + ", got " +
			std::to_string(src.shape(0)) + ")");
	}
	const double* data = src.data();
	for (std::size_t i = 0; i < dst.size(); ++i) {
		dst[i] = static_cast<T>(data[i]);
	}
}

// -- Shared dtype dispatcher (mirrors conditioning_bindings.cpp) -------------

template <template<class> class Impl, class Base, class... Args>
static std::unique_ptr<Base>
make_impl_for_dtype(mpdsp::ArithConfig config, const char* cls, Args&&... args) {
	using mpdsp::ArithConfig;
	using mpdsp::cf24;
	using mpdsp::half_;
	using mpdsp::p32;
	using tiny_posit_t = sw::universal::posit<8, 2>;
	switch (config) {
	case ArithConfig::reference:    return std::make_unique<Impl<double>>(std::forward<Args>(args)...);
	case ArithConfig::gpu_baseline: return std::make_unique<Impl<float>>(std::forward<Args>(args)...);
	case ArithConfig::ml_hw:        return std::make_unique<Impl<half_>>(std::forward<Args>(args)...);
	case ArithConfig::cf24_config:  return std::make_unique<Impl<cf24>>(std::forward<Args>(args)...);
	case ArithConfig::half_config:  return std::make_unique<Impl<half_>>(std::forward<Args>(args)...);
	case ArithConfig::posit_full:   return std::make_unique<Impl<p32>>(std::forward<Args>(args)...);
	case ArithConfig::tiny_posit:   return std::make_unique<Impl<tiny_posit_t>>(std::forward<Args>(args)...);
	}
	throw std::invalid_argument(std::string(cls) + ": unsupported ArithConfig");
}

// ===========================================================================
// KalmanFilter
// ===========================================================================

struct IKalmanImpl {
	virtual ~IKalmanImpl() = default;

	virtual std::size_t state_dim() const = 0;
	virtual std::size_t meas_dim() const = 0;
	virtual std::size_t ctrl_dim() const = 0;

	// Getters are intentionally non-const: sw::dsp::KalmanFilter<T>::B() has
	// only a non-const overload upstream (the other five matrices have
	// const overloads — this looks like an upstream omission). Reading
	// Python matrix properties returns a fresh NumPy copy either way, so
	// non-const virtuals are semantically fine here.
	virtual np_f64_2d get_F() = 0;
	virtual np_f64_2d get_H() = 0;
	virtual np_f64_2d get_Q() = 0;
	virtual np_f64_2d get_R() = 0;
	virtual np_f64_2d get_P() = 0;
	virtual np_f64_2d get_B() = 0;
	virtual np_f64    get_state() = 0;

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

	np_f64_2d get_F() override { return mat_to_numpy(inner.F()); }
	np_f64_2d get_H() override { return mat_to_numpy(inner.H()); }
	np_f64_2d get_Q() override { return mat_to_numpy(inner.Q()); }
	np_f64_2d get_R() override { return mat_to_numpy(inner.R()); }
	np_f64_2d get_P() override { return mat_to_numpy(inner.P()); }
	np_f64_2d get_B() override { return mat_to_numpy(inner.B()); }
	np_f64    get_state() override { return vec_to_numpy(inner.state()); }

	void set_F(np_f64_2d_ro a) override { numpy_to_mat(a, inner.F(), "F"); }
	void set_H(np_f64_2d_ro a) override { numpy_to_mat(a, inner.H(), "H"); }
	void set_Q(np_f64_2d_ro a) override { numpy_to_mat(a, inner.Q(), "Q"); }
	void set_R(np_f64_2d_ro a) override { numpy_to_mat(a, inner.R(), "R"); }
	void set_P(np_f64_2d_ro a) override { numpy_to_mat(a, inner.P(), "P"); }
	void set_B(np_f64_2d_ro a) override { numpy_to_mat(a, inner.B(), "B"); }
	void set_state(np_f64_ro v) override { numpy_to_vec(v, inner.state(), "state"); }

	void predict() override { inner.predict(); }

	void predict_with_control(np_f64_ro u) override {
		mtl::vec::dense_vector<T> uv(u.shape(0));
		numpy_to_vec(u, uv, "u");
		inner.predict(uv);
	}

	void update(np_f64_ro z) override {
		mtl::vec::dense_vector<T> zv(z.shape(0));
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

	np_f64_2d get_F() { return impl_->get_F(); }
	np_f64_2d get_H() { return impl_->get_H(); }
	np_f64_2d get_Q() { return impl_->get_Q(); }
	np_f64_2d get_R() { return impl_->get_R(); }
	np_f64_2d get_P() { return impl_->get_P(); }
	np_f64_2d get_B() { return impl_->get_B(); }
	np_f64    get_state() { return impl_->get_state(); }

	void set_F(np_f64_2d_ro a) { impl_->set_F(a); }
	void set_H(np_f64_2d_ro a) { impl_->set_H(a); }
	void set_Q(np_f64_2d_ro a) { impl_->set_Q(a); }
	void set_R(np_f64_2d_ro a) { impl_->set_R(a); }
	void set_P(np_f64_2d_ro a) { impl_->set_P(a); }
	void set_B(np_f64_2d_ro a) { impl_->set_B(a); }
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
}
