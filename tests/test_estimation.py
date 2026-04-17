"""Tests for state-estimation bindings: KalmanFilter, LMSFilter, NLMSFilter, RLSFilter."""

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


class TestKalmanConstruction:
    def test_default_dtype(self):
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)
        assert kf.dtype == "reference"
        assert kf.state_dim == 2
        assert kf.meas_dim == 1
        assert kf.ctrl_dim == 0

    def test_with_control(self):
        kf = mpdsp.KalmanFilter(state_dim=3, meas_dim=2, ctrl_dim=1)
        assert kf.ctrl_dim == 1

    def test_zero_state_dim_raises(self):
        with pytest.raises(ValueError):
            mpdsp.KalmanFilter(state_dim=0, meas_dim=1)

    def test_zero_meas_dim_raises(self):
        with pytest.raises(ValueError):
            mpdsp.KalmanFilter(state_dim=2, meas_dim=0)

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError):
            mpdsp.KalmanFilter(state_dim=2, meas_dim=1, dtype="not_a_dtype")


class TestKalmanDefaults:
    def test_F_defaults_to_identity(self):
        kf = mpdsp.KalmanFilter(state_dim=3, meas_dim=1)
        np.testing.assert_array_equal(kf.F, np.eye(3))

    def test_Q_and_P_default_to_identity(self):
        kf = mpdsp.KalmanFilter(state_dim=3, meas_dim=1)
        np.testing.assert_array_equal(kf.Q, np.eye(3))
        np.testing.assert_array_equal(kf.P, np.eye(3))

    def test_R_defaults_to_identity(self):
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=2)
        np.testing.assert_array_equal(kf.R, np.eye(2))

    def test_state_defaults_to_zero(self):
        kf = mpdsp.KalmanFilter(state_dim=3, meas_dim=1)
        np.testing.assert_array_equal(kf.state, np.zeros(3))

    def test_matrix_shapes(self):
        kf = mpdsp.KalmanFilter(state_dim=3, meas_dim=2, ctrl_dim=1)
        assert kf.F.shape == (3, 3)
        assert kf.H.shape == (2, 3)
        assert kf.Q.shape == (3, 3)
        assert kf.R.shape == (2, 2)
        assert kf.P.shape == (3, 3)
        assert kf.B.shape == (3, 1)
        assert kf.state.shape == (3,)


class TestKalmanMatrixIO:
    def test_set_and_read_F(self):
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)
        new_F = np.array([[1.0, 0.5], [0.0, 1.0]])
        kf.F = new_F
        np.testing.assert_array_equal(kf.F, new_F)

    def test_set_and_read_H(self):
        kf = mpdsp.KalmanFilter(state_dim=3, meas_dim=2)
        new_H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        kf.H = new_H
        np.testing.assert_array_equal(kf.H, new_H)

    def test_set_state(self):
        kf = mpdsp.KalmanFilter(state_dim=3, meas_dim=1)
        kf.state = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(kf.state, [1.0, 2.0, 3.0])

    def test_wrong_shape_matrix_raises(self):
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)
        with pytest.raises(ValueError):
            kf.F = np.eye(3)  # 3x3 instead of 2x2

    def test_wrong_length_state_raises(self):
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)
        with pytest.raises(ValueError):
            kf.state = np.array([1.0, 2.0, 3.0])  # length 3 instead of 2

    def test_round_trip_preserves_values_at_reference_dtype(self):
        """At reference (double) precision the matrix I/O must be exact."""
        kf = mpdsp.KalmanFilter(state_dim=3, meas_dim=2)
        rng = np.random.default_rng(42)
        # Symmetric positive-definite random covariance
        A = rng.standard_normal((3, 3))
        cov = A @ A.T + np.eye(3)
        kf.P = cov
        np.testing.assert_array_equal(kf.P, cov)


class TestKalmanPredictUpdate:
    def _constant_velocity(self, dt=1.0):
        """Build a 1D constant-velocity filter: state = [position, velocity]."""
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)
        kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.Q = np.eye(2) * 0.01
        kf.R = np.array([[0.1]])
        kf.P = np.eye(2)
        kf.state = np.array([0.0, 0.0])
        return kf

    def test_predict_without_control(self):
        """predict() with F=I should not change the state."""
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)  # F default = I
        kf.state = np.array([1.5, -2.5])
        kf.predict()
        np.testing.assert_allclose(kf.state, [1.5, -2.5])

    def test_predict_advances_state(self):
        """With the constant-velocity F, predict should advance position by velocity."""
        kf = self._constant_velocity()
        kf.state = np.array([1.0, 0.5])  # position=1, velocity=0.5
        kf.predict()
        np.testing.assert_allclose(kf.state, [1.5, 0.5], rtol=1e-10)

    def test_update_pulls_state_toward_measurement(self):
        """A measurement at z=10 should pull position toward 10."""
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)
        kf.H = np.array([[1.0, 0.0]])
        kf.R = np.array([[0.01]])  # high-confidence measurement
        kf.P = np.eye(2) * 10.0     # low-confidence initial state
        kf.state = np.array([0.0, 0.0])

        kf.update(np.array([10.0]))
        # Position should move substantially toward 10
        assert kf.state[0] > 5.0

    def test_tracks_constant_velocity_signal(self):
        """Feeding noisy measurements of a constant-velocity trajectory, the
        filter must recover both position and velocity."""
        kf = self._constant_velocity()
        rng = np.random.default_rng(42)
        true_vel = 0.5
        ests = []
        for t in range(100):
            kf.predict()
            true_pos = true_vel * t
            z = np.array([true_pos + rng.normal(0, 0.3)])
            kf.update(z)
            ests.append(kf.state.copy())
        ests = np.array(ests)
        # RMSE on position vs truth must be small relative to measurement noise.
        truths = true_vel * np.arange(100)
        rmse_pos = np.sqrt(np.mean((ests[:, 0] - truths) ** 2))
        assert rmse_pos < 0.3
        # Estimated velocity converges near the true value.
        assert abs(ests[-1, 1] - true_vel) < 0.15

    def test_wrong_length_measurement_raises(self):
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)
        with pytest.raises(ValueError):
            kf.update(np.array([1.0, 2.0]))  # 2 elements, expected 1


class TestKalmanControl:
    def test_predict_with_control_no_ctrl_dim_raises(self):
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)  # ctrl_dim=0 default
        with pytest.raises(ValueError):
            kf.predict(np.array([1.0]))

    def test_set_B_no_ctrl_dim_raises_clearly(self):
        """Assigning B on a filter constructed without ctrl_dim should raise
        a targeted error rather than a confusing 'expected Nx0' shape error."""
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)  # ctrl_dim=0 default
        with pytest.raises(ValueError, match="ctrl_dim=0"):
            kf.B = np.array([[1.0], [0.0]])

    def test_predict_with_control_wrong_length_raises(self):
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1, ctrl_dim=1)
        with pytest.raises(ValueError):
            kf.predict(np.array([1.0, 2.0]))  # length 2, expected 1

    def test_control_applies_B_u(self):
        """With F=I, B=[[1],[0]], u=[3], state should change by B*u = [3, 0]."""
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1, ctrl_dim=1)
        # F stays identity
        kf.B = np.array([[1.0], [0.0]])
        kf.state = np.array([5.0, 2.0])
        kf.predict(np.array([3.0]))
        np.testing.assert_allclose(kf.state, [8.0, 2.0])


class TestKalmanDtypeDispatch:
    @pytest.mark.parametrize("dtype", [
        "reference", "gpu_baseline", "ml_hw", "cf24", "half",
        "posit_full", "tiny_posit",
    ])
    def test_runs_under_each_dtype(self, dtype):
        """Every dtype must at least instantiate, track through one
        predict/update cycle, and return finite values. Low-precision
        types may track poorly — that's expected and a separate test."""
        kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1, dtype=dtype)
        kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.Q = np.eye(2) * 0.01
        kf.R = np.array([[0.1]])
        kf.state = np.array([0.0, 0.0])

        kf.predict()
        kf.update(np.array([1.0]))
        state = kf.state
        assert state.shape == (2,)
        assert np.all(np.isfinite(state))

    def test_posit_full_produces_different_tracking_than_reference(self):
        """Acceptance criterion: mixed-precision Kalman (posit_full) produces
        measurably different tracking than the double reference."""
        rng = np.random.default_rng(7)
        measurements = [0.5 * t + rng.normal(0, 0.3) for t in range(30)]

        def run(dtype):
            kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1, dtype=dtype)
            kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
            kf.H = np.array([[1.0, 0.0]])
            kf.Q = np.eye(2) * 0.01
            kf.R = np.array([[0.1]])
            kf.state = np.array([0.0, 0.0])
            out = []
            for z in measurements:
                kf.predict()
                kf.update(np.array([z]))
                out.append(kf.state[0])
            return np.array(out)

        ref = run("reference")
        posit = run("posit_full")
        # Traces must differ — posit arithmetic can't reproduce double exactly.
        assert not np.array_equal(ref, posit)
        # But should still be close — within a few percent of the reference.
        err = np.max(np.abs(ref - posit)) / (np.max(np.abs(ref)) + 1e-12)
        assert err < 0.05


# ---------------------------------------------------------------------------
# Adaptive filters — shared test fixtures.
# ---------------------------------------------------------------------------


def _sysid_signals(true_taps, n=4000, seed=0):
    """Generate (x, d) signals for a system-identification task where the
    unknown system is an FIR with `true_taps`. Returns contiguous float64
    arrays of length n each."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    d = np.convolve(x, true_taps, mode="full")[:n]
    return x, d


# ---------------------------------------------------------------------------
# LMSFilter
# ---------------------------------------------------------------------------


class TestLMSFilter:
    def test_default_dtype(self):
        f = mpdsp.LMSFilter(num_taps=4, step_size=0.01)
        assert f.dtype == "reference"
        assert f.num_taps == 4

    def test_zero_num_taps_raises(self):
        with pytest.raises(ValueError):
            mpdsp.LMSFilter(num_taps=0, step_size=0.01)

    @pytest.mark.parametrize("bad_step", [0.0, -0.01, float("nan")])
    def test_non_positive_or_nan_step_size_raises(self, bad_step):
        with pytest.raises(ValueError):
            mpdsp.LMSFilter(num_taps=4, step_size=bad_step)

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError):
            mpdsp.LMSFilter(num_taps=4, step_size=0.01, dtype="not_a_dtype")

    def test_process_returns_tuple(self):
        f = mpdsp.LMSFilter(num_taps=3, step_size=0.05)
        y, e = f.process(1.0, 0.5)
        assert isinstance(y, float)
        assert isinstance(e, float)

    def test_weights_shape(self):
        f = mpdsp.LMSFilter(num_taps=5, step_size=0.01)
        w = f.weights
        assert w.shape == (5,)
        assert w.dtype == np.float64
        # Weights start at zero.
        assert np.all(w == 0.0)

    def test_process_block_shapes(self):
        f = mpdsp.LMSFilter(num_taps=3, step_size=0.05)
        x, d = _sysid_signals([0.3, 0.5, 0.2], n=128)
        ys, es = f.process_block(x, d)
        assert ys.shape == (128,)
        assert es.shape == (128,)

    def test_process_block_length_mismatch_raises(self):
        f = mpdsp.LMSFilter(num_taps=3, step_size=0.01)
        with pytest.raises(ValueError):
            f.process_block(np.zeros(10), np.zeros(11))

    def test_converges_to_known_system(self):
        """LMS should recover a known unknown-system FIR given enough samples."""
        true_taps = np.array([0.3, 0.5, 0.2])
        f = mpdsp.LMSFilter(num_taps=3, step_size=0.05)
        x, d = _sysid_signals(true_taps, n=2000)
        f.process_block(x, d)
        np.testing.assert_allclose(f.weights, true_taps, atol=0.01)

    def test_reset_clears_weights(self):
        f = mpdsp.LMSFilter(num_taps=3, step_size=0.05)
        x, d = _sysid_signals([0.3, 0.5, 0.2], n=500)
        f.process_block(x, d)
        assert np.max(np.abs(f.weights)) > 0.1
        f.reset()
        assert np.all(f.weights == 0.0)


# ---------------------------------------------------------------------------
# NLMSFilter
# ---------------------------------------------------------------------------


class TestNLMSFilter:
    def test_default_dtype(self):
        f = mpdsp.NLMSFilter(num_taps=4, step_size=0.5)
        assert f.dtype == "reference"
        assert f.num_taps == 4

    def test_zero_num_taps_raises(self):
        with pytest.raises(ValueError):
            mpdsp.NLMSFilter(num_taps=0, step_size=0.5)

    @pytest.mark.parametrize("bad_step", [0.0, -0.5, float("nan")])
    def test_non_positive_or_nan_step_size_raises(self, bad_step):
        with pytest.raises(ValueError):
            mpdsp.NLMSFilter(num_taps=4, step_size=bad_step)

    def test_non_positive_epsilon_raises(self):
        with pytest.raises(ValueError):
            mpdsp.NLMSFilter(num_taps=4, step_size=0.5, epsilon=0.0)

    def test_converges_to_known_system(self):
        true_taps = np.array([0.3, 0.5, 0.2])
        f = mpdsp.NLMSFilter(num_taps=3, step_size=0.5)
        x, d = _sysid_signals(true_taps, n=2000)
        f.process_block(x, d)
        np.testing.assert_allclose(f.weights, true_taps, atol=0.01)

    def test_handles_large_input_stably(self):
        """NLMS's normalization is its reason to exist: identical step_size
        that would diverge on LMS at large input amplitudes should track
        stably with NLMS."""
        true_taps = np.array([0.4, 0.3])
        x, d = _sysid_signals(true_taps, n=2000)
        x_loud = 50.0 * x   # 50x amplitude
        d_loud = 50.0 * d
        f = mpdsp.NLMSFilter(num_taps=2, step_size=0.5)
        f.process_block(x_loud, d_loud)
        # Even at huge amplitude NLMS converges close to the truth.
        np.testing.assert_allclose(f.weights, true_taps, atol=0.05)


# ---------------------------------------------------------------------------
# RLSFilter
# ---------------------------------------------------------------------------


class TestRLSFilter:
    def test_default_dtype(self):
        f = mpdsp.RLSFilter(num_taps=4)
        assert f.dtype == "reference"
        assert f.num_taps == 4

    def test_zero_num_taps_raises(self):
        with pytest.raises(ValueError):
            mpdsp.RLSFilter(num_taps=0)

    def test_forgetting_factor_out_of_range_raises(self):
        with pytest.raises(ValueError):
            mpdsp.RLSFilter(num_taps=4, forgetting_factor=0.0)
        with pytest.raises(ValueError):
            mpdsp.RLSFilter(num_taps=4, forgetting_factor=1.5)

    def test_non_positive_delta_raises(self):
        with pytest.raises(ValueError):
            mpdsp.RLSFilter(num_taps=4, delta=0.0)

    def test_converges_to_known_system(self):
        true_taps = np.array([0.3, 0.5, 0.2])
        f = mpdsp.RLSFilter(num_taps=3, forgetting_factor=0.99)
        x, d = _sysid_signals(true_taps, n=500)  # short — RLS is fast
        f.process_block(x, d)
        np.testing.assert_allclose(f.weights, true_taps, atol=1e-3)

    def test_reset_clears_state(self):
        f = mpdsp.RLSFilter(num_taps=3)
        x, d = _sysid_signals([0.3, 0.5, 0.2], n=200)
        f.process_block(x, d)
        assert np.max(np.abs(f.weights)) > 0.1
        f.reset()
        assert np.all(f.weights == 0.0)


# ---------------------------------------------------------------------------
# RLS vs LMS — acceptance criterion from issue #6.
# ---------------------------------------------------------------------------


def test_rls_converges_faster_than_lms():
    """RLS uses a matrix update and should reach low error in far fewer
    samples than LMS. Count how many samples it takes each to get the
    weight error below a fixed tolerance."""
    true_taps = np.array([0.3, 0.5, 0.2])

    def iters_to_converge(filt, x, d, tol=0.02):
        xs, ds = x.copy(), d.copy()
        for i in range(len(xs)):
            filt.process(float(xs[i]), float(ds[i]))
            if np.max(np.abs(np.asarray(filt.weights) - true_taps)) < tol:
                return i + 1
        return len(xs)

    x, d = _sysid_signals(true_taps, n=5000, seed=1)
    lms_iters = iters_to_converge(mpdsp.LMSFilter(num_taps=3, step_size=0.05),
                                  x, d)
    rls_iters = iters_to_converge(mpdsp.RLSFilter(num_taps=3,
                                                   forgetting_factor=0.99),
                                  x, d)
    assert rls_iters < lms_iters, (
        f"expected RLS to converge faster than LMS "
        f"(LMS={lms_iters}, RLS={rls_iters})"
    )
    # Stronger assertion: RLS should be dramatically faster, not merely 1
    # iteration fewer. Factor of 3 is a conservative expectation on this
    # well-conditioned sysid.
    assert rls_iters * 3 < lms_iters


# ---------------------------------------------------------------------------
# Dtype dispatch (adaptive filters).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [
    "reference", "gpu_baseline", "ml_hw", "cf24", "half", "posit_full",
])
@pytest.mark.parametrize("ctor", [
    lambda dt: mpdsp.LMSFilter(num_taps=3, step_size=0.05, dtype=dt),
    lambda dt: mpdsp.NLMSFilter(num_taps=3, step_size=0.5, dtype=dt),
    lambda dt: mpdsp.RLSFilter(num_taps=3, forgetting_factor=0.99, dtype=dt),
], ids=["lms", "nlms", "rls"])
def test_adaptive_filter_runs_under_each_dtype(ctor, dtype):
    """Every dtype must instantiate, process one sample, and return finite
    values. tiny_posit is excluded because its precision floor (< 4 bits of
    fraction at typical weight magnitudes) can't meaningfully represent the
    update step and the test would be checking noise rather than arithmetic.
    """
    f = ctor(dtype)
    x, d = _sysid_signals([0.3, 0.5, 0.2], n=64)
    ys, es = f.process_block(x, d)
    assert ys.shape == x.shape
    assert np.all(np.isfinite(ys))
    assert np.all(np.isfinite(es))
