"""Tests for state-estimation bindings (scaffold: KalmanFilter)."""

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
