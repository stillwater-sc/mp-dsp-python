"""Tests for signal-conditioning bindings (scaffold: PeakEnvelope)."""

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


SAMPLE_RATE = 8000.0


def _step_signal(n=1024, onset=128, amplitude=1.0):
    """Unit-step-like signal: zeros, then a constant-amplitude tone."""
    t = np.arange(n) / SAMPLE_RATE
    env = np.where(np.arange(n) < onset, 0.0, amplitude)
    return env * np.sin(2 * np.pi * 200 * t)


class TestPeakEnvelopeConstruction:
    def test_default_dtype_is_reference(self):
        env = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=5.0, release_ms=50.0)
        assert env.dtype == "reference"

    def test_dtype_is_stored(self):
        env = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=5.0, release_ms=50.0,
                                  dtype="half")
        assert env.dtype == "half"

    @pytest.mark.parametrize("bad_sr", [-1.0, 0.0])
    def test_invalid_sample_rate_raises(self, bad_sr):
        with pytest.raises(ValueError):
            mpdsp.PeakEnvelope(sample_rate=bad_sr,
                                attack_ms=5.0, release_ms=50.0)

    def test_non_positive_attack_raises(self):
        with pytest.raises(ValueError):
            mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                attack_ms=0.0, release_ms=50.0)

    @pytest.mark.parametrize("bad_release", [0.0, -1.0])
    def test_non_positive_release_raises(self, bad_release):
        with pytest.raises(ValueError):
            mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                attack_ms=5.0, release_ms=bad_release)

    def test_unknown_dtype_raises(self):
        # parse_config throws std::invalid_argument -> ValueError in Python.
        with pytest.raises(ValueError):
            mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                attack_ms=5.0, release_ms=50.0,
                                dtype="not_a_dtype")


class TestPeakEnvelopeBehavior:
    def test_process_single_sample_returns_float(self):
        env = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=5.0, release_ms=50.0)
        y = env.process(1.0)
        assert isinstance(y, float)

    def test_initial_value_is_zero(self):
        env = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=5.0, release_ms=50.0)
        assert env.value() == 0.0

    def test_state_persists_across_calls(self):
        env = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=5.0, release_ms=50.0)
        for _ in range(100):
            env.process(1.0)
        # After a burst of ones, envelope should have risen above zero
        assert env.value() > 0.1

    def test_reset_clears_state(self):
        env = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=5.0, release_ms=50.0)
        for _ in range(200):
            env.process(1.0)
        assert env.value() > 0.1
        env.reset()
        assert env.value() == 0.0

    def test_process_block_returns_correct_shape(self):
        env = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=5.0, release_ms=50.0)
        sig = _step_signal()
        out = env.process_block(sig)
        assert out.shape == sig.shape
        assert out.dtype == np.float64

    def test_process_block_accepts_strided_input(self):
        """A non-contiguous view (e.g. sig[::2]) must produce the same output
        as feeding a contiguous copy. The c_contig constraint on the binding
        asks nanobind to copy strided inputs transparently — this test pins
        that behavior so a future relaxation of the constraint would fail
        loudly rather than silently walk the underlying buffer linearly."""
        sig = _step_signal(n=2048, onset=256)
        view = sig[::2]
        copy = view.copy()

        env_view = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                       attack_ms=5.0, release_ms=50.0)
        env_copy = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                       attack_ms=5.0, release_ms=50.0)
        out_view = env_view.process_block(view)
        out_copy = env_copy.process_block(copy)
        np.testing.assert_array_equal(out_view, out_copy)

    def test_envelope_tracks_step_onset(self):
        env = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=1.0, release_ms=100.0)
        sig = _step_signal(n=2048, onset=256, amplitude=1.0)
        out = env.process_block(sig)
        # Before onset: envelope stays near zero
        assert np.max(np.abs(out[:200])) < 0.05
        # After onset with a few attack time-constants: envelope rises
        # toward the peak amplitude (1.0).
        assert out[1000] > 0.5

    def test_attack_faster_than_release(self):
        # With short attack and long release, envelope should rise quickly
        # and decay slowly. Compare attack-time-constant region to release.
        env = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=1.0, release_ms=100.0)
        # 512 samples @ 8 kHz = 64 ms. Burst of amplitude 1 for 256 samples,
        # then zero for the rest. By sample 256 we should be near 1.0; by
        # sample 400 we should still be well above 0.1 (slow release).
        n = 768
        sig = np.zeros(n)
        sig[:256] = 1.0
        out = env.process_block(sig)
        assert out[200] > 0.8      # attack has converged
        assert out[400] > 0.1      # release still holding (long decay)


class TestPeakEnvelopeDtypeDispatch:
    @pytest.mark.parametrize("dtype", [
        "reference", "gpu_baseline", "ml_hw", "cf24", "half",
        "posit_full", "tiny_posit",
    ])
    def test_process_runs_under_each_dtype(self, dtype):
        """Every dtype must at least instantiate, process, and return
        correctly-shaped output — even if arithmetic is lossy enough that
        tracking itself isn't meaningful at that precision."""
        env = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=5.0, release_ms=50.0,
                                  dtype=dtype)
        sig = _step_signal(n=512)
        out = env.process_block(sig)
        assert out.shape == sig.shape
        assert out.dtype == np.float64

    @pytest.mark.parametrize("dtype", [
        "reference", "gpu_baseline", "ml_hw", "cf24", "half", "posit_full",
    ])
    def test_envelope_tracks_step_under_dtype(self, dtype):
        """All dtypes with enough precision to represent the smoothing
        coefficient (1 - exp(-1/(release_ms*fs))) should track the step.

        tiny_posit (posit<8,2>) is deliberately excluded: with 5 ms attack /
        50 ms release at 8 kHz the release coefficient is ~0.9975 and
        (1 - 0.9975) = 0.0025 rounds to 0 in posit<8,2>, pinning the
        envelope to zero. That's a precision characteristic, not a bug;
        it's the kind of thing mixed-precision research is supposed to
        surface.
        """
        env = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=5.0, release_ms=50.0,
                                  dtype=dtype)
        sig = _step_signal(n=512)
        out = env.process_block(sig)
        assert np.max(out[256:]) > 0.1

    def test_reduced_precision_tracks_reference_approximately(self):
        ref = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=5.0, release_ms=50.0,
                                  dtype="reference")
        alt = mpdsp.PeakEnvelope(sample_rate=SAMPLE_RATE,
                                  attack_ms=5.0, release_ms=50.0,
                                  dtype="half")
        sig = _step_signal(n=1024)
        r = ref.process_block(sig)
        a = alt.process_block(sig)
        # Envelope shapes agree to within a reasonable low-precision bound.
        err = np.max(np.abs(r - a)) / (np.max(np.abs(r)) + 1e-12)
        assert err < 0.1


# ---------------------------------------------------------------------------
# RMSEnvelope
# ---------------------------------------------------------------------------


class TestRMSEnvelope:
    def test_default_dtype_is_reference(self):
        env = mpdsp.RMSEnvelope(sample_rate=SAMPLE_RATE, window_ms=10.0)
        assert env.dtype == "reference"

    @pytest.mark.parametrize("bad_sr", [-1.0, 0.0])
    def test_invalid_sample_rate_raises(self, bad_sr):
        with pytest.raises(ValueError):
            mpdsp.RMSEnvelope(sample_rate=bad_sr, window_ms=10.0)

    @pytest.mark.parametrize("bad_window", [0.0, -1.0])
    def test_invalid_window_raises(self, bad_window):
        with pytest.raises(ValueError):
            mpdsp.RMSEnvelope(sample_rate=SAMPLE_RATE, window_ms=bad_window)

    def test_unit_amplitude_sine_rms_is_sqrt_half(self):
        """An amplitude-1 sine has RMS = sqrt(1/2) ≈ 0.707."""
        env = mpdsp.RMSEnvelope(sample_rate=SAMPLE_RATE, window_ms=50.0)
        t = np.arange(8192) / SAMPLE_RATE
        sig = np.sin(2 * np.pi * 200 * t)
        out = env.process_block(sig)
        # Let the window converge before measuring.
        steady = out[4096:]
        assert abs(np.mean(steady) - 1.0 / np.sqrt(2.0)) < 0.02

    def test_reset_clears_state(self):
        env = mpdsp.RMSEnvelope(sample_rate=SAMPLE_RATE, window_ms=10.0)
        for _ in range(500):
            env.process(1.0)
        assert env.value() > 0.1
        env.reset()
        assert env.value() == 0.0

    @pytest.mark.parametrize("dtype", [
        "reference", "gpu_baseline", "ml_hw", "cf24", "half",
        "posit_full", "tiny_posit",
    ])
    def test_process_runs_under_each_dtype(self, dtype):
        env = mpdsp.RMSEnvelope(sample_rate=SAMPLE_RATE, window_ms=10.0,
                                 dtype=dtype)
        sig = _step_signal(n=512)
        out = env.process_block(sig)
        assert out.shape == sig.shape
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class TestCompressor:
    def _params(self, **overrides):
        base = dict(sample_rate=SAMPLE_RATE, threshold_db=-20.0, ratio=4.0,
                    attack_ms=1.0, release_ms=50.0)
        base.update(overrides)
        return base

    def test_default_dtype(self):
        comp = mpdsp.Compressor(**self._params())
        assert comp.dtype == "reference"

    def test_ratio_below_one_raises(self):
        with pytest.raises(ValueError):
            mpdsp.Compressor(**self._params(ratio=0.5))

    def test_negative_knee_raises(self):
        with pytest.raises(ValueError):
            mpdsp.Compressor(**self._params(knee_db=-1.0))

    @pytest.mark.parametrize("bad_sr", [-1.0, 0.0])
    def test_invalid_sample_rate_raises(self, bad_sr):
        with pytest.raises(ValueError):
            mpdsp.Compressor(**self._params(sample_rate=bad_sr))

    def test_below_threshold_is_untouched(self):
        """Signals below the threshold pass through without attenuation."""
        comp = mpdsp.Compressor(**self._params(threshold_db=-6.0))
        # Amplitude 0.1 => -20 dBFS, well under -6 dB threshold.
        t = np.arange(4096) / SAMPLE_RATE
        sig = 0.1 * np.sin(2 * np.pi * 200 * t)
        out = comp.process_block(sig)
        # Compressor with no makeup gain shouldn't change anything appreciably.
        np.testing.assert_allclose(out, sig, rtol=1e-3, atol=1e-3)

    def test_above_threshold_gets_attenuated(self):
        """With ratio=4:1 and a tone well above threshold, the steady-state
        peak should drop."""
        comp = mpdsp.Compressor(**self._params(threshold_db=-20.0, ratio=4.0))
        t = np.arange(4096) / SAMPLE_RATE
        # Amplitude 1.0 => 0 dBFS, 20 dB above threshold. 4:1 compression
        # turns a 20 dB overage into 5 dB, so steady-state peak ~ 10^(-15/20)
        # ≈ 0.178. Allow a broad range to accommodate attack transients.
        sig = np.sin(2 * np.pi * 200 * t)
        out = comp.process_block(sig)
        # Use the tail of the signal where the envelope has converged.
        tail_peak = np.max(np.abs(out[3072:]))
        assert tail_peak < 0.5  # well below the 1.0 input peak
        assert tail_peak > 0.05  # but not completely silenced

    def test_reset_clears_envelope_state(self):
        comp = mpdsp.Compressor(**self._params())
        # Burst of ones to build up envelope state
        burst = np.ones(512)
        comp.process_block(burst)
        # Now reset and process a zero-signal; output should start at zero.
        comp.reset()
        out = comp.process_block(np.zeros(16))
        assert np.all(out == 0.0)

    def test_soft_knee_smooths_transition(self):
        """At threshold exactly, a hard-knee compressor yields one gain; a
        soft-knee compressor yields a different (smoother) gain."""
        hard = mpdsp.Compressor(**self._params(threshold_db=-6.0, knee_db=0.0))
        soft = mpdsp.Compressor(**self._params(threshold_db=-6.0, knee_db=6.0))
        # Drive both with a tone exactly at threshold amplitude (~0.5).
        t = np.arange(4096) / SAMPLE_RATE
        sig = 0.5 * np.sin(2 * np.pi * 200 * t)
        h = hard.process_block(sig)
        s = soft.process_block(sig)
        # Both should do *something*, and the outputs should differ.
        assert not np.allclose(h, s)

    @pytest.mark.parametrize("dtype", ["reference", "gpu_baseline", "half", "posit_full"])
    def test_process_runs_under_each_dtype(self, dtype):
        comp = mpdsp.Compressor(**self._params(dtype=dtype))
        t = np.arange(1024) / SAMPLE_RATE
        sig = np.sin(2 * np.pi * 200 * t)
        out = comp.process_block(sig)
        assert out.shape == sig.shape
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# AGC
# ---------------------------------------------------------------------------


class TestAGC:
    def test_default_dtype(self):
        agc = mpdsp.AGC(sample_rate=SAMPLE_RATE, target_level=0.5)
        assert agc.dtype == "reference"

    @pytest.mark.parametrize("bad", [-1.0, 0.0])
    def test_invalid_params_raise(self, bad):
        with pytest.raises(ValueError):
            mpdsp.AGC(sample_rate=bad, target_level=0.5)
        with pytest.raises(ValueError):
            mpdsp.AGC(sample_rate=SAMPLE_RATE, target_level=bad)
        with pytest.raises(ValueError):
            mpdsp.AGC(sample_rate=SAMPLE_RATE, target_level=0.5, window_ms=bad)
        with pytest.raises(ValueError):
            mpdsp.AGC(sample_rate=SAMPLE_RATE, target_level=0.5, max_gain=bad)

    def test_boost_quiet_signal(self):
        """A quiet input should be amplified toward the target level (up to
        max_gain)."""
        agc = mpdsp.AGC(sample_rate=SAMPLE_RATE, target_level=0.5,
                         window_ms=10.0, max_gain=100.0)
        t = np.arange(16384) / SAMPLE_RATE
        sig = 0.01 * np.sin(2 * np.pi * 200 * t)  # RMS ~ 0.007
        out = agc.process_block(sig)
        # After convergence the RMS should be much closer to the target
        # than the input was.
        tail_rms = np.sqrt(np.mean(out[8192:] ** 2))
        assert tail_rms > 0.1  # clearly boosted
        assert tail_rms < 0.7  # and not wildly overshooting

    def test_attenuate_loud_signal(self):
        """A loud input should be reduced toward the target."""
        agc = mpdsp.AGC(sample_rate=SAMPLE_RATE, target_level=0.5,
                         window_ms=10.0, max_gain=100.0)
        t = np.arange(16384) / SAMPLE_RATE
        sig = 2.0 * np.sin(2 * np.pi * 200 * t)  # RMS ~ sqrt(2)
        out = agc.process_block(sig)
        tail_rms = np.sqrt(np.mean(out[8192:] ** 2))
        assert tail_rms < 1.0
        assert tail_rms > 0.2

    def test_max_gain_bounds_quiet_silence(self):
        """Essentially-zero input should not be amplified beyond max_gain."""
        agc = mpdsp.AGC(sample_rate=SAMPLE_RATE, target_level=0.5,
                         window_ms=10.0, max_gain=10.0)
        sig = 1e-6 * np.ones(1024)
        out = agc.process_block(sig)
        # max_gain = 10 -> output cap should be around 1e-5, not blowing up
        assert np.max(np.abs(out)) < 1e-4

    def test_reset_clears_state(self):
        agc = mpdsp.AGC(sample_rate=SAMPLE_RATE, target_level=0.5)
        t = np.arange(4096) / SAMPLE_RATE
        sig = 0.1 * np.sin(2 * np.pi * 200 * t)
        agc.process_block(sig)
        agc.reset()
        # Process zeros and confirm internal RMS state is clean.
        out = agc.process_block(np.zeros(16))
        # With level == 0, gain defaults to 1 (don't amplify silence), so
        # output equals input (all zeros).
        assert np.all(out == 0.0)

    @pytest.mark.parametrize("dtype", [
        "reference", "gpu_baseline", "ml_hw", "cf24", "half",
        "posit_full", "tiny_posit",
    ])
    def test_process_runs_under_each_dtype(self, dtype):
        agc = mpdsp.AGC(sample_rate=SAMPLE_RATE, target_level=0.5,
                         dtype=dtype)
        t = np.arange(512) / SAMPLE_RATE
        sig = np.sin(2 * np.pi * 200 * t)
        out = agc.process_block(sig)
        assert out.shape == sig.shape
        assert out.dtype == np.float64
