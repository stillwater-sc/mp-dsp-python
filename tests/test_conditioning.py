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
