"""Tests for oscilloscope-style measurement bindings (gap-analysis #101)."""

import math

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


SAMPLE_RATE = 8000.0


def _sine(freq, n=4096, amp=1.0, phase=0.0):
    t = np.arange(n) / SAMPLE_RATE
    return amp * np.sin(2.0 * np.pi * freq * t + phase)


class TestPeakToPeak:
    def test_unit_sine_returns_two(self):
        sig = _sine(300.0, n=4096)
        assert abs(mpdsp.peak_to_peak(sig) - 2.0) < 1e-6

    def test_dc_signal_returns_zero(self):
        sig = np.full(1000, 0.5)
        assert mpdsp.peak_to_peak(sig) == 0.0

    def test_arbitrary_range(self):
        sig = np.array([-3.0, -1.0, 0.0, 4.0, 2.0])
        assert mpdsp.peak_to_peak(sig) == 7.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mpdsp.peak_to_peak(np.array([]))


class TestMean:
    def test_dc(self):
        sig = np.full(500, 2.75)
        assert abs(mpdsp.instrument_mean(sig) - 2.75) < 1e-12

    def test_zero_mean_sine(self):
        sig = _sine(200.0, n=8000)  # integer cycles
        assert abs(mpdsp.instrument_mean(sig)) < 1e-6

    def test_biased_sine(self):
        sig = _sine(200.0, n=8000) + 1.5
        assert abs(mpdsp.instrument_mean(sig) - 1.5) < 1e-6

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mpdsp.instrument_mean(np.array([]))


class TestRms:
    def test_unit_sine(self):
        sig = _sine(200.0, n=8000)
        assert abs(mpdsp.instrument_rms(sig) - 1.0 / math.sqrt(2)) < 1e-3

    def test_unit_square_wave(self):
        sig = np.where(np.arange(1000) % 2 == 0, 1.0, -1.0)
        assert abs(mpdsp.instrument_rms(sig) - 1.0) < 1e-12

    def test_dc_rms_equals_magnitude(self):
        sig = np.full(500, -2.0)
        assert abs(mpdsp.instrument_rms(sig) - 2.0) < 1e-12

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mpdsp.instrument_rms(np.array([]))


class TestRiseFallTime:
    def test_rise_time_linear_ramp_default_thresholds(self):
        # x[n] = n/(N-1) over 100 samples -> ramp from 0 to 1.
        # low_pct=0.1 crosses at sample 9.9, high_pct=0.9 at 89.1.
        # Rise time = 89.1 - 9.9 = 79.2 samples.
        n = 100
        sig = np.linspace(0.0, 1.0, n)
        rt = mpdsp.rise_time(sig)
        assert abs(rt - 79.2) < 0.5

    def test_fall_time_linear_ramp_default_thresholds(self):
        n = 100
        sig = np.linspace(1.0, 0.0, n)
        ft = mpdsp.fall_time(sig)
        assert abs(ft - 79.2) < 0.5

    def test_rise_time_nan_on_flat_signal(self):
        assert math.isnan(mpdsp.rise_time(np.zeros(100)))

    def test_fall_time_nan_on_flat_signal(self):
        assert math.isnan(mpdsp.fall_time(np.zeros(100)))

    def test_rise_time_nan_on_falling_signal(self):
        # Monotonically falling has no rising crossings.
        sig = np.linspace(1.0, 0.0, 100)
        assert math.isnan(mpdsp.rise_time(sig))

    def test_rise_time_custom_thresholds(self):
        n = 100
        sig = np.linspace(0.0, 1.0, n)
        rt_20_80 = mpdsp.rise_time(sig, low_pct=0.2, high_pct=0.8)
        # (0.8 - 0.2) * (n - 1) = 0.6 * 99 = 59.4
        assert abs(rt_20_80 - 59.4) < 0.5

    def test_rise_time_rejects_bad_thresholds(self):
        sig = np.linspace(0.0, 1.0, 100)
        with pytest.raises(ValueError):
            mpdsp.rise_time(sig, low_pct=0.9, high_pct=0.1)  # inverted
        with pytest.raises(ValueError):
            mpdsp.rise_time(sig, low_pct=-0.1, high_pct=0.9)  # below zero
        with pytest.raises(ValueError):
            mpdsp.rise_time(sig, low_pct=0.1, high_pct=1.1)  # above one


class TestPeriodFrequency:
    def test_period_recovers_sine(self):
        # 200 Hz sine at 8 kHz sample rate: period = 40 samples.
        sig = _sine(200.0, n=4096)
        p = mpdsp.period(sig, threshold=0.0)
        assert abs(p - 40.0) < 0.5

    def test_frequency_recovers_sine(self):
        sig = _sine(200.0, n=4096)
        f = mpdsp.frequency(sig, sample_rate=SAMPLE_RATE, threshold=0.0)
        assert abs(f - 200.0) < 2.0

    def test_frequency_recovers_high_frequency(self):
        sig = _sine(1500.0, n=8192)
        f = mpdsp.frequency(sig, sample_rate=SAMPLE_RATE, threshold=0.0)
        assert abs(f - 1500.0) < 5.0

    def test_period_nan_on_dc(self):
        assert math.isnan(mpdsp.period(np.full(1000, 0.5)))

    def test_frequency_nan_on_dc(self):
        result = mpdsp.frequency(np.full(1000, 0.5), sample_rate=SAMPLE_RATE)
        assert math.isnan(result)

    def test_period_with_offset_threshold(self):
        # Biased sine: rising zero-crossings shift by DC offset, but the
        # period between rising *threshold*-crossings should still recover.
        sig = _sine(200.0, n=4096) + 0.3
        p = mpdsp.period(sig, threshold=0.3)
        assert abs(p - 40.0) < 0.5

    def test_frequency_rejects_bad_sample_rate(self):
        sig = _sine(200.0, n=1024)
        with pytest.raises(ValueError):
            mpdsp.frequency(sig, sample_rate=0.0)
        with pytest.raises(ValueError):
            mpdsp.frequency(sig, sample_rate=-1.0)


class TestDtypeDispatch:
    @pytest.mark.parametrize(
        "measure",
        ["peak_to_peak", "instrument_mean", "instrument_rms"])
    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_dispatch_across_dtypes(self, measure, dtype):
        sig = _sine(200.0, n=1024) + 0.1
        fn = getattr(mpdsp, measure)
        ref = fn(sig, dtype="reference")
        out = fn(sig, dtype=dtype)
        assert math.isfinite(out)
        # Narrow dtype should stay in the same ballpark as reference.
        assert abs(out - ref) / (abs(ref) + 1e-12) < 0.1

    def test_frequency_dispatch(self):
        sig = _sine(200.0, n=4096)
        ref = mpdsp.frequency(sig, sample_rate=SAMPLE_RATE, dtype="reference")
        alt = mpdsp.frequency(sig, sample_rate=SAMPLE_RATE, dtype="posit_full")
        assert abs(alt - ref) < 2.0

    def test_unknown_dtype_raises(self):
        sig = _sine(200.0, n=64)
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.peak_to_peak(sig, dtype="not_a_dtype")


# ---------------------------------------------------------------------------
# PeakDetectDecimator — scope-style min/max preserving decimator (#102).
# ---------------------------------------------------------------------------


class TestPeakDetectDecimatorStreaming:
    def test_process_returns_none_until_window_closes(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=4)
        assert d.process(1.0) is None
        assert d.process(2.0) is None
        assert d.process(3.0) is None
        result = d.process(4.0)
        assert result == (1.0, 4.0)

    def test_process_cycles(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=3)
        # First window: 1, 2, 3 -> (1, 3)
        assert d.process(1.0) is None
        assert d.process(2.0) is None
        assert d.process(3.0) == (1.0, 3.0)
        # Second window: -1, 0, 5 -> (-1, 5)
        assert d.process(-1.0) is None
        assert d.process(0.0) is None
        assert d.process(5.0) == (-1.0, 5.0)

    def test_glitch_within_window_is_preserved(self):
        # This is the defining scope-vs-generic-decimator invariant:
        # a spike shorter than the decimation interval still shows up.
        d = mpdsp.PeakDetectDecimator(decimation_factor=8)
        for i in range(7):
            d.process(0.0)
        # A single-sample glitch inside the 8-sample window
        result = d.process(5.0)
        assert result == (0.0, 5.0)

    def test_ratio_one_is_passthrough(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=1)
        for x in [1.5, -2.0, 0.25]:
            assert d.process(x) == (x, x)

    def test_reset_drops_partial_window(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=4)
        d.process(1.0)
        d.process(2.0)
        assert d.samples_in_window == 2
        d.reset()
        assert d.samples_in_window == 0
        # New window seeds fresh from the next push
        assert d.process(10.0) is None
        assert d.process(20.0) is None
        assert d.process(30.0) is None
        assert d.process(40.0) == (10.0, 40.0)

    def test_samples_in_window_property(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=5)
        assert d.samples_in_window == 0
        d.process(1.0)
        assert d.samples_in_window == 1
        d.process(2.0)
        d.process(3.0)
        d.process(4.0)
        assert d.samples_in_window == 4
        d.process(5.0)  # closes the window
        assert d.samples_in_window == 0


class TestPeakDetectDecimatorBlock:
    def test_process_block_lengths(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=4)
        sig = np.arange(20, dtype=np.float64)
        mins, maxs = d.process_block(sig)
        # 20 / 4 = 5 complete windows
        assert mins.shape == (5,)
        assert maxs.shape == (5,)

    def test_process_block_values_on_ramp(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=4)
        sig = np.arange(20, dtype=np.float64)  # 0..19
        mins, maxs = d.process_block(sig)
        # Window k holds samples [4k, 4k+3]
        np.testing.assert_allclose(mins, [0.0, 4.0, 8.0, 12.0, 16.0])
        np.testing.assert_allclose(maxs, [3.0, 7.0, 11.0, 15.0, 19.0])

    def test_process_block_min_max_convenience(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=2)
        sig = np.array([1.0, 3.0, -2.0, 5.0])
        mins_only = d.process_block_min(sig)
        # process_block_min consumed the buffer; use a fresh decimator
        # for process_block_max so we're not confused by carry-over state.
        d2 = mpdsp.PeakDetectDecimator(decimation_factor=2)
        maxs_only = d2.process_block_max(sig)
        np.testing.assert_allclose(mins_only, [1.0, -2.0])
        np.testing.assert_allclose(maxs_only, [3.0, 5.0])

    def test_process_block_partial_leaves_carryover(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=4)
        # 10 samples with R=4 -> 2 complete windows, 2 carry over
        sig = np.arange(10, dtype=np.float64)
        mins, maxs = d.process_block(sig)
        assert mins.shape == (2,)
        assert d.samples_in_window == 2

    def test_streaming_and_block_agree(self):
        rng = np.random.default_rng(0xC0FFEE)
        sig = rng.standard_normal(120)
        R = 8

        # Streaming
        d1 = mpdsp.PeakDetectDecimator(decimation_factor=R)
        stream_mins, stream_maxs = [], []
        for x in sig:
            r = d1.process(float(x))
            if r is not None:
                stream_mins.append(r[0])
                stream_maxs.append(r[1])

        # Block
        d2 = mpdsp.PeakDetectDecimator(decimation_factor=R)
        block_mins, block_maxs = d2.process_block(sig)

        np.testing.assert_allclose(stream_mins, block_mins)
        np.testing.assert_allclose(stream_maxs, block_maxs)

    def test_block_respects_prior_streaming_state(self):
        # Push some samples via streaming, then finish with a block. Total
        # closed-window count must include the prior samples_in_window.
        R = 5
        d = mpdsp.PeakDetectDecimator(decimation_factor=R)
        # Push 3 via streaming (count_ = 3)
        for x in [1.0, 2.0, 3.0]:
            assert d.process(x) is None
        assert d.samples_in_window == 3
        # Now feed a block of 7 samples: (3 + 7) / 5 = 2 windows close
        sig = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        mins, maxs = d.process_block(sig)
        assert mins.shape == (2,)
        # First window is [1,2,3,4,5] -> (1, 5); second is [6,7,8,9,10] -> (6, 10)
        np.testing.assert_allclose(mins, [1.0, 6.0])
        np.testing.assert_allclose(maxs, [5.0, 10.0])

    def test_process_block_empty_input(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=4)
        mins, maxs = d.process_block(np.array([], dtype=np.float64))
        assert mins.shape == (0,)
        assert maxs.shape == (0,)


class TestPeakDetectDecimatorConstruction:
    def test_rejects_zero_decimation_factor(self):
        with pytest.raises(ValueError):
            mpdsp.PeakDetectDecimator(decimation_factor=0)

    def test_decimation_factor_property(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=17)
        assert d.decimation_factor == 17

    def test_dtype_property(self):
        d = mpdsp.PeakDetectDecimator(decimation_factor=4, dtype="posit_full")
        assert d.dtype == "posit_full"

    def test_unknown_dtype_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.PeakDetectDecimator(decimation_factor=4, dtype="not_a_dtype")

    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_dispatch_across_dtypes(self, dtype):
        # min/max are precision-insensitive at this signal amplitude —
        # verify only that the class constructs and produces correct-shape
        # output under each dtype.
        d = mpdsp.PeakDetectDecimator(decimation_factor=4, dtype=dtype)
        sig = np.linspace(-1.0, 1.0, 32)
        mins, maxs = d.process_block(sig)
        assert mins.shape == (8,)
        assert maxs.shape == (8,)
        assert np.all(np.isfinite(mins))
        assert np.all(np.isfinite(maxs))


# ---------------------------------------------------------------------------
# TriggerRingBuffer — pre/post-trigger scope capture (#103).
# ---------------------------------------------------------------------------


class TestTriggerRingBufferBasics:
    def test_construction_stores_capacities(self):
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=32,
                                      post_trigger_samples=48)
        assert buf.pre_trigger_capacity == 32
        assert buf.post_trigger_capacity == 48
        assert buf.dtype == "reference"

    def test_fresh_instance_not_complete(self):
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=8,
                                      post_trigger_samples=4)
        assert buf.capture_complete is False
        # Empty segment while not complete
        assert buf.captured_segment().shape == (0,)

    def test_dtype_property(self):
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=4,
                                      post_trigger_samples=4,
                                      dtype="posit_full")
        assert buf.dtype == "posit_full"

    def test_unknown_dtype_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.TriggerRingBuffer(pre_trigger_samples=4,
                                    post_trigger_samples=4,
                                    dtype="not_a_dtype")


class TestTriggerRingBufferCapture:
    def test_capture_after_full_prefill(self):
        pre, post = 4, 3
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=pre,
                                      post_trigger_samples=post)
        # Fill the pre-trigger ring
        for x in [1.0, 2.0, 3.0, 4.0]:
            buf.push(x)
        assert buf.capture_complete is False
        # Fire trigger and push post-trigger samples
        buf.push_trigger(99.0)
        assert buf.capture_complete is False  # still waiting for post
        for x in [10.0, 20.0, 30.0]:
            buf.push(x)
        assert buf.capture_complete is True
        seg = buf.captured_segment()
        # Full capture = pre + trigger + post = 4 + 1 + 3 = 8 samples
        np.testing.assert_allclose(
            seg, [1.0, 2.0, 3.0, 4.0, 99.0, 10.0, 20.0, 30.0])

    def test_short_capture_when_triggered_during_prefill(self):
        # Trigger arrives before pre-trigger ring is full: capture uses
        # whatever pre-context has accumulated (partial).
        pre, post = 8, 2
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=pre,
                                      post_trigger_samples=post)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        buf.push_trigger(50.0)
        buf.push(60.0)
        buf.push(70.0)
        assert buf.capture_complete is True
        seg = buf.captured_segment()
        # Only 3 pre-context samples + trigger + 2 post = 6 samples total
        np.testing.assert_allclose(seg, [1.0, 2.0, 3.0, 50.0, 60.0, 70.0])

    def test_ring_shows_only_most_recent_pre_samples(self):
        # Push more than pre_size samples pre-trigger; only the last
        # pre_size should appear as pre-context.
        pre, post = 3, 1
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=pre,
                                      post_trigger_samples=post)
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            buf.push(x)
        buf.push_trigger(9.0)
        buf.push(11.0)
        assert buf.capture_complete is True
        seg = buf.captured_segment()
        # Last 3 pre samples are [3, 4, 5], then trigger, then post
        np.testing.assert_allclose(seg, [3.0, 4.0, 5.0, 9.0, 11.0])

    def test_zero_post_trigger_completes_immediately(self):
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=4,
                                      post_trigger_samples=0)
        for x in [1.0, 2.0, 3.0, 4.0]:
            buf.push(x)
        assert buf.capture_complete is False
        buf.push_trigger(5.0)
        assert buf.capture_complete is True
        np.testing.assert_allclose(buf.captured_segment(),
                                   [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_zero_pre_trigger_captures_trigger_and_post(self):
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=0,
                                      post_trigger_samples=3)
        buf.push_trigger(7.0)
        buf.push(8.0)
        buf.push(9.0)
        buf.push(10.0)
        assert buf.capture_complete is True
        np.testing.assert_allclose(buf.captured_segment(),
                                   [7.0, 8.0, 9.0, 10.0])


class TestTriggerRingBufferLifecycle:
    def test_pushes_after_complete_are_dropped(self):
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=2,
                                      post_trigger_samples=1)
        buf.push(1.0)
        buf.push(2.0)
        buf.push_trigger(9.0)
        buf.push(3.0)
        assert buf.capture_complete is True
        before = buf.captured_segment().copy()
        # Extra pushes are silently ignored
        buf.push(999.0)
        buf.push(-999.0)
        assert buf.capture_complete is True
        np.testing.assert_array_equal(buf.captured_segment(), before)

    def test_push_trigger_during_capturing_is_ignored(self):
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=2,
                                      post_trigger_samples=2)
        buf.push(1.0)
        buf.push(2.0)
        buf.push_trigger(9.0)
        # Second trigger call during Capturing must be a no-op — the
        # sample it carries must NOT be treated as post-trigger data.
        buf.push_trigger(-999.0)
        buf.push(3.0)
        buf.push(4.0)
        assert buf.capture_complete is True
        seg = buf.captured_segment()
        # -999.0 must not appear anywhere in the captured window.
        assert -999.0 not in seg.tolist()

    def test_rearm_allows_second_capture(self):
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=2,
                                      post_trigger_samples=1)
        buf.push(1.0)
        buf.push(2.0)
        buf.push_trigger(9.0)
        buf.push(3.0)
        assert buf.capture_complete is True
        buf.rearm()
        assert buf.capture_complete is False
        # After rearm, the pre-ring retains [1, 2] so an immediate trigger
        # gets the full pre-context (from upstream contract).
        buf.push_trigger(19.0)
        buf.push(29.0)
        assert buf.capture_complete is True
        np.testing.assert_allclose(buf.captured_segment(),
                                   [1.0, 2.0, 19.0, 29.0])

    def test_reset_wipes_everything(self):
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=3,
                                      post_trigger_samples=1)
        for x in [1.0, 2.0, 3.0]:
            buf.push(x)
        buf.push_trigger(9.0)
        buf.push(4.0)
        assert buf.capture_complete is True
        buf.reset()
        assert buf.capture_complete is False
        assert buf.captured_segment().shape == (0,)
        # After reset, an immediate trigger produces a short capture
        # because the ring is empty (returned to PreFill).
        buf.push_trigger(50.0)
        buf.push(60.0)
        assert buf.capture_complete is True
        np.testing.assert_allclose(buf.captured_segment(), [50.0, 60.0])


class TestTriggerRingBufferDtypeDispatch:
    @pytest.mark.parametrize(
        "dtype", ["gpu_baseline", "half", "cf24", "posit_full"])
    def test_captures_correctly_across_dtypes(self, dtype):
        # Use small integer-valued samples so any narrow dtype represents
        # them exactly; capture ordering / length is what matters here.
        buf = mpdsp.TriggerRingBuffer(pre_trigger_samples=3,
                                      post_trigger_samples=2,
                                      dtype=dtype)
        for x in [1.0, 2.0, 3.0]:
            buf.push(x)
        buf.push_trigger(9.0)
        buf.push(10.0)
        buf.push(11.0)
        assert buf.capture_complete is True
        seg = buf.captured_segment()
        assert seg.shape == (6,)
        np.testing.assert_allclose(seg, [1.0, 2.0, 3.0, 9.0, 10.0, 11.0])
