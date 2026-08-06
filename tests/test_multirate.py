"""Tests for the sw::dsp::multirate bindings — Channelizer, FractionalDelay.

Both are upstream v0.7.0 modules that had no Python entry points until now.
"""

from __future__ import annotations

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


_DTYPES = ["reference", "posit_full"]
SAMPLE_RATE = 8000.0


# =============================================================================
# Channelizer
# =============================================================================

def _channel_power(channelizer, signal, skip_blocks=32):
    """Mean power per channel, past the filter startup transient."""
    real, imag = channelizer.process_block(np.ascontiguousarray(signal))
    z = (real + 1j * imag)[skip_blocks:]
    return np.mean(np.abs(z) ** 2, axis=0)


class TestChannelizer:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_construction(self, dtype):
        ch = mpdsp.Channelizer(num_channels=8, taps_per_phase=16, dtype=dtype)
        assert ch.num_channels == 8
        assert ch.taps_per_phase == 16
        assert ch.num_taps == 8 * 16

    @pytest.mark.parametrize("bad", [0, 3, 6, 12, 100])
    def test_non_power_of_two_raises(self, bad):
        """The library FFT requires it, and the message says so."""
        with pytest.raises((ValueError, RuntimeError), match="power of two"):
            mpdsp.Channelizer(num_channels=bad)

    def test_process_requires_exactly_one_block(self):
        ch = mpdsp.Channelizer(num_channels=8)
        real, imag = ch.process(np.ones(8))
        assert real.shape == imag.shape == (8,)
        for wrong in (7, 9, 16):
            with pytest.raises((ValueError, RuntimeError)):
                ch.process(np.ones(wrong))

    def test_process_block_shape(self):
        ch = mpdsp.Channelizer(num_channels=8)
        real, imag = ch.process_block(np.ones(8 * 40))
        assert real.shape == imag.shape == (40, 8)
        assert real.dtype == imag.dtype == np.float64

    def test_partial_trailing_block_is_dropped(self):
        """Zero-padding would inject a transient the caller did not ask for."""
        ch = mpdsp.Channelizer(num_channels=8)
        real, _ = ch.process_block(np.ones(8 * 5 + 3))
        assert real.shape == (5, 8)

    @pytest.mark.parametrize("channel", range(8))
    def test_tone_lands_in_its_own_channel(self, channel):
        """A real tone at channel c's centre appears in channels c and M-c.

        Both, not one: a real cosine carries equal energy at +f and -f, and
        channel M-c is the negative-frequency image. Expecting a single
        channel here would be wrong about the signal, not about the filter.
        """
        channels = 8
        ch = mpdsp.Channelizer(num_channels=channels, taps_per_phase=16)
        freq = channel * SAMPLE_RATE / channels
        signal = np.cos(2 * np.pi * freq * np.arange(8192) / SAMPLE_RATE)

        power = _channel_power(ch, signal)
        expected = {channel, (channels - channel) % channels}
        assert int(np.argmax(power)) in expected

    @pytest.mark.parametrize("channel", [1, 2, 3])
    def test_out_of_band_rejection(self, channel):
        """Everything outside the conjugate pair is ~100 dB down."""
        channels = 8
        ch = mpdsp.Channelizer(num_channels=channels, taps_per_phase=16,
                               kaiser_beta=8.0)
        freq = channel * SAMPLE_RATE / channels
        signal = np.cos(2 * np.pi * freq * np.arange(8192) / SAMPLE_RATE)

        power = _channel_power(ch, signal)
        expected = {channel, (channels - channel) % channels}
        in_band = max(power[k] for k in expected)
        worst_other = max(power[k] for k in range(channels)
                          if k not in expected)
        assert 10 * np.log10(in_band / worst_other) > 80.0

    def test_longer_prototype_rejects_better(self):
        """taps_per_phase buys adjacent-channel rejection — the reason to
        pay for it."""
        signal = np.cos(2 * np.pi * (3 * SAMPLE_RATE / 8)
                        * np.arange(8192) / SAMPLE_RATE)

        def rejection(taps):
            ch = mpdsp.Channelizer(num_channels=8, taps_per_phase=taps)
            power = _channel_power(ch, signal)
            expected = {3, 5}
            in_band = max(power[k] for k in expected)
            other = max(power[k] for k in range(8) if k not in expected)
            return 10 * np.log10(in_band / other)

        assert rejection(24) > rejection(4)

    def test_reset_restores_initial_output(self):
        ch = mpdsp.Channelizer(num_channels=8)
        signal = np.random.default_rng(0).standard_normal(8 * 50)
        first = ch.process_block(np.ascontiguousarray(signal))
        ch.reset()
        second = ch.process_block(np.ascontiguousarray(signal))
        np.testing.assert_allclose(first[0], second[0], atol=1e-12)
        np.testing.assert_allclose(first[1], second[1], atol=1e-12)

    @pytest.mark.parametrize("dtype", ["reference", "gpu_baseline",
                                       "posit_full", "half", "posit_8_2"])
    def test_dtype_dispatch_produces_finite_output(self, dtype):
        ch = mpdsp.Channelizer(num_channels=8, dtype=dtype)
        real, imag = ch.process_block(np.ones(8 * 40))
        assert np.all(np.isfinite(real)) and np.all(np.isfinite(imag))

    def test_reduced_precision_degrades(self):
        """Pins that dtype= reaches the arithmetic — identical SNRs would
        mean it was accepted and ignored."""
        signal = np.cos(2 * np.pi * (3 * SAMPLE_RATE / 8)
                        * np.arange(4096) / SAMPLE_RATE)

        def output(dtype):
            ch = mpdsp.Channelizer(num_channels=8, dtype=dtype)
            real, imag = ch.process_block(np.ascontiguousarray(signal))
            return (real + 1j * imag)[32:]

        reference = output("reference")

        def snr(dtype):
            got = output(dtype)
            n = min(len(reference), len(got))
            return 10 * np.log10(
                np.mean(np.abs(reference[:n]) ** 2)
                / np.mean(np.abs(reference[:n] - got[:n]) ** 2))

        assert snr("posit_8_2") < snr("half") < snr("posit_full")


class TestChannelizerPrototypeBank:
    def test_shape(self):
        bank = mpdsp.channelizer_prototype_bank(8, 16)
        assert len(bank) == 8
        assert all(sub.shape == (16,) for sub in bank)
        assert all(np.all(np.isfinite(sub)) for sub in bank)

    def test_matches_the_channelizer_it_describes(self):
        """The bank exists so an analysis and a synthesis bank can share one
        prototype — so its size must track the channelizer's."""
        ch = mpdsp.Channelizer(num_channels=16, taps_per_phase=8)
        bank = mpdsp.channelizer_prototype_bank(16, 8)
        assert sum(sub.size for sub in bank) == ch.num_taps

    def test_rejects_non_power_of_two(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.channelizer_prototype_bank(6, 16)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_dtype(self, dtype):
        bank = mpdsp.channelizer_prototype_bank(8, 16, dtype=dtype)
        assert len(bank) == 8


# =============================================================================
# FractionalDelay
# =============================================================================

def _measured_delay(reference, delayed, skip=500, tail=500):
    """Delay in samples, from the phase slope at the dominant bin.

    Only valid for delays shorter than one period of the test tone — a phase
    fit wraps beyond that. Callers pick a tone low enough that it does not.
    """
    seg = slice(skip, len(reference) - tail)
    ref_f = np.fft.rfft(reference[seg])
    got_f = np.fft.rfft(delayed[seg])
    k = int(np.argmax(np.abs(ref_f)))
    length = len(reference[seg])
    return -np.angle(got_f[k] / ref_f[k]) / (2 * np.pi * k / length)


class TestFractionalDelay:
    @staticmethod
    def _tone(freq=5.0, length=6000, rate=1000.0):
        # Period 200 samples at the default, comfortably longer than any
        # delay tested, so the phase fit is unambiguous.
        return np.sin(2 * np.pi * freq * np.arange(length) / rate)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_construction_with_defaults(self, dtype):
        """Defaults must be usable.

        They are not upstream: `taps_per_phase` defaults to 12 there, which
        its own validator rejects for being even (mixed-precision-dsp#208).
        This binding defaults to 11.
        """
        fd = mpdsp.FractionalDelay(num_phases=64, dtype=dtype)
        assert fd.num_phases == 64
        assert fd.taps_per_phase == 11
        assert fd.num_taps == 64 * 11
        assert fd.base_group_delay_samples == pytest.approx(5.0)

    @pytest.mark.parametrize("bad", [0, 1, 2, 4, 12, 16])
    def test_even_or_tiny_taps_raise(self, bad):
        with pytest.raises((ValueError, RuntimeError), match="odd"):
            mpdsp.FractionalDelay(num_phases=64, taps_per_phase=bad)

    def test_zero_phases_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.FractionalDelay(num_phases=0)

    @pytest.mark.parametrize("offset", [5.0, 5.5, 8.25, 12.75, 20.0, 36.0])
    def test_delay_is_accurate(self, offset):
        """Measured to within a thousandth of a sample, at unity gain."""
        fd = mpdsp.FractionalDelay(num_phases=64)
        signal = self._tone()
        delayed = fd.delay_block(np.ascontiguousarray(signal), offset)
        assert _measured_delay(signal, delayed) == pytest.approx(offset,
                                                                 abs=0.01)

    def test_passband_gain_is_unity(self):
        fd = mpdsp.FractionalDelay(num_phases=64)
        signal = self._tone()
        delayed = fd.delay_block(np.ascontiguousarray(signal), 8.25)
        seg = slice(500, 5500)
        ref_f = np.fft.rfft(signal[seg])
        got_f = np.fft.rfft(delayed[seg])
        k = int(np.argmax(np.abs(ref_f)))
        assert abs(got_f[k] / ref_f[k]) == pytest.approx(1.0, abs=0.01)

    def test_below_the_floor_rounds_up(self):
        """A filter cannot reconstruct samples from the future, so requests
        under the group delay round up rather than failing silently."""
        fd = mpdsp.FractionalDelay(num_phases=64)
        signal = self._tone()
        delayed = fd.delay_block(np.ascontiguousarray(signal), 0.0)
        assert _measured_delay(signal, delayed) == pytest.approx(
            fd.base_group_delay_samples, abs=0.01)

    def test_beyond_the_ring_buffer_raises(self):
        fd = mpdsp.FractionalDelay(num_phases=64, max_int_delay=8)
        with pytest.raises((ValueError, RuntimeError)):
            fd.delay_block(np.ascontiguousarray(self._tone()), 500.0)

    def test_finer_resolution_with_more_phases(self):
        """num_phases sets the offset quantum, 1/L samples."""
        signal = self._tone()
        offset = 8.0 + 1.0 / 512.0        # inexpressible at L=8

        def error(phases):
            fd = mpdsp.FractionalDelay(num_phases=phases)
            delayed = fd.delay_block(np.ascontiguousarray(signal), offset)
            return abs(_measured_delay(signal, delayed) - offset)

        assert error(256) < error(8)

    def test_single_sample_matches_block(self):
        fd_block = mpdsp.FractionalDelay(num_phases=64)
        fd_stream = mpdsp.FractionalDelay(num_phases=64)
        signal = self._tone(length=600)

        block = fd_block.delay_block(np.ascontiguousarray(signal), 8.25)
        stream = np.array([fd_stream.delay(float(s), 8.25) for s in signal])
        np.testing.assert_allclose(stream, block, atol=1e-12)

    def test_reset_restores_initial_output(self):
        fd = mpdsp.FractionalDelay(num_phases=64)
        signal = self._tone(length=800)
        first = fd.delay_block(np.ascontiguousarray(signal), 8.25)
        fd.reset()
        second = fd.delay_block(np.ascontiguousarray(signal), 8.25)
        np.testing.assert_allclose(first, second, atol=1e-12)

    @pytest.mark.parametrize("dtype", ["reference", "gpu_baseline",
                                       "posit_full", "half", "posit_8_2"])
    def test_dtype_dispatch_produces_finite_output(self, dtype):
        fd = mpdsp.FractionalDelay(num_phases=64, dtype=dtype)
        out = fd.delay_block(np.ascontiguousarray(self._tone(length=1000)),
                             8.25)
        assert np.all(np.isfinite(out))

    def test_reduced_precision_degrades(self):
        signal = self._tone(length=2000)

        def snr(dtype):
            fd = mpdsp.FractionalDelay(num_phases=64, dtype=dtype)
            got = fd.delay_block(np.ascontiguousarray(signal), 8.25)
            ref = mpdsp.FractionalDelay(num_phases=64).delay_block(
                np.ascontiguousarray(signal), 8.25)
            return 10 * np.log10(np.mean(ref ** 2)
                                 / np.mean((ref - got) ** 2))

        assert snr("posit_8_2") < snr("half") < snr("posit_full")
