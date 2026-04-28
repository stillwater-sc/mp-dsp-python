"""Tests for the high-rate data-acquisition bindings (Phase 3 / Issue #86).

Covers NCO, CICDecimator/Interpolator, HalfBandFilter, PolyphaseDecimator/
Interpolator, and the design helpers design_halfband and polyphase_decompose.

These are smoke + invariant tests rather than full numerical-accuracy
sweeps — the depth lives in the upstream C++ test suite. Here we verify
that the bindings construct, dispatch dtype, and produce the right shapes
and basic properties.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import mpdsp


# Two dtypes exercised everywhere: the reference path and one mixed-precision
# representative. Adding more dtypes is a `parametrize` away.
_DTYPES = ["reference", "posit_full"]


# =============================================================================
# Free design helpers
# =============================================================================

class TestDesignHalfband:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_basic_shape_and_finiteness(self, dtype):
        # 4K+3 form: 11 = 4*2+3
        h = mpdsp.design_halfband(11, transition_width=0.1, dtype=dtype)
        assert h.shape == (11,)
        assert h.dtype == np.float64
        assert np.all(np.isfinite(h))

    def test_halfband_structure_alternating_zeros(self):
        # h[center] = 0.5 and h[center +/- 2k] = 0 for k >= 1.
        h = mpdsp.design_halfband(15, transition_width=0.1)
        center = (len(h) - 1) // 2
        assert abs(h[center] - 0.5) < 1e-10
        for k in range(2, center + 1, 2):
            assert abs(h[center - k]) < 1e-10
            assert abs(h[center + k]) < 1e-10

    def test_invalid_num_taps_raises(self):
        # not 4K+3 → upstream rejects
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.design_halfband(10, transition_width=0.1)


class TestPolyphaseDecompose:
    def test_decompose_round_trip(self):
        # Recompose from sub-taps: sub[q][p] = h[p*M + q] for q in [0,M).
        h = np.arange(12, dtype=np.float64)
        sub = mpdsp.polyphase_decompose(h, factor=4)
        assert len(sub) == 4
        # Reconstruct
        N = len(h)
        recon = np.zeros_like(h)
        for q, branch in enumerate(sub):
            for p in range(len(branch)):
                idx = p * 4 + q
                if idx < N:
                    recon[idx] = branch[p]
        np.testing.assert_array_equal(h, recon)

    def test_factor_zero_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.polyphase_decompose(np.ones(10), factor=0)


# =============================================================================
# NCO
# =============================================================================

class TestNCO:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_construction(self, dtype):
        nco = mpdsp.NCO(frequency=1000.0, sample_rate=48000.0, dtype=dtype)
        # phase increment = freq/sr (in normalized units, 1.0 = full cycle)
        assert abs(nco.phase_increment - 1000.0 / 48000.0) < 1e-6

    def test_invalid_sample_rate_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.NCO(frequency=1000.0, sample_rate=0.0)

    def test_generate_real_block_frequency(self):
        # Generated cos at fs/4 should hit ~1, 0, -1, 0 pattern for 4 samples.
        # Tolerance accommodates the upstream NCO's denormal-prevention AC
        # dither (~1e-8 magnitude per sample), which is intentional to avoid
        # denormal-flush stalls on x86.
        nco = mpdsp.NCO(frequency=12000.0, sample_rate=48000.0)
        block = nco.generate_block_real(4)
        assert block.shape == (4,)
        np.testing.assert_allclose(block, [1.0, 0.0, -1.0, 0.0], atol=1e-6)

    def test_generate_block_is_complex_tuple(self):
        nco = mpdsp.NCO(frequency=1000.0, sample_rate=48000.0)
        re, im = nco.generate_block(128)
        assert re.shape == (128,)
        assert im.shape == (128,)
        # |z| should be ~1 throughout
        mag = np.hypot(re, im)
        np.testing.assert_allclose(mag, np.ones(128), atol=1e-6)

    def test_reset_returns_phase_to_zero(self):
        nco = mpdsp.NCO(frequency=1000.0, sample_rate=48000.0)
        nco.generate_block_real(50)
        assert nco.phase != 0.0
        nco.reset()
        assert nco.phase == 0.0

    def test_mix_down_dc_at_center_freq(self):
        # A tone at fc, mixed down by an NCO at fc, ends up at DC.
        fs, fc, N = 48000.0, 4000.0, 2048
        n = np.arange(N)
        tone = np.cos(2 * math.pi * fc * n / fs)
        nco = mpdsp.NCO(frequency=fc, sample_rate=fs)
        re, im = nco.mix_down(tone)
        # DC (mean) should be the dominant content
        assert abs(re.mean()) > 0.4   # half the tone amplitude after sum/N
        # AC content amplitude is much smaller than the DC residue
        spectrum = np.abs(np.fft.fft(re + 1j * im))
        assert spectrum[0] > 10 * spectrum[1:N // 2].max()


# =============================================================================
# CICDecimator
# =============================================================================

class TestCICDecimator:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_construction_and_props(self, dtype):
        cic = mpdsp.CICDecimator(decimation_ratio=8, num_stages=3,
                                 differential_delay=1, dtype=dtype)
        assert cic.decimation_ratio == 8
        assert cic.num_stages == 3
        assert cic.differential_delay == 1

    def test_decimation_emits_every_R(self):
        cic = mpdsp.CICDecimator(decimation_ratio=4, num_stages=2)
        emit_count = 0
        for i in range(64):
            ok, _ = cic.push(1.0)
            if ok:
                emit_count += 1
        assert emit_count == 64 // 4

    def test_process_block_decimates_count(self):
        cic = mpdsp.CICDecimator(decimation_ratio=4, num_stages=2)
        out = cic.process_block(np.random.RandomState(42).randn(128))
        assert out.shape == (128 // 4,)


# =============================================================================
# CICInterpolator
# =============================================================================

class TestCICInterpolator:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_construction(self, dtype):
        cic = mpdsp.CICInterpolator(interpolation_ratio=4, num_stages=2,
                                    dtype=dtype)
        assert cic.interpolation_ratio == 4
        assert cic.num_stages == 2

    def test_process_block_upsamples(self):
        cic = mpdsp.CICInterpolator(interpolation_ratio=4, num_stages=2)
        out = cic.process_block(np.ones(16))
        assert out.shape == (16 * 4,)


# =============================================================================
# HalfBandFilter
# =============================================================================

class TestHalfBandFilter:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_construction(self, dtype):
        taps = mpdsp.design_halfband(11, transition_width=0.1)
        hb = mpdsp.HalfBandFilter(taps=taps, dtype=dtype)
        assert hb.num_taps == 11
        # Half-band: roughly half the taps are zero (center + every other).
        assert hb.num_nonzero_taps < hb.num_taps

    def test_non_halfband_taps_raises(self):
        # Even-offset non-zero taps violate the half-band property.
        bad_taps = np.ones(11)
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.HalfBandFilter(taps=bad_taps)

    def test_decimate_emits_every_other_call(self):
        taps = mpdsp.design_halfband(11, transition_width=0.1)
        hb = mpdsp.HalfBandFilter(taps=taps)
        emits = [hb.process_decimate(1.0)[0] for _ in range(10)]
        # Should alternate after the warmup transient
        assert sum(emits) == 5

    def test_process_block_decimate_halves_length(self):
        taps = mpdsp.design_halfband(11, transition_width=0.1)
        hb = mpdsp.HalfBandFilter(taps=taps)
        out = hb.process_block_decimate(np.random.RandomState(0).randn(64))
        assert out.shape == (32,)


# =============================================================================
# PolyphaseDecimator / PolyphaseInterpolator
# =============================================================================

class TestPolyphaseDecimator:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_construction(self, dtype):
        taps = np.ones(20) / 20.0
        pd = mpdsp.PolyphaseDecimator(taps=taps, factor=4, dtype=dtype)
        assert pd.factor == 4

    def test_process_block_decimates(self):
        taps = np.ones(20) / 20.0
        pd = mpdsp.PolyphaseDecimator(taps=taps, factor=4)
        out = pd.process_block(np.ones(80))
        # Approximately N/factor outputs (modulo startup phase)
        assert abs(len(out) - 80 // 4) <= 1


class TestPolyphaseInterpolator:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_construction(self, dtype):
        taps = np.ones(20) / 20.0
        pi = mpdsp.PolyphaseInterpolator(taps=taps, factor=4, dtype=dtype)
        assert pi.factor == 4

    def test_process_block_upsamples_by_factor(self):
        taps = np.ones(20) / 20.0
        pi = mpdsp.PolyphaseInterpolator(taps=taps, factor=4)
        out = pi.process_block(np.ones(20))
        assert out.shape == (20 * 4,)
