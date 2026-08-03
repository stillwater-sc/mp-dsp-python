"""Tests for the high-rate data-acquisition bindings (Issues #86, #87).

Covers NCO, CICDecimator/Interpolator, HalfBandFilter, PolyphaseDecimator/
Interpolator, the design helpers design_halfband and polyphase_decompose,
and the DDC digital down-converter.

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
        for _ in range(64):
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

    def test_factor_zero_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.PolyphaseDecimator(taps=np.ones(8), factor=0)


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

    def test_factor_zero_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.PolyphaseInterpolator(taps=np.ones(8), factor=0)


# =============================================================================
# DDC (digital down-converter) — Issue #87
# =============================================================================

# Matches the upstream C++ test fixture (tests/test_ddc.cpp): a Hamming-windowed
# lowpass at 0.45/R of the input rate, long enough that the transition band
# doesn't leak the image back into the passband.
_DDC_FS = 48000.0
_DDC_IF = 6000.0
_DDC_R = 4
# Output samples to discard before measuring: the FIR group delay is
# (num_taps-1)/2 input samples, which is ~32 output samples after decimation.
_DDC_SKIP = 32


def _ddc_taps(factor=_DDC_R, sample_rate=_DDC_FS):
    num_taps = 64 * factor + 1
    return mpdsp.fir_lowpass(
        num_taps=num_taps,
        sample_rate=sample_rate,
        cutoff=(0.45 / factor) * sample_rate,
        window="hamming",
    ).coefficients()


def _tone(freq, length, sample_rate=_DDC_FS):
    n = np.arange(length)
    return np.cos(2.0 * np.pi * freq * n / sample_rate)


class TestDDC:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_construction_exposes_parameters(self, dtype):
        ddc = mpdsp.DDC(_DDC_IF, _DDC_FS, _ddc_taps(), _DDC_R, dtype=dtype)
        assert ddc.center_frequency == pytest.approx(_DDC_IF)
        assert ddc.sample_rate == pytest.approx(_DDC_FS)
        assert ddc.decimation_factor == _DDC_R

    def test_nonpositive_sample_rate_raises(self):
        for bad in (0.0, -_DDC_FS):
            with pytest.raises((ValueError, RuntimeError)):
                mpdsp.DDC(_DDC_IF, bad, _ddc_taps(), _DDC_R)

    def test_zero_decimation_factor_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.DDC(_DDC_IF, _DDC_FS, _ddc_taps(), 0)

    def test_empty_taps_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.DDC(_DDC_IF, _DDC_FS, np.array([]), _DDC_R)

    def test_process_block_returns_real_imag_pair(self):
        n_in = 4096
        ddc = mpdsp.DDC(_DDC_IF, _DDC_FS, _ddc_taps(), _DDC_R)
        re, im = ddc.process_block(_tone(_DDC_IF, n_in))
        assert re.dtype == np.float64 and im.dtype == np.float64
        assert re.shape == im.shape
        # One complex output per decimation_factor inputs, modulo startup phase.
        assert abs(len(re) - n_in // _DDC_R) <= 1

    def test_tone_at_center_translates_to_dc(self):
        """A real tone at the tuned frequency lands at baseband DC.

        Mean magnitude is ~0.5, not 1.0: a real cosine is two conjugate
        half-amplitude complex exponentials, and the mixer keeps one of them.
        """
        ddc = mpdsp.DDC(_DDC_IF, _DDC_FS, _ddc_taps(), _DDC_R)
        re, im = ddc.process_block(_tone(_DDC_IF, 4096))
        z = (re + 1j * im)[_DDC_SKIP:]

        assert np.abs(z).mean() == pytest.approx(0.5, abs=0.05)

        # DC must be the dominant bin of the down-converted output.
        spectrum = np.abs(np.fft.fft(z[:512]))
        assert np.argmax(spectrum) == 0

    def test_offset_tone_translates_to_offset(self):
        """A tone `offset` above the tuned frequency lands at `offset`."""
        offset = 1000.0
        ddc = mpdsp.DDC(_DDC_IF, _DDC_FS, _ddc_taps(), _DDC_R)
        re, im = ddc.process_block(_tone(_DDC_IF + offset, 4096))
        z = (re + 1j * im)[_DDC_SKIP:]

        n_fft = 512
        output_rate = _DDC_FS / _DDC_R
        bin_hz = output_rate / n_fft
        peak_hz = np.argmax(np.abs(np.fft.fft(z[:n_fft]))) * bin_hz
        # Within one FFT bin of the expected offset.
        assert peak_hz == pytest.approx(offset, abs=bin_hz)

    def test_streaming_matches_block(self):
        """process() sample-by-sample reproduces process_block() exactly."""
        x = _tone(_DDC_IF, 1024)
        block = mpdsp.DDC(_DDC_IF, _DDC_FS, _ddc_taps(), _DDC_R)
        re, im = block.process_block(x)

        stream = mpdsp.DDC(_DDC_IF, _DDC_FS, _ddc_taps(), _DDC_R)
        emitted = [value for sample in x
                   for ready, value in [stream.process(sample)] if ready]

        assert len(emitted) == len(re)
        np.testing.assert_allclose(np.real(emitted), re, atol=1e-12)
        np.testing.assert_allclose(np.imag(emitted), im, atol=1e-12)

    def test_set_center_frequency_retunes(self):
        new_if = 10000.0
        ddc = mpdsp.DDC(_DDC_IF, _DDC_FS, _ddc_taps(), _DDC_R)
        ddc.set_center_frequency(new_if)
        assert ddc.center_frequency == pytest.approx(new_if)

        # The retuned DDC brings a tone at the new IF down to DC.
        ddc.reset()
        re, im = ddc.process_block(_tone(new_if, 4096))
        z = (re + 1j * im)[_DDC_SKIP:]
        assert np.abs(z).mean() == pytest.approx(0.5, abs=0.05)

    def test_reset_restores_initial_output(self):
        x = _tone(_DDC_IF, 1024)
        ddc = mpdsp.DDC(_DDC_IF, _DDC_FS, _ddc_taps(), _DDC_R)
        re_first, im_first = ddc.process_block(x)
        ddc.reset()
        re_second, im_second = ddc.process_block(x)
        np.testing.assert_allclose(re_first, re_second, atol=1e-12)
        np.testing.assert_allclose(im_first, im_second, atol=1e-12)

    def test_nco_phase_advances_and_resets(self):
        ddc = mpdsp.DDC(_DDC_IF, _DDC_FS, _ddc_taps(), _DDC_R)
        # Phase is in normalized cycles, not radians: increment is f/fs and
        # the accumulator wraps within [0, 1).
        assert ddc.nco_phase_increment == pytest.approx(_DDC_IF / _DDC_FS)
        assert ddc.nco_phase == pytest.approx(0.0)

        # 65 samples at 0.125 cycles/sample lands mid-cycle rather than back
        # at the origin, so a nonzero phase is a meaningful check.
        ddc.process_block(_tone(_DDC_IF, 65))
        assert 0.0 < ddc.nco_phase < 1.0

        ddc.reset()
        assert ddc.nco_phase == pytest.approx(0.0)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_down_conversion_holds_across_dtypes(self, dtype):
        ddc = mpdsp.DDC(_DDC_IF, _DDC_FS, _ddc_taps(), _DDC_R, dtype=dtype)
        re, im = ddc.process_block(_tone(_DDC_IF, 4096))
        z = (re + 1j * im)[_DDC_SKIP:]
        assert np.all(np.isfinite(z))
        assert np.abs(z).mean() == pytest.approx(0.5, abs=0.05)

    def test_reduced_precision_degrades_against_reference(self):
        """posit<8,2> sample path is measurably worse than the double path.

        Pins the mixed-precision dispatch as actually reaching the arithmetic:
        if `dtype=` were silently ignored, the two SQNRs would be identical.
        """
        x = _tone(_DDC_IF, 4096)
        taps = _ddc_taps()

        ref_re, ref_im = mpdsp.DDC(
            _DDC_IF, _DDC_FS, taps, _DDC_R, dtype="reference").process_block(x)
        coarse_re, coarse_im = mpdsp.DDC(
            _DDC_IF, _DDC_FS, taps, _DDC_R, dtype="posit_8_2").process_block(x)

        assert np.all(np.isfinite(coarse_re)) and np.all(np.isfinite(coarse_im))
        ref_sqnr = mpdsp.sqnr_db(ref_re[_DDC_SKIP:], ref_re[_DDC_SKIP:])
        coarse_sqnr = mpdsp.sqnr_db(ref_re[_DDC_SKIP:], coarse_re[_DDC_SKIP:])
        assert coarse_sqnr < ref_sqnr
