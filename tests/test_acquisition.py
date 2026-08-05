"""Tests for the high-rate data-acquisition bindings (Issues #86, #87, #88).

Covers NCO, CICDecimator/Interpolator, HalfBandFilter, PolyphaseDecimator/
Interpolator, the design helpers design_halfband, polyphase_decompose and
design_cic_compensator, the DDC digital down-converter, and the
DecimationChain multi-stage cascade.

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


# =============================================================================
# DecimationChain + CIC droop compensator — Issue #88
# =============================================================================

_CHAIN_FS = 48000.0


def _cic(ratio=4, stages=3, delay=1):
    return mpdsp.CICDecimator(decimation_ratio=ratio, num_stages=stages,
                              differential_delay=delay)


def _halfband():
    return mpdsp.HalfBandFilter(taps=mpdsp.design_halfband(11, 0.1))


def _polyphase(factor=5):
    return mpdsp.PolyphaseDecimator(taps=np.ones(20) / 20.0, factor=factor)


def _cic_magnitude(freqs, ratio, stages, delay=1):
    """|H_cic(f)| at output-rate-normalized f, normalized to unit DC gain.

    |H(f)| = |sin(pi f D) / (R D sin(pi f / R))| ** M, with the f = 0 limit
    taken as 1 (the 0/0 point of that expression).
    """
    out = np.ones_like(freqs, dtype=float)
    nz = freqs != 0.0
    num = np.sin(np.pi * freqs[nz] * delay)
    den = ratio * delay * np.sin(np.pi * freqs[nz] / ratio)
    out[nz] = np.abs(num / den) ** stages
    return out


class TestDecimationChain:
    def test_two_stage_cic_halfband(self):
        chain = mpdsp.DecimationChain(_CHAIN_FS, [_cic(ratio=4), _halfband()])
        assert chain.num_stages == 2
        # HalfBandFilter is structurally 2:1 and reports so.
        assert chain.stage_ratios() == [4, 2]
        assert chain.total_decimation == 8
        assert chain.input_rate == pytest.approx(_CHAIN_FS)
        assert chain.output_rate == pytest.approx(_CHAIN_FS / 8)

        n_in = 8000
        out = chain.process_block(np.ones(n_in))
        assert abs(len(out) - n_in // 8) <= 1

    def test_three_stage_rates_match_expected_sequence(self):
        chain = mpdsp.DecimationChain(
            _CHAIN_FS, [_cic(ratio=4), _halfband(), _polyphase(factor=5)])
        assert chain.stage_ratios() == [4, 2, 5]
        assert chain.total_decimation == 40
        # Rate at the output of each stage, in input order.
        assert chain.stage_rates() == pytest.approx(
            [_CHAIN_FS / 4, _CHAIN_FS / 8, _CHAIN_FS / 40])
        # Documented invariant: the last stage rate is the chain output rate.
        assert chain.stage_rates()[-1] == pytest.approx(chain.output_rate)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_dtype_dispatch_constructs_and_runs(self, dtype):
        chain = mpdsp.DecimationChain(
            _CHAIN_FS, [_cic(), _halfband()], dtype=dtype)
        out = chain.process_block(np.ones(2000))
        assert np.all(np.isfinite(out))
        assert len(out) > 0

    def test_streaming_matches_block(self):
        x = np.random.default_rng(0).standard_normal(2000)
        block = mpdsp.DecimationChain(_CHAIN_FS, [_cic(), _halfband()])
        streamed = mpdsp.DecimationChain(_CHAIN_FS, [_cic(), _halfband()])

        expected = block.process_block(x)
        got = [y for sample in x
               for ready, y in [streamed.process(sample)] if ready]

        assert len(got) == len(expected)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_reset_restores_initial_output(self):
        x = np.random.default_rng(1).standard_normal(1000)
        chain = mpdsp.DecimationChain(_CHAIN_FS, [_cic(), _halfband()])
        first = chain.process_block(x)
        chain.reset()
        second = chain.process_block(x)
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_prototype_stages_are_not_consumed(self):
        """Prototypes supply parameters only — they stay independently usable.

        The chain rebuilds equivalent stages at its own dtype rather than
        adopting the prototype objects, so feeding a prototype afterwards must
        behave as though the chain never existed.
        """
        cic = _cic(ratio=4)
        before = cic.process_block(np.ones(400))

        chain = mpdsp.DecimationChain(_CHAIN_FS, [cic, _halfband()])
        chain.process_block(np.ones(4000))

        cic.reset()
        after = cic.process_block(np.ones(400))
        np.testing.assert_allclose(before, after, atol=1e-12)

    def test_single_stage_chain_is_allowed(self):
        chain = mpdsp.DecimationChain(_CHAIN_FS, [_cic(ratio=8)])
        assert chain.num_stages == 1
        assert chain.total_decimation == 8
        assert chain.output_rate == pytest.approx(_CHAIN_FS / 8)

    def test_empty_stage_list_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.DecimationChain(_CHAIN_FS, [])

    def test_too_many_stages_raises(self):
        # Well past any supported arity; the message names the cap.
        with pytest.raises((ValueError, RuntimeError)):
            mpdsp.DecimationChain(_CHAIN_FS, [_cic() for _ in range(12)])

    def test_non_stage_object_raises(self):
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            mpdsp.DecimationChain(_CHAIN_FS, [1.0])

    def test_interpolator_is_rejected_as_a_stage(self):
        # PolyphaseInterpolator upsamples; it is not a decimation stage.
        pi = mpdsp.PolyphaseInterpolator(taps=np.ones(20) / 20.0, factor=4)
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            mpdsp.DecimationChain(_CHAIN_FS, [pi])

    def test_nonpositive_input_rate_raises(self):
        for bad in (0.0, -_CHAIN_FS):
            with pytest.raises((ValueError, RuntimeError)):
                mpdsp.DecimationChain(bad, [_cic()])


class TestStageTapsAccessors:
    """`taps` readback, added so DecimationChain can rebuild stages."""

    def test_halfband_taps_round_trip(self):
        taps = mpdsp.design_halfband(11, 0.1)
        hb = mpdsp.HalfBandFilter(taps=taps)
        np.testing.assert_allclose(hb.taps, taps, atol=1e-15)
        assert hb.taps.dtype == np.float64

    def test_polyphase_taps_round_trip(self):
        taps = np.arange(20, dtype=np.float64) / 20.0
        pd = mpdsp.PolyphaseDecimator(taps=taps, factor=4)
        np.testing.assert_allclose(pd.taps, taps, atol=1e-15)

    def test_taps_getter_returns_independent_copy(self):
        # Each read builds a fresh capsule-owned array; mutating one must not
        # disturb the filter or a subsequent read.
        hb = mpdsp.HalfBandFilter(taps=mpdsp.design_halfband(11, 0.1))
        first = hb.taps
        first[0] = 999.0
        assert hb.taps[0] != 999.0


class TestDesignCICCompensator:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_shape_and_unit_dc_gain(self, dtype):
        h = mpdsp.design_cic_compensator(
            31, cic_stages=3, cic_ratio=4, passband=0.2, dtype=dtype)
        assert h.shape == (31,)
        assert np.all(np.isfinite(h))
        # Normalized to unit DC gain so the compensator preserves scale.
        assert h.sum() == pytest.approx(1.0, abs=1e-6)

    # (cic_ratio, cic_stages, passband, minimum uncompensated droop in dB).
    # The droop floors are measured, held a little under the observed values
    # so the test pins the premise without being brittle about it.
    @pytest.mark.parametrize("ratio,stages,passband,min_droop_db", [
        (4, 3, 0.4, 6.0),    # observed 6.83 dB
        (8, 4, 0.2, 2.0),    # observed 2.28 dB
        (16, 5, 0.25, 4.0),  # observed 4.54 dB
    ])
    def test_flattens_cic_passband_droop(self, ratio, stages, passband,
                                         min_droop_db):
        """The compensated passband is far flatter than the raw CIC passband."""
        h = mpdsp.design_cic_compensator(
            41, cic_stages=stages, cic_ratio=ratio, passband=passband)

        f = np.linspace(0.0, passband, 128)
        cic_mag = _cic_magnitude(f, ratio, stages)
        # Compensator magnitude response at the same normalized frequencies.
        comp_mag = np.abs(
            np.exp(-2j * np.pi * np.outer(f, np.arange(len(h)))) @ h)

        def ripple_db(mag):
            return 20.0 * np.log10(mag.max() / mag.min())

        # The premise: the raw CIC droops materially across this passband.
        assert ripple_db(cic_mag) > min_droop_db
        # The claim: compensation leaves the passband within 1 dB. Measured
        # residuals are 0.04–0.33 dB, so this has real headroom.
        assert ripple_db(cic_mag * comp_mag) < 1.0

    def test_invalid_parameters_raise(self):
        bad_kwargs = [
            dict(num_taps=2, cic_stages=3, cic_ratio=4, passband=0.2),
            dict(num_taps=31, cic_stages=0, cic_ratio=4, passband=0.2),
            dict(num_taps=31, cic_stages=3, cic_ratio=1, passband=0.2),
            dict(num_taps=31, cic_stages=3, cic_ratio=4, passband=0.0),
            dict(num_taps=31, cic_stages=3, cic_ratio=4, passband=0.5),
            dict(num_taps=31, cic_stages=3, cic_ratio=4, passband=0.2,
                 differential_delay=0),
        ]
        for kwargs in bad_kwargs:
            with pytest.raises((ValueError, RuntimeError)):
                mpdsp.design_cic_compensator(**kwargs)


# =============================================================================
# Phase-accumulator overflow guard (Issue #117)
# =============================================================================

class TestPhaseOverflowGuard:
    """NCO/DDC must not silently produce a NaN phase increment.

    Upstream holds `frequency` and `sample_rate` at the configuration's state
    scalar and divides only afterwards, so absolute RF-scale rates overflow
    narrow state types before `frequency / sample_rate` is evaluated. fixpnt
    trips upstream's own positivity check; the cfloat types used to construct
    successfully and then emit NaN for every sample, which is far worse.
    """

    # Rates that overflow a 16- or 24-bit float significand's exponent range.
    _RF_CARRIER = 1.2e9
    _RF_RATE = 5.0e9

    # These configurations hold the ratio fine but not the absolute rates.
    _NARROW = ["cf24", "half"]

    @pytest.mark.parametrize("dtype", _NARROW)
    def test_nco_rejects_absolute_rf_rates(self, dtype):
        with pytest.raises((ValueError, RuntimeError)) as excinfo:
            mpdsp.NCO(self._RF_CARRIER, self._RF_RATE, dtype=dtype)
        # The message has to name the workaround; the symptom does not imply it.
        assert "normalized" in str(excinfo.value)

    @pytest.mark.parametrize("dtype", _NARROW)
    def test_ddc_rejects_absolute_rf_rates(self, dtype):
        with pytest.raises((ValueError, RuntimeError)) as excinfo:
            mpdsp.DDC(self._RF_CARRIER, self._RF_RATE,
                      mpdsp.design_halfband(11, 0.1), 2, dtype=dtype)
        assert "normalized" in str(excinfo.value)

    @pytest.mark.parametrize("dtype", _NARROW)
    def test_retuning_is_guarded_too(self, dtype):
        """set_frequency can push a healthy oscillator into NaN."""
        nco = mpdsp.NCO(0.24, 1.0, dtype=dtype)
        assert np.isfinite(nco.phase_increment)
        with pytest.raises((ValueError, RuntimeError)):
            nco.set_frequency(self._RF_CARRIER, self._RF_RATE)

    @pytest.mark.parametrize("dtype", _NARROW)
    def test_ddc_retuning_is_guarded(self, dtype):
        ddc = mpdsp.DDC(0.24, 1.0, mpdsp.design_halfband(11, 0.1), 2,
                        dtype=dtype)
        with pytest.raises((ValueError, RuntimeError)):
            ddc.set_center_frequency(self._RF_CARRIER)

    @pytest.mark.parametrize("dtype", ["reference", "gpu_baseline",
                                       "posit_full", "cf24", "half",
                                       "fpga_fixed"])
    def test_normalized_rates_work_for_every_dtype(self, dtype):
        """The documented workaround has to actually work everywhere."""
        nco = mpdsp.NCO(0.24, 1.0, dtype=dtype)
        assert np.isfinite(nco.phase_increment)
        assert nco.phase_increment == pytest.approx(0.24, abs=1e-3)

        ddc = mpdsp.DDC(0.24, 1.0, mpdsp.design_halfband(11, 0.1), 2,
                        dtype=dtype)
        assert np.isfinite(ddc.nco_phase_increment)

    @pytest.mark.parametrize("dtype", ["reference", "gpu_baseline",
                                       "posit_full"])
    def test_wide_dtypes_still_accept_absolute_rates(self, dtype):
        """The guard must not reject configurations that were always fine."""
        nco = mpdsp.NCO(self._RF_CARRIER, self._RF_RATE, dtype=dtype)
        assert nco.phase_increment == pytest.approx(0.24, abs=1e-6)
