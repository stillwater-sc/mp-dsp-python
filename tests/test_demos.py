"""Tests for the reference demonstrations under demos/ (Issue #62).

Demo 1 — audio dynamics processor (#58). These are integration tests: they
run the real chain, on a short clip, at every dtype the demo advertises, and
compile the header it emits. Short clips keep the suite fast; correctness of
the individual primitives is covered by their own test modules.
"""

from __future__ import annotations

import contextlib
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)

_DEMO_DIR = Path(__file__).resolve().parents[1] / "demos" / "01_audio_dynamics"

# Every bare module name any demo imports. Demos are standalone scripts run as
# `python demos/<n>/design.py`, so they import each other by bare name — and
# more than one demo defines `design` and `simulate`. Loading a second demo
# would otherwise pick up the first one's cached module and fail obscurely, so
# these names are purged around each import.
_DEMO_MODULE_NAMES = (
    "design", "simulate", "emit_c_header",
    "quantize_coefficients", "emit_coeffs_vivado", "emit_coeffs_verilog",
)


def types_ns(**kwargs):
    """Tiny attribute bag — clearer at the call site than a dict."""
    return type("DemoModules", (), kwargs)


@contextlib.contextmanager
def _demo_modules(directory: Path, names: tuple[str, ...]):
    """Import `names` from `directory`, isolated from other demos."""
    if not directory.is_dir():
        pytest.skip(f"demo directory missing: {directory}")

    saved = {n: sys.modules.pop(n) for n in _DEMO_MODULE_NAMES
             if n in sys.modules}
    sys.path.insert(0, str(directory))
    try:
        yield {n: importlib.import_module(n) for n in names}
    finally:
        for n in _DEMO_MODULE_NAMES:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(directory))


@pytest.fixture
def demo():
    with _demo_modules(_DEMO_DIR,
                       ("design", "simulate", "emit_c_header")) as mods:
        yield types_ns(design=mods["design"], simulate=mods["simulate"],
                       emit_c_header=mods["emit_c_header"])


@pytest.fixture
def clip(demo):
    # 0.25 s is enough for the compressor and limiter to engage on at least
    # one transient while keeping the full-dtype sweep quick.
    return demo.simulate.synthesize(duration_s=0.25)


class TestSimulate:
    def test_synthesize_shape_and_headroom(self, demo):
        clip = demo.simulate.synthesize(duration_s=0.2)
        assert clip.shape[0] == 2
        assert clip.shape[1] == int(round(0.2 * demo.simulate.SAMPLE_RATE))
        assert np.all(np.isfinite(clip))
        # Normalized just under full scale, leaving room for makeup gain.
        assert 0.8 < np.max(np.abs(clip)) <= 0.9

    def test_channels_are_decorrelated(self, demo):
        clip = demo.simulate.synthesize(duration_s=0.2)
        correlation = np.corrcoef(clip[0], clip[1])[0, 1]
        assert abs(correlation) < 0.999, "channels are effectively identical"

    def test_wav_round_trip(self, demo, tmp_path):
        clip = demo.simulate.synthesize(duration_s=0.1)
        path = tmp_path / "rt.wav"
        demo.simulate.write_wav(path, clip, demo.simulate.SAMPLE_RATE)
        back, sample_rate = demo.simulate.load_wav(path)
        assert sample_rate == pytest.approx(demo.simulate.SAMPLE_RATE)
        assert back.shape == clip.shape
        # 16-bit PCM quantization step is ~3e-5.
        np.testing.assert_allclose(back, clip, atol=1e-4)


class TestChain:
    @pytest.mark.parametrize("dtype", ["reference"] + [
        d for d in ["gpu_baseline", "ml_hw", "cf24", "half", "posit_full",
                    "posit_8_2"]])
    def test_runs_at_every_advertised_dtype(self, demo, clip, dtype):
        """Acceptance: the chain runs cleanly at all default dtypes.

        'Cleanly' means it completes and produces finite audio of the right
        shape — not that the audio is *good*. posit_8_2 is expected to be
        badly degraded; that is the demo's point, not a failure.
        """
        config = demo.design.ChainConfig()
        result = demo.design.run_chain(clip, config, dtype)
        assert result.audio.shape == clip.shape
        assert np.all(np.isfinite(result.audio))
        assert result.gain_reduction_db.shape == (clip.shape[1],)
        assert np.all(result.gain_reduction_db <= 1e-9), \
            "gain reduction must never be a gain boost"

    def test_default_dtype_list_is_all_valid(self, demo):
        available = set(mpdsp.available_dtypes())
        assert set(demo.design.DEFAULT_DTYPES) <= available

    def test_reference_output_stays_under_full_scale(self, demo, clip):
        """The limiter has to actually hold the ceiling at reference.

        This is the regression guard for the limiter's structure: an
        envelope follower alone (no lookahead, no sliding max) let peaks of
        1.8 through a 0.95 ceiling.
        """
        config = demo.design.ChainConfig()
        result = demo.design.run_chain(clip, config, "reference")
        assert result.peak < 1.0, (
            f"reference run peaks at {result.peak:.3f} — the limiter is not "
            f"holding the {config.ceiling} ceiling")
        assert result.clipped_samples == 0

    def test_limiter_actually_engages(self, demo, clip):
        """A limiter that never reduces gain is not being demonstrated."""
        config = demo.design.ChainConfig()
        result = demo.design.run_chain(clip, config, "reference")
        assert float(np.min(result.gain_reduction_db)) < -1.0

    def test_eq_mid_band_is_parallel_not_series(self, demo, clip):
        """The mid band must leave out-of-band content alone.

        A bandpass in series would gut everything outside the mid band. The
        check: energy well below the mid band survives the EQ.
        """
        config = demo.design.ChainConfig()
        eq = demo.design.design_eq(config)
        low_tone = mpdsp.sine(4096, frequency=80.0,
                              sample_rate=config.sample_rate)
        out = demo.design.apply_eq(low_tone, eq, config, "reference")
        in_rms = float(np.sqrt(np.mean(low_tone ** 2)))
        out_rms = float(np.sqrt(np.mean(out ** 2)))
        # The lowshelf boosts here, so out should be at least comparable.
        assert out_rms > 0.5 * in_rms

    def test_run_all_scores_against_reference(self, demo, clip):
        config = demo.design.ChainConfig()
        results = demo.design.run_all(
            clip, config, ["reference", "gpu_baseline", "posit_8_2"])
        by_dtype = {r.dtype: r for r in results}

        assert by_dtype["reference"].sqnr_db is None
        # A 32-bit float path should track double closely...
        assert by_dtype["gpu_baseline"].sqnr_db > 40.0
        # ...and an 8-bit posit path should not.
        assert by_dtype["posit_8_2"].sqnr_db < 20.0

    def test_reference_is_scored_first_even_if_listed_last(self, demo, clip):
        """run_all reorders so `reference` exists before anything is scored."""
        config = demo.design.ChainConfig()
        results = demo.design.run_all(
            clip, config, ["gpu_baseline", "reference"])
        scored = {r.dtype: r.sqnr_db for r in results}
        assert scored["reference"] is None
        assert scored["gpu_baseline"] is not None

    def test_artifacts_are_written(self, demo, clip, tmp_path):
        """Acceptance: artifacts/ gets per-dtype WAVs and a summary."""
        config = demo.design.ChainConfig()
        results = demo.design.run_all(
            clip, config, ["reference", "gpu_baseline"])

        for r in results:
            demo.simulate.write_wav(tmp_path / f"output_{r.dtype}.wav",
                                    r.audio, config.sample_rate)
        demo.design.write_summary_csv(tmp_path / "summary.csv", results)

        assert (tmp_path / "output_reference.wav").is_file()
        assert (tmp_path / "output_gpu_baseline.wav").is_file()

        rows = (tmp_path / "summary.csv").read_text().strip().splitlines()
        assert rows[0].startswith("dtype,")
        assert len(rows) == 1 + len(results)

    def test_summary_png_written_when_matplotlib_present(self, demo, clip,
                                                         tmp_path):
        pytest.importorskip("matplotlib")
        config = demo.design.ChainConfig()
        results = demo.design.run_all(clip, config,
                                      ["reference", "gpu_baseline"])
        assert demo.design.plot_summary(tmp_path / "summary.png", clip,
                                        results, config)
        assert (tmp_path / "summary.png").stat().st_size > 0


class TestEmitCHeader:
    def test_header_contains_the_designed_coefficients(self, demo):
        config = demo.design.ChainConfig()
        text = demo.emit_c_header.render_header(config)

        assert "#ifndef CHAIN_H" in text and "#endif" in text
        assert "CHAIN_NUM_BIQUADS     3" in text
        for macro in ("CHAIN_SAMPLE_RATE_HZ", "CHAIN_MID_GAIN",
                      "CHAIN_COMP_THRESHOLD_DB", "CHAIN_LIMIT_CEILING",
                      "CHAIN_LIMIT_LOOKAHEAD_SAMPLES"):
            assert macro in text, f"missing {macro}"

        # Every biquad's b0 must appear verbatim at full precision, which is
        # what makes the header a faithful export rather than an approximation.
        eq = demo.design.design_eq(config)
        for band in ("low", "mid", "high"):
            b0 = eq[band].coefficients()[0][0]
            assert f"{b0:.17g}" in text, f"{band} b0 not exported exactly"

    def test_header_notes_the_parallel_mid_band(self, demo):
        """A porter who wires the mid band in series gets a broken EQ, so the
        header has to say so."""
        text = demo.emit_c_header.render_header(demo.design.ChainConfig())
        assert "PARALLEL" in text

    @pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not present")
    @pytest.mark.parametrize("std", ["c89", "c99"])
    def test_header_compiles(self, demo, tmp_path, std):
        """Acceptance: the header compiles on a bare-metal-style toolchain.

        Compiled via a translation unit that includes it, rather than
        `gcc -c chain.h` — that form builds a *precompiled header* rather
        than an object, which both tests less and fails for unrelated
        environment reasons. Warnings are errors: a header destined for an
        embedded build should not add noise to someone else's clean build.
        """
        header = tmp_path / "chain.h"
        header.write_text(
            demo.emit_c_header.render_header(demo.design.ChainConfig()))

        source = tmp_path / "tu.c"
        source.write_text(
            '#include "chain.h"\n'
            "int main(void) {\n"
            "    return (CHAIN_NUM_BIQUADS == 3\n"
            "            && chain_biquads[0].b0 != 0.0\n"
            "            && CHAIN_LIMIT_LOOKAHEAD_SAMPLES > 0) ? 0 : 1;\n"
            "}\n")

        binary = tmp_path / "tu"
        compile_result = subprocess.run(
            ["gcc", f"-std={std}", "-Wall", "-Wextra", "-pedantic", "-Werror",
             "-I", str(tmp_path), str(source), "-o", str(binary)],
            capture_output=True, text=True, timeout=120)
        assert compile_result.returncode == 0, (
            f"header failed to compile as {std}:\n{compile_result.stderr}")

        run_result = subprocess.run([str(binary)], timeout=60)
        assert run_result.returncode == 0, "header constants did not load"

    def test_coeff_dtype_changes_the_exported_coefficients(self, demo):
        """`--coeff-dtype` is meant to show what a target's design-time
        arithmetic would have produced, so it must actually change them."""
        reference = demo.emit_c_header.render_header(
            demo.design.ChainConfig(coeff_dtype="reference"))
        coarse = demo.emit_c_header.render_header(
            demo.design.ChainConfig(coeff_dtype="posit_8_2"))
        assert reference != coarse


class TestDemoScriptsRun:
    """End-to-end: the documented command lines work."""

    def test_design_main(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(_DEMO_DIR / "design.py"),
             "--duration", "0.2", "--outdir", str(tmp_path),
             "--dtypes", "reference", "half"],
            capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "output_reference.wav").is_file()
        assert (tmp_path / "output_half.wav").is_file()
        assert (tmp_path / "summary.csv").is_file()

    def test_emit_c_header_main(self, tmp_path):
        out = tmp_path / "chain.h"
        result = subprocess.run(
            [sys.executable, str(_DEMO_DIR / "emit_c_header.py"),
             "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr
        assert out.is_file() and out.stat().st_size > 0

    def test_simulate_main(self, tmp_path):
        out = tmp_path / "input.wav"
        result = subprocess.run(
            [sys.executable, str(_DEMO_DIR / "simulate.py"),
             "--out", str(out), "--duration", "0.2"],
            capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr
        assert out.is_file()


# =============================================================================
# Demo 4 — SDR DDC cascade (#61)
# =============================================================================

_DDC_DIR = Path(__file__).resolve().parents[1] / "demos" / "04_sdr_ddc_cascade"


@pytest.fixture
def ddc():
    names = ("design", "simulate", "quantize_coefficients",
             "emit_coeffs_vivado", "emit_coeffs_verilog")
    with _demo_modules(_DDC_DIR, names) as mods:
        yield types_ns(design=mods["design"], simulate=mods["simulate"],
                       quantize=mods["quantize_coefficients"],
                       vivado=mods["emit_coeffs_vivado"],
                       verilog=mods["emit_coeffs_verilog"])


@pytest.fixture
def cascade(ddc):
    return ddc.design.design_cascade(ddc.design.CascadeSpec())


class TestCascadeDesign:
    def test_four_stages_decimating_by_sixteen(self, ddc, cascade):
        """Acceptance: 4-stage cascade with total decimation >= 16."""
        assert len(cascade) == ddc.design.NUM_STAGES == 4
        assert ddc.design.TOTAL_DECIMATION >= 16
        for taps in cascade:
            assert len(taps) >= 3
            assert np.all(np.isfinite(taps))
            # Linear phase: symmetric taps.
            np.testing.assert_allclose(taps, taps[::-1], atol=1e-12)

    def test_meets_the_spec(self, ddc, cascade):
        """Acceptance: 60 dB stopband, < 0.5 dB passband ripple."""
        spec = ddc.design.CascadeSpec()
        analysis = ddc.design.analyze_response(cascade)
        assert ddc.design.check_spec(analysis, spec) == []
        assert analysis["passband_ripple_db"] < spec.max_passband_ripple_db
        assert (analysis["alias_attenuation_db"]
                > spec.min_alias_attenuation_db)

    def test_alias_bands_are_measured_not_just_stopband(self, ddc, cascade):
        """The analysis must identify which image is worst.

        A decimator's spec is about the bands that fold onto the passband,
        not the response between them.
        """
        analysis = ddc.design.analyze_response(cascade)
        assert analysis["worst_image"] is not None
        assert 1 <= analysis["worst_image"] <= ddc.design.TOTAL_DECIMATION // 2

    def test_composite_is_unity_at_dc(self, ddc, cascade):
        response = ddc.design.composite_response(cascade, np.array([0.0]))
        assert abs(response[0]) == pytest.approx(1.0, abs=1e-3)

    def test_spec_failure_is_detected(self, ddc):
        """check_spec must actually reject a bad cascade."""
        spec = ddc.design.CascadeSpec()
        starved = [np.asarray(mpdsp.fir_lowpass(
            num_taps=7, sample_rate=1.0, cutoff=0.25,
            window="hamming").coefficients()) for _ in range(4)]
        assert ddc.design.check_spec(
            ddc.design.analyze_response(starved), spec) != []


class TestSignal:
    def test_metadata_and_alias_trap_placement(self, ddc):
        signal, meta = ddc.simulate.synthesize(num_samples=8192)
        assert signal.shape == (8192,)
        assert np.all(np.isfinite(signal))
        # The alias trap sits exactly one output-rate step above the carrier,
        # which is what makes it fold onto the signal at /16.
        step = meta["sample_rate_hz"] / ddc.design.TOTAL_DECIMATION
        assert meta["alias_hz"] - meta["carrier_hz"] == pytest.approx(step)

    def test_wanted_band_brackets_the_carrier(self, ddc):
        _, meta = ddc.simulate.synthesize(num_samples=4096)
        low, high = ddc.simulate.wanted_band(meta)
        assert low < meta["carrier_hz"] < high


class TestCascadeRun:
    @pytest.mark.parametrize("dtype", ["reference", "gpu_baseline",
                                       "fpga_fixed", "posit_full", "half"])
    def test_runs_at_every_advertised_dtype(self, ddc, cascade, dtype):
        signal, meta = ddc.simulate.synthesize(num_samples=16384)
        baseband, traces = ddc.design.run_cascade(signal, cascade, meta, dtype)
        assert len(traces) == ddc.design.NUM_STAGES
        assert np.all(np.isfinite(baseband))
        assert np.iscomplexobj(baseband)
        # Roughly N/16 output samples, allowing for per-stage startup phase.
        assert abs(len(baseband) - len(signal) // 16) <= 4

    def test_rates_are_normalized_not_hz(self, ddc, cascade):
        """Regression guard for the NCO/DDC absolute-rate overflow.

        Passing GHz values to DDC raises for fpga_fixed and silently yields
        NaN for cf24/half. run_cascade must normalize, so these dtypes have
        to produce finite output.
        """
        signal, meta = ddc.simulate.synthesize(num_samples=8192)
        for dtype in ("fpga_fixed", "cf24", "half"):
            baseband, _ = ddc.design.run_cascade(signal, cascade, meta, dtype)
            assert np.all(np.isfinite(baseband)), \
                f"{dtype} produced non-finite output — rates not normalized?"

    def test_run_all_scores_stages_against_reference(self, ddc, cascade):
        signal, meta = ddc.simulate.synthesize(num_samples=16384)
        results = ddc.design.run_all(signal, cascade, meta,
                                     ["reference", "gpu_baseline"])
        by_dtype = {r.dtype: r for r in results}
        assert by_dtype["reference"].output_snr_db is None
        assert by_dtype["reference"].stage_snr_db == []
        assert len(by_dtype["gpu_baseline"].stage_snr_db) == 4
        assert by_dtype["gpu_baseline"].output_snr_db > 30.0


class TestQuantization:
    def test_fixed_point_round_trip(self, ddc, cascade):
        taps = cascade[0]
        dequantized, codes, frac_bits = ddc.quantize.quantize_taps(taps, 18)
        assert codes.dtype == np.int64
        assert np.all(np.abs(codes) <= 2 ** 17)
        np.testing.assert_allclose(dequantized, codes / 2.0 ** frac_bits,
                                   atol=0)
        # 18 bits should track the float taps closely.
        assert np.max(np.abs(dequantized - taps)) < 1e-4

    def test_degradation_is_monotone_in_width(self, ddc, cascade):
        """Acceptance: the sweep shows degradation vs. coefficient width."""
        rows = ddc.quantize.sweep_bit_widths(
            cascade, ddc.design.CascadeSpec(), widths=[8, 12, 16, 20])
        atten = [r["alias_attenuation_db"] for r in rows]
        # Narrow words must be clearly worse than wide ones.
        assert atten[0] < atten[-1] - 20.0
        assert rows[0]["passes"] is False
        assert rows[-1]["passes"] is True

    def test_narrowest_passing_width_is_reported(self, ddc, cascade):
        rows = ddc.quantize.sweep_bit_widths(cascade,
                                             ddc.design.CascadeSpec())
        chosen = ddc.quantize.smallest_passing_width(rows)
        assert chosen is not None
        assert 8 < chosen <= 18
        # And it really does pass at that width.
        assert next(r for r in rows if r["bits"] == chosen)["passes"]

    def test_dtype_projection_sweep(self, ddc, cascade):
        rows = ddc.quantize.sweep_coeff_dtypes(
            cascade, ddc.design.CascadeSpec(),
            dtypes=["reference", "posit_8_2"])
        by_dtype = {r["dtype"]: r for r in rows}
        assert by_dtype["reference"]["max_projection_error"] == 0.0
        assert by_dtype["reference"]["passes"]
        # An 8-bit posit cannot hold these coefficients.
        assert by_dtype["posit_8_2"]["max_projection_error"] > 1e-4
        assert not by_dtype["posit_8_2"]["passes"]


class TestCoefficientEmitters:
    def test_coe_format(self, ddc, cascade):
        text = ddc.vivado.render_coe(cascade[2], 18, "stage3", 3, 2)
        assert "radix = 10;" in text
        assert "coefdata =" in text
        assert text.rstrip().endswith(";")
        # Metadata a porter needs.
        assert "scale         : 2^" in text
        assert "word length   : 18 bits, signed" in text
        # One entry per tap.
        body = text.split("coefdata =", 1)[1]
        assert len([v for v in body.replace(";", "").split(",")
                    if v.strip()]) == len(cascade[2])

    def test_coe_values_are_integers_in_range(self, ddc, cascade):
        text = ddc.vivado.render_coe(cascade[0], 12, "stage1", 1, 2)
        body = text.split("coefdata =", 1)[1].replace(";", "")
        values = [int(v.strip()) for v in body.split(",") if v.strip()]
        assert all(-2048 <= v <= 2047 for v in values)

    def test_readmemh_is_twos_complement(self, ddc):
        assert ddc.verilog.to_twos_complement(-95, 18) == 0x3FFA1
        assert ddc.verilog.to_twos_complement(4, 18) == 0x00004
        assert ddc.verilog.to_twos_complement(-1, 8) == 0xFF
        with pytest.raises(ValueError):
            ddc.verilog.to_twos_complement(200, 8)

    def test_hex_file_shape(self, ddc, cascade):
        text = ddc.verilog.render_hex(cascade[2], 18)
        lines = text.strip().splitlines()
        assert len(lines) == len(cascade[2])
        # 18 bits -> 5 hex digits, no prefixes (readmemh wants bare words).
        assert all(len(l) == 5 and not l.startswith("0x") for l in lines)
        assert all(int(l, 16) < 2 ** 18 for l in lines)

    def test_hex_round_trips_to_the_quantized_taps(self, ddc, cascade):
        """The hex file must carry the same numbers as the .coe."""
        taps, bits = cascade[1], 16
        _, codes, _ = ddc.quantize.quantize_taps(taps, bits)
        hex_lines = ddc.verilog.render_hex(taps, bits).strip().splitlines()
        for code, line in zip(codes, hex_lines):
            decoded = int(line, 16)
            if decoded >= 2 ** (bits - 1):
                decoded -= 2 ** bits
            assert decoded == int(code)

    def test_verilog_params_header(self, ddc, cascade):
        text = ddc.verilog.render_params(cascade, 18)
        for macro in ("`define COEFF_WIDTH", "`define NUM_STAGES",
                      "`define TOTAL_DECIMATION", "`define STAGE1_TAPS",
                      "`define STAGE4_FRAC"):
            assert macro in text, f"missing {macro}"
        assert "`ifndef CASCADE_PARAMS_VH" in text
        assert "`endif" in text

    def test_emitters_write_all_stages(self, ddc, tmp_path):
        coe = ddc.vivado.emit_all(tmp_path / "vivado", 18)
        assert len(coe) == 4 and all(p.suffix == ".coe" for p in coe)

        hexes = ddc.verilog.emit_all(tmp_path / "verilog", 18)
        assert len(hexes) == 5  # four stages plus the params header
        assert (tmp_path / "verilog" / "cascade_params.vh").is_file()
        for path in coe + hexes:
            assert path.stat().st_size > 0


class TestDDCScriptsRun:
    def test_design_main(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(_DDC_DIR / "design.py"),
             "--num-samples", "16384", "--outdir", str(tmp_path),
             "--dtypes", "reference", "posit_full"],
            capture_output=True, text=True, timeout=900)
        assert result.returncode == 0, result.stderr
        assert "spec: PASS" in result.stdout
        assert (tmp_path / "summary.csv").is_file()

    def test_quantize_main(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(_DDC_DIR / "quantize_coefficients.py"),
             "--outdir", str(tmp_path)],
            capture_output=True, text=True, timeout=900)
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "quantization.csv").is_file()

    def test_emitter_mains(self, tmp_path):
        for script, subdir, suffix in (
                ("emit_coeffs_vivado.py", "vivado", ".coe"),
                ("emit_coeffs_verilog.py", "verilog", ".hex")):
            out = tmp_path / subdir
            result = subprocess.run(
                [sys.executable, str(_DDC_DIR / script),
                 "--outdir", str(out), "--bits", "16"],
                capture_output=True, text=True, timeout=600)
            assert result.returncode == 0, result.stderr
            assert list(out.glob(f"*{suffix}"))


# =============================================================================
# Demo 3 — active vibration cancellation (#60)
# =============================================================================

_ANC_DIR = Path(__file__).resolve().parents[1] / "demos" / "03_vibration_cancellation"


@pytest.fixture
def anc():
    names = ("design", "simulate", "emit_c_header")
    with _demo_modules(_ANC_DIR, names) as mods:
        yield types_ns(design=mods["design"], simulate=mods["simulate"],
                       emit=mods["emit_c_header"])


# Adaptive training over software-float types is slow, so the tests use a
# short run. Long enough for LMS/NLMS to converge and for the narrow RLS
# configurations to fail, which is what is being asserted.
_ANC_SAMPLES = 6000


@pytest.fixture
def disturbance(anc):
    return anc.simulate.synthesize(num_samples=_ANC_SAMPLES)


class TestDisturbance:
    def test_shapes_and_metadata(self, anc, disturbance):
        reference, primary, meta = disturbance
        assert reference.shape == primary.shape == (_ANC_SAMPLES,)
        assert np.all(np.isfinite(reference)) and np.all(np.isfinite(primary))
        assert meta["speed_change_at"] == _ANC_SAMPLES // 2
        assert meta["achievable_reduction_db"] == meta["sensor_snr_db"]

    def test_sensor_floor_is_real(self, anc, disturbance):
        """The noise floor must actually bound cancellation.

        Without it the identification is exact and every dtype reports
        300+ dB, which makes the whole comparison meaningless.
        """
        reference, primary, meta = disturbance
        clean = np.convolve(reference,
                            meta["primary_path"])[:len(primary)]
        residual_noise = primary - clean
        measured_snr = 10.0 * np.log10(
            np.mean(clean ** 2) / np.mean(residual_noise ** 2))
        assert measured_snr == pytest.approx(meta["sensor_snr_db"], abs=1.0)

    def test_speed_change_is_present(self, anc, disturbance):
        """The second half must genuinely differ, or there is nothing to
        re-converge on."""
        reference, _, meta = disturbance
        split = meta["speed_change_at"]
        first = np.abs(np.fft.rfft(reference[:split]))
        second = np.abs(np.fft.rfft(reference[split:]))
        assert np.argmax(first) != np.argmax(second)


class TestTraining:
    @pytest.mark.parametrize("algorithm", ["LMS", "NLMS", "RLS"])
    def test_reference_reaches_the_sensor_floor(self, anc, disturbance,
                                                algorithm):
        """Acceptance: all three variants train. In double each must reach
        the physical limit, or the comparison has no baseline."""
        reference, primary, meta = disturbance
        result = anc.design.train(algorithm, "reference", reference, primary,
                                  meta)
        assert result.status == "ok"
        assert result.reduction_db == pytest.approx(
            meta["achievable_reduction_db"], abs=4.0)

    @pytest.mark.parametrize("algorithm", ["LMS", "NLMS"])
    def test_gradient_filters_degrade_gracefully(self, anc, disturbance,
                                                 algorithm):
        """LMS and NLMS must still cancel at 8-bit, just less.

        The contrast with RLS is the demo's point, so it needs pinning on
        both sides.
        """
        reference, primary, meta = disturbance
        coarse = anc.design.train(algorithm, "posit_8_2", reference, primary,
                                  meta)
        assert coarse.status == "ok"
        assert coarse.reduction_db > 3.0        # still cancelling
        fine = anc.design.train(algorithm, "reference", reference, primary,
                                meta)
        assert coarse.reduction_db < fine.reduction_db - 5.0   # but worse

    def test_rls_fails_at_narrow_precision(self, anc, disturbance):
        """Acceptance: at least one dtype/filter combination diverges.

        Reproduces the notebooks/06_estimation finding that RLS's Kalman-form
        update loses P-matrix symmetry at narrow precision.
        """
        reference, primary, meta = disturbance
        result = anc.design.train("RLS", "half", reference, primary, meta)
        assert result.status in ("DIVERGED", "AMPLIFIED")

    def test_divergence_is_distinguished_from_amplification(self, anc):
        """They are different failures: one produces NaN, the other produces
        a filter that drives the structure. Conflating them would hide the
        more dangerous case, which looks like a working filter."""
        result = anc.design.TrainingResult(
            algorithm="RLS", dtype="x", errors=np.zeros(4),
            weights=np.zeros(4), diverged=True)
        assert result.status == "DIVERGED"
        result.diverged = False
        result.amplified = True
        assert result.status == "AMPLIFIED"

    def test_weight_drift_is_scored_against_the_same_algorithm(
            self, anc, disturbance):
        reference, primary, meta = disturbance
        results = anc.design.run_all(reference, primary, meta,
                                     ["LMS"], ["reference", "half"])
        by_dtype = {r.dtype: r for r in results}
        assert np.isnan(by_dtype["reference"].weight_drift)
        assert by_dtype["half"].weight_drift > 0.0

    def test_diverged_runs_are_not_scored(self, anc, disturbance):
        """A diverged run must not report a reduction figure — a number
        there would read as a measurement rather than a failure."""
        reference, primary, meta = disturbance
        result = anc.design.train("RLS", "posit_8_2", reference, primary, meta)
        if result.diverged:
            assert np.isnan(result.reduction_db)
            assert result.notes


class TestAncArtifacts:
    def test_summary_csv(self, anc, disturbance, tmp_path):
        reference, primary, meta = disturbance
        results = anc.design.run_all(reference, primary, meta,
                                     ["LMS"], ["reference", "half"])
        anc.design.write_summary_csv(tmp_path / "summary.csv", results, meta)
        rows = (tmp_path / "summary.csv").read_text().strip().splitlines()
        assert rows[0].startswith("algorithm,dtype,")
        assert len(rows) == 1 + len(results)

    def test_summary_png(self, anc, disturbance, tmp_path):
        pytest.importorskip("matplotlib")
        reference, primary, meta = disturbance
        results = anc.design.run_all(reference, primary, meta,
                                     ["LMS", "RLS"], ["reference", "half"])
        assert anc.design.plot_summary(tmp_path / "summary.png", results,
                                       primary, meta)
        assert (tmp_path / "summary.png").stat().st_size > 0

    def test_diverged_trace_breaks_rather_than_flatlines(self, anc):
        """A diverged run must leave a gap in the plot. Flat-lining at the
        floor would read as perfect cancellation."""
        errors = np.concatenate([np.ones(512), np.full(512, np.nan)])
        trace = anc.design._smoothed_error_power_db(errors, window=64)
        assert np.all(np.isfinite(trace[:400]))
        assert np.all(np.isnan(trace[-400:]))


class TestAncCHeader:
    def test_header_contents(self, anc):
        text = anc.emit.render_header("NLMS", "reference",
                                      num_samples=_ANC_SAMPLES)
        assert "#ifndef CANCELLER_H" in text and "#endif" in text
        assert "CANCELLER_NUM_TAPS   24" in text
        assert "canceller_taps[CANCELLER_NUM_TAPS]" in text
        # Adaptation parameters ship alongside the taps — see the README.
        assert "CANCELLER_STEP_SIZE" in text
        assert "-y[n]" in text          # sign convention stated

    def test_refuses_a_diverged_run(self, anc):
        """Exporting NaN weights into an actuator loop is the worst possible
        place to discover a divergence."""
        with pytest.raises(ValueError, match="diverged|amplified"):
            anc.emit.render_header("RLS", "half", num_samples=_ANC_SAMPLES)

    def test_rejects_an_unknown_algorithm(self, anc):
        with pytest.raises(ValueError, match="unknown algorithm"):
            anc.emit.render_header("SGD", "reference",
                                   num_samples=_ANC_SAMPLES)

    @pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not present")
    @pytest.mark.parametrize("std", ["c89", "c99"])
    def test_header_compiles(self, anc, tmp_path, std):
        header = tmp_path / "canceller.h"
        header.write_text(anc.emit.render_header("NLMS", "reference",
                                                 num_samples=_ANC_SAMPLES))
        source = tmp_path / "tu.c"
        source.write_text(
            '#include "canceller.h"\n'
            "int main(void) {\n"
            "    return (CANCELLER_NUM_TAPS == 24\n"
            "            && canceller_taps[0] == canceller_taps[0]) ? 0 : 1;\n"
            "}\n")
        binary = tmp_path / "tu"
        compiled = subprocess.run(
            ["gcc", f"-std={std}", "-Wall", "-Wextra", "-pedantic", "-Werror",
             "-I", str(tmp_path), str(source), "-o", str(binary)],
            capture_output=True, text=True, timeout=120)
        assert compiled.returncode == 0, compiled.stderr
        assert subprocess.run([str(binary)], timeout=60).returncode == 0


class TestAncScriptsRun:
    def test_design_main(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(_ANC_DIR / "design.py"),
             "--num-samples", "4000", "--outdir", str(tmp_path),
             "--algorithms", "LMS", "--dtypes", "reference", "half"],
            capture_output=True, text=True, timeout=900)
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "summary.csv").is_file()

    def test_simulate_main(self):
        result = subprocess.run(
            [sys.executable, str(_ANC_DIR / "simulate.py"),
             "--num-samples", "4000"],
            capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr
        assert "sensor floor" in result.stdout

    def test_emit_main(self, tmp_path):
        out = tmp_path / "canceller.h"
        result = subprocess.run(
            [sys.executable, str(_ANC_DIR / "emit_c_header.py"),
             "--out", str(out), "--num-samples", "6000"],
            capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr
        assert out.is_file() and out.stat().st_size > 0


# =============================================================================
# Demo 2 — motor current loop + resonance notch (#59)
# =============================================================================

_MOTOR_DIR = Path(__file__).resolve().parents[1] / "demos" / "02_motor_servo"


@pytest.fixture
def motor_demo():
    with _demo_modules(_MOTOR_DIR,
                       ("design", "simulate", "emit_c_header")) as mods:
        yield types_ns(design=mods["design"], simulate=mods["simulate"],
                       emit=mods["emit_c_header"])


class TestMotorPlant:
    def test_discrete_plant_matches_the_analytic_model(self, motor_demo):
        """ZOH of 1/(Ls+R): DC gain must be 1/R and the pole exp(-T/tau)."""
        motor = motor_demo.simulate.Motor()
        num, den = motor.discrete_plant()
        assert num.sum() / den.sum() == pytest.approx(
            1.0 / motor.resistance_ohm, rel=1e-9)
        assert -den[1] == pytest.approx(
            np.exp(-1.0 / (motor_demo.simulate.SAMPLE_RATE_HZ
                           * motor.electrical_tau_s)), rel=1e-12)

    def test_electrical_time_constant(self, motor_demo):
        motor = motor_demo.simulate.Motor(resistance_ohm=0.05,
                                          inductance_h=50e-6)
        assert motor.electrical_tau_s == pytest.approx(1e-3)

    def test_plant_is_stable(self, motor_demo):
        motor = motor_demo.simulate.Motor()
        assert 0.0 < motor.plant_pole() < 1.0

    def test_close_loop_pads_before_adding(self, motor_demo):
        """T = L/(1+L). Adding unequal-length polynomials without padding
        shifts the system by a sample and yields a plausible wrong answer."""
        num, den = motor_demo.simulate.close_loop(
            np.array([0.5]), np.array([1.0, -0.9]))
        assert len(num) == len(den)
        np.testing.assert_allclose(den, [1.0, -0.4])

    def test_step_response_of_a_known_system(self, motor_demo):
        """First-order lag: y[n] = 0.5 y[n-1] + 0.5 x[n] settles at 1."""
        step = motor_demo.simulate.step_response(
            np.array([0.5]), np.array([1.0, -0.5]), num_samples=60)
        assert step[-1] == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.diff(step) >= -1e-12)      # monotone, no overshoot

    def test_divergent_system_is_truncated(self, motor_demo):
        step = motor_demo.simulate.step_response(
            np.array([1.0]), np.array([1.0, -1.5]), num_samples=4000)
        assert not np.all(np.isfinite(step))

    def test_metrics_flag_divergence(self, motor_demo):
        metrics = motor_demo.simulate.response_metrics(
            np.array([1.0, np.nan]), 20000.0)
        assert metrics["diverged"]


class TestMotorDesign:
    def test_pi_zero_cancels_the_plant_pole(self, motor_demo):
        """The tuning premise. If this drifts, the loop is not what the
        crossover formula assumes."""
        motor = motor_demo.simulate.Motor()
        num, den, _ = motor_demo.design.design_pi(motor, 1200.0)
        np.testing.assert_allclose(den, [1.0, -1.0])          # integrator
        assert -num[1] / num[0] == pytest.approx(motor.plant_pole(),
                                                 rel=1e-12)

    def test_crossover_scales_the_gain(self, motor_demo):
        motor = motor_demo.simulate.Motor()
        _, _, slow = motor_demo.design.design_pi(motor, 600.0)
        _, _, fast = motor_demo.design.design_pi(motor, 1200.0)
        assert fast == pytest.approx(2.0 * slow, rel=1e-9)

    def test_notch_attenuates_at_its_centre(self, motor_demo):
        point = motor_demo.design.DESIGN_POINTS["conservative"]
        num, den = motor_demo.design.design_notch(point)
        depth = motor_demo.design.notch_depth_db(num, den, point.notch_hz)
        assert depth < -40.0
        # ...and passes DC and Nyquist essentially untouched.
        assert motor_demo.design.notch_depth_db(num, den, 10.0) > -1.0

    @pytest.mark.parametrize("point_name", ["conservative", "aggressive"])
    def test_reference_design_is_stable_and_accurate(self, motor_demo,
                                                     point_name):
        point = motor_demo.design.DESIGN_POINTS[point_name]
        result = next(iter(motor_demo.design.sweep(point, ["reference"])))
        assert result.status == "ok"
        assert result.max_pole < 1.0
        assert result.metrics["steady_state"] == pytest.approx(1.0, abs=0.01)

    def test_all_seven_dtypes_sweep(self, motor_demo):
        """Acceptance: plant + controller + notch across 7 dtypes."""
        point = motor_demo.design.DESIGN_POINTS["conservative"]
        results = motor_demo.design.sweep(
            point, motor_demo.design.DEFAULT_DTYPES)
        assert len(results) == 7
        assert len(motor_demo.design.DEFAULT_DTYPES) == 7
        for r in results:
            assert np.isfinite(r.max_pole)
            assert np.isfinite(r.notch_depth_db)

    def test_notch_depth_degrades_with_precision(self, motor_demo):
        """The conservative point's headline: precision costs notch depth."""
        point = motor_demo.design.DESIGN_POINTS["conservative"]
        by_dtype = {r.dtype: r for r in motor_demo.design.sweep(
            point, ["reference", "half", "posit_8_2"])}
        assert by_dtype["reference"].notch_depth_db < -100.0
        assert by_dtype["half"].notch_depth_db < -40.0
        # 8-bit barely notches at all.
        assert by_dtype["posit_8_2"].notch_depth_db > -25.0

    def test_conservative_point_is_stable_at_every_dtype(self, motor_demo):
        point = motor_demo.design.DESIGN_POINTS["conservative"]
        results = motor_demo.design.sweep(
            point, motor_demo.design.DEFAULT_DTYPES)
        assert all(r.stable for r in results)

    def test_aggressive_point_destabilizes_at_8_bit(self, motor_demo):
        """Acceptance: the step-response sweep must show which dtypes are
        stable and which are not — so at least one must not be."""
        point = motor_demo.design.DESIGN_POINTS["aggressive"]
        by_dtype = {r.dtype: r for r in motor_demo.design.sweep(
            point, ["reference", "posit_8_2"])}
        assert by_dtype["reference"].stable
        assert not by_dtype["posit_8_2"].stable
        assert by_dtype["posit_8_2"].status == "UNSTABLE"

    def test_the_failure_is_a_pole_on_the_unit_circle(self, motor_demo):
        """Pin the mechanism, not just the symptom.

        At posit<8,2> the notch denominator rounds to [1, -1.5, 0.5], whose
        roots are exactly 1.0 and 0.5 — quantization lands a pole on the
        boundary and converts the notch into an integrator.
        """
        point = motor_demo.design.DESIGN_POINTS["aggressive"]
        _, den = motor_demo.design.design_notch(point)
        quantized = motor_demo.design._quantize(den, "posit_8_2")
        np.testing.assert_allclose(quantized, [1.0, -1.5, 0.5], atol=1e-12)
        assert np.max(np.abs(np.roots(quantized))) == pytest.approx(1.0,
                                                                    abs=1e-12)

    def test_design_margin_is_what_differs(self, motor_demo):
        """The demo's thesis: same dtype, different margin, opposite outcome."""
        conservative = next(iter(motor_demo.design.sweep(
            motor_demo.design.DESIGN_POINTS["conservative"], ["posit_8_2"])))
        aggressive = next(iter(motor_demo.design.sweep(
            motor_demo.design.DESIGN_POINTS["aggressive"], ["posit_8_2"])))
        assert conservative.stable
        assert not aggressive.stable


class TestMotorArtifacts:
    def test_summary_and_step_csv(self, motor_demo, tmp_path):
        point = motor_demo.design.DESIGN_POINTS["conservative"]
        results = motor_demo.design.sweep(point, ["reference", "half"])
        motor_demo.design.write_summary_csv(tmp_path / "summary.csv", results)
        motor_demo.design.write_step_csv(tmp_path / "steps.csv", results)

        summary = (tmp_path / "summary.csv").read_text().strip().splitlines()
        assert summary[0].startswith("design_point,dtype,")
        assert len(summary) == 1 + len(results)

        steps = (tmp_path / "steps.csv").read_text().strip().splitlines()
        assert steps[0].startswith("time_ms,")
        assert len(steps) > 100

    def test_summary_png(self, motor_demo, tmp_path):
        pytest.importorskip("matplotlib")
        by_point = {
            name: motor_demo.design.sweep(
                motor_demo.design.DESIGN_POINTS[name], ["reference", "half"])
            for name in ("conservative", "aggressive")}
        assert motor_demo.design.plot_summary(
            tmp_path / "summary.png", by_point, motor_demo.simulate.Motor())
        assert (tmp_path / "summary.png").stat().st_size > 0


class TestMotorCHeader:
    def test_header_contents(self, motor_demo):
        text = motor_demo.emit.render_header("conservative", "reference")
        assert "#ifndef CURRENT_LOOP_H" in text and "#endif" in text
        for macro in ("LOOP_RATE_HZ", "PI_B0", "PI_B1", "PI_KP",
                      "NOTCH_B0", "NOTCH_A1", "NOTCH_A2", "NOTCH_CENTER_HZ"):
            assert macro in text, f"missing {macro}"
        # The difference equations a porter needs.
        assert "u[n] = u[n-1]" in text
        # And the warning that the PI zero is motor-specific.
        assert "Re-run this exporter if R or L change" in text

    def test_refuses_an_unstable_configuration(self, motor_demo):
        with pytest.raises(ValueError, match="unstable"):
            motor_demo.emit.render_header("aggressive", "posit_8_2")

    def test_rejects_an_unknown_design_point(self, motor_demo):
        with pytest.raises(ValueError, match="unknown design point"):
            motor_demo.emit.render_header("reckless", "reference")

    def test_coefficients_match_the_design(self, motor_demo):
        """The header must carry the numbers the sweep actually validated."""
        motor = motor_demo.simulate.Motor()
        point = motor_demo.design.DESIGN_POINTS["conservative"]
        c_num, _, _ = motor_demo.design.design_pi(motor, point.crossover_hz)
        text = motor_demo.emit.render_header("conservative", "reference")
        assert f"{c_num[0]:.17g}" in text
        assert f"{c_num[1]:.17g}" in text

    @pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not present")
    @pytest.mark.parametrize("std", ["c89", "c99"])
    def test_header_compiles(self, motor_demo, tmp_path, std):
        header = tmp_path / "current_loop.h"
        header.write_text(
            motor_demo.emit.render_header("conservative", "reference",
                                          name="current_loop.h"))
        source = tmp_path / "tu.c"
        source.write_text(
            '#include "current_loop.h"\n'
            "int main(void) {\n"
            "    return (LOOP_RATE_HZ > 0.0 && PI_KP > 0.0\n"
            "            && NOTCH_CENTER_HZ > 0.0) ? 0 : 1;\n"
            "}\n")
        binary = tmp_path / "tu"
        compiled = subprocess.run(
            ["gcc", f"-std={std}", "-Wall", "-Wextra", "-pedantic", "-Werror",
             "-I", str(tmp_path), str(source), "-o", str(binary)],
            capture_output=True, text=True, timeout=120)
        assert compiled.returncode == 0, compiled.stderr
        assert subprocess.run([str(binary)], timeout=60).returncode == 0


class TestMotorScriptsRun:
    def test_design_main(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(_MOTOR_DIR / "design.py"),
             "--outdir", str(tmp_path)],
            capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr
        assert "conservative" in result.stdout and "aggressive" in result.stdout
        assert "UNSTABLE" in result.stdout      # the 8-bit aggressive case
        for name in ("summary.csv", "step_responses.csv"):
            assert (tmp_path / name).is_file()

    def test_simulate_main(self):
        result = subprocess.run(
            [sys.executable, str(_MOTOR_DIR / "simulate.py")],
            capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr
        assert "electrical tau" in result.stdout

    def test_emit_main_and_refusal_exit_code(self, tmp_path):
        out = tmp_path / "current_loop.h"
        ok = subprocess.run(
            [sys.executable, str(_MOTOR_DIR / "emit_c_header.py"),
             "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        assert ok.returncode == 0, ok.stderr
        assert out.is_file()

        refused = subprocess.run(
            [sys.executable, str(_MOTOR_DIR / "emit_c_header.py"),
             "--design-point", "aggressive", "--dtype", "posit_8_2",
             "--out", str(tmp_path / "bad.h")],
            capture_output=True, text=True, timeout=300)
        assert refused.returncode == 1
        assert not (tmp_path / "bad.h").exists()
