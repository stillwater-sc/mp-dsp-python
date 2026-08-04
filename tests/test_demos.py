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
