"""Tests for the reference demonstrations under demos/ (Issue #62).

Demo 1 — audio dynamics processor (#58). These are integration tests: they
run the real chain, on a short clip, at every dtype the demo advertises, and
compile the header it emits. Short clips keep the suite fast; correctness of
the individual primitives is covered by their own test modules.
"""

from __future__ import annotations

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


@pytest.fixture(scope="module")
def demo():
    """Import the demo modules with their own directory on sys.path.

    They import each other by bare name (`from design import ...`), the way
    they are meant to be run — `python demos/01_audio_dynamics/design.py`.
    """
    if not _DEMO_DIR.is_dir():
        pytest.skip(f"demo directory missing: {_DEMO_DIR}")
    sys.path.insert(0, str(_DEMO_DIR))
    try:
        import design
        import emit_c_header
        import simulate
        yield types_ns(design=design, simulate=simulate,
                       emit_c_header=emit_c_header)
    finally:
        sys.path.remove(str(_DEMO_DIR))


def types_ns(**kwargs):
    """Tiny attribute bag — clearer at the call site than a dict."""
    return type("DemoModules", (), kwargs)


@pytest.fixture(scope="module")
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
