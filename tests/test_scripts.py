"""Tests for the CSV visualization scripts."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def csv_dir():
    """Create a temp directory with minimal valid CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # iir_precision_sweep.csv
        with open(os.path.join(tmpdir, "iir_precision_sweep.csv"), "w") as f:
            f.write("filter_family,arith_type,bits,max_abs_error,max_rel_error,"
                    "sqnr_db,pole_displacement,stability_margin\n")
            f.write("Butterworth,double,64,0,0,300,0,0.1\n")
            f.write("Butterworth,float,32,1.5e-7,1.4e-6,119.1,0,0.1\n")
            f.write('Butterworth,"cfloat<24,5>",24,1.4e-6,1.3e-5,95.1,0,0.1\n')
            f.write("Butterworth,half,16,5e-4,4.7e-3,50.1,0,0.1\n")
            f.write('Butterworth,"posit<32,2>",32,9.8e-9,9e-8,149.4,0,0.1\n')
            f.write('Butterworth,"posit<16,1>",16,1.8e-4,1.7e-3,51.5,0,0.1\n')
            f.write("Bessel,double,64,0,0,300,0,0.41\n")
            f.write("Bessel,float,32,2e-8,8e-8,141.8,0,0.41\n")

        # frequency_response.csv
        with open(os.path.join(tmpdir, "frequency_response.csv"), "w") as f:
            f.write("filter_family,arith_type,freq_hz,magnitude_db,phase_deg,"
                    "ref_magnitude_db,ref_phase_deg\n")
            for freq in [0, 1000, 2000, 5000, 10000]:
                f.write(f"Butterworth,double,{freq},-3.0,-45.0,-3.0,-45.0\n")
                f.write(f"Butterworth,float,{freq},-3.01,-45.1,-3.0,-45.0\n")

        # pole_positions.csv
        with open(os.path.join(tmpdir, "pole_positions.csv"), "w") as f:
            f.write("filter_family,arith_type,pole_index,real,imag,"
                    "ref_real,ref_imag,displacement\n")
            f.write("Butterworth,double,0,0.866,0.234,0.866,0.234,0\n")
            f.write("Butterworth,double,1,0.866,-0.234,0.866,-0.234,0\n")
            f.write("Butterworth,float,0,0.866,0.234,0.866,0.234,0\n")
            f.write("Butterworth,float,1,0.866,-0.234,0.866,-0.234,0\n")

        # impulse_response.csv — long-format per-sample rows
        with open(os.path.join(tmpdir, "impulse_response.csv"), "w") as f:
            f.write("filter_family,arith_type,sample_index,value,ref_value\n")
            import math as _math
            for atype, scale in (("double", 1.0), ("float", 0.999)):
                for n in range(8):
                    v = _math.exp(-n * 0.4) * _math.cos(n * 0.6) * scale
                    f.write(f"Butterworth,{atype},{n},{v:.6f},"
                            f"{_math.exp(-n*0.4)*_math.cos(n*0.6):.6f}\n")

        yield tmpdir


def _run_script(script_name: str, csv_dir: str, output_dir: str) -> subprocess.CompletedProcess:
    """Run a plotting script and return the result.

    The subprocess.run calls in this file all receive the active Python
    interpreter and a script path resolved from `__file__` plus
    tempdir arguments constructed locally — no user/network input
    reaches the command line. Ruff's S603 fires on every
    subprocess.run regardless of provenance, so tag these specific
    call sites as intentional.
    """
    script_path = Path(__file__).parent.parent / "scripts" / script_name
    return subprocess.run(  # noqa: S603 - test-controlled script + tempdirs
        [sys.executable, str(script_path), csv_dir, "--output", output_dir],
        capture_output=True, text=True, timeout=30)


def test_plot_precision_generates_output(csv_dir):
    """Legacy invocation: positional csv_dir + --output."""
    with tempfile.TemporaryDirectory() as outdir:
        result = _run_script("plot_precision.py", csv_dir, outdir)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        # All five figures x two formats.
        expected_stems = ["magnitude_response", "phase_response",
                          "magnitude_error", "phase_error",
                          "impulse_response"]
        for stem in expected_stems:
            for ext in ("png", "pdf"):
                assert os.path.exists(os.path.join(outdir, f"{stem}.{ext}")), (
                    f"Missing {stem}.{ext}")


def test_plot_precision_new_cli_flags(csv_dir):
    """Issue #12-spec'd invocation: --input-dir / --output-dir."""
    script_path = Path(__file__).parent.parent / "scripts" / "plot_precision.py"
    with tempfile.TemporaryDirectory() as outdir:
        result = subprocess.run(  # noqa: S603 - test-controlled script + tempdirs
            [sys.executable, str(script_path),
             "--input-dir", csv_dir, "--output-dir", outdir],
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(os.path.join(outdir, "magnitude_response.png"))
        assert os.path.exists(os.path.join(outdir, "magnitude_response.pdf"))


def test_plot_precision_skips_impulse_when_csv_missing(csv_dir):
    """impulse_response.csv is documented as optional — script must still
    emit the other four figures if it's absent."""
    os.remove(os.path.join(csv_dir, "impulse_response.csv"))
    with tempfile.TemporaryDirectory() as outdir:
        result = _run_script("plot_precision.py", csv_dir, outdir)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        # Four figures still present.
        for stem in ("magnitude_response", "phase_response",
                     "magnitude_error", "phase_error"):
            assert os.path.exists(os.path.join(outdir, f"{stem}.png"))
        # Impulse is absent.
        assert not os.path.exists(os.path.join(outdir, "impulse_response.png"))


def test_plot_precision_publication_flag(csv_dir):
    """--publication should not error and should still produce outputs."""
    script_path = Path(__file__).parent.parent / "scripts" / "plot_precision.py"
    with tempfile.TemporaryDirectory() as outdir:
        result = subprocess.run(  # noqa: S603 - test-controlled script + tempdirs
            [sys.executable, str(script_path),
             "--input-dir", csv_dir, "--output-dir", outdir,
             "--publication"],
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(os.path.join(outdir, "magnitude_response.pdf"))


def test_plot_heatmap_generates_output(csv_dir):
    """All four summary figures land in both PNG and PDF."""
    with tempfile.TemporaryDirectory() as outdir:
        result = _run_script("plot_heatmap.py", csv_dir, outdir)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        expected_stems = ["error_heatmap", "sqnr_heatmap",
                          "sqnr_bar_chart", "precision_cost_frontier"]
        for stem in expected_stems:
            for ext in ("png", "pdf"):
                assert os.path.exists(os.path.join(outdir, f"{stem}.{ext}")), (
                    f"Missing {stem}.{ext}")


def test_plot_heatmap_new_cli_flags(csv_dir):
    """Issue #13 spec'd invocation: --input-dir / --output-dir."""
    script_path = Path(__file__).parent.parent / "scripts" / "plot_heatmap.py"
    with tempfile.TemporaryDirectory() as outdir:
        result = subprocess.run(  # noqa: S603 - test-controlled script + tempdirs
            [sys.executable, str(script_path),
             "--input-dir", csv_dir, "--output-dir", outdir],
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(os.path.join(outdir, "error_heatmap.pdf"))
        assert os.path.exists(os.path.join(outdir, "sqnr_heatmap.pdf"))


def test_plot_pole_zero_generates_output(csv_dir):
    """Both pole-zero and displacement-summary figures in PNG and PDF."""
    with tempfile.TemporaryDirectory() as outdir:
        result = _run_script("plot_pole_zero.py", csv_dir, outdir)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        for stem in ("pole_zero", "pole_displacement"):
            for ext in ("png", "pdf"):
                assert os.path.exists(os.path.join(outdir, f"{stem}.{ext}")), (
                    f"Missing {stem}.{ext}")


def test_plot_pole_zero_new_cli_flags(csv_dir):
    """Issue #14 spec'd invocation: --input-dir / --output-dir.

    Mirrors the full artifact set asserted in
    `test_plot_pole_zero_generates_output` — both figures, both formats —
    so a regression that skips one of the two plots on the new CLI path
    still fails this test. (Initial version only checked pole_zero.pdf,
    which left pole_displacement.pdf unguarded.)
    """
    script_path = Path(__file__).parent.parent / "scripts" / "plot_pole_zero.py"
    with tempfile.TemporaryDirectory() as outdir:
        result = subprocess.run(  # noqa: S603 - test-controlled script + tempdirs
            [sys.executable, str(script_path),
             "--input-dir", csv_dir, "--output-dir", outdir],
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        for stem in ("pole_zero", "pole_displacement"):
            for ext in ("png", "pdf"):
                assert os.path.exists(os.path.join(outdir, f"{stem}.{ext}")), (
                    f"Missing {stem}.{ext}")


# ---------------------------------------------------------------------------
# build_api_ref.py (Issue #116)
#
# This script rotted silently for months: it did not parse on Python < 3.12
# (a backslash inside an f-string expression, legal only from PEP 701), and
# its CATEGORIES/CLASSES tables fell 61 names behind the bindings while the
# committed docs/api_reference.md was hand-edited around it. These tests
# make both failure modes loud.
# ---------------------------------------------------------------------------

_API_REF_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_api_ref.py"


class TestBuildApiRef:
    def test_script_parses_on_this_interpreter(self):
        """Syntax-only check, so it fails on 3.9-3.11 too rather than only
        where the generator happens to be run."""
        import ast
        ast.parse(_API_REF_SCRIPT.read_text(), filename=str(_API_REF_SCRIPT))

    def test_every_public_name_is_in_a_table(self):
        """No public `mpdsp` name may be missing from CATEGORIES/CLASSES.

        A binding that lands without a table entry is silently omitted from
        the generated document, which still looks complete — that is exactly
        how the tables fell 61 names behind.
        """
        pytest.importorskip("mpdsp")
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_build_api_ref", _API_REF_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        missing = module.check_coverage()
        assert missing == [], (
            f"{len(missing)} public mpdsp name(s) are in no CATEGORIES or "
            f"CLASSES entry: {missing}. Add each to the right category "
            f"(or CLASSES for a stateful class), or to "
            f"UNDOCUMENTED_BY_DESIGN if it is not public API."
        )

    def test_generates_without_error(self, tmp_path):
        """End-to-end run. Executed in a temp cwd so the committed
        docs/api_reference.md is never rewritten as a test side effect."""
        pytest.importorskip("mpdsp")
        (tmp_path / "docs").mkdir()
        result = subprocess.run(
            [sys.executable, str(_API_REF_SCRIPT)],
            cwd=tmp_path, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, (
            f"build_api_ref.py failed:\n{result.stdout}\n{result.stderr}")

        generated = (tmp_path / "docs" / "api_reference.md").read_text()
        assert generated.startswith("# ")
        # Sanity: the document actually covers the surface, not just a stub.
        for expected in ("## Signal generators", "## Classes",
                         "### `IIRFilter`", "`butterworth_lowpass`"):
            assert expected in generated, f"missing {expected!r}"

    def test_committed_doc_is_up_to_date(self, tmp_path):
        """The checked-in doc must match what the generator produces.

        Guards the other half of the original failure: hand-editing
        docs/api_reference.md instead of the generator, which makes a later
        regeneration silently destructive.
        """
        pytest.importorskip("mpdsp")
        (tmp_path / "docs").mkdir()
        result = subprocess.run(
            [sys.executable, str(_API_REF_SCRIPT)],
            cwd=tmp_path, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr

        generated = (tmp_path / "docs" / "api_reference.md").read_text()
        committed = (_API_REF_SCRIPT.parents[1]
                     / "docs" / "api_reference.md").read_text()
        assert generated == committed, (
            "docs/api_reference.md is out of date with the generator. "
            "Run `python scripts/build_api_ref.py` and commit the result. "
            "If the change is prose, edit INTROS / CLASS_INTROS in "
            "scripts/build_api_ref.py rather than the document."
        )


# ---------------------------------------------------------------------------
# plot_dashboard.py — analog-prototype pane (Issue #78)
#
# The dashboard imports streamlit at module scope but never calls it at import
# time, so the plotting functions are testable behind a stub. That matters:
# streamlit is an optional extra that CI does not install, so importorskip
# would mean this logic is never exercised anywhere.
# ---------------------------------------------------------------------------

_DASHBOARD = Path(__file__).resolve().parents[1] / "scripts" / "plot_dashboard.py"


@pytest.fixture(scope="module")
def dashboard():
    pytest.importorskip("mpdsp")
    pytest.importorskip("matplotlib")
    import importlib
    import types as _types

    import matplotlib
    matplotlib.use("Agg")

    # Stub streamlit only if it is genuinely absent, so a real install is
    # exercised when present.
    installed = sys.modules.get("streamlit")
    if installed is None:
        try:
            import streamlit  # noqa: F401
        except ImportError:
            sys.modules["streamlit"] = _types.ModuleType("streamlit")

    sys.path.insert(0, str(_DASHBOARD.parent))
    try:
        yield importlib.import_module("plot_dashboard")
    finally:
        sys.path.remove(str(_DASHBOARD.parent))


_FREQ_PARAMS = {"cutoff": 1000.0, "center": 1000.0, "width": 400.0}
_SAMPLE_RATE = 8000.0


class TestAnalogPrototypePane:
    @pytest.mark.parametrize("family", [
        "Butterworth", "Chebyshev I", "Chebyshev II", "Bessel", "Elliptic"])
    def test_supported_families_are_available(self, dashboard, family):
        available, reason = dashboard.analog_prototype_available(family)
        assert available and reason == ""

    @pytest.mark.parametrize("family,needle", [
        ("RBJ", "z-plane"),
        ("Legendre", "no analog-prototype factory"),
    ])
    def test_unsupported_families_explain_themselves(self, dashboard, family,
                                                     needle):
        """RBJ has no analog prototype by construction; Legendre has none
        bound upstream. Both must say so rather than render nothing."""
        available, reason = dashboard.analog_prototype_available(family)
        assert not available
        assert needle in reason

    @pytest.mark.parametrize("topology,expected_kind", [
        ("lowpass", "lowpass"),
        ("highpass", "highpass"),
    ])
    def test_prototype_topology(self, dashboard, topology, expected_kind):
        plot = dashboard.build_analog_prototype(
            "Butterworth", topology, 4, _FREQ_PARAMS, {})
        assert plot is not None
        assert plot.kind == expected_kind
        assert len(plot.s_poles) == 4

    def test_lowpass_prototype_view_is_untransformed(self, dashboard):
        plot = dashboard.build_analog_prototype(
            "Butterworth", "bandpass", 4, _FREQ_PARAMS, {},
            as_lowpass_prototype=True)
        assert plot.kind == "lowpass"
        assert len(plot.s_poles) == 4

    def test_family_specific_parameters_reach_the_factory(self, dashboard):
        """Changing ripple must change the constellation, or the pane is
        silently ignoring the sidebar."""
        gentle = dashboard.build_analog_prototype(
            "Chebyshev I", "lowpass", 4, _FREQ_PARAMS, {"ripple_db": 0.1})
        steep = dashboard.build_analog_prototype(
            "Chebyshev I", "lowpass", 4, _FREQ_PARAMS, {"ripple_db": 3.0})
        assert gentle.ripple_db != steep.ripple_db
        assert not np.allclose(np.asarray(gentle.s_poles),
                               np.asarray(steep.s_poles))

    def test_elliptic_selectivity_is_wired(self, dashboard):
        loose = dashboard.build_analog_prototype(
            "Elliptic", "lowpass", 4, _FREQ_PARAMS, {"ripple_db": 1.0},
            selectivity_k=0.5)
        tight = dashboard.build_analog_prototype(
            "Elliptic", "lowpass", 4, _FREQ_PARAMS, {"ripple_db": 1.0},
            selectivity_k=0.95)
        assert not np.allclose(np.asarray(loose.s_poles),
                               np.asarray(tight.s_poles))

    def test_response_is_peak_normalized(self, dashboard):
        plot = dashboard.build_analog_prototype(
            "Butterworth", "lowpass", 4, _FREQ_PARAMS, {})
        omega = 2 * np.pi * np.logspace(1, 5, 500)
        mag_db, phase_deg = dashboard.analog_response(plot, omega)
        assert mag_db.max() == pytest.approx(0.0, abs=1e-9)
        assert np.all(np.isfinite(mag_db)) and np.all(np.isfinite(phase_deg))

    def test_response_matches_the_analog_prototype(self, dashboard):
        """-3 dB at cutoff — the pane must plot the real H(s), not a stand-in."""
        plot = dashboard.build_analog_prototype(
            "Butterworth", "lowpass", 4, _FREQ_PARAMS, {})
        mag_db, _ = dashboard.analog_response(
            plot, 2 * np.pi * np.array([100.0, 1000.0, 10000.0]))
        assert mag_db[0] == pytest.approx(0.0, abs=0.01)
        assert mag_db[1] == pytest.approx(-3.01, abs=0.05)
        assert mag_db[2] == pytest.approx(-80.0, abs=0.5)

    def test_phase_is_unwrapped(self, dashboard):
        plot = dashboard.build_analog_prototype(
            "Butterworth", "lowpass", 6, _FREQ_PARAMS, {})
        _, phase_deg = dashboard.analog_response(
            plot, 2 * np.pi * np.logspace(1, 5, 2000))
        # Unwrapped phase has no 360-degree jumps between adjacent samples.
        assert np.abs(np.diff(phase_deg)).max() < 180.0

    @pytest.mark.parametrize("family,extra", [
        ("Butterworth", {}),
        ("Chebyshev I", {"ripple_db": 1.0}),
        ("Chebyshev II", {"stopband_db": 40.0}),
        ("Bessel", {}),
        ("Elliptic", {"ripple_db": 1.0}),
    ])
    @pytest.mark.parametrize("topology",
                             ["lowpass", "highpass", "bandpass", "bandstop"])
    def test_figure_renders(self, dashboard, family, extra, topology):
        import matplotlib.pyplot as plt
        plot = dashboard.build_analog_prototype(
            family, topology, 4, _FREQ_PARAMS, extra)
        assert plot is not None
        fig = dashboard.plot_analog_bode(plot, _SAMPLE_RATE)
        try:
            assert len(dashboard.figure_to_png_bytes(fig)) > 0
        finally:
            plt.close(fig)

    def test_degenerate_band_is_rejected(self, dashboard):
        """A band whose lower edge reaches DC has no bandpass prototype."""
        assert dashboard.build_analog_prototype(
            "Butterworth", "bandpass", 4,
            {"center": 100.0, "width": 400.0}, {}) is None


class TestDtypeMagnitudeOverlay:
    """The frequency-response pane must draw real per-dtype curves (#77).

    Before this landed the pane added empty legend artists carrying only an
    SQNR number, which read as colour-coded curves that were not there.
    """

    @staticmethod
    def _filter():
        import mpdsp
        return mpdsp.butterworth_lowpass(order=6, sample_rate=8000.0,
                                         cutoff=800.0)

    @staticmethod
    def _plotted(ax):
        """Lines with actual data — the empty-artist trick is what we are
        guarding against, so length matters."""
        return [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 0]

    @pytest.mark.parametrize("x_scale", ["linear", "log"])
    def test_one_curve_per_dtype(self, dashboard, x_scale):
        import matplotlib.pyplot as plt
        import mpdsp
        signal = mpdsp.white_noise(length=1024, amplitude=0.5, seed=1)
        dtypes = ["reference", "half", "posit_8_2"]

        fig = dashboard.plot_magnitude_phase(
            self._filter(), 8000.0, dtypes, signal,
            x_scale=x_scale, cutoff=800.0)
        try:
            ax = fig.axes[0]
            # reference + one per non-reference dtype.
            assert len(self._plotted(ax)) == len(dtypes)
            labels = [ln.get_label() for ln in self._plotted(ax)]
            assert any("half" in l for l in labels)
            assert any("posit_8_2" in l for l in labels)
        finally:
            plt.close(fig)

    def test_curves_actually_differ(self, dashboard):
        """A coarse dtype must visibly separate from the reference line."""
        import matplotlib.pyplot as plt
        fig = dashboard.plot_magnitude_phase(
            self._filter(), 8000.0, ["reference", "posit_8_2"], None,
            cutoff=800.0)
        try:
            curves = self._plotted(fig.axes[0])
            reference, coarse = curves[0].get_ydata(), curves[-1].get_ydata()
            assert np.max(np.abs(coarse - reference)) > 1.0
        finally:
            plt.close(fig)

    def test_sqnr_annotation_is_kept(self, dashboard):
        """The SQNR number carries the sample-path effect the curve cannot,
        so it stays in the label when a signal is supplied."""
        import matplotlib.pyplot as plt
        import mpdsp
        signal = mpdsp.white_noise(length=1024, amplitude=0.5, seed=1)
        fig = dashboard.plot_magnitude_phase(
            self._filter(), 8000.0, ["reference", "half"], signal,
            cutoff=800.0)
        try:
            labels = [ln.get_label() for ln in self._plotted(fig.axes[0])]
            assert any("SQNR" in l for l in labels)
        finally:
            plt.close(fig)

    def test_works_without_a_signal(self, dashboard):
        """Curves do not depend on the test signal; only the SQNR does."""
        import matplotlib.pyplot as plt
        fig = dashboard.plot_magnitude_phase(
            self._filter(), 8000.0, ["reference", "half"], None, cutoff=800.0)
        try:
            labels = [ln.get_label() for ln in self._plotted(fig.axes[0])]
            assert len(self._plotted(fig.axes[0])) == 2
            assert not any("SQNR" in l for l in labels)
        finally:
            plt.close(fig)

    def test_coefficient_preserving_dtype_is_labelled(self, dashboard):
        """sensor_* leaves the coefficients alone, so its curve lies exactly
        on the reference. Without a label that reads as a missing curve."""
        import matplotlib.pyplot as plt
        fig = dashboard.plot_magnitude_phase(
            self._filter(), 8000.0, ["reference", "sensor_8bit"], None,
            cutoff=800.0)
        try:
            labels = [ln.get_label() for ln in self._plotted(fig.axes[0])]
            assert any("coefficients unchanged" in l for l in labels)
        finally:
            plt.close(fig)

    def test_bad_dtype_degrades_to_a_legend_note(self, dashboard):
        """One broken dtype must not take the whole pane down."""
        import matplotlib.pyplot as plt
        fig = dashboard.plot_magnitude_phase(
            self._filter(), 8000.0, ["reference", "not_a_dtype"], None,
            cutoff=800.0)
        try:
            ax = fig.axes[0]
            assert len(self._plotted(ax)) == 1          # reference survives
            all_labels = [ln.get_label() for ln in ax.get_lines()]
            assert any("not_a_dtype" in l for l in all_labels)
        finally:
            plt.close(fig)
