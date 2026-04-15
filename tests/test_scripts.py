"""Tests for the CSV visualization scripts."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

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

        yield tmpdir


def _run_script(script_name: str, csv_dir: str, output_dir: str) -> subprocess.CompletedProcess:
    """Run a plotting script and return the result."""
    script_path = Path(__file__).parent.parent / "scripts" / script_name
    return subprocess.run(
        [sys.executable, str(script_path), csv_dir, "--output", output_dir],
        capture_output=True, text=True, timeout=30)


def test_plot_precision_generates_output(csv_dir):
    with tempfile.TemporaryDirectory() as outdir:
        result = _run_script("plot_precision.py", csv_dir, outdir)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(os.path.join(outdir, "magnitude_response.png"))
        assert os.path.exists(os.path.join(outdir, "phase_response.png"))
        assert os.path.exists(os.path.join(outdir, "magnitude_error.png"))


def test_plot_heatmap_generates_output(csv_dir):
    with tempfile.TemporaryDirectory() as outdir:
        result = _run_script("plot_heatmap.py", csv_dir, outdir)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(os.path.join(outdir, "sqnr_heatmap.png"))
        assert os.path.exists(os.path.join(outdir, "sqnr_bar_chart.png"))
        assert os.path.exists(os.path.join(outdir, "precision_cost_frontier.png"))


def test_plot_pole_zero_generates_output(csv_dir):
    with tempfile.TemporaryDirectory() as outdir:
        result = _run_script("plot_pole_zero.py", csv_dir, outdir)
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(os.path.join(outdir, "pole_zero.png"))
