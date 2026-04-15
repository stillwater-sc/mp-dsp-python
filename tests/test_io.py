"""Tests for mpdsp.io — CSV loading from precision sweep output."""

import os
import tempfile
from pathlib import Path

import pytest


def _write_test_csvs(directory: str) -> None:
    """Write minimal valid CSV files for testing."""
    Path(directory).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(directory, "iir_precision_sweep.csv"), "w") as f:
        f.write("filter_family,arith_type,bits,max_abs_error,max_rel_error,"
                "sqnr_db,pole_displacement,stability_margin\n")
        f.write("Butterworth,double,64,0,0,300,0,0.1\n")
        f.write('Butterworth,"cfloat<24,5>",24,1e-6,1e-5,95.1,0,0.1\n')

    with open(os.path.join(directory, "frequency_response.csv"), "w") as f:
        f.write("filter_family,arith_type,freq_hz,magnitude_db,phase_deg,"
                "ref_magnitude_db,ref_phase_deg\n")
        f.write("Butterworth,double,0,0,0,0,0\n")

    with open(os.path.join(directory, "pole_positions.csv"), "w") as f:
        f.write("filter_family,arith_type,pole_index,real,imag,"
                "ref_real,ref_imag,displacement\n")
        f.write("Butterworth,double,0,0.866,0.234,0.866,0.234,0\n")


def test_load_sweep():
    from mpdsp.io import load_sweep

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_test_csvs(tmpdir)
        data = load_sweep(tmpdir)

        assert "metrics" in data
        assert "frequency" in data
        assert "poles" in data


def test_load_sweep_missing_file():
    from mpdsp.io import load_sweep

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            load_sweep(tmpdir)


def test_load_sweep_parses_quoted_fields():
    """Verify that quoted fields like "cfloat<24,5>" are parsed correctly."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    from mpdsp.io import load_sweep

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_test_csvs(tmpdir)
        data = load_sweep(tmpdir)

        df = data["metrics"]
        types = df["arith_type"].tolist()
        assert "cfloat<24,5>" in types, f"Expected cfloat<24,5> in {types}"
