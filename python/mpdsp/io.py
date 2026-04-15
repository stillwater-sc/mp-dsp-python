"""CSV I/O for consuming precision sweep output from the C++ application."""

import os
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_sweep(directory: str) -> dict:
    """Load all CSV files from an iir_precision_sweep output directory.

    Parameters
    ----------
    directory : str
        Path to directory containing the CSV files.

    Returns
    -------
    dict with keys:
        'metrics'   — DataFrame from iir_precision_sweep.csv
        'frequency' — DataFrame from frequency_response.csv
        'poles'     — DataFrame from pole_positions.csv

    If pandas is not installed, returns dict of file paths instead.
    """
    d = Path(directory)
    files = {
        'metrics': d / 'iir_precision_sweep.csv',
        'frequency': d / 'frequency_response.csv',
        'poles': d / 'pole_positions.csv',
    }

    # Verify files exist
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected CSV not found: {path}")

    if HAS_PANDAS:
        return {
            'metrics': pd.read_csv(files['metrics']),
            'frequency': pd.read_csv(files['frequency']),
            'poles': pd.read_csv(files['poles']),
        }
    else:
        return {name: str(path) for name, path in files.items()}
