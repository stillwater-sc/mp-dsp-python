"""Tests for mpdsp I/O — CSV sweeps (pure Python) and WAV audio (C++)."""

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

import mpdsp


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


# ---------------------------------------------------------------------------
# WAV file I/O — read_wav / write_wav (issue #51).
#
# Upstream writes integer PCM only (8/16/24/32-bit) and reads both integer
# and 32-bit-float PCM. Every test below uses the narrowest round-trip
# tolerance defensible for the bit depth — wider for 8-bit, tighter for 24
# and 32 — since the PCM truncation error is what actually sets the floor,
# not any mpdsp-introduced loss.
# ---------------------------------------------------------------------------


# Worst-case absolute error introduced by integer-PCM quantization.
# Derivation: n-bit signed PCM has 2^(n-1) positive grid points covering
# the [0, 1] range, so one quantum ≈ 1 / 2^(n-1). Allow 1.5 quanta of slack
# for the rescaling round-trip (scale to int, cast, rescale back).
_PCM_TOL = {
    8: 1.5 / (2 ** 7),    # ~0.0117
    16: 1.5 / (2 ** 15),  # ~4.6e-5
    24: 1.5 / (2 ** 23),  # ~1.8e-7
    32: 1.5 / (2 ** 31),  # ~7.0e-10
}


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_wav_round_trip_mono(bits):
    """Sine wave written and read back lands within the PCM quantum."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"sine_{bits}.wav")
        sample_rate = 44100
        signal = mpdsp.sine(length=2048, frequency=440.0,
                            sample_rate=float(sample_rate), amplitude=0.5)

        mpdsp.write_wav(path, signal, sample_rate=sample_rate,
                         bits_per_sample=bits)

        data, rate = mpdsp.read_wav(path)
        assert rate == sample_rate
        assert data.shape == signal.shape
        assert data.dtype == np.float64
        assert np.max(np.abs(data - signal)) <= _PCM_TOL[bits]


def test_wav_round_trip_stereo_16bit():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "stereo.wav")
        sample_rate = 48000
        left = mpdsp.sine(length=1024, frequency=440.0,
                           sample_rate=float(sample_rate), amplitude=0.4)
        right = mpdsp.sine(length=1024, frequency=880.0,
                            sample_rate=float(sample_rate), amplitude=0.4)
        stereo = np.column_stack([left, right])   # (N, 2) C-contiguous
        assert stereo.shape == (1024, 2)

        mpdsp.write_wav(path, stereo, sample_rate=sample_rate,
                         bits_per_sample=16)

        data, rate = mpdsp.read_wav(path)
        assert rate == sample_rate
        assert data.shape == (1024, 2)
        assert np.max(np.abs(data - stereo)) <= _PCM_TOL[16]


def test_wav_shape_convention():
    """Mono -> 1D, multi-channel -> 2D. Matches scipy.io.wavfile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mono_path = os.path.join(tmpdir, "mono.wav")
        mono_signal = mpdsp.sine(length=256, frequency=100.0,
                                  sample_rate=8000.0)
        mpdsp.write_wav(mono_path, mono_signal, sample_rate=8000)
        data, _ = mpdsp.read_wav(mono_path)
        assert data.ndim == 1

        stereo_path = os.path.join(tmpdir, "stereo.wav")
        stereo_signal = np.column_stack([mono_signal, mono_signal])
        mpdsp.write_wav(stereo_path, stereo_signal, sample_rate=8000)
        data, _ = mpdsp.read_wav(stereo_path)
        assert data.ndim == 2
        assert data.shape[1] == 2


def test_wav_sample_rate_preserved():
    with tempfile.TemporaryDirectory() as tmpdir:
        for sr in (8000, 16000, 22050, 44100, 48000, 96000, 192000):
            path = os.path.join(tmpdir, f"sr_{sr}.wav")
            signal = mpdsp.sine(length=128, frequency=100.0,
                                sample_rate=float(sr))
            mpdsp.write_wav(path, signal, sample_rate=sr)
            _, rate = mpdsp.read_wav(path)
            assert rate == sr


def test_wav_clipping_outside_unit_range():
    """Upstream clamp_sample() clips to [-1, 1]; verify that happens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "loud.wav")
        # Intentional overdrive — maxima should land at +/- 1 after round-trip.
        signal = np.array([1.5, -1.5, 0.5, -0.5], dtype=np.float64)
        mpdsp.write_wav(path, signal, sample_rate=8000, bits_per_sample=16)
        data, _ = mpdsp.read_wav(path)
        assert data[0] == pytest.approx(1.0, abs=_PCM_TOL[16])
        assert data[1] == pytest.approx(-1.0, abs=_PCM_TOL[16])
        assert data[2] == pytest.approx(0.5, abs=_PCM_TOL[16])
        assert data[3] == pytest.approx(-0.5, abs=_PCM_TOL[16])


def test_wav_invalid_bits_per_sample_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.wav")
        signal = mpdsp.sine(length=100, frequency=100.0, sample_rate=8000.0)
        for bad in (7, 12, 20, 64):
            with pytest.raises(ValueError):
                mpdsp.write_wav(path, signal, sample_rate=8000,
                                 bits_per_sample=bad)


def test_wav_invalid_sample_rate_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.wav")
        signal = mpdsp.sine(length=100, frequency=100.0, sample_rate=8000.0)
        with pytest.raises(ValueError):
            mpdsp.write_wav(path, signal, sample_rate=0)
        with pytest.raises(ValueError):
            mpdsp.write_wav(path, signal, sample_rate=-44100)


def test_wav_read_nonexistent_file_raises():
    with pytest.raises(RuntimeError):
        mpdsp.read_wav("/this/path/does/not/exist.wav")


def test_wav_read_corrupt_file_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "corrupt.wav")
        with open(path, "wb") as f:
            f.write(b"NOTAWAVE" + b"\x00" * 40)
        with pytest.raises(RuntimeError):
            mpdsp.read_wav(path)


def test_wav_3d_data_raises():
    """write_wav only accepts 1D or 2D input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.wav")
        cube = np.zeros((4, 4, 4), dtype=np.float64)
        with pytest.raises(ValueError):
            mpdsp.write_wav(path, cube, sample_rate=8000)


def test_wav_non_float64_raises():
    """Only float64 ndarrays are accepted — keeps the normalization
    contract unambiguous (samples in [-1, 1] as doubles)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.wav")
        int_signal = np.array([0, 1000, -1000, 0], dtype=np.int16)
        with pytest.raises(ValueError):
            mpdsp.write_wav(path, int_signal, sample_rate=8000)
