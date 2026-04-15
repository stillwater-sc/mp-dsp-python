"""
mpdsp — Python integration layer for the mixed-precision DSP library.

Provides nanobind bindings to sw::dsp (C++20), matplotlib visualizations,
and Jupyter notebooks for mixed-precision DSP research.

Modules:
    signals     — signal generators (sine, chirp, noise, impulse, ...)
    windows     — window functions (hamming, hanning, blackman, kaiser, ...)
    quantization — ADC/DAC modeling, dithering, SQNR measurement
    filters     — IIR/FIR filter design and mixed-precision processing
    spectral    — FFT, PSD, spectrogram, Z-transform, Laplace
    conditioning — envelope followers, compressor, AGC
    estimation  — Kalman filter, LMS, NLMS, RLS adaptive filters
    image       — 2D convolution, morphology, edge detection, generators
    analysis    — stability, sensitivity, condition number
    io          — WAV, PGM, PPM, BMP, CSV file I/O
    plotting    — matplotlib convenience functions

Usage:
    import mpdsp

    signal = mpdsp.sine(length=2000, frequency=440, sample_rate=44100)
    filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100, cutoff=1000)
    result = filt.process(signal, dtype="posit_full")
"""

__version__ = "0.1.0"

# Phase 1: CSV I/O (pure Python, no nanobind needed)
from mpdsp.io import load_sweep

# Nanobind bindings will be imported as they become available:
# from mpdsp._core import ...  # Phase 2+
