"""
mpdsp — Python integration layer for the mixed-precision DSP library.

Provides nanobind bindings to sw::dsp (C++20), matplotlib visualizations,
and Jupyter notebooks for mixed-precision DSP research.
"""

__version__ = "0.1.0"

# CSV I/O (pure Python, always available)
from mpdsp.io import load_sweep

# C++ bindings (available after building the nanobind module)
try:
    from mpdsp._core import (
        # Signal generators
        sine, cosine, chirp, square, triangle, sawtooth,
        impulse, step, white_noise,
        # Window functions
        hamming, hanning, blackman, kaiser, rectangular, flat_top,
        # Quantization
        adc, sqnr_db, measure_sqnr_db,
        max_absolute_error, max_relative_error,
        # Spectral
        fft, fft_magnitude_db, ifft, periodogram, psd, spectrogram,
        # Filters
        IIRFilter, butterworth_lowpass, butterworth_highpass,
        # Introspection
        available_dtypes,
    )
    HAS_CORE = True
except ImportError:
    HAS_CORE = False

# Plotting (requires matplotlib)
try:
    from mpdsp.plotting import (
        plot_signal, plot_spectrum, plot_signal_and_spectrum,
        plot_quantization_comparison, plot_sqnr_comparison,
        plot_window_comparison, plot_spectrogram, plot_psd,
    )
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
