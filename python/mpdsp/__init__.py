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
        IIRFilter,
        butterworth_lowpass, butterworth_highpass,
        butterworth_bandpass, butterworth_bandstop,
        chebyshev1_lowpass, chebyshev1_highpass,
        chebyshev1_bandpass, chebyshev1_bandstop,
        chebyshev2_lowpass, chebyshev2_highpass,
        chebyshev2_bandpass, chebyshev2_bandstop,
        bessel_lowpass, bessel_highpass,
        bessel_bandpass, bessel_bandstop,
        legendre_lowpass, legendre_highpass,
        legendre_bandpass, legendre_bandstop,
        elliptic_lowpass, elliptic_highpass,
        elliptic_bandpass, elliptic_bandstop,
        rbj_lowpass, rbj_highpass,
        rbj_bandpass, rbj_bandstop,
        rbj_allpass, rbj_lowshelf, rbj_highshelf,
        FIRFilter, fir_filter,
        fir_lowpass, fir_highpass,
        fir_bandpass, fir_bandstop,
        # Conditioning
        PeakEnvelope, RMSEnvelope, Compressor, AGC,
        # Estimation
        KalmanFilter,
        LMSFilter, NLMSFilter, RLSFilter,
        # Image — generators
        checkerboard,
        stripes_horizontal, stripes_vertical, grid,
        gradient_horizontal, gradient_vertical, gradient_radial,
        gaussian_blob, circle, rectangle, zone_plate,
        uniform_noise_image, gaussian_noise_image, salt_and_pepper,
        add_noise, threshold,
        # Image — processing
        convolve2d, separable_filter,
        gaussian_blur, box_blur,
        # Image — edge detection
        sobel_x, sobel_y, prewitt_x, prewitt_y,
        gradient_magnitude, canny,
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

# Filter helpers (pure Python; requires mpdsp._core; plotting needs matplotlib)
if HAS_CORE:
    from mpdsp.filters import compare_filters, plot_filter_comparison

# Estimation helpers (pure Python; plotting needs matplotlib)
if HAS_CORE:
    from mpdsp.estimation import (
        plot_kalman_tracking,
        plot_adaptive_convergence,
        collect_adaptive_weights,
    )
