"""
mpdsp — Python integration layer for the mixed-precision DSP library.

Provides nanobind bindings to sw::dsp (C++20), matplotlib visualizations,
and Jupyter notebooks for mixed-precision DSP research.

Versioning: __version__ is sourced from the installed wheel metadata,
which in turn is read from CMakeLists.txt at build time. We track the
mixed-precision-dsp C++ library version in lockstep. __dsp_version__
(available once the C++ module loads) reports the version of the
upstream library the wheel was compiled against — useful for runtime
verification in research setups where the two can drift during
development.
"""

# Read the installed-package version from wheel metadata so there's
# exactly one source of truth. Falls back to "0+unknown" when run from
# an unbuilt source checkout (no wheel metadata present).
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        __version__ = _pkg_version("mpdsp")
    except PackageNotFoundError:
        __version__ = "0+unknown"
    del _pkg_version, PackageNotFoundError
except ImportError:  # pragma: no cover  -- pre-3.8 fallback not used
    __version__ = "0+unknown"

# CSV I/O (pure Python, always available)
from mpdsp.io import load_sweep

# C++ bindings (available after building the nanobind module).
# The try/except exists so pure-Python paths (mpdsp.io, mpdsp.plotting) still
# work in unbuilt source checkouts. But a *failed* import in an installed
# wheel is a packaging bug — silently falling through to HAS_CORE=False once
# shipped wheels with `_core.so` installed to the wrong path (see the
# CMakeLists.txt install() comment). Stash the underlying exception on the
# module so users can diagnose without re-importing.
__core_import_error__ = None
try:
    from mpdsp._core import (
        # Signal generators
        sine, cosine, chirp, square, triangle, sawtooth,
        impulse, step, white_noise, gaussian_noise, pink_noise,
        # Window functions
        hamming, hanning, blackman, kaiser, rectangular, flat_top,
        # Quantization
        adc, dac, sqnr_db, measure_sqnr_db,
        max_absolute_error, max_relative_error,
        RPDFDither, TPDFDither, FirstOrderNoiseShaper,
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
        # Image — morphology
        make_rect_element, make_cross_element, make_ellipse_element,
        dilate, erode,
        morphological_open, morphological_close,
        morphological_gradient, tophat, blackhat,
        # Image — multi-channel
        rgb_to_gray,
        # Image — file I/O
        read_pgm, write_pgm,
        read_ppm, write_ppm,
        read_bmp, write_bmp, write_bmp_rgb,
        # Types — rational transfer function + type projection
        TransferFunction,
        project_onto, projection_error,
        # Introspection
        available_dtypes,
    )
    # The underlying mixed-precision-dsp C++ library version the wheel
    # was built against, sourced from sw::dsp::version.hpp. Matches
    # __version__ when the lockstep convention holds; divergence means
    # someone is running a mismatched source checkout, which is what
    # runtime introspection is for.
    from mpdsp._core import dsp_version as __dsp_version__
    from mpdsp._core import dsp_version_info as __dsp_version_info__
    HAS_CORE = True
except ImportError as _e:
    HAS_CORE = False
    __dsp_version__ = None
    __dsp_version_info__ = None
    __core_import_error__ = _e
    import warnings as _warnings
    _warnings.warn(
        f"mpdsp._core failed to import: {_e}. "
        f"C++ bindings unavailable (filters, spectral, quantization, image). "
        f"This is expected for unbuilt source checkouts but indicates a "
        f"packaging bug in an installed wheel.",
        ImportWarning,
        stacklevel=2,
    )
    del _warnings, _e

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
    from mpdsp.filters import (compare_filters, plot_filter_comparison,
                                 to_transfer_function)

# Analysis helpers (pure Python; build only on stdlib + numpy + _core methods)
from mpdsp.analysis import biquad_poles, is_stable, max_pole_radius

# Estimation helpers (pure Python; plotting needs matplotlib)
if HAS_CORE:
    from mpdsp.estimation import (
        plot_kalman_tracking,
        plot_adaptive_convergence,
        collect_adaptive_weights,
    )

# Image helpers (pure Python; plotting requires matplotlib)
if HAS_CORE:
    from mpdsp.image import (
        apply_per_channel,
        plot_image, plot_image_grid, plot_pipeline,
    )
