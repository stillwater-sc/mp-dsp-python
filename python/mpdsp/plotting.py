"""Matplotlib convenience functions for mixed-precision DSP visualization.

These functions work with NumPy arrays returned by mpdsp signal generators
and quantization tools. Import with:

    from mpdsp.plotting import plot_signal, plot_spectrum, plot_sqnr_comparison
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_matplotlib():
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plotting. "
                          "Install with: pip install matplotlib")


def plot_signal(signal, sample_rate=1.0, title="Signal", ax=None, **kwargs):
    """Plot a time-domain signal.

    Parameters
    ----------
    signal : numpy.ndarray
        1D signal array.
    sample_rate : float
        Sample rate in Hz (for time axis). Default 1.0 gives sample indices.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.
    **kwargs : dict
        Passed to ax.plot().

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    t = np.arange(len(signal)) / sample_rate
    ax.plot(t, signal, **kwargs)
    ax.set_xlabel("Time (s)" if sample_rate != 1.0 else "Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_spectrum(signal, sample_rate=1.0, title="Spectrum", ax=None,
                  db=True, **kwargs):
    """Plot the magnitude spectrum of a signal.

    Parameters
    ----------
    signal : numpy.ndarray
        1D signal array.
    sample_rate : float
        Sample rate in Hz.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    db : bool
        If True, plot in dB. Otherwise linear magnitude.
    **kwargs : dict
        Passed to ax.plot().

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal)) / N

    if db:
        spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-20))
        ax.plot(freqs, spectrum_db, **kwargs)
        ax.set_ylabel("Magnitude (dB)")
    else:
        ax.plot(freqs, spectrum, **kwargs)
        ax.set_ylabel("Magnitude")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_signal_and_spectrum(signal, sample_rate=1.0, title=""):
    """Plot signal in time domain and frequency domain side by side.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.5))
    plot_signal(signal, sample_rate, title=f"{title} (time)", ax=ax1)
    plot_spectrum(signal, sample_rate, title=f"{title} (spectrum)", ax=ax2)
    plt.tight_layout()
    return fig


def plot_quantization_comparison(signal, dtypes, sample_rate=1.0,
                                  title="Quantization Comparison"):
    """Plot a signal quantized through multiple arithmetic types.

    Requires mpdsp._core to be available.

    Parameters
    ----------
    signal : numpy.ndarray
        Reference signal (float64).
    dtypes : list of str
        Arithmetic config names (e.g., ["gpu_baseline", "half", "posit_full"]).
    sample_rate : float
        Sample rate for time axis.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    try:
        from mpdsp._core import adc, sqnr_db
    except ImportError:
        raise ImportError("mpdsp._core not available. Build the nanobind module first.")

    n_types = len(dtypes)
    fig, axes = plt.subplots(n_types + 1, 1, figsize=(12, 2.5 * (n_types + 1)),
                             sharex=True)

    # Reference signal
    t = np.arange(len(signal)) / sample_rate
    axes[0].plot(t, signal, "k-", linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Reference (double)")
    axes[0].grid(True, alpha=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, n_types))
    for i, dtype in enumerate(dtypes):
        quantized = adc(signal, dtype=dtype)
        sqnr = sqnr_db(signal, quantized)
        error = signal - quantized

        ax = axes[i + 1]
        ax.plot(t, error, color=colors[i], linewidth=0.5)
        ax.set_ylabel("Error")
        ax.set_title(f"{dtype} — SQNR: {sqnr:.1f} dB")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)" if sample_rate != 1.0 else "Sample")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_sqnr_comparison(signal, dtypes=None, title="SQNR Comparison"):
    """Bar chart comparing SQNR across arithmetic types.

    Parameters
    ----------
    signal : numpy.ndarray
        Reference signal (float64).
    dtypes : list of str, optional
        Arithmetic config names. If None, uses all available.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    try:
        from mpdsp._core import adc, sqnr_db, available_dtypes
    except ImportError:
        raise ImportError("mpdsp._core not available. Build the nanobind module first.")

    if dtypes is None:
        dtypes = [d for d in available_dtypes() if d != "reference"]

    sqnrs = []
    for dtype in dtypes:
        quantized = adc(signal, dtype=dtype)
        sqnrs.append(sqnr_db(signal, quantized))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.array(sqnrs) / max(sqnrs))
    bars = ax.barh(dtypes, sqnrs, color=colors, edgecolor="black", linewidth=0.5)

    # Annotate bars
    for bar, sqnr in zip(bars, sqnrs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{sqnr:.1f} dB", va="center", fontsize=9)

    ax.set_xlabel("SQNR (dB)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_window_comparison(window_funcs, N=256, title="Window Comparison"):
    """Plot multiple windows and their frequency responses.

    Parameters
    ----------
    window_funcs : dict
        Mapping of name → callable(N) returning NumPy array.
        E.g., {"Hamming": mpdsp.hamming, "Blackman": mpdsp.blackman}
    N : int
        Window length.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    for name, func in window_funcs.items():
        w = func(N)
        ax1.plot(w, label=name)

        # Frequency response
        W = np.fft.rfft(w, n=4096)
        W_db = 20 * np.log10(np.maximum(np.abs(W) / np.abs(W).max(), 1e-10))
        freqs = np.linspace(0, 0.5, len(W_db))
        ax2.plot(freqs, W_db, label=name)

    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time domain")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Normalized frequency")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_title("Frequency response")
    ax2.set_ylim(-120, 5)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_spectrogram(times, freqs, magnitude_db, title="Spectrogram",
                     vmin=-80, vmax=0, ax=None):
    """Plot a spectrogram from mpdsp.spectrogram() output.

    Parameters
    ----------
    times : numpy.ndarray
        Time axis (seconds).
    freqs : numpy.ndarray
        Frequency axis (Hz).
    magnitude_db : numpy.ndarray
        2D magnitude array [n_frames x n_freqs] in dB.
    title : str
        Plot title.
    vmin, vmax : float
        Color scale limits (dB).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    im = ax.pcolormesh(times, freqs, magnitude_db.T, shading="auto",
                       cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    return ax


def plot_psd(freqs, power, title="Power Spectral Density", ax=None, **kwargs):
    """Plot power spectral density from mpdsp.psd() output.

    Parameters
    ----------
    freqs : numpy.ndarray
        Frequency axis (Hz).
    power : numpy.ndarray
        Power values.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    ax.semilogy(freqs, power, **kwargs)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
