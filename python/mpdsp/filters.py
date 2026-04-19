"""Mixed-precision filter comparison helpers.

Built on top of the C++ bindings in `mpdsp._core` (IIRFilter and FIRFilter).
Use `compare_filters` to quantify precision loss across dtypes and
`plot_filter_comparison` to visualize magnitude, phase, and pole locations
side by side.
"""

import numpy as np

import mpdsp

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Configurations to compare by default — the pre-instantiated ones.
DEFAULT_DTYPES = [
    "reference",
    "gpu_baseline",
    "ml_hw",
    "cf24",
    "half",
    "posit_full",
    "tiny_posit",
]


def to_transfer_function(filt):
    """Fold an `IIRFilter` cascade into a single `TransferFunction`.

    The cascade ``H(z) = H_0(z) * H_1(z) * ... * H_{k-1}(z)`` is computed by
    polynomial convolution of the per-stage numerators and denominators.
    Each biquad `(b0, b1, b2, a1, a2)` contributes a numerator `[b0, b1, b2]`
    and denominator `[1, a1, a2]`; the cascade numerator is the convolution
    of all stage numerators, and similarly for the denominator. The leading
    `1` is stripped from the cascade denominator before returning (matching
    upstream `TransferFunction<T>`'s implicit-a0 convention).

    Parameters
    ----------
    filt : mpdsp.IIRFilter
        A designed IIR filter.

    Returns
    -------
    mpdsp.TransferFunction
        A new TransferFunction with equivalent frequency response.
    """
    # Deferred import so the module imports cleanly even when the C++
    # extension (which provides TransferFunction) isn't available.
    from mpdsp._core import TransferFunction

    num = np.array([1.0])
    # Cascade denominator including the implicit leading 1 for convolution;
    # strip it before handing to TransferFunction which wants the a1, a2, ... tail.
    den_with_a0 = np.array([1.0])
    for b0, b1, b2, a1, a2 in filt.coefficients():
        num = np.convolve(num, [b0, b1, b2])
        den_with_a0 = np.convolve(den_with_a0, [1.0, a1, a2])
    return TransferFunction(num, den_with_a0[1:])


def compare_filters(filt, signal, dtypes=None):
    """Process `signal` through `filt` at multiple dtypes and report error metrics.

    For each dtype the filter is re-run; SQNR and max absolute error are
    measured against the "reference" (double-precision) output.

    Parameters
    ----------
    filt : mpdsp.IIRFilter or mpdsp.FIRFilter
        A filter object with a `process(signal, dtype)` method.
    signal : numpy.ndarray
        1D input signal (float64).
    dtypes : list of str, optional
        Dtype keys to compare. Defaults to all pre-instantiated configurations.
        The first entry is treated as the reference if "reference" isn't listed.

    Returns
    -------
    pandas.DataFrame (if pandas is available), else list of dict
        Columns: dtype, sqnr_db, max_abs_error, max_rel_error.
        Failed dtypes (e.g. posit when disabled) record the exception message
        and NaN error values.
    """
    if dtypes is None:
        dtypes = list(DEFAULT_DTYPES)

    reference_key = "reference" if "reference" in dtypes else dtypes[0]
    ref = filt.process(signal, dtype=reference_key)

    rows = []
    for dt in dtypes:
        try:
            out = filt.process(signal, dtype=dt)
            row = {
                "dtype": dt,
                "sqnr_db": mpdsp.sqnr_db(ref, out),
                "max_abs_error": mpdsp.max_absolute_error(ref, out),
                "max_rel_error": mpdsp.max_relative_error(ref, out),
                "error": None,
            }
        except Exception as e:
            row = {
                "dtype": dt,
                "sqnr_db": float("nan"),
                "max_abs_error": float("nan"),
                "max_rel_error": float("nan"),
                "error": str(e),
            }
        rows.append(row)

    if HAS_PANDAS:
        return pd.DataFrame(rows)
    return rows


def plot_filter_comparison(filt, dtypes=None, num_freqs=512, signal=None,
                           sample_rate=1.0, title=None, figsize=(12, 4)):
    """Plot magnitude, phase, and pole locations for a filter.

    Parameters
    ----------
    filt : mpdsp.IIRFilter or mpdsp.FIRFilter
        A filter object. Pole-zero subplot is only populated for IIRFilter.
    dtypes : list of str, optional
        If provided AND a `signal` is given, overlays SQNR-annotated
        magnitude curves from each dtype onto the main magnitude subplot.
        For design-only visualization, leave this as None.
    num_freqs : int
        Number of frequency points on [0, 0.5] (normalized).
    signal : numpy.ndarray, optional
        If provided with `dtypes`, each dtype's processed output is evaluated
        for SQNR annotation (purely informational).
    sample_rate : float
        For axis labels only. Default 1.0 labels x-axis in normalized units.
    title : str, optional
        Overall figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plot_filter_comparison. "
                          "Install with: pip install matplotlib")

    freqs = np.linspace(0.0, 0.5, num_freqs)
    H = filt.frequency_response(freqs)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase = np.unwrap(np.angle(H))

    # Plot in Hz if sample_rate was provided, otherwise normalized.
    x_freqs = freqs * sample_rate
    x_label = "Frequency (Hz)" if sample_rate != 1.0 else "Normalized frequency"

    is_iir = isinstance(filt, mpdsp.IIRFilter)
    n_cols = 3 if is_iir else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    ax_mag = axes[0]
    ax_phase = axes[1]

    ax_mag.plot(x_freqs, mag_db, label="reference")
    ax_mag.set_xlabel(x_label)
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_title("Magnitude response")
    ax_mag.grid(True)

    ax_phase.plot(x_freqs, phase)
    ax_phase.set_xlabel(x_label)
    ax_phase.set_ylabel("Phase (rad, unwrapped)")
    ax_phase.set_title("Phase response")
    ax_phase.grid(True)

    # If the caller asks for per-dtype SQNR annotation, run the signal through
    # each dtype once and tag the legend with the result.
    if dtypes is not None and signal is not None:
        ref_out = filt.process(signal, dtype="reference")
        for dt in dtypes:
            if dt == "reference":
                continue
            try:
                out = filt.process(signal, dtype=dt)
                sqnr = mpdsp.sqnr_db(ref_out, out)
                ax_mag.plot([], [], " ", label=f"{dt}: SQNR={sqnr:.1f} dB")
            except Exception as e:
                ax_mag.plot([], [], " ", label=f"{dt}: {e}")
        ax_mag.legend(loc="best", fontsize="small")

    if is_iir:
        ax_pz = axes[2]
        poles = np.asarray(filt.poles())
        # Unit circle for reference
        theta = np.linspace(0, 2 * np.pi, 256)
        ax_pz.plot(np.cos(theta), np.sin(theta), color="0.7", linewidth=0.8)
        ax_pz.scatter(poles.real, poles.imag, marker="x", color="C3",
                      s=60, label="poles")
        ax_pz.set_aspect("equal")
        ax_pz.set_xlabel("Re(z)")
        ax_pz.set_ylabel("Im(z)")
        ax_pz.set_title(f"Pole locations (margin = "
                        f"{filt.stability_margin():.3f})")
        ax_pz.grid(True)
        ax_pz.legend(loc="best", fontsize="small")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
