"""Streamlit filter-designer dashboard for mpdsp.

Run with:
    pip install mpdsp[dashboard]
    streamlit run scripts/plot_dashboard.py

The dashboard exposes every IIR family, every topology (LP/HP/BP/BS plus
RBJ shelves/allpass), all 7 mixed-precision arithmetic configurations,
and the numerical-analysis metrics that mpdsp's IIRFilter carries
(stability margin, condition number, worst-case sensitivity, per-dtype
pole displacement).

The goal is feature parity with the classic DSPFilters demo app, plus
the mixed-precision comparison view that's the whole point of this
library.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import mpdsp
from mpdsp.filters import compare_filters


# ---------------------------------------------------------------------------
# Filter-family registry.
#
# Each family binds four topologies (LP/HP/BP/BS) to constructor functions,
# plus any family-specific parameters (ripple, stopband, rolloff). RBJ is
# an outlier — its constructors don't take `order` and add shelf/allpass
# variants — so it gets its own branch in `build_filter()` rather than
# being forced into this schema.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilySpec:
    name: str
    has_order: bool = True
    extra_params: tuple = ()          # names of extra kwargs (e.g. "ripple_db")
    makers: dict[str, Callable] = field(default_factory=dict)  # topology -> callable

    def __post_init__(self):
        # A FamilySpec with no topologies is useless and would TypeError at
        # the first `spec.makers[topology]` lookup. Fail fast at construction.
        if not self.makers:
            raise ValueError(f"FamilySpec({self.name!r}) must define at least "
                             "one topology in `makers`.")


ORDER_FAMILIES: dict[str, FamilySpec] = {
    "Butterworth": FamilySpec(
        "Butterworth",
        makers={
            "lowpass":  mpdsp.butterworth_lowpass,
            "highpass": mpdsp.butterworth_highpass,
            "bandpass": mpdsp.butterworth_bandpass,
            "bandstop": mpdsp.butterworth_bandstop,
        },
    ),
    "Chebyshev I": FamilySpec(
        "Chebyshev I",
        extra_params=("ripple_db",),
        makers={
            "lowpass":  mpdsp.chebyshev1_lowpass,
            "highpass": mpdsp.chebyshev1_highpass,
            "bandpass": mpdsp.chebyshev1_bandpass,
            "bandstop": mpdsp.chebyshev1_bandstop,
        },
    ),
    "Chebyshev II": FamilySpec(
        "Chebyshev II",
        extra_params=("stopband_db",),
        makers={
            "lowpass":  mpdsp.chebyshev2_lowpass,
            "highpass": mpdsp.chebyshev2_highpass,
            "bandpass": mpdsp.chebyshev2_bandpass,
            "bandstop": mpdsp.chebyshev2_bandstop,
        },
    ),
    "Bessel": FamilySpec(
        "Bessel",
        makers={
            "lowpass":  mpdsp.bessel_lowpass,
            "highpass": mpdsp.bessel_highpass,
            "bandpass": mpdsp.bessel_bandpass,
            "bandstop": mpdsp.bessel_bandstop,
        },
    ),
    "Legendre": FamilySpec(
        "Legendre",
        makers={
            "lowpass":  mpdsp.legendre_lowpass,
            "highpass": mpdsp.legendre_highpass,
            "bandpass": mpdsp.legendre_bandpass,
            "bandstop": mpdsp.legendre_bandstop,
        },
    ),
    "Elliptic": FamilySpec(
        "Elliptic",
        extra_params=("ripple_db", "rolloff"),
        makers={
            "lowpass":  mpdsp.elliptic_lowpass,
            "highpass": mpdsp.elliptic_highpass,
            "bandpass": mpdsp.elliptic_bandpass,
            "bandstop": mpdsp.elliptic_bandstop,
        },
    ),
}


# RBJ biquads are a different shape: no `order`, per-variant parameter set,
# and include shelf/allpass topologies that don't exist elsewhere. Each
# entry is (maker, list of (param_name, default, min, max)).
RBJ_VARIANTS: dict[str, tuple[Callable, tuple]] = {
    "lowpass":   (mpdsp.rbj_lowpass,   (("q", 0.7071, 0.1, 10.0),)),
    "highpass":  (mpdsp.rbj_highpass,  (("q", 0.7071, 0.1, 10.0),)),
    "bandpass":  (mpdsp.rbj_bandpass,  (("bandwidth", 1.0, 0.1, 4.0),)),
    "bandstop":  (mpdsp.rbj_bandstop,  (("bandwidth", 1.0, 0.1, 4.0),)),
    "allpass":   (mpdsp.rbj_allpass,   (("q", 0.7071, 0.1, 10.0),)),
    "lowshelf":  (mpdsp.rbj_lowshelf,  (("gain_db", 6.0, -24.0, 24.0),
                                         ("q", 0.7071, 0.1, 10.0))),
    "highshelf": (mpdsp.rbj_highshelf, (("gain_db", 6.0, -24.0, 24.0),
                                         ("q", 0.7071, 0.1, 10.0))),
}


# ---------------------------------------------------------------------------
# Filter construction.
# Isolated in a function so Streamlit's caching keeps re-design cheap when
# only downstream controls (dtype selection, signal params) change. Filter
# objects aren't picklable (they hold C++ state), so we cache coefficients
# and rebuild — cheap enough that the user won't notice.
# ---------------------------------------------------------------------------


def build_filter(family: str, topology: str, order: int, sample_rate: float,
                 freq_params: dict, extra: dict):
    """Construct an IIRFilter from sidebar selections.

    `freq_params` holds cutoff / center / width keyed to the topology; `extra`
    holds ripple / stopband / rolloff / q keyed to the family or RBJ variant.
    """
    if family == "RBJ":
        maker, _ = RBJ_VARIANTS[topology]
        if topology in ("lowpass", "highpass", "allpass"):
            return maker(sample_rate, freq_params["cutoff"], extra["q"])
        if topology in ("bandpass", "bandstop"):
            return maker(sample_rate, freq_params["center"], extra["bandwidth"])
        if topology in ("lowshelf", "highshelf"):
            return maker(sample_rate, freq_params["cutoff"],
                         extra["gain_db"], extra["q"])
        raise ValueError(f"Unknown RBJ topology: {topology}")

    spec = ORDER_FAMILIES[family]
    maker = spec.makers[topology]
    args = [order, sample_rate]
    if topology in ("lowpass", "highpass"):
        args.append(freq_params["cutoff"])
    else:  # bandpass / bandstop
        args.extend([freq_params["center"], freq_params["width"]])
    for p in spec.extra_params:
        args.append(extra[p])
    return maker(*args)


# ---------------------------------------------------------------------------
# Plotting helpers. Each returns a matplotlib Figure for st.pyplot().
# ---------------------------------------------------------------------------


def plot_magnitude_phase(filt, sample_rate: float, dtypes: list[str] | None,
                         signal: np.ndarray | None, num_freqs: int = 1024):
    """Magnitude (dB) + unwrapped phase across the full [0, fs/2] band.

    If `dtypes` and `signal` are provided, overlays per-dtype magnitude
    responses computed from the filter's quantized coefficients. Today this
    just annotates the reference curve with SQNR per dtype; true per-dtype
    frequency response requires upstream support (#40) for mixed-precision
    spectral analysis.
    """
    freqs = np.linspace(0.0, 0.5, num_freqs)
    H = filt.frequency_response(freqs)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase = np.unwrap(np.angle(H))

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax_mag.plot(freqs * sample_rate, mag_db, linewidth=1.6, label="reference")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.grid(True, alpha=0.4)
    ax_mag.set_ylim(bottom=max(-160.0, mag_db.min() - 6.0))

    if dtypes and signal is not None:
        ref_out = filt.process(signal, dtype="reference")
        for dt in dtypes:
            if dt == "reference":
                continue
            try:
                out = filt.process(signal, dtype=dt)
                sqnr = mpdsp.sqnr_db(ref_out, out)
                ax_mag.plot([], [], " ", label=f"{dt}: SQNR={sqnr:.1f} dB")
            except Exception as e:  # noqa: BLE001 - surface whatever upstream throws
                ax_mag.plot([], [], " ", label=f"{dt}: error")
        ax_mag.legend(loc="best", fontsize="small")

    ax_phase.plot(freqs * sample_rate, phase, linewidth=1.4, color="C1")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (rad, unwrapped)")
    ax_phase.grid(True, alpha=0.4)

    fig.tight_layout()
    return fig


def plot_pole_zero(filt):
    poles = np.asarray(filt.poles())
    theta = np.linspace(0.0, 2 * np.pi, 256)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.cos(theta), np.sin(theta), color="0.6", linewidth=0.8)
    ax.axhline(0.0, color="0.85", linewidth=0.5)
    ax.axvline(0.0, color="0.85", linewidth=0.5)
    ax.scatter(poles.real, poles.imag, marker="x", s=90, color="C3",
               linewidths=2, label=f"poles (n={len(poles)})")
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title(f"Pole locations — stability margin = "
                 f"{filt.stability_margin():.4f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_impulse_step(filt, dtypes: list[str], length: int = 256
                       ) -> tuple[plt.Figure, list[str]]:
    """Impulse and step response across selected dtypes.

    Returns `(fig, failures)` — each `failures` entry is a human-readable
    string like "tiny_posit: unsupported operation" that the caller can
    surface through `st.warning`. Swallowing silently would leave the
    user with an empty plot and no indication why.
    """
    impulse = np.zeros(length)
    impulse[0] = 1.0
    step = np.ones(length)

    fig, (ax_imp, ax_step) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax_imp.set_ylabel("Impulse response")
    ax_step.set_ylabel("Step response")
    ax_step.set_xlabel("Sample")
    for ax in (ax_imp, ax_step):
        ax.grid(True, alpha=0.4)
        ax.axhline(0.0, color="0.85", linewidth=0.5)

    failures: list[str] = []
    for dt in dtypes:
        try:
            ax_imp.plot(filt.process(impulse, dtype=dt), label=dt, linewidth=1.2)
            ax_step.plot(filt.process(step, dtype=dt), label=dt, linewidth=1.2)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{dt}: {e}")
            # Render the failure as a legend-only entry so the user sees
            # which dtype didn't contribute a trace.
            ax_imp.plot([], [], " ", label=f"{dt}: error")
    ax_imp.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig, failures


def figure_to_png_bytes(fig, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def freq_slider_bounds(nyquist: float, preferred_min: float,
                       preferred_default: float, margin_frac: float = 0.02
                       ) -> tuple[float, float, float]:
    """Derive (min, max, default) for a frequency slider keyed to Nyquist.

    At very low sample rates the "preferred" constants (e.g. 20 Hz minimum
    cutoff, 1 kHz default) would invert `min_value > max_value` or place
    the default outside `[min, max]`, which Streamlit rejects with an
    API exception. Clamp everything back into a valid range, preserving
    the preferred values when the sample rate is high enough. `margin_frac`
    keeps the upper bound strictly below Nyquist so edge-frequency designs
    don't land on numerical edge cases in the filter constructors.
    """
    hi = max(1.0, nyquist * (1.0 - margin_frac))
    # preferred_min must not exceed half the usable band, else the slider
    # becomes single-valued or degenerate.
    lo = min(preferred_min, hi * 0.5) if hi > 2.0 else hi * 0.5
    default = min(max(preferred_default, lo), hi)
    return lo, hi, default


def plot_pole_displacement(filt, dtypes: list[str]
                            ) -> tuple[plt.Figure, list[str]]:
    """Bar chart: how far each dtype drifts the poles from the reference.

    Returns `(fig, failures)` so the caller can surface per-dtype error
    messages through `st.warning` — a bar silently dropped from the plot
    would leave no trail. A short placeholder appears on the chart for
    every failing dtype so the user can see what was attempted.
    """
    labels, values = [], []
    failures: list[str] = []
    for dt in dtypes:
        if dt == "reference":
            continue
        try:
            d = filt.pole_displacement(dt)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{dt}: {e}")
            labels.append(f"{dt} (err)")
            values.append(0.0)
            continue
        labels.append(dt)
        values.append(d)
    fig, ax = plt.subplots(figsize=(8, 4))
    if not labels:
        ax.text(0.5, 0.5, "No non-reference dtypes selected.",
                ha="center", va="center", transform=ax.transAxes)
    else:
        ax.bar(labels, values, color="C2")
        ax.set_ylabel("Pole displacement (unitless)")
        ax.set_title("How far do quantized coefficients drift the poles?")
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig, failures


# ---------------------------------------------------------------------------
# The Streamlit app.
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="mpdsp filter designer",
                       layout="wide", initial_sidebar_state="expanded")
    st.title("Mixed-Precision IIR Filter Designer")
    st.caption(
        "Interactive DSPFilters-style explorer built on `mpdsp "
        f"{mpdsp.__version__}` (wraps `sw::dsp {mpdsp.__dsp_version__}`). "
        "Every IIR family, every topology, compared across all 7 mixed-"
        "precision arithmetic configurations."
    )

    # --- Sidebar: design controls ---
    st.sidebar.header("Filter design")
    families = ["RBJ"] + list(ORDER_FAMILIES.keys())
    family = st.sidebar.selectbox("Family", families, index=1)  # Butterworth default

    if family == "RBJ":
        topology = st.sidebar.selectbox("Topology", list(RBJ_VARIANTS.keys()))
        order = None
    else:
        spec = ORDER_FAMILIES[family]
        topology = st.sidebar.selectbox("Topology", list(spec.makers.keys()))
        order = st.sidebar.slider("Order", 1, 8, 4)

    # 1 kHz sample-rate floor keeps all derived slider bounds comfortably
    # above the fixed preferred minimums; the previous 100 Hz floor could
    # drive nyquist below some slider lower bounds and crash the app.
    sample_rate = st.sidebar.number_input(
        "Sample rate (Hz)", min_value=1_000.0, max_value=384_000.0,
        value=44_100.0, step=1000.0)

    nyquist = sample_rate / 2.0
    freq_params: dict = {}
    if topology in ("lowpass", "highpass", "allpass", "lowshelf", "highshelf"):
        lo, hi, default = freq_slider_bounds(nyquist, 20.0, nyquist / 4)
        freq_params["cutoff"] = st.sidebar.slider(
            "Cutoff (Hz)", lo, hi, default, step=10.0)
    else:  # bandpass / bandstop
        lo, hi, default = freq_slider_bounds(nyquist, 50.0, nyquist / 4)
        freq_params["center"] = st.sidebar.slider(
            "Center frequency (Hz)", lo, hi, default, step=10.0)
        if family != "RBJ":
            lo, hi, default = freq_slider_bounds(nyquist, 20.0, nyquist / 8)
            freq_params["width"] = st.sidebar.slider(
                "Bandwidth (Hz)", lo, hi, default, step=10.0)

    # Family-specific extra parameters.
    extra: dict = {}
    if family == "RBJ":
        for pname, default, pmin, pmax in RBJ_VARIANTS[topology][1]:
            extra[pname] = st.sidebar.slider(pname, float(pmin), float(pmax),
                                             float(default), step=0.01)
    else:
        spec = ORDER_FAMILIES[family]
        for p in spec.extra_params:
            default_range = {
                "ripple_db":   (0.01, 3.0, 0.5, 0.01),
                "stopband_db": (10.0, 80.0, 40.0, 1.0),
                "rolloff":     (0.1, 2.0, 1.0, 0.05),
            }[p]
            pmin, pmax, default, step = default_range
            extra[p] = st.sidebar.slider(p, pmin, pmax, default, step=step)

    # --- Build the filter ---
    try:
        filt = build_filter(family, topology, order, sample_rate,
                            freq_params, extra)
    except Exception as e:  # noqa: BLE001
        st.error(f"Filter design failed: {e}")
        st.stop()

    # --- Top metrics strip ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stages", f"{filt.num_stages()}")
    col2.metric("Stability margin", f"{filt.stability_margin():.4f}")
    col3.metric("Condition number", f"{filt.condition_number():.2e}")
    col4.metric("Worst-case sens.", f"{filt.worst_case_sensitivity():.4f}")

    # --- Precision selector (shared across tabs) ---
    st.sidebar.header("Mixed precision")
    all_dtypes = mpdsp.available_dtypes()
    selected_dtypes = st.sidebar.multiselect(
        "Compare dtypes", all_dtypes,
        default=["reference", "gpu_baseline", "posit_full", "tiny_posit"])

    # --- Test signal (used for SQNR measurements) ---
    st.sidebar.header("Test signal")
    sig_kind = st.sidebar.selectbox("Shape", ["sine", "chirp", "white_noise"])
    sig_length = st.sidebar.slider("Length (samples)", 128, 8192, 2048, step=128)
    if sig_kind == "sine":
        lo, hi, default = freq_slider_bounds(nyquist, 20.0, nyquist / 10)
        sig_freq = st.sidebar.slider("Frequency (Hz)", lo, hi, default, step=10.0)
        signal = mpdsp.sine(length=sig_length, frequency=sig_freq,
                            sample_rate=sample_rate)
    elif sig_kind == "chirp":
        # f_start/f_end: use the same Nyquist margin as the sliders so the
        # chirp doesn't drive the design up to the exact Nyquist edge.
        _, hi, _ = freq_slider_bounds(nyquist, 1.0, nyquist / 2)
        f_start = min(20.0, hi * 0.5)
        signal = mpdsp.chirp(length=sig_length, f_start=f_start,
                             f_end=hi, sample_rate=sample_rate)
    else:
        signal = mpdsp.white_noise(length=sig_length, amplitude=0.5, seed=1)

    # --- Tabs ---
    tab_freq, tab_pz, tab_time, tab_prec = st.tabs(
        ["Frequency response", "Pole / zero", "Time domain",
         "Mixed-precision comparison"])

    # Build the shared descriptor used in export filenames so the same
    # design can be tagged across multiple downloads.
    if family == "RBJ":
        tag = f"rbj_{topology}"
    else:
        tag = f"{family.lower().replace(' ', '')}_{topology}_n{order}"

    # Each tab renders a matplotlib Figure, then pipes bytes out for the
    # PNG download button, then calls `plt.close(fig)`. Without the close,
    # matplotlib's pyplot registry retains every figure generated on every
    # slider change, steadily leaking memory in long-lived sessions.

    with tab_freq:
        fig = plot_magnitude_phase(filt, sample_rate, selected_dtypes, signal)
        st.pyplot(fig)
        st.download_button("Download PNG", figure_to_png_bytes(fig),
                           f"{tag}_freq.png", "image/png")
        plt.close(fig)

    with tab_pz:
        fig = plot_pole_zero(filt)
        st.pyplot(fig)
        st.download_button("Download PNG", figure_to_png_bytes(fig),
                           f"{tag}_polezero.png", "image/png")
        plt.close(fig)
        if filt.num_stages() > 0:
            coefs = filt.coefficients()
            st.caption(f"{len(coefs)} biquad stage(s). "
                       "Coefficients are always designed in `double`.")
            coef_dict = {"stage": list(range(len(coefs))),
                          "b0": [c[0] for c in coefs],
                          "b1": [c[1] for c in coefs],
                          "b2": [c[2] for c in coefs],
                          "a1": [c[3] for c in coefs],
                          "a2": [c[4] for c in coefs]}
            st.dataframe(coef_dict, hide_index=True)
            # CSV export via pandas if available; fallback to manual join.
            try:
                import pandas as pd
                csv_bytes = pd.DataFrame(coef_dict).to_csv(index=False).encode()
            except ImportError:
                rows = [",".join(coef_dict.keys())]
                for i in range(len(coefs)):
                    rows.append(",".join(str(coef_dict[k][i]) for k in coef_dict))
                csv_bytes = "\n".join(rows).encode()
            st.download_button("Download coefficients CSV", csv_bytes,
                               f"{tag}_coefficients.csv", "text/csv")

    with tab_time:
        fig, time_failures = plot_impulse_step(filt, selected_dtypes)
        st.pyplot(fig)
        st.download_button("Download PNG", figure_to_png_bytes(fig),
                           f"{tag}_time.png", "image/png")
        plt.close(fig)
        for msg in time_failures:
            st.warning(f"Time-domain plot skipped {msg}", icon="⚠️")

    with tab_prec:
        if not selected_dtypes:
            st.info("Pick at least one dtype in the sidebar.")
        else:
            df = compare_filters(filt, signal, dtypes=selected_dtypes)
            st.dataframe(df, hide_index=True)
            # pandas is already in mpdsp's runtime deps so `to_csv` is safe.
            st.download_button("Download comparison CSV",
                               df.to_csv(index=False).encode(),
                               f"{tag}_comparison.csv", "text/csv")
            fig, disp_failures = plot_pole_displacement(filt, selected_dtypes)
            st.pyplot(fig)
            st.download_button("Download displacement PNG",
                               figure_to_png_bytes(fig),
                               f"{tag}_displacement.png", "image/png")
            plt.close(fig)
            for msg in disp_failures:
                st.warning(f"Pole displacement skipped {msg}", icon="⚠️")

    st.caption(
        "Free-function analysis primitives (`biquad_poles`, "
        "`coefficient_sensitivity`, `biquad_condition_number`) and dtype "
        "dispatch on spectral transforms are planned — see issue #8 and #40."
    )


if __name__ == "__main__":
    main()
