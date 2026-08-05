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
import pandas as pd
import seaborn as sns
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


def _reference_cutoff(topology: str, freq_params: dict) -> float | None:
    """Return the reference frequency (Hz) for normalizing the x-axis.

    - LP / HP / RBJ shelves: the user-specified `cutoff`.
    - Non-RBJ and RBJ bandpass / bandstop: the user-specified `center` —
      that's what the sliders exposed and what the user is thinking about,
      more faithful than reconstructing geometric-mean edges from (center,
      width). For filter families defined by (f_lo, f_hi) edges the
      geometric mean would be the canonical choice, but none of our
      current constructors take that shape.
    - RBJ allpass: returns None so the caller falls back to the absolute-Hz
      view. The allpass magnitude is flat at 0 dB for every ω, so a
      normalized plot would be a featureless horizontal line; the phase
      twist that *is* interesting reads fine on absolute Hz too.

    The topology check for allpass is explicit because the allpass slider
    lives in the same "lowpass/highpass/allpass/lowshelf/highshelf" branch
    that populates `freq_params["cutoff"]`, so keying off the dict alone
    would silently enable normalization for allpass.
    """
    if topology == "allpass":
        return None
    if "cutoff" in freq_params:
        return freq_params["cutoff"]
    if "center" in freq_params:
        return freq_params["center"]
    return None


def plot_magnitude_phase(filt, sample_rate: float, dtypes: list[str] | None,
                         signal: np.ndarray | None, num_freqs: int = 1024,
                         x_units: str = "Hz",
                         x_scale: str = "linear",
                         cutoff: float | None = None):
    """Magnitude (dB) + unwrapped phase.

    `x_units` selects the x-axis framing:
      - "Hz": absolute frequency (deployment view — current default)
      - "ratio": normalized to cutoff (f / f_c). Needs `cutoff` to be
        non-None; falls back to "Hz" otherwise (e.g. RBJ allpass).

    `x_scale` is "linear" or "log". Log skips the DC sample — we pick a
    small positive lower bound instead (fs/10000 for "Hz", 0.01·f_c for
    "ratio") so the log sweep gets ~4 decades of context.

    If `dtypes` and `signal` are provided, overlays per-dtype magnitude
    responses computed from the filter's quantized coefficients. Today this
    just annotates the reference curve with SQNR per dtype; true per-dtype
    frequency response requires upstream support (#40) for mixed-precision
    spectral analysis.
    """
    nyquist = sample_rate / 2.0

    normalize = (x_units == "ratio" and cutoff is not None and cutoff > 0)

    # Build the normalized-frequency sweep (what filt.frequency_response
    # consumes — values in [0, 0.5]) based on the two toggles. For log
    # scale we pick a small positive lower bound so log10 is finite and
    # the plot shows ~4 decades of context before the interesting region.
    if x_scale == "log":
        if normalize:
            # 2 decades below cutoff, 1 decade above (clamped to Nyquist).
            f_lo_hz = max(cutoff * 0.01, sample_rate / 100000.0)
            f_hi_hz = min(cutoff * 10.0, nyquist * 0.999)
        else:
            f_lo_hz = max(sample_rate / 10000.0, 1e-6)
            f_hi_hz = nyquist * 0.999
        freqs = np.logspace(np.log10(f_lo_hz / sample_rate),
                             np.log10(f_hi_hz / sample_rate), num_freqs)
    else:
        if normalize:
            f_hi_hz = min(cutoff * 3.0, nyquist)
            freqs = np.linspace(0.0, f_hi_hz / sample_rate, num_freqs)
        else:
            freqs = np.linspace(0.0, 0.5, num_freqs)

    H = filt.frequency_response(freqs)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase = np.unwrap(np.angle(H))

    # X values and labels. Normalization expresses f/f_c directly so the
    # axis number reads "1.0 = cutoff" — matches the Wikipedia convention.
    if normalize:
        x = freqs * sample_rate / cutoff
        xlabel = "f / f_c"
    else:
        x = freqs * sample_rate
        xlabel = "Frequency (Hz)"

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax_mag.plot(x, mag_db, linewidth=1.6, label="reference")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.grid(True, alpha=0.4, which="both")
    ax_mag.set_ylim(bottom=max(-160.0, mag_db.min() - 6.0))
    if x_scale == "log":
        ax_mag.set_xscale("log")
    # A horizontal guide at -3 dB is the canonical cutoff marker on a
    # normalized Bode plot; show it only when normalization is active to
    # keep the default Hz view uncluttered.
    if normalize:
        ax_mag.axhline(-3.0, color="0.7", linewidth=0.5, linestyle="--")

    if dtypes and signal is not None:
        ref_out = filt.process(signal, dtype="reference")
        for dt in dtypes:
            if dt == "reference":
                continue
            try:
                out = filt.process(signal, dtype=dt)
                sqnr = mpdsp.sqnr_db(ref_out, out)
                ax_mag.plot([], [], " ", label=f"{dt}: SQNR={sqnr:.1f} dB")
            except Exception:  # noqa: BLE001 - surface whatever upstream throws
                ax_mag.plot([], [], " ", label=f"{dt}: error")
        ax_mag.legend(loc="best", fontsize="small")

    ax_phase.plot(x, phase, linewidth=1.4, color="C1")
    ax_phase.set_xlabel(xlabel)
    ax_phase.set_ylabel("Phase (rad, unwrapped)")
    ax_phase.grid(True, alpha=0.4, which="both")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Analog prototype (pre-bilinear H(s)) — Issue #78
#
# Every classical IIR family is designed as an analog prototype in the s-plane
# and then bilinear-transformed to a digital cascade. The digital response the
# "Frequency response" tab shows therefore bakes in the bilinear frequency
# warp: it squashes asymptotically toward Nyquist, while the prototype extends
# linearly in omega. Showing both is how the warp becomes visible, and it is
# the only way to read a family's signature without warp artifacts — Bessel's
# flat group delay, for instance, is an omega-space property that the digital
# form only approximates near DC.
#
# The prototypes come from mpdsp's *_prototype factories (#115). Those are
# keyed on design parameters rather than extracted from a designed IIRFilter,
# which is fine here because the dashboard owns the parameters it just used.
# ---------------------------------------------------------------------------

# Families whose analog prototype is available. Legendre is absent upstream:
# sw::dsp::transfer_function has no legendre_prototype, so there is nothing to
# plot for it even though its digital designer exists.
PROTOTYPE_FACTORIES = {
    "Butterworth": mpdsp.butterworth_prototype,
    "Chebyshev I": mpdsp.chebyshev1_prototype,
    "Chebyshev II": mpdsp.chebyshev2_prototype,
    "Bessel": mpdsp.bessel_prototype,
    "Elliptic": mpdsp.elliptic_prototype,
}


def analog_prototype_available(family: str) -> tuple[bool, str]:
    """(available, reason) for the analog-prototype pane."""
    if family == "RBJ":
        return False, (
            "RBJ biquads are designed directly in the z-plane from the audio "
            "EQ cookbook formulas — there is no analog prototype to show. "
            "The digital response on the *Frequency response* tab is the "
            "whole design."
        )
    if family not in PROTOTYPE_FACTORIES:
        return False, (
            f"{family} has no analog-prototype factory upstream "
            f"(`sw::dsp::transfer_function` provides Butterworth, "
            f"Chebyshev I/II, Bessel, and Elliptic). The digital design is "
            f"unaffected — only this pedagogical view is unavailable."
        )
    return True, ""


def build_analog_prototype(family: str, topology: str, order: int,
                           freq_params: dict, extra: dict,
                           selectivity_k: float = 0.9,
                           as_lowpass_prototype: bool = False):
    """Return the PoleZeroPlot for the current design, or None.

    Lowpass designs are built directly at the requested cutoff. The other
    topologies build a normalized lowpass prototype and then apply the same
    frequency transformation the digital designer applies before the bilinear
    step, so the constellation corresponds to *this* filter rather than to a
    generic lowpass.

    `as_lowpass_prototype` short-circuits that and returns the underlying
    lowpass instead — the more pedagogical view, since it shows the family
    signature without the band transformation folded in.
    """
    factory = PROTOTYPE_FACTORIES.get(family)
    if factory is None:
        return None

    kwargs = {}
    if family == "Chebyshev I":
        kwargs["ripple_db"] = extra.get("ripple_db", 1.0)
    elif family == "Chebyshev II":
        kwargs["stopband_db"] = extra.get("stopband_db", 40.0)
    elif family == "Elliptic":
        # NOTE: upstream's elliptic prototype is parameterized by the elliptic
        # modulus (selectivity_k), while the digital designer takes a
        # `rolloff` factor. They are not the same knob and there is no exact
        # mapping, so this pane exposes selectivity_k separately and the
        # caption says the two curves will not correspond exactly.
        kwargs["ripple_db"] = extra.get("ripple_db", 1.0)
        kwargs["selectivity_k"] = selectivity_k

    if topology == "lowpass" and not as_lowpass_prototype:
        return factory(order, freq_params["cutoff"], **kwargs)

    # Normalized lowpass prototype, then transform.
    plot = factory(order, 1.0, **kwargs)
    if as_lowpass_prototype or topology == "lowpass":
        return plot
    if topology == "highpass":
        return mpdsp.lp_to_hp(plot, freq_params["cutoff"])

    center = freq_params["center"]
    width = freq_params["width"]
    low, high = center - width / 2.0, center + width / 2.0
    if low <= 0.0:
        return None
    if topology == "bandpass":
        return mpdsp.lp_to_bp(plot, low, high)
    if topology == "bandstop":
        return mpdsp.lp_to_bs(plot, low, high)
    return None


def analog_response(plot, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate H(j*omega) from the s-plane constellation.

    H(s) = prod(s - z_i) / prod(s - p_j), evaluated directly from the poles
    and zeros rather than by expanding to polynomial coefficients — the
    expansion is ill-conditioned at the orders a bandstop reaches, and
    `laplace_freqs` wants a uniform grid anyway, which is the wrong sampling
    for a Bode plot.

    Returns (magnitude_db, phase_deg), normalized so the peak is 0 dB. Peak
    normalization works across every topology, where fixing |H(0)| = 1 would
    divide by zero for highpass and bandpass.
    """
    s = 1j * omega
    numerator = np.ones_like(s)
    for zero in plot.s_zeros:
        numerator = numerator * (s - zero)
    denominator = np.ones_like(s)
    for pole in plot.s_poles:
        denominator = denominator * (s - pole)

    h = numerator / denominator
    magnitude = np.abs(h)
    peak = magnitude.max()
    if peak > 0:
        magnitude = magnitude / peak
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    phase_deg = np.degrees(np.unwrap(np.angle(h)))
    return magnitude_db, phase_deg


def plot_analog_bode(plot, sample_rate: float, num_points: int = 1024):
    """Bode plot of the analog prototype, with Nyquist marked.

    The sweep spans four decades around the constellation so the asymptotic
    roll-off is visible; the digital design's Nyquist is drawn as a reference
    line, because everything to the right of it is what the bilinear
    transform has to compress into the region below it.
    """
    radii = np.abs(np.concatenate([
        np.asarray(plot.s_poles, dtype=complex),
        np.asarray(plot.s_zeros, dtype=complex)
        if len(plot.s_zeros) else np.zeros(0, dtype=complex)]))
    radii = radii[radii > 0]
    center = float(np.median(radii)) if len(radii) else 1.0
    omega = np.logspace(np.log10(center) - 2.0, np.log10(center) + 2.0,
                        num_points)

    magnitude_db, phase_deg = analog_response(plot, omega)

    fig, (ax_mag, ax_phase) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})

    nyquist_omega = 2.0 * np.pi * (sample_rate / 2.0)

    ax_mag.semilogx(omega, magnitude_db, linewidth=1.4)
    ax_mag.axhline(-3.0, color="grey", linestyle=":", linewidth=0.9,
                   label="-3 dB")
    if omega[0] < nyquist_omega < omega[-1]:
        ax_mag.axvline(nyquist_omega, color="tab:red", linestyle="--",
                       linewidth=1.0, label="digital Nyquist")
    ax_mag.set_ylabel("|H(jw)| (dB)")
    ax_mag.set_ylim(-100, 10)
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(fontsize=8)
    ax_mag.set_title(f"Analog prototype — {plot.design}, order {plot.order}, "
                     f"{plot.kind}")

    ax_phase.semilogx(omega, phase_deg, linewidth=1.2, color="tab:orange")
    if omega[0] < nyquist_omega < omega[-1]:
        ax_phase.axvline(nyquist_omega, color="tab:red", linestyle="--",
                         linewidth=1.0)
    ax_phase.set_xlabel("angular frequency w (rad/s)")
    ax_phase.set_ylabel("phase (deg)")
    ax_phase.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_group_delay(filt, sample_rate: float, num_points: int = 512):
    """Group delay τ(f) = -d(phase)/d(omega), plotted across [0, fs/2].

    Group delay quantifies per-frequency latency: a passband with flat
    group delay preserves waveform shape (Bessel's signature property),
    while sharp-rolloff IIR filters (elliptic, high-order Chebyshev) peak
    hard at the transition band. Folds the IIR cascade into a single
    TransferFunction via `to_transfer_function`, then calls upstream's
    central-difference group_delay — the same path DSPFilters' classic
    GUI used for its group-delay pane.
    """
    tf = mpdsp.to_transfer_function(filt)
    gd = mpdsp.group_delay(tf, num_points)
    freqs = np.linspace(0.0, 0.5, num_points, endpoint=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(freqs * sample_rate, gd, linewidth=1.4, color="C2")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Group delay (samples)")
    ax.set_title(f"Group delay — mean {gd.mean():.2f} samples, "
                 f"peak {gd.max():.2f} samples")
    ax.grid(True, alpha=0.4)
    ax.axhline(0.0, color="0.85", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_pole_zero(filt):
    poles = np.asarray(filt.poles())
    zeros = np.asarray(filt.zeros())
    theta = np.linspace(0.0, 2 * np.pi, 256)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.cos(theta), np.sin(theta), color="0.6", linewidth=0.8)
    ax.axhline(0.0, color="0.85", linewidth=0.5)
    ax.axvline(0.0, color="0.85", linewidth=0.5)
    # Conventional markers: 'x' for poles, 'o' for zeros. All-pole families
    # (Butterworth / Chebyshev I / Bessel / Legendre) place every zero at
    # z = -1, so they render as a visible N-fold cluster on the negative
    # real axis rather than disappearing from the plot.
    ax.scatter(poles.real, poles.imag, marker="x", s=90, color="C3",
               linewidths=2, label=f"poles (n={len(poles)})")
    ax.scatter(zeros.real, zeros.imag, marker="o", s=90,
               facecolors="none", edgecolors="C0", linewidths=1.5,
               label=f"zeros (n={len(zeros)})")
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title(f"Pole / zero locations — stability margin = "
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
# Dedicated two-type comparison: issue #16 asks for "select two arithmetic
# types, show overlay and error" which the existing multi-dtype SQNR table
# doesn't quite match. This renders an A-vs-B side-by-side at higher fidelity
# (both magnitude + phase of A and B, with reference shown in grey as context)
# plus a tight numeric summary below.
# ---------------------------------------------------------------------------


def plot_two_type_compare(filt, sample_rate: float, dtype_a: str, dtype_b: str,
                           signal: np.ndarray, num_freqs: int = 1024):
    """Magnitude/phase overlay of two dtypes + impulse-response overlay.

    The reference (`double`) frequency response is drawn in grey as visual
    context so the reader can see which of A or B is closer to reference.
    Impulse response is plotted for both dtypes on the lower subplot.

    Returns (fig, metrics_dict). `metrics_dict` carries the numbers the
    caller should surface: SQNR and max|Δ| between A and B computed over
    the provided `signal`, plus each dtype's SQNR vs reference.
    """
    freqs = np.linspace(0.0, 0.5, num_freqs)
    H_ref = filt.frequency_response(freqs)
    mag_ref_db = 20.0 * np.log10(np.maximum(np.abs(H_ref), 1e-12))

    fig, (ax_mag, ax_phase, ax_imp) = plt.subplots(
        3, 1, figsize=(10, 9), sharex=False)

    # Reference mag in grey — gives the reader the ideal curve alongside A/B.
    ax_mag.plot(freqs * sample_rate, mag_ref_db,
                 color="0.6", linewidth=2.2, linestyle="-",
                 label="reference")
    ax_phase.plot(freqs * sample_rate, np.unwrap(np.angle(H_ref)),
                   color="0.6", linewidth=2.2, linestyle="-",
                   label="reference")

    # Can't get per-dtype frequency response today (spectral dtype dispatch
    # is on the #40 backlog). Use the same designed-coefficient response
    # for both lines and distinguish with color/style — downstream work
    # at #40 will tighten this.
    for dtype, color, ls in ((dtype_a, "C0", "-"), (dtype_b, "C3", "--")):
        ax_mag.plot(freqs * sample_rate, mag_ref_db,
                     color=color, linewidth=1.4, linestyle=ls,
                     label=dtype, alpha=0.85)
        ax_phase.plot(freqs * sample_rate, np.unwrap(np.angle(H_ref)),
                       color=color, linewidth=1.4, linestyle=ls,
                       label=dtype, alpha=0.85)

    ax_mag.set_xscale("log")
    ax_mag.set(ylabel="Magnitude (dB)", ylim=(-80, 5))
    ax_mag.set_title(f"A = {dtype_a}   vs   B = {dtype_b}", fontweight="bold")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(loc="lower left", ncol=3)

    ax_phase.set_xscale("log")
    ax_phase.set(ylabel="Phase (rad, unwrapped)", xlabel="Frequency (Hz)")
    ax_phase.grid(True, which="both", alpha=0.3)
    ax_phase.legend(loc="lower left", ncol=3)

    # Impulse response overlay on the bottom subplot — this is where the
    # two dtypes actually differ visibly, because filt.process(impulse,
    # dtype=...) goes through the quantized state/sample types.
    impulse = np.zeros(256)
    impulse[0] = 1.0
    metrics = {"dtype_a": dtype_a, "dtype_b": dtype_b}
    try:
        ya = filt.process(impulse, dtype=dtype_a)
        ax_imp.plot(ya, color="C0", linewidth=1.6, label=dtype_a)
    except Exception as e:  # noqa: BLE001
        metrics["error_a"] = str(e)
        ya = None
    try:
        yb = filt.process(impulse, dtype=dtype_b)
        ax_imp.plot(yb, color="C3", linewidth=1.2, linestyle="--", label=dtype_b)
    except Exception as e:  # noqa: BLE001
        metrics["error_b"] = str(e)
        yb = None
    ax_imp.set(xlabel="Sample", ylabel="Impulse response")
    ax_imp.grid(True, alpha=0.4)
    ax_imp.legend(loc="upper right")
    ax_imp.set_title("Impulse response (samples 0–255)")

    # Metrics on the caller-supplied signal so a realistic SQNR is available.
    try:
        sig_a = filt.process(signal, dtype=dtype_a)
        sig_b = filt.process(signal, dtype=dtype_b)
        sig_ref = filt.process(signal, dtype="reference")
        metrics["sqnr_ab_db"] = float(mpdsp.sqnr_db(sig_a, sig_b))
        metrics["max_abs_ab"] = float(np.max(np.abs(sig_a - sig_b)))
        metrics["sqnr_a_vs_ref_db"] = float(mpdsp.sqnr_db(sig_ref, sig_a))
        metrics["sqnr_b_vs_ref_db"] = float(mpdsp.sqnr_db(sig_ref, sig_b))
    except Exception as e:  # noqa: BLE001
        metrics["error_metrics"] = str(e)

    fig.tight_layout()
    return fig, metrics


# ---------------------------------------------------------------------------
# Summary panel: heatmap + precision-cost frontier.
# Computes on-demand for the current design, rather than reading the
# static CSVs — gives the researcher immediate feedback as sliders change.
# ---------------------------------------------------------------------------


def plot_summary_heatmap(filt, signal: np.ndarray):
    """Single-row heatmap of SQNR across all available dtypes.

    For a summary view driven by a single currently-designed filter the
    interesting axis is "which arithmetic configs deliver what SQNR". A
    single-row heatmap keeps that axis horizontal and annotation-friendly.
    """
    dtypes = list(mpdsp.available_dtypes())
    sqnrs: list[float] = []
    ref_out = filt.process(signal, dtype="reference")
    for dt in dtypes:
        try:
            out = filt.process(signal, dtype=dt)
            sqnrs.append(float(mpdsp.sqnr_db(ref_out, out)))
        except Exception:  # noqa: BLE001
            sqnrs.append(float("nan"))

    row = pd.DataFrame([sqnrs], columns=dtypes, index=["current design"])
    display = row.clip(upper=200)
    annot = row.map(
        lambda v: "nan" if pd.isna(v) else ("inf" if v >= 290 else f"{v:.0f}")
    ).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 2.5))
    sns.heatmap(display, annot=annot, fmt="", cmap="RdYlGn",
                 vmin=0, vmax=200, ax=ax,
                 cbar_kws={"label": "SQNR (dB), capped at 200"},
                 linewidths=0.5, linecolor="white")
    ax.set_title("SQNR across arithmetic configurations — current design",
                  fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig, row.iloc[0].to_dict()


def plot_precision_cost_frontier(filt, signal: np.ndarray):
    """SQNR vs bits-per-sample scatter: identifies Pareto-optimal dtypes.

    Bit widths come from `mpdsp.bits_of()` (issue #55) — the dashboard
    used to duplicate this mapping as a local dict, which drifted every
    time a new config was added. Now it queries the binding directly and
    will automatically pick up any future sensor/FPGA configs.
    """
    ref_out = filt.process(signal, dtype="reference")

    rows = []
    for dt in mpdsp.available_dtypes():
        try:
            out = filt.process(signal, dtype=dt)
            sqnr = float(mpdsp.sqnr_db(ref_out, out))
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(sqnr) and sqnr < 290:
            rows.append({"dtype": dt, "bits": mpdsp.bits_of(dt),
                          "sqnr_db": sqnr})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    if df.empty:
        ax.text(0.5, 0.5, "No non-reference dtypes produced finite SQNR.",
                 ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        ax.scatter(df["bits"], df["sqnr_db"], s=120, c="C1",
                    edgecolors="black", linewidth=0.6, zorder=3)
        for _, r in df.iterrows():
            ax.annotate(r["dtype"], (r["bits"], r["sqnr_db"]),
                         fontsize=8, xytext=(5, 5),
                         textcoords="offset points")
        ax.set(xlabel="Bits per sample", ylabel="SQNR (dB)",
                title="Precision-cost frontier — current design")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, df


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
    # Frequency-response pane axis toggles (issue #75). The "Hz + linear"
    # default is the deployment-view framing (concrete Hz from DC to
    # Nyquist); "ratio + log" is the Bode / Wikipedia convention that makes
    # rolloff a straight line with slope 20·N dB/decade and collapses all
    # same-family-same-order filters onto the same curve regardless of
    # cutoff. Independent toggles so "Hz + log" (Bode of the deployed
    # filter) is reachable in one step.
    st.sidebar.header("Freq-response axis")
    x_units = st.sidebar.radio(
        "X-axis units",
        ["Hz", "ratio"],
        format_func=lambda s: "Absolute Hz" if s == "Hz" else "f / f_c (normalized)",
        horizontal=True,
    )
    x_scale = st.sidebar.radio(
        "X-axis scale",
        ["linear", "log"],
        horizontal=True,
    )

    # Two-type comparison picker — separate from the multiselect above so
    # the A-vs-B tab has its own controls the user can change without
    # disturbing the broader multi-dtype comparison.
    st.sidebar.header("Two-type compare")
    dtype_a = st.sidebar.selectbox("Type A", all_dtypes,
                                     index=all_dtypes.index("reference"))
    default_b = "posit_full" if "posit_full" in all_dtypes else all_dtypes[-1]
    dtype_b = st.sidebar.selectbox("Type B", all_dtypes,
                                     index=all_dtypes.index(default_b))

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
    (tab_freq, tab_analog, tab_pz, tab_gd, tab_time, tab_prec,
     tab_two_type, tab_summary) = st.tabs(
        ["Frequency response", "Analog prototype", "Pole / zero",
         "Group delay", "Time domain", "Mixed-precision comparison",
         "Compare A vs B", "Summary"])

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
        fig = plot_magnitude_phase(
            filt, sample_rate, selected_dtypes, signal,
            x_units=x_units, x_scale=x_scale,
            cutoff=_reference_cutoff(topology, freq_params))
        st.pyplot(fig)
        st.download_button("Download PNG", figure_to_png_bytes(fig),
                           f"{tag}_freq.png", "image/png")
        plt.close(fig)

    with tab_analog:
        available, reason = analog_prototype_available(family)
        if not available:
            st.info(reason)
        else:
            st.markdown(
                "The **pre-bilinear** H(s): the analog prototype this filter "
                "was designed from, before the bilinear transform mapped it "
                "to the z-plane. Compare against the *Frequency response* "
                "tab — the digital curve compresses asymptotically toward "
                "Nyquist, while this one continues linearly in omega. That "
                "difference is bilinear frequency warping."
            )

            # Elliptic's prototype takes the elliptic modulus, which is not
            # the `rolloff` knob the digital designer takes; expose it here
            # rather than pretend a mapping exists.
            selectivity_k = 0.9
            if family == "Elliptic":
                selectivity_k = st.slider(
                    "selectivity_k (elliptic modulus)", 0.05, 0.99, 0.9,
                    step=0.01, key="analog_selectivity")

            # Bandpass/bandstop can only be shown as the untransformed
            # lowpass prototype: upstream's lp_to_bp / lp_to_bs produce
            # constellations whose responses are not bandpass and bandstop
            # (mixed-precision-dsp#204 — lp_to_bs emits no notch zeros at
            # all). Rendering the transformed view would put a knowingly
            # wrong curve on screen, so it is withheld rather than captioned.
            # lp_to_hp is correct, so highpass keeps the choice.
            as_lp = False
            if topology == "highpass":
                view = st.radio(
                    "View", ["Transformed H(s)", "Lowpass prototype"],
                    horizontal=True, key="analog_view")
                as_lp = view == "Lowpass prototype"
            elif topology in ("bandpass", "bandstop"):
                as_lp = True
                st.warning(
                    f"Showing the **lowpass prototype** this {topology} is "
                    f"built from. The transformed H(s) view is withheld: "
                    f"upstream's `lp_to_bp` / `lp_to_bs` currently produce "
                    f"constellations whose responses are not {topology} "
                    f"(see [mixed-precision-dsp#204](https://github.com/"
                    f"stillwater-sc/mixed-precision-dsp/issues/204)). The "
                    f"prototype below is correct, and is the view this "
                    f"pane's bilinear-warp comparison needs anyway."
                )

            try:
                proto = build_analog_prototype(
                    family, topology, order, freq_params, extra,
                    selectivity_k=selectivity_k, as_lowpass_prototype=as_lp)
            except Exception as exc:                      # noqa: BLE001
                proto = None
                st.error(f"Prototype construction failed: {exc}")

            if proto is None:
                st.warning("No analog prototype available for this "
                           "parameter combination.")
            else:
                fig = plot_analog_bode(proto, sample_rate)
                st.pyplot(fig)
                st.download_button("Download PNG", figure_to_png_bytes(fig),
                                   f"{tag}_analog.png", "image/png")
                plt.close(fig)

                if as_lp:
                    st.caption(
                        "Showing the **normalized lowpass prototype** the "
                        "band transformation starts from — the family "
                        "signature without the transformation folded in. "
                        "Switch to *Transformed H(s)* for the constellation "
                        "belonging to this specific filter."
                    )
                else:
                    st.caption(
                        f"Showing the **transformed H(s)** for this specific "
                        f"{topology} — the lowpass prototype after the "
                        f"frequency transformation, which is what the "
                        f"bilinear step actually consumes. Magnitude is "
                        f"normalized to a 0 dB peak, since |H(0)| = 0 for "
                        f"highpass and bandpass."
                    )

                if family == "Elliptic":
                    st.caption(
                        ":warning: Upstream's elliptic prototype is "
                        "parameterized by the elliptic modulus "
                        "(`selectivity_k`), while the digital designer takes "
                        "a `rolloff` factor. There is no exact mapping "
                        "between them, so this curve will not correspond "
                        "exactly to the digital response — treat it as "
                        "representative of the family, not of this design."
                    )

                st.markdown(
                    f"**s-plane constellation** — {len(proto.s_poles)} pole(s), "
                    f"{len(proto.s_zeros)} zero(s). "
                    + ("All-pole family: the stopband roll-off comes entirely "
                       "from the poles."
                       if not len(proto.s_zeros)
                       else "Finite zeros on the jw axis produce the stopband "
                            "nulls.")
                )

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

    with tab_gd:
        fig = None
        try:
            fig = plot_group_delay(filt, sample_rate)
            st.pyplot(fig)
            st.download_button("Download PNG", figure_to_png_bytes(fig),
                               f"{tag}_groupdelay.png", "image/png")
            st.caption(
                "Group delay is the per-frequency latency τ(f) = "
                "-d(phase)/d(ω). Bessel prototypes target a flat group "
                "delay across the passband (waveform-preserving); "
                "elliptic and high-order Chebyshev filters trade group "
                "delay for sharper rolloff, producing a peak near the "
                "transition band.")
        except Exception as e:  # noqa: BLE001 - surface whatever upstream throws
            st.warning(f"Group-delay computation failed: {e}", icon="⚠️")
        finally:
            # Close the figure even if st.pyplot / figure_to_png_bytes /
            # st.download_button throws after it was created — otherwise the
            # pyplot registry accumulates figures across slider changes.
            # plot_group_delay may have thrown before returning, in which
            # case fig stays None and there's nothing to close.
            if fig is not None:
                plt.close(fig)

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

    with tab_two_type:
        st.markdown(
            f"**A = `{dtype_a}`** (blue, solid)   **B = `{dtype_b}`** "
            f"(red, dashed)   reference in grey as visual context.")
        if dtype_a == dtype_b:
            st.info("Pick two different dtypes in the sidebar to see a "
                    "meaningful comparison.")
        else:
            fig, metrics = plot_two_type_compare(
                filt, sample_rate, dtype_a, dtype_b, signal)
            st.pyplot(fig)
            st.download_button(
                "Download PNG", figure_to_png_bytes(fig),
                f"{tag}_{dtype_a}_vs_{dtype_b}.png".replace("<","").replace(">","")
                .replace(",", "_"),
                "image/png")
            plt.close(fig)
            # Metrics row
            cols = st.columns(4)
            cols[0].metric(f"SQNR A vs B (dB)",
                            f"{metrics.get('sqnr_ab_db', float('nan')):.1f}")
            cols[1].metric(f"max|A − B|",
                            f"{metrics.get('max_abs_ab', float('nan')):.2e}")
            cols[2].metric(f"SQNR {dtype_a} vs ref",
                            f"{metrics.get('sqnr_a_vs_ref_db', float('nan')):.1f}")
            cols[3].metric(f"SQNR {dtype_b} vs ref",
                            f"{metrics.get('sqnr_b_vs_ref_db', float('nan')):.1f}")
            for k in ("error_a", "error_b", "error_metrics"):
                if k in metrics:
                    st.warning(f"{k}: {metrics[k]}", icon="⚠️")

    with tab_summary:
        st.markdown(
            "Computed live for the **current design** — not from a "
            "pre-collected CSV — so these numbers move as you slide "
            "family / order / cutoff in the sidebar.")
        fig_hm, _ = plot_summary_heatmap(filt, signal)
        st.pyplot(fig_hm)
        st.download_button("Download heatmap PNG",
                            figure_to_png_bytes(fig_hm),
                            f"{tag}_summary_heatmap.png", "image/png")
        plt.close(fig_hm)
        fig_fr, frontier_df = plot_precision_cost_frontier(filt, signal)
        st.pyplot(fig_fr)
        st.download_button("Download frontier PNG",
                            figure_to_png_bytes(fig_fr),
                            f"{tag}_precision_cost_frontier.png", "image/png")
        plt.close(fig_fr)
        if not frontier_df.empty:
            st.download_button("Download frontier CSV",
                                frontier_df.to_csv(index=False).encode(),
                                f"{tag}_precision_cost_frontier.csv",
                                "text/csv")

    st.caption(
        "Free-function analysis primitives (`biquad_poles`, "
        "`coefficient_sensitivity`, `biquad_condition_number`) and dtype "
        "dispatch on spectral transforms are planned — see issue #8 and #40."
    )


if __name__ == "__main__":
    main()
