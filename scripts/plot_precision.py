#!/usr/bin/env python3
"""Publication-quality overlay plots from `iir_precision_sweep` CSV output.

Reads three CSVs from a directory and produces five figures overlaying
every arithmetic type against the reference (double) per filter family.

Expected inputs (column names):

- `frequency_response.csv`:
    filter_family, arith_type, freq_hz, magnitude_db, phase_deg,
    ref_magnitude_db, ref_phase_deg
- `impulse_response.csv` (optional — impulse plots are skipped if absent):
    filter_family, arith_type, sample_index, value, ref_value

Outputs saved to `--output-dir` (both PNG + PDF):
    magnitude_response.{png,pdf}
    phase_response.{png,pdf}
    magnitude_error.{png,pdf}
    phase_error.{png,pdf}
    impulse_response.{png,pdf}  (only if impulse_response.csv present)

Usage:
    python scripts/plot_precision.py --input-dir results/ --output-dir figures/
    python scripts/plot_precision.py results/ --output-dir figures/     # legacy positional
    python scripts/plot_precision.py results/ --output figures/         # legacy --output
    python scripts/plot_precision.py results/                           # interactive

Optional styling:
    --publication    Serif fonts, tighter margins, larger labels
    --latex          Use LaTeX for text (requires a working TeX install)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REFERENCE_TYPE = "double"       # which arith_type is the ground truth
IMPULSE_WINDOW = 100            # first N samples shown on impulse plots


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_frequency_response(csv_dir: str) -> pd.DataFrame:
    """Required CSV — missing it is a hard error."""
    path = os.path.join(csv_dir, "frequency_response.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def load_impulse_response(csv_dir: str) -> Optional[pd.DataFrame]:
    """Optional CSV — returns None when absent so the caller can skip."""
    path = os.path.join(csv_dir, "impulse_response.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def save_figure(fig, output_dir: Optional[str], stem: str) -> None:
    """Write `fig` to `{stem}.png` and `{stem}.pdf` in `output_dir`.

    When `output_dir` is None, show the figure interactively instead.
    PDF output is vector and prints cleanly at any size — preferred for
    papers; PNG is the raster fallback for slides and web pages.
    """
    if output_dir is None:
        plt.show()
        return
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def apply_publication_style(use_latex: bool = False) -> None:
    """Opt-in serif/tight rcParams for papers.

    LaTeX is optional because not every environment has a working TeX
    install; without it we still get a clean, paper-ready look via the
    default mathtext engine.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
    })
    if use_latex:
        plt.rcParams["text.usetex"] = True


def type_color_map(types: Iterable[str]) -> dict[str, tuple]:
    """Stable color assignment per arithmetic type.

    Using `tab10` gives 10 distinct colors at the start and then cycles —
    fine for the 5–8 types this sweep typically produces. Sorting the
    types first makes the legend ordering reproducible across runs.
    """
    t_list = list(types)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(t_list), 1)))
    return dict(zip(t_list, colors))


def style_for(atype: str) -> tuple[float, str]:
    """Reference gets a thick solid line; everything else is a thin dashed
    overlay. This reads clearly when many types are plotted together."""
    if atype == REFERENCE_TYPE:
        return 2.5, "-"
    return 1.0, "--"


# ---------------------------------------------------------------------------
# Plot functions. Each is pure in the sense that it just builds a figure —
# saving / showing is centralized in `save_figure`.
# ---------------------------------------------------------------------------


def _per_family_subplots(families):
    fig, axes = plt.subplots(len(families), 1,
                              figsize=(12, 3.5 * len(families)), sharex=True)
    if len(families) == 1:
        axes = [axes]
    return fig, axes


def plot_magnitude_response(df: pd.DataFrame, output_dir: Optional[str]):
    families = df["filter_family"].unique()
    types = df["arith_type"].unique()
    colors = type_color_map(types)
    fig, axes = _per_family_subplots(families)

    for ax, family in zip(axes, families):
        fam_df = df[df["filter_family"] == family]
        for atype in types:
            sub = fam_df[fam_df["arith_type"] == atype]
            if sub.empty:
                continue
            lw, ls = style_for(atype)
            ax.plot(sub["freq_hz"].values, sub["magnitude_db"].values,
                    label=atype, color=colors[atype], linewidth=lw, linestyle=ls)
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(family)
        ax.set_ylim(-80, 5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", ncol=3)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Magnitude Response — mixed-precision IIR comparison",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "magnitude_response")


def plot_phase_response(df: pd.DataFrame, output_dir: Optional[str]):
    families = df["filter_family"].unique()
    types = df["arith_type"].unique()
    colors = type_color_map(types)
    fig, axes = _per_family_subplots(families)

    for ax, family in zip(axes, families):
        fam_df = df[df["filter_family"] == family]
        for atype in types:
            sub = fam_df[fam_df["arith_type"] == atype]
            if sub.empty:
                continue
            lw, ls = style_for(atype)
            ax.plot(sub["freq_hz"].values, sub["phase_deg"].values,
                    label=atype, color=colors[atype], linewidth=lw, linestyle=ls)
        ax.set_ylabel("Phase (degrees)")
        ax.set_title(family)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", ncol=3)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Phase Response — mixed-precision IIR comparison",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "phase_response")


def _plot_error(df: pd.DataFrame, value_col: str, ref_col: str,
                ylabel: str, title: str, stem: str,
                output_dir: Optional[str]):
    """Shared helper for the magnitude/phase error plots.

    Both have the same shape: absolute difference from reference, log y
    so near-zero errors don't dominate the axis, one subplot per family.
    Reference entries are excluded since |ref - ref| = 0 by construction.
    """
    families = df["filter_family"].unique()
    non_ref_types = [t for t in df["arith_type"].unique() if t != REFERENCE_TYPE]
    # Reserve tab10[0] for the reference (not drawn), so errors sit at 1..N.
    raw_colors = plt.cm.tab10(np.linspace(0, 1, len(non_ref_types) + 1))[1:]
    colors = dict(zip(non_ref_types, raw_colors))
    fig, axes = _per_family_subplots(families)

    for ax, family in zip(axes, families):
        fam_df = df[df["filter_family"] == family]
        for atype in non_ref_types:
            sub = fam_df[fam_df["arith_type"] == atype]
            if sub.empty:
                continue
            freq = sub["freq_hz"].values
            # Floor at 1e-15 so log scale doesn't blow up on exact matches.
            err = np.maximum(
                np.abs(sub[value_col].values - sub[ref_col].values), 1e-15)
            ax.semilogy(freq, err, label=atype, color=colors[atype], linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(family)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", ncol=2)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def plot_magnitude_error(df: pd.DataFrame, output_dir: Optional[str]):
    _plot_error(df, "magnitude_db", "ref_magnitude_db",
                ylabel="|Mag error| (dB)",
                title="Magnitude error vs reference (double)",
                stem="magnitude_error", output_dir=output_dir)


def plot_phase_error(df: pd.DataFrame, output_dir: Optional[str]):
    _plot_error(df, "phase_deg", "ref_phase_deg",
                ylabel="|Phase error| (deg)",
                title="Phase error vs reference (double)",
                stem="phase_error", output_dir=output_dir)


def plot_impulse_response(df: pd.DataFrame, output_dir: Optional[str]):
    """Impulse response overlay, first IMPULSE_WINDOW samples per family.

    Reference shows as a thick solid line; other types as thin dashed
    overlays. We clip to IMPULSE_WINDOW even if the CSV has more rows —
    the early samples are where quantization behavior is visible; later
    the reference has usually decayed to near zero and the overlays
    collapse onto each other.
    """
    families = df["filter_family"].unique()
    types = df["arith_type"].unique()
    colors = type_color_map(types)
    fig, axes = _per_family_subplots(families)

    for ax, family in zip(axes, families):
        fam_df = df[df["filter_family"] == family]
        for atype in types:
            sub = fam_df[fam_df["arith_type"] == atype]
            if sub.empty:
                continue
            # Respect either a sample_index column (the documented schema)
            # or, if producers emit sorted rows without an explicit index,
            # just fall back to positional ordering.
            sub = sub.head(IMPULSE_WINDOW)
            if "sample_index" in sub.columns:
                x = sub["sample_index"].values
            else:
                x = np.arange(len(sub))
            lw, ls = style_for(atype)
            ax.plot(x, sub["value"].values, label=atype,
                    color=colors[atype], linewidth=lw, linestyle=ls)
        ax.set_ylabel("Amplitude")
        ax.set_title(family)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=3)

    axes[-1].set_xlabel("Sample index")
    fig.suptitle(f"Impulse response (first {IMPULSE_WINDOW} samples)",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "impulse_response")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Publication-quality mixed-precision filter plots from CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input location: accept either positional `csv_dir` (legacy) or
    # `--input-dir` (spec'd in issue #12). Exactly one must resolve.
    parser.add_argument("csv_dir", nargs="?", default=None,
                        help="Directory with frequency_response.csv (legacy "
                             "positional; prefer --input-dir)")
    parser.add_argument("--input-dir", "-i", default=None,
                        help="Directory with frequency_response.csv "
                             "(impulse_response.csv optional)")
    # Output location: accept both `--output` (legacy) and `--output-dir`
    # (spec'd in issue #12). Omit to show interactively.
    parser.add_argument("--output-dir", "--output", "-o", default=None,
                        dest="output_dir",
                        help="Output directory for figures. Omit for "
                             "interactive display.")
    parser.add_argument("--publication", action="store_true",
                        help="Apply publication rcParams (serif fonts, etc.)")
    parser.add_argument("--latex", action="store_true",
                        help="Use LaTeX to render text (requires TeX install). "
                             "Implies --publication.")
    args = parser.parse_args()

    if args.csv_dir and args.input_dir and args.csv_dir != args.input_dir:
        parser.error("pass either the positional csv_dir OR --input-dir, not both")
    args.input_dir = args.input_dir or args.csv_dir
    if not args.input_dir:
        parser.error("need an input directory: --input-dir DIR (or positional)")
    return args


def main():
    args = _parse_args()
    if args.publication or args.latex:
        apply_publication_style(use_latex=args.latex)

    df = load_frequency_response(args.input_dir)
    print(f"Loaded {len(df)} rows: {df['filter_family'].nunique()} families, "
          f"{df['arith_type'].nunique()} types")

    plot_magnitude_response(df, args.output_dir)
    plot_phase_response(df, args.output_dir)
    plot_magnitude_error(df, args.output_dir)
    plot_phase_error(df, args.output_dir)

    imp = load_impulse_response(args.input_dir)
    if imp is not None:
        print(f"Loaded {len(imp)} impulse-response rows")
        plot_impulse_response(imp, args.output_dir)
    else:
        print("No impulse_response.csv — skipping impulse plot")


if __name__ == "__main__":
    main()
