#!/usr/bin/env python3
"""Pole-zero displacement visualization.

Reads `pole_positions.csv` and produces two figures:

- `pole_zero.{png,pdf}` — unit-circle diagram with reference (double)
  poles as large black × markers, quantized poles as smaller colored
  circles, arrows drawn from reference to quantized location. Legend
  annotates each non-reference arith_type with its maximum pole
  displacement so readers can compare quantization sensitivity across
  types without leaving the legend.

- `pole_displacement.{png,pdf}` — grouped bar chart, max pole
  displacement per (family, type), log-y. Only non-reference types.

Expected input columns:
    filter_family, arith_type, pole_index, real, imag, ref_real,
    ref_imag, displacement

Usage:
    python scripts/plot_pole_zero.py --input-dir results/ --output-dir figures/
    python scripts/plot_pole_zero.py results/ --output-dir figures/     # legacy positional
    python scripts/plot_pole_zero.py results/ --output figures/         # legacy --output
    python scripts/plot_pole_zero.py results/                           # interactive

Optional styling:
    --publication    Serif fonts, tighter margins, larger labels
    --latex          Use LaTeX for text (requires a working TeX install)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REFERENCE_TYPE = "double"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_poles(csv_dir: str) -> pd.DataFrame:
    path = os.path.join(csv_dir, "pole_positions.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def save_figure(fig, output_dir: Optional[str], stem: str) -> None:
    """Write `fig` to `{stem}.png` and `{stem}.pdf` in `output_dir`.

    Mirrors the helper in scripts/plot_precision.py and
    scripts/plot_heatmap.py so all three scripts produce consistent
    PNG + PDF output.
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


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def _draw_unit_circle(ax):
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.4)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_aspect("equal")


def _grid_shape(n_families: int) -> tuple[int, int]:
    """Default to 2×3 (issue #14 spec); fall back to an adaptive grid when
    the fixture or a custom sweep contains a different number of families
    — a rigid 2×3 with only 1-2 filled subplots looks sparse.
    """
    if n_families == 6:
        return 2, 3
    ncols = min(3, max(n_families, 1))
    nrows = (n_families + ncols - 1) // ncols
    return nrows, ncols


def plot_pole_zero(df: pd.DataFrame, output_dir: Optional[str]):
    families = sorted(df["filter_family"].unique())
    types = sorted(df["arith_type"].unique())
    non_ref_types = [t for t in types if t != REFERENCE_TYPE]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(non_ref_types), 1)))
    type_colors = dict(zip(non_ref_types, colors))

    nrows, ncols = _grid_shape(len(families))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    # Normalize to a flat array regardless of rows/cols — handles the
    # single-family test fixture and the 2×3 production case uniformly.
    axes = np.atleast_1d(axes).flatten()

    for idx, family in enumerate(families):
        ax = axes[idx]
        _draw_unit_circle(ax)
        fam_df = df[df["filter_family"] == family]

        # Reference poles — large black × markers per issue spec.
        ref = fam_df[fam_df["arith_type"] == REFERENCE_TYPE]
        if not ref.empty:
            ax.plot(ref["ref_real"], ref["ref_imag"], "kx",
                    markersize=10, markeredgewidth=2.5, zorder=10,
                    label=f"{REFERENCE_TYPE} (ref)")

        # Non-reference types — colored circles + arrow from ref to quantized.
        for atype in non_ref_types:
            sub = fam_df[fam_df["arith_type"] == atype]
            if sub.empty:
                continue
            color = type_colors[atype]

            # Legend entry includes Δmax — the quantization-sensitivity
            # number the issue asks to surface. Formatted in scientific
            # notation because displacements span many orders of magnitude
            # (1e-9 for posit<32,2>, 1e-3 for tiny_posit).
            dmax = sub["displacement"].max()
            label = f"{atype}  (Δmax={dmax:.1e})"

            ax.plot(sub["real"], sub["imag"], "o",
                    markersize=7, markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=1.5,
                    label=label)

            # Draw displacement vectors only where visible — sub-1e-10
            # displacements produce arrow heads that overlap the markers.
            for _, row in sub.iterrows():
                if row["displacement"] > 1e-10:
                    ax.annotate("", xy=(row["real"], row["imag"]),
                                 xytext=(row["ref_real"], row["ref_imag"]),
                                 arrowprops=dict(arrowstyle="->", color=color,
                                                  lw=0.8, alpha=0.6))

        ax.set_title(family, fontweight="bold")
        ax.legend(loc="lower left", ncol=1, fontsize=7)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")

        all_real = fam_df["real"].tolist() + fam_df["ref_real"].tolist()
        all_imag = fam_df["imag"].tolist() + fam_df["ref_imag"].tolist()
        margin = 0.15
        ax.set_xlim(min(all_real) - margin, max(all_real) + margin)
        ax.set_ylim(min(all_imag) - margin, max(all_imag) + margin)

    # Hide unused subplots in the 2×3 grid when <6 families were provided.
    for idx in range(len(families), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Pole positions: reference vs mixed-precision",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "pole_zero")


def plot_displacement_summary(df: pd.DataFrame, output_dir: Optional[str]):
    """Max pole displacement per (family, type), log-y bar chart."""
    df_no_ref = df[df["arith_type"] != REFERENCE_TYPE]
    summary = (df_no_ref
               .groupby(["filter_family", "arith_type"])["displacement"]
               .max().reset_index())
    # Only rows with visible displacement — a fixture with all CoeffScalar=double
    # has no displacement anywhere, in which case we emit a hint and a blank
    # figure rather than crashing downstream.
    summary = summary[summary["displacement"] > 1e-15]

    if summary.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5,
                 "No pole displacement detected.\n"
                 "This happens when all arithmetic types share CoeffScalar=double;\n"
                 "displacement becomes visible when coefficients are projected\n"
                 "to narrower types via project_onto<T>().",
                 ha="center", va="center", transform=ax.transAxes,
                 fontsize=10)
        ax.set_axis_off()
        fig.tight_layout()
        save_figure(fig, output_dir, "pole_displacement")
        return

    pivot = summary.pivot_table(index="filter_family",
                                 columns="arith_type",
                                 values="displacement")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Max pole displacement (unitless)")
    ax.set_title("Pole displacement from reference (double)",
                  fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_dir, "pole_displacement")


# ---------------------------------------------------------------------------
# CLI — same shape as scripts/plot_precision.py and scripts/plot_heatmap.py.
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Pole-zero displacement visualization from CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv_dir", nargs="?", default=None,
                        help="Directory with pole_positions.csv "
                             "(legacy positional; prefer --input-dir)")
    parser.add_argument("--input-dir", "-i", default=None,
                        help="Directory with pole_positions.csv")
    parser.add_argument("--output-dir", "--output", "-o", default=None,
                        dest="output_dir",
                        help="Output directory for figures. Omit for "
                             "interactive display.")
    parser.add_argument("--publication", action="store_true",
                        help="Apply publication rcParams (serif fonts, etc.)")
    parser.add_argument("--latex", action="store_true",
                        help="Use LaTeX to render text (requires TeX). "
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

    df = load_poles(args.input_dir)
    print(f"Loaded {len(df)} pole entries: "
          f"{df['filter_family'].nunique()} families, "
          f"{df['arith_type'].nunique()} types")

    plot_pole_zero(df, args.output_dir)
    plot_displacement_summary(df, args.output_dir)


if __name__ == "__main__":
    main()
