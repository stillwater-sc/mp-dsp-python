#!/usr/bin/env python3
"""Summary visualizations: error + SQNR heatmaps, bar chart, precision-cost frontier.

Reads `iir_precision_sweep.csv` and produces four publication-quality figures
giving the full mixed-precision comparison matrix at a glance.

Expected input columns:
    filter_family, arith_type, bits, max_abs_error, max_rel_error, sqnr_db,
    pole_displacement, stability_margin

The issue spec (#13) mentions `sqnr.csv` and `impulse_errors.csv` as two
separate files; in practice `iir_precision_sweep.csv` already carries both
the SQNR column and the `max_abs_error` column, and that's what upstream
`iir_precision_sweep` emits. Splitting would add ceremony without adding
information, so we read the single file directly.

Outputs saved to `--output-dir` (both PNG + PDF):
    error_heatmap.{png,pdf}         — new in #13: log10(max_abs_error)
                                     per (family, type), seaborn annot.
    sqnr_heatmap.{png,pdf}          — seaborn heatmap, SQNR (dB) capped
                                     at 200 for display (double → inf).
    sqnr_bar_chart.{png,pdf}        — grouped bars per family, one bar
                                     per non-reference arithmetic type.
    precision_cost_frontier.{png,pdf} — SQNR vs bits scatter per family,
                                       each point labeled with its type.

Usage:
    python scripts/plot_heatmap.py --input-dir results/ --output-dir figures/
    python scripts/plot_heatmap.py results/ --output-dir figures/     # legacy positional
    python scripts/plot_heatmap.py results/ --output figures/         # legacy --output
    python scripts/plot_heatmap.py results/                           # interactive

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
import seaborn as sns


REFERENCE_TYPE = "double"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_metrics(csv_dir: str) -> pd.DataFrame:
    path = os.path.join(csv_dir, "iir_precision_sweep.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def save_figure(fig, output_dir: Optional[str], stem: str) -> None:
    """Write `fig` to `{stem}.png` and `{stem}.pdf` in `output_dir`.

    Mirrors the pattern in scripts/plot_precision.py so both scripts
    produce consistent PNG + PDF output for paper use.
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
    """Opt-in serif/tight rcParams for papers."""
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
# Pivot helpers — both heatmaps need the same (family × type) matrix shape.
# ---------------------------------------------------------------------------


def _pivot_by_family_type(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot long-format metrics into a (filter_family × arith_type) matrix.

    Column order is sorted by `bits` descending — widest types leftmost,
    narrowest rightmost — so the visual "precision cliff" reads left-to-
    right, which matches how most readers scan a heatmap.
    """
    pivot = df.pivot_table(index="filter_family", columns="arith_type",
                            values=value_col, aggfunc="first")
    # Reorder columns by bit width (wide → narrow).
    bits_map = df.drop_duplicates("arith_type").set_index("arith_type")["bits"]
    col_order = [c for c in bits_map.sort_values(ascending=False).index
                  if c in pivot.columns]
    return pivot[col_order]


# ---------------------------------------------------------------------------
# Plot functions — seaborn for the two heatmaps (issue #13 requirement),
# matplotlib for the bar chart and scatter.
# ---------------------------------------------------------------------------


def plot_error_heatmap(df: pd.DataFrame, output_dir: Optional[str]):
    """log10(max_abs_error) per (filter_family, arith_type), annotated.

    Uses a log-error color scale because precision types span many orders
    of magnitude (double hits 0, tiny_posit hits 1e-3). Cells are annotated
    with the actual error values in scientific notation — log-scale color
    shows the ordering; annotations give the magnitude.
    """
    pivot = _pivot_by_family_type(df, "max_abs_error")

    # Floor errors at 1e-16 so log10(0) doesn't blow up (double is exactly
    # zero error against itself by construction).
    log_matrix = np.log10(pivot.to_numpy(dtype=float).clip(min=1e-16))

    # Format each cell as its raw error; the color conveys the log scale.
    annot = pivot.map(lambda v: "0" if v == 0 else f"{v:.1e}").to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(log_matrix, annot=annot, fmt="", cmap="RdYlGn_r",
                 xticklabels=pivot.columns, yticklabels=pivot.index,
                 cbar_kws={"label": "log10(max |error|)"}, ax=ax,
                 linewidths=0.5, linecolor="white")
    ax.set_title("Max absolute error — filter family × arithmetic type",
                 fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_figure(fig, output_dir, "error_heatmap")


def plot_sqnr_heatmap(df: pd.DataFrame, output_dir: Optional[str]):
    """SQNR (dB) per (filter_family, arith_type), capped at 200 for display.

    `double` returns `inf` SQNR against itself; we display it as "inf" in
    the annotation but cap the color scale at 200 so the color range is
    meaningful for the non-reference types. Matches the existing `plot_sqnr_heatmap`
    behavior, now routed through seaborn for consistency with `plot_error_heatmap`.
    """
    pivot = _pivot_by_family_type(df, "sqnr_db")
    display = pivot.clip(upper=200)

    annot = pivot.map(
        lambda v: "inf" if v >= 290 else f"{v:.0f}").to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(display, annot=annot, fmt="", cmap="RdYlGn",
                 vmin=0, vmax=200, ax=ax,
                 cbar_kws={"label": "SQNR (dB), capped at 200"},
                 linewidths=0.5, linecolor="white")
    ax.set_title("SQNR (dB) — filter family × arithmetic type",
                 fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_figure(fig, output_dir, "sqnr_heatmap")


def plot_sqnr_bar_chart(df: pd.DataFrame, output_dir: Optional[str]):
    """Grouped bars — SQNR per family, bar per non-reference arithmetic type.

    The reference type (`double`) has infinite SQNR against itself by
    definition and would dominate the chart, so it's filtered out. Other
    dtypes get clipped at 200 dB to keep the bar heights legible.
    """
    df_no_ref = df[df["arith_type"] != REFERENCE_TYPE].copy()
    df_no_ref["sqnr_db"] = df_no_ref["sqnr_db"].clip(upper=200)

    families = sorted(df_no_ref["filter_family"].unique())
    types = sorted(df_no_ref["arith_type"].unique())
    n_types = len(types)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_types, 1)))
    x = np.arange(len(families))
    width = 0.8 / max(n_types, 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, atype in enumerate(types):
        vals = []
        for fam in families:
            row = df_no_ref[(df_no_ref["filter_family"] == fam) &
                             (df_no_ref["arith_type"] == atype)]
            vals.append(row["sqnr_db"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=atype, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("SQNR (dB), capped at 200")
    ax.set_title("SQNR by filter family and arithmetic type "
                  f"(reference={REFERENCE_TYPE} excluded)", fontweight="bold")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir, "sqnr_bar_chart")


def plot_precision_cost_frontier(df: pd.DataFrame, output_dir: Optional[str]):
    """SQNR vs bits/sample, one series per filter family.

    Reveals the Pareto frontier: for a given bit budget, which
    arithmetic type + family combination gives the highest SQNR?
    """
    df_no_ref = df[df["arith_type"] != REFERENCE_TYPE].copy()

    families = sorted(df_no_ref["filter_family"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(families), 1)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, fam in enumerate(families):
        sub = df_no_ref[df_no_ref["filter_family"] == fam]
        ax.scatter(sub["bits"], sub["sqnr_db"],
                    label=fam, marker=markers[i % len(markers)],
                    color=colors[i], s=80, edgecolors="black", linewidth=0.5)
        for _, row in sub.iterrows():
            ax.annotate(row["arith_type"], (row["bits"], row["sqnr_db"]),
                         fontsize=6, ha="left", va="bottom",
                         xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Bits per sample")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("Precision-cost frontier: SQNR vs bit width",
                  fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir, "precision_cost_frontier")


# ---------------------------------------------------------------------------
# CLI — same shape as scripts/plot_precision.py so the two feel consistent.
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Publication-quality summary visualizations from CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv_dir", nargs="?", default=None,
                        help="Directory with iir_precision_sweep.csv "
                             "(legacy positional; prefer --input-dir)")
    parser.add_argument("--input-dir", "-i", default=None,
                        help="Directory with iir_precision_sweep.csv")
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

    df = load_metrics(args.input_dir)
    print(f"Loaded {len(df)} rows: {df['filter_family'].nunique()} families, "
          f"{df['arith_type'].nunique()} types")

    plot_error_heatmap(df, args.output_dir)
    plot_sqnr_heatmap(df, args.output_dir)
    plot_sqnr_bar_chart(df, args.output_dir)
    plot_precision_cost_frontier(df, args.output_dir)


if __name__ == "__main__":
    main()
