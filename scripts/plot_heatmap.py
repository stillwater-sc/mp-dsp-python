#!/usr/bin/env python3
"""Plot SQNR heatmap, bar chart, and precision-cost frontier from iir_precision_sweep.csv.

Usage:
    python scripts/plot_heatmap.py /path/to/csv_directory
    python scripts/plot_heatmap.py /path/to/csv_directory --output figures/
"""

import argparse
from typing import Optional
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metrics(csv_dir: str) -> pd.DataFrame:
    path = os.path.join(csv_dir, "iir_precision_sweep.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def plot_sqnr_heatmap(df: pd.DataFrame, output_dir: Optional[str] = None):
    """SQNR heatmap: filter families (rows) x arithmetic types (columns)."""
    # Pivot to matrix form
    pivot = df.pivot_table(index="filter_family", columns="arith_type",
                           values="sqnr_db", aggfunc="first")

    # Order columns by bits (descending) for visual clarity
    bits_map = df.drop_duplicates("arith_type").set_index("arith_type")["bits"]
    col_order = bits_map.sort_values(ascending=False).index
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]

    # Cap at 200 dB for display (double is "inf" at 300)
    display = pivot.copy()
    display = display.clip(upper=200)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(display.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=200)

    # Labels
    ax.set_xticks(range(len(display.columns)))
    ax.set_xticklabels(display.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(display.index)))
    ax.set_yticklabels(display.index, fontsize=10)

    # Annotate cells with SQNR values
    for i in range(len(display.index)):
        for j in range(len(display.columns)):
            val = pivot.iloc[i, j]
            text = "inf" if val >= 290 else f"{val:.0f}"
            color = "white" if display.iloc[i, j] < 60 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_title("SQNR (dB): Filter Family x Arithmetic Type",
                 fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="SQNR (dB)", shrink=0.8)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "sqnr_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()


def plot_sqnr_bar_chart(df: pd.DataFrame, output_dir: Optional[str] = None):
    """Grouped bar chart: SQNR per filter family, bars per arithmetic type."""
    # Exclude double (inf SQNR dominates the scale)
    df_no_ref = df[df["arith_type"] != "double"].copy()
    df_no_ref["sqnr_db"] = df_no_ref["sqnr_db"].clip(upper=200)

    families = df_no_ref["filter_family"].unique()
    types = df_no_ref["arith_type"].unique()
    n_families = len(families)
    n_types = len(types)

    colors = plt.cm.tab10(np.linspace(0, 1, n_types))
    x = np.arange(n_families)
    width = 0.8 / n_types

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
    ax.set_xticklabels(families, fontsize=10)
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("SQNR by Filter Family and Arithmetic Type (reference=double excluded)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "sqnr_bar_chart.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()


def plot_precision_cost_frontier(df: pd.DataFrame, output_dir: Optional[str] = None):
    """SQNR vs bits-per-sample across all filter families.

    Each point is one (filter_family, arith_type) cell. This reveals
    which types are Pareto-optimal: highest SQNR for a given bit width.
    """
    df_no_ref = df[df["arith_type"] != "double"].copy()

    families = df_no_ref["filter_family"].unique()
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    colors = plt.cm.Set1(np.linspace(0, 1, len(families)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, fam in enumerate(families):
        sub = df_no_ref[df_no_ref["filter_family"] == fam]
        ax.scatter(sub["bits"], sub["sqnr_db"],
                   label=fam, marker=markers[i % len(markers)],
                   color=colors[i], s=80, edgecolors="black", linewidth=0.5)
        # Label each point with the type name
        for _, row in sub.iterrows():
            ax.annotate(row["arith_type"], (row["bits"], row["sqnr_db"]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Bits per sample")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("Precision-Cost Frontier: SQNR vs Bit Width",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "precision_cost_frontier.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot SQNR heatmap and bar chart from precision sweep CSV")
    parser.add_argument("csv_dir", help="Directory containing iir_precision_sweep.csv")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for PNG files (omit for interactive)")
    args = parser.parse_args()

    df = load_metrics(args.csv_dir)
    print(f"Loaded {len(df)} rows: {df['filter_family'].nunique()} families, "
          f"{df['arith_type'].nunique()} types")

    plot_sqnr_heatmap(df, args.output)
    plot_sqnr_bar_chart(df, args.output)
    plot_precision_cost_frontier(df, args.output)


if __name__ == "__main__":
    main()
