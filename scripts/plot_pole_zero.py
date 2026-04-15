#!/usr/bin/env python3
"""Plot pole-zero diagrams from pole_positions.csv.

Unit circle with reference poles (filled) and displaced poles (open markers).
One subplot per filter family. Displacement vectors drawn between positions.

Usage:
    python scripts/plot_pole_zero.py /path/to/csv_directory
    python scripts/plot_pole_zero.py /path/to/csv_directory --output figures/
"""

import argparse
from typing import Optional
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_poles(csv_dir: str) -> pd.DataFrame:
    path = os.path.join(csv_dir, "pole_positions.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def draw_unit_circle(ax):
    """Draw the unit circle and axes."""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.4)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_aspect("equal")


def plot_pole_zero(df: pd.DataFrame, output_dir: Optional[str] = None):
    """Pole-zero diagrams: one subplot per filter family."""
    families = df["filter_family"].unique()
    types = df["arith_type"].unique()
    non_ref_types = [t for t in types if t != "double"]

    colors = plt.cm.tab10(np.linspace(0, 1, len(non_ref_types)))
    type_colors = dict(zip(non_ref_types, colors))

    ncols = min(3, len(families))
    nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if len(families) == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, family in enumerate(families):
        ax = axes[idx]
        draw_unit_circle(ax)

        fam_df = df[df["filter_family"] == family]

        # Plot reference poles (filled black 'x')
        ref = fam_df[fam_df["arith_type"] == "double"]
        if not ref.empty:
            ax.plot(ref["ref_real"], ref["ref_imag"], "kx",
                    markersize=10, markeredgewidth=2.5, label="double (ref)",
                    zorder=10)

        # Plot displaced poles for each type
        for atype in non_ref_types:
            sub = fam_df[fam_df["arith_type"] == atype]
            if sub.empty:
                continue

            color = type_colors[atype]
            ax.plot(sub["real"], sub["imag"], "o",
                    markersize=7, markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=1.5,
                    label=atype)

            # Draw displacement vectors (if displacement > threshold)
            for _, row in sub.iterrows():
                if row["displacement"] > 1e-10:
                    ax.annotate("", xy=(row["real"], row["imag"]),
                                xytext=(row["ref_real"], row["ref_imag"]),
                                arrowprops=dict(arrowstyle="->", color=color,
                                                lw=0.8, alpha=0.6))

        ax.set_title(family, fontsize=11, fontweight="bold")
        ax.legend(fontsize=6, loc="lower left", ncol=2)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")

        # Zoom to pole region
        all_real = fam_df["real"].tolist() + fam_df["ref_real"].tolist()
        all_imag = fam_df["imag"].tolist() + fam_df["ref_imag"].tolist()
        margin = 0.15
        ax.set_xlim(min(all_real) - margin, max(all_real) + margin)
        ax.set_ylim(min(all_imag) - margin, max(all_imag) + margin)

    # Hide unused subplots
    for idx in range(len(families), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Pole Positions: Reference vs Mixed-Precision",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "pole_zero.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()


def plot_displacement_summary(df: pd.DataFrame, output_dir: Optional[str] = None):
    """Bar chart of maximum pole displacement per filter family × type."""
    df_no_ref = df[df["arith_type"] != "double"]
    summary = df_no_ref.groupby(["filter_family", "arith_type"])["displacement"].max()
    summary = summary.reset_index()

    # Only show types with non-zero displacement
    summary = summary[summary["displacement"] > 1e-15]

    if summary.empty:
        print("  No pole displacement detected (all CoeffScalar=double)")
        print("  Displacement becomes visible when coefficients are projected")
        print("  to narrower types via project_onto<T>()")
        return

    pivot = summary.pivot_table(index="filter_family", columns="arith_type",
                                values="displacement")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Max Pole Displacement")
    plt.title("Pole Displacement from Reference (double)",
              fontsize=13, fontweight="bold")
    plt.yscale("log")
    plt.legend(fontsize=8)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "pole_displacement.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot pole-zero diagrams from precision sweep CSV")
    parser.add_argument("csv_dir", help="Directory containing pole_positions.csv")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for PNG files (omit for interactive)")
    args = parser.parse_args()

    df = load_poles(args.csv_dir)
    print(f"Loaded {len(df)} pole entries: {df['filter_family'].nunique()} families, "
          f"{df['arith_type'].nunique()} types")

    plot_pole_zero(df, args.output)
    plot_displacement_summary(df, args.output)


if __name__ == "__main__":
    main()
