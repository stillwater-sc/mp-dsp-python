#!/usr/bin/env python3
"""Plot magnitude and phase response overlays from frequency_response.csv.

One subplot per filter family, all arithmetic types overlaid.

Usage:
    python scripts/plot_precision.py /path/to/csv_directory
    python scripts/plot_precision.py /path/to/csv_directory --output figures/
"""

import argparse
from typing import Optional
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_frequency_response(csv_dir: str) -> pd.DataFrame:
    path = os.path.join(csv_dir, "frequency_response.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def plot_magnitude_response(df: pd.DataFrame, output_dir: Optional[str] = None):
    """Magnitude response: one subplot per filter family."""
    families = df["filter_family"].unique()
    types = df["arith_type"].unique()

    # Color map for arithmetic types
    colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
    type_colors = dict(zip(types, colors))

    fig, axes = plt.subplots(len(families), 1, figsize=(12, 3.5 * len(families)),
                             sharex=True)
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        fam_df = df[df["filter_family"] == family]
        for atype in types:
            sub = fam_df[fam_df["arith_type"] == atype]
            if sub.empty:
                continue
            freq = sub["freq_hz"].values
            mag = sub["magnitude_db"].values
            linewidth = 2.0 if atype == "double" else 1.0
            linestyle = "-" if atype == "double" else "--"
            ax.plot(freq, mag, label=atype, color=type_colors[atype],
                    linewidth=linewidth, linestyle=linestyle)

        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(family)
        ax.set_ylim(-80, 5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower left", ncol=3)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Magnitude Response: Mixed-Precision IIR Filter Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "magnitude_response.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()


def plot_phase_response(df: pd.DataFrame, output_dir: Optional[str] = None):
    """Phase response: one subplot per filter family."""
    families = df["filter_family"].unique()
    types = df["arith_type"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
    type_colors = dict(zip(types, colors))

    fig, axes = plt.subplots(len(families), 1, figsize=(12, 3.5 * len(families)),
                             sharex=True)
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        fam_df = df[df["filter_family"] == family]
        for atype in types:
            sub = fam_df[fam_df["arith_type"] == atype]
            if sub.empty:
                continue
            freq = sub["freq_hz"].values
            phase = sub["phase_deg"].values
            linewidth = 2.0 if atype == "double" else 1.0
            linestyle = "-" if atype == "double" else "--"
            ax.plot(freq, phase, label=atype, color=type_colors[atype],
                    linewidth=linewidth, linestyle=linestyle)

        ax.set_ylabel("Phase (degrees)")
        ax.set_title(family)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower left", ncol=3)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Phase Response: Mixed-Precision IIR Filter Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "phase_response.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()


def plot_magnitude_error(df: pd.DataFrame, output_dir: Optional[str] = None):
    """Magnitude error vs reference: one subplot per filter family."""
    families = df["filter_family"].unique()
    types = [t for t in df["arith_type"].unique() if t != "double"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(types) + 1))[1:]
    type_colors = dict(zip(types, colors))

    fig, axes = plt.subplots(len(families), 1, figsize=(12, 3.5 * len(families)),
                             sharex=True)
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        fam_df = df[df["filter_family"] == family]
        for atype in types:
            sub = fam_df[fam_df["arith_type"] == atype]
            if sub.empty:
                continue
            freq = sub["freq_hz"].values
            error = np.abs(sub["magnitude_db"].values - sub["ref_magnitude_db"].values)
            # Avoid log(0)
            error = np.maximum(error, 1e-15)
            ax.semilogy(freq, error, label=atype, color=type_colors[atype],
                        linewidth=1.0)

        ax.set_ylabel("| Mag Error | (dB)")
        ax.set_title(family)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Magnitude Error vs Reference (double)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "magnitude_error.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot magnitude/phase response from precision sweep CSV")
    parser.add_argument("csv_dir", help="Directory containing frequency_response.csv")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for PNG files (omit for interactive)")
    args = parser.parse_args()

    df = load_frequency_response(args.csv_dir)
    print(f"Loaded {len(df)} rows: {df['filter_family'].nunique()} families, "
          f"{df['arith_type'].nunique()} types")

    plot_magnitude_response(df, args.output)
    plot_phase_response(df, args.output)
    plot_magnitude_error(df, args.output)


if __name__ == "__main__":
    main()
