#!/usr/bin/env python3
"""Reproduce Universal's decimals-of-accuracy tutorial in one run.

Universal's tutorial (docs/tutorials/decimals-of-accuracy.md) builds a C++
application, runs it to write one CSV per number system, and calls a
plotting script once per figure. Here the profiles come straight from
`mpdsp.precision_profile()`, so one command regenerates every figure and
table on `docs/decimals_of_accuracy.md`:

    16bit-full-range          int16, Q15, fp16, lns<16,10>, bposit<16,6,5>
    16bit-region-of-interest  the same, zoomed to [2^-32, 2^24] with DSP markers
    16bit-bposit-fit          Q15, fp16, bposit<16,6,5>, <16,5,2>, <16,4,2>
    32bit-bposit-fit          Q31, float, bposit<32,6,5>, <32,4,2>

and prints the encoding-share and range-fit tables as Markdown.

Usage:
    python scripts/plot_decimals_of_accuracy.py --output-dir figures/
    python scripts/plot_decimals_of_accuracy.py --output-dir docs/img/precision-profiles --format png
    python scripts/plot_decimals_of_accuracy.py --tables-only

Optional styling:
    --publication    Serif fonts, tighter margins, larger labels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FIGURES = [
    # (file stem, types, dsp markers, xmin, xmax, title)
    ("16bit-full-range",
     ["int16", "Q15", "fp16", "lns<16,10>", "bposit<16,6,5>"],
     False, None, None,
     "16-bit number systems: decimals of accuracy across the full range"),
    ("16bit-region-of-interest",
     ["int16", "Q15", "fp16", "lns<16,10>", "bposit<16,6,5>"],
     True, -32, 24,
     "16-bit number systems: the region of interest"),
    ("16bit-bposit-fit",
     ["Q15", "fp16", "bposit<16,6,5>", "bposit<16,5,2>", "bposit<16,4,2>"],
     True, -32, 24,
     "Fitting the b-posit range to the region of interest"),
    ("32bit-bposit-fit",
     ["Q31", "float", "bposit<32,6,5>", "bposit<32,4,2>"],
     True, -40, 24,
     "32-bit: the same fit"),
]

SHARE_TYPES = ["bposit<16,6,5>", "fp16", "lns<16,10>"]
FIT16_TYPES = ["bposit<16,6,5>", "bposit<16,6,3>", "bposit<16,6,2>",
               "bposit<16,5,2>", "bposit<16,4,2>", "bposit<16,3,1>"]
FIT32_TYPES = ["float", "bposit<32,6,5>", "bposit<32,5,2>", "bposit<32,4,2>"]


def apply_publication_style() -> None:
    """Opt-in serif/tight rcParams for papers (matches plot_precision.py)."""
    import matplotlib.pyplot as plt
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


def tables(mpdsp) -> str:
    """The tutorial's three tables, as Markdown."""
    from mpdsp import precision as pr

    def profiles(names):
        return [mpdsp.precision_profile(n) for n in names]

    share = pr.format_markdown_table(
        pr.range_fit_table(profiles(SHARE_TYPES)),
        [("type", "label", pr.fmt_label),
         ("range (log2)", "range", pr.fmt_range),
         ("positive encodings in [2^-15, 2^12]", "share", pr.fmt_share)])
    fit16 = pr.format_markdown_table(
        pr.range_fit_table(profiles(FIT16_TYPES)),
        [("type", "label", pr.fmt_label),
         ("range (log2)", "range", pr.fmt_range),
         ("encodings in the region", "share", pr.fmt_share),
         ("decimals in [0.5, 1)", "band", pr.fmt_decimals),
         ("worst in the region", "worst", pr.fmt_decimals),
         ("floor (bits)", "floor_bits", pr.fmt_bits)])
    fit32 = pr.format_markdown_table(
        pr.range_fit_table(profiles(FIT32_TYPES)),
        [("type", "label", pr.fmt_label),
         ("range (log2)", "range", pr.fmt_range),
         ("decimals in [0.5, 1)", "band", pr.fmt_decimals),
         ("worst in the region", "worst", pr.fmt_decimals),
         ("floor (bits)", "floor_bits", pr.fmt_bits)])
    return ("Encodings in the region of interest\n\n" + share +
            "\n\nFitting the range (16-bit)\n\n" + fit16 +
            "\n\nThe same at 32 bits\n\n" + fit32 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce Universal's decimals-of-accuracy tutorial.")
    parser.add_argument("--output-dir", "-o", default="figures",
                        help="directory for the figures (default: figures/)")
    parser.add_argument("--format", nargs="+", default=["png", "pdf"],
                        choices=["png", "pdf", "svg"],
                        help="output formats (default: png pdf)")
    parser.add_argument("--publication", action="store_true",
                        help="Apply publication rcParams (serif fonts, etc.)")
    parser.add_argument("--tables-only", action="store_true",
                        help="print the tables and skip the figures")
    args = parser.parse_args()

    try:
        import mpdsp
    except ImportError:
        sys.exit("plot_decimals_of_accuracy.py needs mpdsp: pip install -e .")
    if not mpdsp.HAS_CORE:
        sys.exit(f"mpdsp._core is not available: "
                 f"{mpdsp.__core_import_error__}")

    print(tables(mpdsp))
    if args.tables_only:
        return 0

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("plot_decimals_of_accuracy.py needs matplotlib: "
                 "pip install matplotlib")
    from mpdsp.plotting import plot_precision_profiles

    if args.publication:
        apply_publication_style()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stem, names, dsp, xmin, xmax, title in FIGURES:
        fig, ax = plt.subplots(figsize=(11, 6))
        plot_precision_profiles(
            [mpdsp.precision_profile(n) for n in names],
            dsp=dsp, xmin=xmin, xmax=xmax, title=title, ax=ax)
        fig.tight_layout()
        for ext in args.format:
            path = out / f"{stem}.{ext}"
            fig.savefig(path)
            print(f"wrote {path}")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
