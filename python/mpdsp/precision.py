"""Precision-profile reductions: reading a number system against a DSP need.

These are the summaries Universal's
``applications/mixed-precision/dsp/precision_profiles.cpp`` prints for its
decimals-of-accuracy tutorial, ported so the tables regenerate from
Python. They are reductions over a bound ``mpdsp.PrecisionProfile``, not
DSP algorithms; the profile itself (every encoding's ulp) comes from
Universal's ``precision_profile_of<T>()`` through
``mpdsp.precision_profile()``.

The region of interest of a DSP pipeline defaults to ``[2^-15, 2^12]``:
small filter coefficients and spectral tails at about -90 dBFS, up to the
log2(N) growth of a 4096-point FFT.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

ROI_LOG2_LO = -15
ROI_LOG2_HI = 12

# 6.02 dB of SQNR per fraction bit
_DB_PER_BIT = 20.0 * math.log10(2.0)


def log2_range(profile) -> tuple[int, int]:
    """``(log2 minpos, log2 maxpos)``, rounded to whole binades."""
    return (int(round(math.log2(profile.minpos))),
            int(round(math.log2(profile.maxpos))))


def share_in_roi(profile, lo: int = ROI_LOG2_LO,
                 hi: int = ROI_LOG2_HI) -> float | None:
    """Fraction of the positive encodings whose magnitude is in [2^lo, 2^hi].

    Defined for exhaustive profiles only: a sampled profile does not hold
    one point per encoding, so it returns ``None``.
    """
    if not profile.exhaustive or len(profile) == 0:
        return None
    m = profile.magnitude
    inside = np.count_nonzero((m >= math.ldexp(1.0, lo))
                              & (m <= math.ldexp(1.0, hi)))
    return inside / m.size


def min_decimals_over(profile, lo: int = ROI_LOG2_LO,
                      hi: int = ROI_LOG2_HI) -> float:
    """Worst decimals of accuracy at the binade boundaries 2^lo .. 2^hi.

    A type that does not cover the whole region scores 0.
    """
    return min(profile.decimals_at(math.ldexp(1.0, k))
               for k in range(lo, hi + 1))


def min_fraction_bits(profile) -> float:
    """The precision floor: the fewest fraction bits at any magnitude."""
    return float(profile.fraction_bits.min())


def sqnr_db(decimals: float) -> float:
    """SQNR implied by ``decimals`` at the start of a binade.

    Decimals d at the start of a binade correspond to ``d / log10(2) - 1``
    fraction bits, at 6.02 dB each. 0 decimals (not representable) is 0 dB.
    """
    if decimals <= 0.0:
        return 0.0
    return _DB_PER_BIT * (decimals / math.log10(2.0) - 1.0)


def summary_table(profiles: Iterable) -> list[dict]:
    """The numbers a DSP designer reads off the curves, one row per type.

    Columns: ``label``, ``range`` (log2 minpos, maxpos), ``band`` (decimals
    in [0.5, 1)), ``sqnr_db``, ``at_floor`` (decimals at 2^-15, -90 dBFS),
    ``at_fft_1024`` (decimals at 2^10) and ``floor_bits``.
    """
    rows = []
    for p in profiles:
        band = p.decimals_at(0.5)
        rows.append({
            "label": p.label,
            "range": log2_range(p),
            "band": band,
            "sqnr_db": sqnr_db(band),
            "at_floor": p.decimals_at(math.ldexp(1.0, -15)),
            "at_fft_1024": p.decimals_at(1024.0),
            "floor_bits": min_fraction_bits(p),
        })
    return rows


def range_fit_table(profiles: Iterable, lo: int = ROI_LOG2_LO,
                    hi: int = ROI_LOG2_HI) -> list[dict]:
    """How well each type fits the region of interest [2^lo, 2^hi].

    Columns: ``label``, ``range``, ``share`` (fraction of encodings in the
    region, ``None`` for sampled profiles), ``band`` (decimals in
    [0.5, 1)), ``worst`` (worst decimals in the region) and
    ``floor_bits``.
    """
    return [{
        "label": p.label,
        "range": log2_range(p),
        "share": share_in_roi(p, lo, hi),
        "band": p.decimals_at(0.5),
        "worst": min_decimals_over(p, lo, hi),
        "floor_bits": min_fraction_bits(p),
    } for p in profiles]


def format_markdown_table(
        rows: Sequence[dict],
        columns: Sequence[tuple[str, str, Callable[[object], str]]]) -> str:
    """Render ``rows`` as a Markdown table.

    ``columns`` is a sequence of ``(header, key, formatter)``; the
    formatter turns the row's value at ``key`` into a cell string.
    """
    lines = ["| " + " | ".join(h for h, _, _ in columns) + " |",
             "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key])
                                       for _, key, fmt in columns) + " |")
    return "\n".join(lines)


def fmt_label(label: str) -> str:
    """A type name as a table cell, following the tutorial's convention.

    C++ spellings (``bposit<16,4,2>``, ``float``, ``int16``) are set as
    inline code; the informal names fp16, Q15 and Q31 are not.
    """
    return label if label in ("fp16", "Q15", "Q31") else f"`{label}`"


def fmt_range(r: tuple[int, int]) -> str:
    """A log2 range as ``-192 .. 192``."""
    return f"{r[0]} .. {r[1]}"


def fmt_share(share: float | None) -> str:
    """An encoding share as a percentage; sampled profiles have none."""
    return "(sampled)" if share is None else f"{100.0 * share:.1f} %"


def fmt_decimals(d: float) -> str:
    return f"{d:.2f}"


def fmt_bits(b: float) -> str:
    return f"{b:.0f}"
