"""Pick the FPGA coefficient width for the DDC cascade.

    python demos/04_sdr_ddc_cascade/quantize_coefficients.py

Two sweeps, answering two different questions:

**How many bits does the coefficient ROM need?** Taps are quantized to a
signed fixed-point word of N bits and the composite response re-measured.
The answer is the smallest N whose cascade still clears the spec — the
number that goes into the Vivado FIR Compiler and sizes the BRAM.

**Which arithmetic type would a soft-float datapath want?** The same taps
projected onto each `mpdsp` configuration via `project_onto`, scored by
`projection_error`. Relevant if the target is a soft-float or posit core
rather than a fixed-point DSP slice.

Fixed-point quantization here is the FPGA convention: one shared scale per
stage, chosen as the largest power of two that keeps every tap inside the
signed range, so the hardware scaling is a shift rather than a multiply.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import mpdsp

from design import (CascadeSpec, analyze_response, check_spec, design_cascade)

# Widths worth asking about: 8 covers a DSP48 pre-adder path, 18 is the
# native DSP48E coefficient width on Xilinx parts, 25 is the wide port.
BIT_WIDTHS = [8, 10, 12, 14, 16, 18, 20, 22, 24]

# Configurations a soft datapath might use for the coefficients themselves.
COEFF_DTYPES = ["reference", "gpu_baseline", "cf24", "fpga_fixed",
                "posit_32_2", "posit_16_1", "half", "posit_8_2"]


def quantize_taps(taps: np.ndarray, num_bits: int) -> tuple[np.ndarray,
                                                            np.ndarray, int]:
    """Quantize to `num_bits` signed fixed point.

    Returns (dequantized_float, integer_codes, fractional_bits).

    The scale is a power of two so the hardware undoes it with a shift. The
    integer codes are what the `.coe` and `$readmemh` files carry.
    """
    if num_bits < 2:
        raise ValueError("num_bits must be >= 2")
    peak = float(np.max(np.abs(taps)))
    if peak == 0.0:
        return taps.copy(), np.zeros(len(taps), dtype=np.int64), num_bits - 1

    # Largest power-of-two scale that keeps every tap representable in a
    # signed num_bits word: codes live in [-2^(n-1), 2^(n-1) - 1].
    max_code = 2 ** (num_bits - 1) - 1
    frac_bits = int(np.floor(np.log2(max_code / peak)))
    scale = float(2 ** frac_bits)

    codes = np.clip(np.rint(taps * scale), -(max_code + 1), max_code)
    return codes / scale, codes.astype(np.int64), frac_bits


def sweep_bit_widths(stages: list[np.ndarray], spec: CascadeSpec,
                     widths: list[int] = None) -> list[dict]:
    """Re-measure the cascade with taps quantized to each width."""
    rows = []
    for num_bits in (widths or BIT_WIDTHS):
        quantized = [quantize_taps(t, num_bits)[0] for t in stages]
        analysis = analyze_response(quantized)
        failures = check_spec(analysis, spec)
        rows.append({
            "bits": num_bits,
            "passband_ripple_db": analysis["passband_ripple_db"],
            "alias_attenuation_db": analysis["alias_attenuation_db"],
            "rom_bits": num_bits * sum(len(t) for t in stages),
            "passes": not failures,
        })
    return rows


def sweep_coeff_dtypes(stages: list[np.ndarray], spec: CascadeSpec,
                       dtypes: list[str] = None) -> list[dict]:
    """Project the taps onto each arithmetic configuration and re-measure."""
    rows = []
    for dtype in (dtypes or COEFF_DTYPES):
        projected, errors = [], []
        for taps in stages:
            contiguous = np.ascontiguousarray(taps)
            projected.append(mpdsp.project_onto(contiguous, dtype))
            errors.append(mpdsp.projection_error(contiguous, dtype))
        analysis = analyze_response(projected)
        failures = check_spec(analysis, spec)
        rows.append({
            "dtype": dtype,
            "sample_bits": mpdsp.bits_of(dtype),
            "max_projection_error": max(errors),
            "passband_ripple_db": analysis["passband_ripple_db"],
            "alias_attenuation_db": analysis["alias_attenuation_db"],
            "passes": not failures,
        })
    return rows


def smallest_passing_width(rows: list[dict]) -> int | None:
    """The narrowest coefficient word that still clears the spec."""
    passing = [r["bits"] for r in rows if r["passes"]]
    return min(passing) if passing else None


def write_csv(path: Path, bit_rows: list[dict], dtype_rows: list[dict]
              ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sweep", "key", "sample_bits_or_rom_bits",
                         "passband_ripple_db", "alias_attenuation_db",
                         "passes"])
        for r in bit_rows:
            writer.writerow(["fixed_point", r["bits"], r["rom_bits"],
                             f"{r['passband_ripple_db']:.4f}",
                             f"{r['alias_attenuation_db']:.2f}",
                             int(r["passes"])])
        for r in dtype_rows:
            writer.writerow(["arithmetic", r["dtype"], r["sample_bits"],
                             f"{r['passband_ripple_db']:.4f}",
                             f"{r['alias_attenuation_db']:.2f}",
                             int(r["passes"])])


def plot_tradeoff(path: Path, bit_rows: list[dict], spec: CascadeSpec
                  ) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    bits = [r["bits"] for r in bit_rows]
    atten = [r["alias_attenuation_db"] for r in bit_rows]
    rom = [r["rom_bits"] / 8.0 for r in bit_rows]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Coefficient width vs. cascade quality")

    ax_left.plot(bits, atten, marker="o")
    ax_left.axhline(spec.min_alias_attenuation_db, color="k", linestyle=":",
                    label=f"spec {spec.min_alias_attenuation_db:.0f} dB")
    ax_left.set_xlabel("coefficient word (bits)")
    ax_left.set_ylabel("alias attenuation (dB)")
    ax_left.set_title("Rejection vs. width")
    ax_left.legend(fontsize=8)
    ax_left.grid(alpha=0.3)

    ax_right.plot(rom, atten, marker="s", color="tab:orange")
    ax_right.axhline(spec.min_alias_attenuation_db, color="k", linestyle=":")
    for r, a, b in zip(rom, atten, bits):
        ax_right.annotate(f"{b}b", (r, a), fontsize=7,
                          textcoords="offset points", xytext=(4, 3))
    ax_right.set_xlabel("coefficient ROM (bytes)")
    ax_right.set_ylabel("alias attenuation (dB)")
    ax_right.set_title("Rejection vs. table size")
    ax_right.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = (Path(args.outdir) if args.outdir
              else Path(__file__).parent / "artifacts")
    spec = CascadeSpec()
    stages = design_cascade(spec)

    bit_rows = sweep_bit_widths(stages, spec)
    dtype_rows = sweep_coeff_dtypes(stages, spec)

    print("Fixed-point coefficient width")
    print(f"{'bits':>5}{'ripple dB':>12}{'alias dB':>11}"
          f"{'ROM bytes':>12}{'spec':>7}")
    print("-" * 47)
    for r in bit_rows:
        print(f"{r['bits']:>5}{r['passband_ripple_db']:>12.4f}"
              f"{r['alias_attenuation_db']:>11.1f}{r['rom_bits'] // 8:>12}"
              f"{'PASS' if r['passes'] else 'fail':>7}")

    chosen = smallest_passing_width(bit_rows)
    print(f"\nnarrowest width meeting spec: "
          f"{chosen if chosen else 'none'} bits")

    print("\nCoefficients projected onto each arithmetic configuration")
    print(f"{'dtype':<14}{'bits':>5}{'proj err':>12}{'ripple dB':>12}"
          f"{'alias dB':>11}{'spec':>7}")
    print("-" * 61)
    for r in dtype_rows:
        print(f"{r['dtype']:<14}{r['sample_bits']:>5}"
              f"{r['max_projection_error']:>12.2e}"
              f"{r['passband_ripple_db']:>12.4f}"
              f"{r['alias_attenuation_db']:>11.1f}"
              f"{'PASS' if r['passes'] else 'fail':>7}")

    write_csv(outdir / "quantization.csv", bit_rows, dtype_rows)
    plotted = plot_tradeoff(outdir / "quantization.png", bit_rows, spec)
    print(f"\nartifacts -> {outdir}")
    if not plotted:
        print("  (quantization.png skipped: matplotlib not installed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
