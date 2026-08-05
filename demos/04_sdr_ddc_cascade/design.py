"""Design and validate a 4-stage DDC decimation cascade for a 5 GSPS front end.

    python demos/04_sdr_ddc_cascade/design.py
    python demos/04_sdr_ddc_cascade/design.py --dtypes reference fpga_fixed

Cascade, total decimation 16 (5 GSPS -> 312.5 MSPS):

    RF in --> [NCO mix to baseband + stage 1 decimate /2]   (mpdsp.DDC)
          --> [stage 2 /2] --> [stage 3 /2] --> [stage 4 /2, sharp]
                                          (mpdsp.DecimationChain)

Why the stages are window-method FIRs
-------------------------------------
`design_halfband` is the natural fit for a decimate-by-2 stage and is still
not used, but for a measured reason rather than the bug that originally
forced it. A half-band's transition band is structurally centred on 0.25:
at its best setting here (51 taps, `transition_width=0.1`) a single stage
reaches 87 dB, but several of the cascade's alias bands land inside that
transition, where nothing is attenuating them. Measured on this spec, a
half-band cascade gives 51.6 dB of alias rejection against the Kaiser
design's 89.9 dB at the same 208 taps — and narrowing the transition makes
it worse, not better (25 dB at `transition_width=0.02`), since a tighter
spec on a fixed tap count buys nothing.

The half-band does win on multiplies — 156 non-zero taps against 208 — so
if the alias-band spec were looser it would be the right call. It is not
looser here.

Validation
----------
Composite response is computed by the noble identity: stage k runs at
fs/2^k, so referred to the input rate its response is H_k(f * 2^k) and the
cascade is the product. Two numbers come out of it:

* **passband ripple** across the band the receiver keeps;
* **alias-band attenuation** — the worst level over the frequency intervals
  that *fold onto* the passband when decimating by 16. That is the number
  that matters for a decimator, and it is not the same as "stopband
  attenuation": an image at k/16 +/- f_pass lands on top of the signal no
  matter how good the response looks in between.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mpdsp

from simulate import SAMPLE_RATE_HZ, synthesize, wanted_band

# Decimation is 2 per stage; four stages give 16 overall.
STAGE_DECIMATION = 2
NUM_STAGES = 4
TOTAL_DECIMATION = STAGE_DECIMATION ** NUM_STAGES

# Fraction of the final Nyquist that counts as passband. 0.8 leaves a
# transition band the last stage can actually realize.
PASSBAND_FRACTION = 0.8

DEFAULT_DTYPES = ["reference", "gpu_baseline", "cf24", "fpga_fixed",
                  "posit_full", "half"]


@dataclass
class CascadeSpec:
    """Tap counts and cutoffs, tuned so the composite clears the spec.

    Cutoffs are in each stage's *own* normalized frequency (cycles/sample at
    that stage's input rate), which is what `fir_lowpass` wants once
    `sample_rate` is 1.0. Every decimate-by-2 stage needs its cutoff under
    0.25 or it aliases into itself.
    """

    stage_taps: tuple[int, ...] = (51, 35, 27)
    stage_cutoff: float = 0.25
    final_taps: int = 95
    final_cutoff: float = 0.23
    window: str = "kaiser"

    # Spec the composite must meet.
    max_passband_ripple_db: float = 0.5
    min_alias_attenuation_db: float = 60.0


@dataclass
class CascadeResult:
    dtype: str
    baseband: np.ndarray          # complex baseband at the output rate
    stage_snr_db: list[float] = field(default_factory=list)
    output_snr_db: float | None = None


def design_cascade(spec: CascadeSpec) -> list[np.ndarray]:
    """Return the four stages' tap arrays, in signal order."""
    stages = [
        np.asarray(mpdsp.fir_lowpass(
            num_taps=n, sample_rate=1.0, cutoff=spec.stage_cutoff,
            window=spec.window).coefficients())
        for n in spec.stage_taps
    ]
    stages.append(np.asarray(mpdsp.fir_lowpass(
        num_taps=spec.final_taps, sample_rate=1.0, cutoff=spec.final_cutoff,
        window=spec.window).coefficients()))
    return stages


def stage_response(taps: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """H(f) for `taps`, with f in cycles/sample at that stage's own rate."""
    n = np.arange(len(taps))
    return np.exp(-2j * np.pi * np.outer(freqs, n)) @ taps


def composite_response(stages: list[np.ndarray],
                       freqs_input: np.ndarray) -> np.ndarray:
    """Cascade response referred to the *input* rate (noble identity)."""
    out = np.ones_like(freqs_input, dtype=complex)
    for k, taps in enumerate(stages):
        out = out * stage_response(taps, freqs_input * (STAGE_DECIMATION ** k))
    return out


def passband_edge() -> float:
    """Input-referred normalized edge of the band the receiver keeps."""
    return PASSBAND_FRACTION * (0.5 / TOTAL_DECIMATION)


def analyze_response(stages: list[np.ndarray], num_points: int = 120001
                     ) -> dict:
    """Measure passband ripple and worst alias-band level."""
    freqs = np.linspace(0.0, 0.5, num_points)
    magnitude = np.abs(composite_response(stages, freqs))
    magnitude = magnitude / magnitude[0]
    db = 20.0 * np.log10(np.maximum(magnitude, 1e-14))

    edge = passband_edge()
    in_passband = freqs <= edge
    ripple = float(db[in_passband].max() - db[in_passband].min())

    worst_level, worst_image = -np.inf, None
    for k in range(1, TOTAL_DECIMATION // 2 + 1):
        low = max(k / TOTAL_DECIMATION - edge, 0.0)
        high = min(k / TOTAL_DECIMATION + edge, 0.5)
        band = (freqs >= low) & (freqs <= high)
        if band.any() and db[band].max() > worst_level:
            worst_level, worst_image = float(db[band].max()), k

    return {
        "freqs": freqs,
        "response_db": db,
        "passband_edge": edge,
        "passband_ripple_db": ripple,
        "alias_attenuation_db": -worst_level,
        "worst_image": worst_image,
    }


def check_spec(analysis: dict, spec: CascadeSpec) -> list[str]:
    """Return a list of spec violations; empty means the design passes."""
    failures = []
    if analysis["passband_ripple_db"] > spec.max_passband_ripple_db:
        failures.append(
            f"passband ripple {analysis['passband_ripple_db']:.3f} dB "
            f"exceeds {spec.max_passband_ripple_db} dB")
    if analysis["alias_attenuation_db"] < spec.min_alias_attenuation_db:
        failures.append(
            f"alias attenuation {analysis['alias_attenuation_db']:.1f} dB "
            f"below {spec.min_alias_attenuation_db} dB "
            f"(worst image k={analysis['worst_image']})")
    return failures


def run_cascade(signal: np.ndarray, stages: list[np.ndarray], meta: dict,
                dtype: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Down-convert and decimate. Returns (baseband, per-stage I traces).

    Stage 1 is folded into `mpdsp.DDC`, which does the NCO mix and the first
    decimation together. The remaining three run as a `DecimationChain` over
    the I and Q streams separately — the chain is real-valued, and I/Q are
    independent once the mixer has separated them.

    Rates are passed **normalized** (sample rate 1.0, carrier as a fraction
    of it), not in Hz. This is not cosmetic. `NCO` and `DDC` take frequency
    and sample rate as the configuration's state scalar and divide only
    afterwards, so absolute GHz values overflow every narrow state type
    before the division can bring them back into range. At 1.2 GHz / 5 GSPS
    every type narrower than float fails: `fpga_fixed` (fixpnt<32,24>,
    integer range +/-128) cannot hold the rate at all, and `cf24` and `half`
    produce a non-finite phase increment.

    Those now raise rather than returning silent NaN (the guard added in
    mp-dsp-python#117), but raising is still a failure — normalized rates
    are what actually works, and they mean the same thing, since an
    oscillator only ever uses the ratio. The underlying fix is upstream
    mixed-precision-dsp#207: divide in double and convert only the ratio.
    """
    carrier_normalized = meta["carrier_hz"] / meta["sample_rate_hz"]

    ddc = mpdsp.DDC(carrier_normalized, 1.0,
                    np.ascontiguousarray(stages[0]), STAGE_DECIMATION,
                    dtype=dtype)
    real, imag = ddc.process_block(np.ascontiguousarray(signal))
    traces = [real.copy()]

    for taps in stages[1:]:
        def decimate(stream):
            stage = [mpdsp.PolyphaseDecimator(
                taps=np.ascontiguousarray(taps), factor=STAGE_DECIMATION,
                dtype=dtype)]
            return mpdsp.DecimationChain(1.0, stage, dtype=dtype
                                         ).process_block(
                np.ascontiguousarray(stream))

        real, imag = decimate(real), decimate(imag)
        traces.append(real.copy())

    return real + 1j * imag, traces


def run_all(signal: np.ndarray, stages: list[np.ndarray], meta: dict,
            dtypes: list[str]) -> list[CascadeResult]:
    """Run every dtype, scoring each against the reference run."""
    ordered = (["reference"] + [d for d in dtypes if d != "reference"]
               if "reference" in dtypes else list(dtypes))

    results, reference_traces, reference_bb = [], None, None
    for dtype in ordered:
        baseband, traces = run_cascade(signal, stages, meta, dtype)
        result = CascadeResult(dtype=dtype, baseband=baseband)

        if dtype == "reference":
            reference_traces, reference_bb = traces, baseband
        elif reference_traces is not None:
            for ref_trace, trace in zip(reference_traces, traces):
                n = min(len(ref_trace), len(trace))
                result.stage_snr_db.append(mpdsp.sqnr_db(
                    np.ascontiguousarray(ref_trace[:n]),
                    np.ascontiguousarray(trace[:n])))
            n = min(len(reference_bb), len(baseband))
            result.output_snr_db = mpdsp.sqnr_db(
                np.ascontiguousarray(np.real(reference_bb[:n])),
                np.ascontiguousarray(np.real(baseband[:n])))
        results.append(result)
    return results


def write_summary_csv(path: Path, analysis: dict, results: list[CascadeResult]
                      ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dtype", "sample_bits", "output_snr_db"]
                        + [f"stage{i + 1}_snr_db" for i in range(NUM_STAGES)])
        for r in results:
            row = [r.dtype, mpdsp.bits_of(r.dtype),
                   "" if r.output_snr_db is None else f"{r.output_snr_db:.2f}"]
            row += [f"{v:.2f}" for v in r.stage_snr_db]
            row += [""] * (NUM_STAGES - len(r.stage_snr_db))
            writer.writerow(row)


def plot_summary(path: Path, stages: list[np.ndarray], analysis: dict,
                 results: list[CascadeResult], spec: CascadeSpec) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("SDR DDC cascade — 5 GSPS front end, decimate by 16", y=0.98)

    # Composite response with the alias bands marked.
    ax = axes[0][0]
    ax.plot(analysis["freqs"], analysis["response_db"], linewidth=0.8)
    edge = analysis["passband_edge"]
    ax.axvspan(0, edge, color="tab:green", alpha=0.2, label="passband")
    for k in range(1, TOTAL_DECIMATION // 2 + 1):
        ax.axvspan(max(k / TOTAL_DECIMATION - edge, 0),
                   min(k / TOTAL_DECIMATION + edge, 0.5),
                   color="tab:red", alpha=0.15,
                   label="alias bands" if k == 1 else None)
    ax.axhline(-spec.min_alias_attenuation_db, color="k", linestyle=":",
               linewidth=0.9, label=f"-{spec.min_alias_attenuation_db:.0f} dB")
    ax.set_ylim(-140, 10)
    ax.set_xlabel("normalized frequency (input rate)")
    ax.set_ylabel("dB")
    ax.set_title("Composite response — bands that fold onto the passband")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Passband detail.
    ax = axes[0][1]
    band = analysis["freqs"] <= edge * 1.5
    ax.plot(analysis["freqs"][band], analysis["response_db"][band],
            linewidth=1.0)
    ax.axvline(edge, color="k", linestyle=":", linewidth=0.9)
    ax.set_title(f"Passband detail — ripple "
                 f"{analysis['passband_ripple_db']:.3f} dB")
    ax.set_xlabel("normalized frequency (input rate)")
    ax.set_ylabel("dB")
    ax.grid(alpha=0.3)

    # SNR after each stage.
    ax = axes[1][0]
    for r in results:
        if r.stage_snr_db:
            ax.plot(range(1, len(r.stage_snr_db) + 1), r.stage_snr_db,
                    marker="o", label=r.dtype)
    ax.set_xlabel("stage")
    ax.set_ylabel("SNR vs reference (dB)")
    ax.set_title("SNR through the cascade")
    ax.set_xticks(range(1, NUM_STAGES + 1))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Per-stage tap counts.
    ax = axes[1][1]
    labels = [f"stage {i + 1}" for i in range(len(stages))]
    ax.bar(labels, [len(t) for t in stages], color="tab:blue")
    ax.set_ylabel("taps")
    ax.set_title(f"Coefficient budget — {sum(len(t) for t in stages)} taps "
                 f"total")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=262144)
    parser.add_argument("--dtypes", nargs="+", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = (Path(args.outdir) if args.outdir
              else Path(__file__).parent / "artifacts")
    spec = CascadeSpec()
    stages = design_cascade(spec)
    analysis = analyze_response(stages)

    print(f"cascade: {NUM_STAGES} stages, decimate by {TOTAL_DECIMATION} "
          f"({SAMPLE_RATE_HZ / 1e9:.1f} GSPS -> "
          f"{SAMPLE_RATE_HZ / TOTAL_DECIMATION / 1e6:.1f} MSPS)")
    print(f"  taps per stage : {[len(t) for t in stages]} "
          f"({sum(len(t) for t in stages)} total)")
    print(f"  passband ripple: {analysis['passband_ripple_db']:.3f} dB "
          f"(spec < {spec.max_passband_ripple_db})")
    print(f"  alias rejection: {analysis['alias_attenuation_db']:.1f} dB "
          f"(spec > {spec.min_alias_attenuation_db}, "
          f"worst image k={analysis['worst_image']})")

    failures = check_spec(analysis, spec)
    if failures:
        for failure in failures:
            print(f"  SPEC FAIL: {failure}")
        return 1
    print("  spec: PASS")

    signal, meta = synthesize(num_samples=args.num_samples)
    low, high = wanted_band(meta)
    print(f"\nsignal: wanted {low / 1e9:.4f}-{high / 1e9:.4f} GHz, "
          f"alias trap at {meta['alias_hz'] / 1e9:.4f} GHz")

    dtypes = list(args.dtypes) if args.dtypes else list(DEFAULT_DTYPES)
    results = run_all(signal, stages, meta, dtypes)

    write_summary_csv(outdir / "summary.csv", analysis, results)
    plotted = plot_summary(outdir / "summary.png", stages, analysis, results,
                           spec)

    print(f"\n{'dtype':<14}{'bits':>5}{'out SNR dB':>12}   per-stage SNR dB")
    print("-" * 62)
    for r in results:
        out = "     (ref)" if r.output_snr_db is None else f"{r.output_snr_db:10.2f}"
        stages_txt = ("  ".join(f"{v:6.1f}" for v in r.stage_snr_db)
                      if r.stage_snr_db else "  (reference)")
        print(f"{r.dtype:<14}{mpdsp.bits_of(r.dtype):>5}{out:>12}   "
              f"{stages_txt}")

    print(f"\nartifacts -> {outdir}")
    if not plotted:
        print("  (summary.png skipped: matplotlib not installed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
