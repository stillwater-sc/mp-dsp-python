"""Train LMS / NLMS / RLS cancellers across arithmetic types.

    python demos/03_vibration_cancellation/design.py
    python demos/03_vibration_cancellation/design.py --algorithms RLS

Each (algorithm, dtype) pair is trained on the same disturbance, and three
things are measured: how far the residual falls, whether it survives the
machine speed change halfway through, and how far its converged taps drift
from the reference run's.

What the demo is actually for
-----------------------------
Adaptive filters fail differently from fixed ones. A quantized IIR filter
gets a slightly wrong frequency response; a quantized *adaptive* filter can
stop converging, converge to the wrong answer, or diverge outright — and in
an active-cancellation loop the last of those does not mean "no
cancellation", it means the actuator is now driving the structure. That
distinction is invisible in a SQNR number and obvious in a residual trace,
which is why this demo plots one.

RLS is the sharp case, and it is not a surprise: `notebooks/06_estimation`
already recorded that its Kalman-form update loses P-matrix symmetry at
narrow precision. This demo pins it as a reproducible measurement.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mpdsp

from simulate import synthesize

NUM_TAPS = 24

# Constructors, so an algorithm is one entry rather than a branch.
ALGORITHMS = {
    "LMS": lambda dtype: mpdsp.LMSFilter(
        num_taps=NUM_TAPS, step_size=0.01, dtype=dtype),
    "NLMS": lambda dtype: mpdsp.NLMSFilter(
        num_taps=NUM_TAPS, step_size=0.3, dtype=dtype),
    # forgetting_factor is 0.9995, not the more common 0.999, and the
    # difference is load bearing. At 0.999 this problem suffers covariance
    # windup: P loses positive-definiteness through accumulated rounding and
    # the filter diverges *even in double* given enough samples — measured,
    # reference amplifies by 30.9 dB over a 40000-sample run. Demoing that
    # as a precision result would be a lie, because the algorithm is
    # unstable at every precision. At 0.9995 double is stable indefinitely
    # (30.0 dB, peak residual 0.22x the disturbance), so precision is the
    # only variable left and the failures below are real ones.
    #
    # Memory is ~1/(1-lambda) = 2000 samples = 0.5 s, still short enough to
    # track the speed change at 3 s.
    "RLS": lambda dtype: mpdsp.RLSFilter(
        num_taps=NUM_TAPS, forgetting_factor=0.9995, delta=1000.0,
        dtype=dtype),
}

# Six configurations: the reference, a float pair, and three narrow types
# that bracket where adaptive filters start to break.
DEFAULT_DTYPES = ["reference", "gpu_baseline", "posit_full", "cf24",
                  "half", "posit_8_2"]

# Samples of steady state to average over, at the end of each regime.
_WINDOW = 2000


@dataclass
class TrainingResult:
    algorithm: str
    dtype: str
    errors: np.ndarray
    weights: np.ndarray
    reduction_db: float = float("nan")        # after the speed change
    reduction_before_db: float = float("nan")  # before it
    weight_drift: float = float("nan")         # vs the reference run
    diverged: bool = False
    amplified: bool = False
    notes: str = ""

    @property
    def status(self) -> str:
        if self.diverged:
            return "DIVERGED"
        if self.amplified:
            return "AMPLIFIED"
        return "ok"


def _reduction_db(primary: np.ndarray, errors: np.ndarray,
                  stop: int, window: int = _WINDOW) -> float:
    """Residual power reduction over the `window` samples ending at `stop`."""
    start = max(0, stop - window)
    before = float(np.mean(primary[start:stop] ** 2))
    after = float(np.mean(errors[start:stop] ** 2))
    if not np.isfinite(after) or after <= 0.0 or before <= 0.0:
        return float("nan")
    return 10.0 * np.log10(before / after)


def train(algorithm: str, dtype: str, reference: np.ndarray,
          primary: np.ndarray, meta: dict) -> TrainingResult:
    """Run one (algorithm, dtype) pair over the whole disturbance."""
    filt = ALGORITHMS[algorithm](dtype)
    try:
        _, errors = filt.process_block(np.ascontiguousarray(reference),
                                       np.ascontiguousarray(primary))
    except Exception as exc:                                # noqa: BLE001
        return TrainingResult(
            algorithm=algorithm, dtype=dtype,
            errors=np.full(len(primary), np.nan),
            weights=np.full(NUM_TAPS, np.nan),
            diverged=True, notes=f"{type(exc).__name__}: {exc}")

    errors = np.asarray(errors)
    weights = np.asarray(filt.weights)

    result = TrainingResult(algorithm=algorithm, dtype=dtype,
                            errors=errors, weights=weights)

    if not (np.all(np.isfinite(errors)) and np.all(np.isfinite(weights))):
        result.diverged = True
        first_bad = int(np.argmax(~np.isfinite(errors)))
        result.notes = f"non-finite from sample {first_bad}"
        return result

    split = meta["speed_change_at"]
    result.reduction_before_db = _reduction_db(primary, errors, split)
    result.reduction_db = _reduction_db(primary, errors, len(errors))

    # Negative reduction is not "poor cancellation" — the residual is louder
    # than the untreated disturbance, which in a real rig means the actuator
    # is pumping energy into the structure.
    result.amplified = result.reduction_db < 0.0
    return result


def run_all(reference: np.ndarray, primary: np.ndarray, meta: dict,
            algorithms: list[str], dtypes: list[str]
            ) -> list[TrainingResult]:
    """Train every pair, scoring weight drift against each algorithm's own
    reference run (comparing across algorithms would be meaningless — they
    converge to the same solution by different paths)."""
    results: list[TrainingResult] = []
    for algorithm in algorithms:
        ordered = (["reference"] + [d for d in dtypes if d != "reference"]
                   if "reference" in dtypes else list(dtypes))
        baseline: np.ndarray | None = None
        for dtype in ordered:
            result = train(algorithm, dtype, reference, primary, meta)
            if dtype == "reference" and not result.diverged:
                baseline = result.weights
            elif baseline is not None and not result.diverged:
                result.weight_drift = float(
                    np.max(np.abs(result.weights - baseline)))
            results.append(result)
    return results


def write_summary_csv(path: Path, results: list[TrainingResult],
                      meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["algorithm", "dtype", "sample_bits", "status",
                         "reduction_before_db", "reduction_after_db",
                         "weight_drift", "achievable_db", "notes"])
        for r in results:
            writer.writerow([
                r.algorithm, r.dtype, mpdsp.bits_of(r.dtype), r.status,
                "" if np.isnan(r.reduction_before_db)
                else f"{r.reduction_before_db:.2f}",
                "" if np.isnan(r.reduction_db) else f"{r.reduction_db:.2f}",
                "" if np.isnan(r.weight_drift) else f"{r.weight_drift:.4e}",
                f"{meta['achievable_reduction_db']:.1f}", r.notes])


def _smoothed_error_power_db(errors: np.ndarray, window: int = 256
                             ) -> np.ndarray:
    """Residual power in dB, boxcar-smoothed so the trace is readable.

    Non-finite samples are held at a floor rather than dropped, so a
    diverging run stays visible on the plot instead of vanishing from it.
    """
    finite = np.isfinite(errors)
    # Square only the finite samples: squaring a NaN warns, and an inf
    # squared overflows to a warning too.
    power = np.zeros_like(errors)
    power[finite] = errors[finite] ** 2
    kernel = np.ones(window) / window
    trace = 10.0 * np.log10(np.maximum(
        np.convolve(power, kernel, mode="same"), 1e-30))
    # Break the line where the run went non-finite, so a diverged filter
    # visibly stops rather than flat-lining at the floor as if converged.
    trace[~finite] = np.nan
    return trace


def plot_summary(path: Path, results: list[TrainingResult],
                 primary: np.ndarray, meta: dict) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    algorithms = sorted({r.algorithm for r in results},
                        key=lambda a: list(ALGORITHMS).index(a))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Active vibration cancellation — convergence vs arithmetic",
                 y=0.98)

    split = meta["speed_change_at"]
    rate = meta["sample_rate"]
    time = np.arange(len(primary)) / rate
    baseline_db = 10.0 * np.log10(np.mean(primary ** 2))

    # One convergence panel per algorithm (first two), then summary panels.
    for ax, algorithm in zip(axes[0], algorithms[:2]):
        for r in (x for x in results if x.algorithm == algorithm):
            ax.plot(time, _smoothed_error_power_db(r.errors),
                    linewidth=1.0, label=f"{r.dtype}", alpha=0.85)
        ax.axhline(baseline_db, color="k", linestyle=":", linewidth=0.9,
                   label="untreated")
        ax.axhline(baseline_db - meta["achievable_reduction_db"],
                   color="tab:green", linestyle="--", linewidth=0.9,
                   label="sensor floor")
        ax.axvline(split / rate, color="tab:red", linestyle="--",
                   linewidth=0.9)
        ax.set_title(f"{algorithm} — residual power "
                     f"(speed change at {split / rate:.1f} s)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("dB")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    # RLS gets its own panel: it is the one that diverges.
    ax = axes[1][0]
    for r in (x for x in results if x.algorithm == "RLS"):
        label = r.dtype + ("" if r.status == "ok" else f"  [{r.status}]")
        ax.plot(time, _smoothed_error_power_db(r.errors), linewidth=1.0,
                label=label, alpha=0.85)
    ax.axhline(baseline_db, color="k", linestyle=":", linewidth=0.9)
    ax.axvline(split / rate, color="tab:red", linestyle="--", linewidth=0.9)
    ax.set_title("RLS — the precision-fragile one")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dB")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Achieved reduction per pair, with the physical limit drawn on.
    ax = axes[1][1]
    dtypes = [d for d in DEFAULT_DTYPES
              if any(r.dtype == d for r in results)]
    width = 0.8 / max(len(algorithms), 1)
    for index, algorithm in enumerate(algorithms):
        by_dtype = {r.dtype: r for r in results if r.algorithm == algorithm}
        values = []
        for dtype in dtypes:
            r = by_dtype.get(dtype)
            # A diverged run has no reduction; plot it at the floor of the
            # axis so the gap is visible rather than absent.
            values.append(0.0 if r is None or not np.isfinite(r.reduction_db)
                          else r.reduction_db)
        positions = np.arange(len(dtypes)) + index * width
        ax.bar(positions, values, width=width, label=algorithm)
    ax.axhline(meta["achievable_reduction_db"], color="tab:green",
               linestyle="--", linewidth=1.0, label="sensor floor")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xticks(np.arange(len(dtypes)) + 0.4 - width / 2)
    ax.set_xticklabels(dtypes, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("reduction (dB)")
    ax.set_title("Achieved cancellation (0 = diverged, < 0 = amplified)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=24000)
    parser.add_argument("--algorithms", nargs="+", default=None,
                        choices=list(ALGORITHMS))
    parser.add_argument("--dtypes", nargs="+", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = (Path(args.outdir) if args.outdir
              else Path(__file__).parent / "artifacts")
    algorithms = list(args.algorithms) if args.algorithms else list(ALGORITHMS)
    dtypes = list(args.dtypes) if args.dtypes else list(DEFAULT_DTYPES)

    reference, primary, meta = synthesize(num_samples=args.num_samples)
    print(f"disturbance: {meta['num_samples']} samples @ "
          f"{meta['sample_rate']:.0f} Hz, tones {meta['tones_hz']} Hz "
          f"stepping x{meta['speed_change']} at "
          f"{meta['speed_change_at'] / meta['sample_rate']:.1f} s")
    print(f"sensor floor: {meta['sensor_snr_db']:.0f} dB SNR — cancellation "
          f"cannot exceed ~{meta['achievable_reduction_db']:.0f} dB\n")

    results = run_all(reference, primary, meta, algorithms, dtypes)

    write_summary_csv(outdir / "summary.csv", results, meta)
    plotted = plot_summary(outdir / "summary.png", results, primary, meta)

    print(f"{'algo':6s}{'dtype':14s}{'bits':>5}{'before dB':>11}"
          f"{'after dB':>10}{'drift':>11}   status")
    print("-" * 68)
    for r in results:
        before = ("     —" if np.isnan(r.reduction_before_db)
                  else f"{r.reduction_before_db:9.1f}")
        after = ("     —" if np.isnan(r.reduction_db)
                 else f"{r.reduction_db:8.1f}")
        drift = ("     —" if np.isnan(r.weight_drift)
                 else f"{r.weight_drift:9.2e}")
        flag = "" if r.status == "ok" else f"   {r.status}"
        print(f"{r.algorithm:6s}{r.dtype:14s}{mpdsp.bits_of(r.dtype):>5}"
              f"{before:>11}{after:>10}{drift:>11}{flag}")
        if r.notes:
            print(f"{'':37s}{r.notes}")

    print(f"\nartifacts -> {outdir}")
    if not plotted:
        print("  (summary.png skipped: matplotlib not installed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
