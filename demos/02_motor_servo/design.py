"""Tune the current-loop PI + resonance notch, and sweep arithmetic types.

    python demos/02_motor_servo/design.py
    python demos/02_motor_servo/design.py --design-point aggressive

Two design points, because the interesting result is the difference between
them:

* **conservative** — notch at 2.5 kHz, above a 1.2 kHz crossover. Every dtype
  down to 8-bit posit stays stable; what precision costs is *notch depth*,
  which falls from a numerical null to 16 dB.
* **aggressive** — notch at 1 kHz, inside a 1.5 kHz crossover, as you are
  forced into when the resonance is a low-frequency blade-pass rather than a
  structural mode. Here 8-bit posit does not merely blunt the notch, it
  destabilizes the loop.

Same arithmetic, same plant; only the design margin differs. That is the
actionable question a controls engineer actually has — not "is 8 bits
enough" but "how much margin does 8 bits cost me".

What is quantized, and what is not
----------------------------------
The controller and notch *coefficients* are round-tripped through the target
type via `project_onto`; the simulation itself runs in double. This is the
deployment question — what does storing these coefficients in T do to my
closed loop — and it is the same model `IIRFilter.frequency_response(dtype=)`
and `pole_displacement(dtype)` use.

It does not model the arithmetic of the control loop's own state updates. On
a fixed-point MCU that matters too, and a full treatment would need the loop
run sample-by-sample at the target type — which `IIRFilter.process` cannot do
today, since it builds fresh state per call. Stated rather than glossed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mpdsp

from simulate import (SAMPLE_RATE_HZ, Motor, close_loop, closed_loop_poles,
                      response_metrics, series, step_response)

# Seven configurations, as the acceptance asks: the reference, both float
# widths, the two 16-bit contenders, the FPGA fixed-point path, and 8-bit.
DEFAULT_DTYPES = ["reference", "gpu_baseline", "cf24", "posit_full",
                  "half", "fpga_fixed", "posit_8_2"]


@dataclass(frozen=True)
class DesignPoint:
    name: str
    crossover_hz: float
    notch_hz: float
    notch_bandwidth_oct: float
    rationale: str


DESIGN_POINTS = {
    "conservative": DesignPoint(
        name="conservative", crossover_hz=1200.0, notch_hz=2500.0,
        notch_bandwidth_oct=1.0,
        rationale="notch above crossover — a structural or PWM-related mode"),
    "aggressive": DesignPoint(
        name="aggressive", crossover_hz=1500.0, notch_hz=1000.0,
        notch_bandwidth_oct=0.5,
        rationale="notch inside the loop — a low-frequency blade-pass mode"),
}


@dataclass
class SweepResult:
    dtype: str
    design_point: str
    max_pole: float
    notch_depth_db: float
    step: np.ndarray
    metrics: dict = field(default_factory=dict)
    pole_displacement: float = float("nan")

    @property
    def stable(self) -> bool:
        return bool(np.isfinite(self.max_pole) and self.max_pole < 1.0)

    @property
    def status(self) -> str:
        if not self.stable:
            return "UNSTABLE"
        if self.metrics.get("diverged"):
            return "DIVERGED"
        if abs(self.metrics.get("steady_state", 1.0) - 1.0) > 0.05:
            return "OFFSET"
        return "ok"


def design_pi(motor: Motor, crossover_hz: float,
              sample_rate_hz: float = SAMPLE_RATE_HZ
              ) -> tuple[np.ndarray, np.ndarray, float]:
    """PI current controller by plant-pole cancellation.

    C(z) = Kp (1 - a z^-1) / (1 - z^-1)

    The zero is placed on the plant's pole, so the compensated loop is a
    pure integrator and the crossover is set by Kp alone. This is the
    standard current-loop tuning and it is also why the demo has something
    to say about quantization: the cancellation is only as good as the
    precision of `a`, and a coefficient error there leaves a slow
    pole-zero doublet behind.
    """
    pole = motor.plant_pole(sample_rate_hz)
    period = 1.0 / sample_rate_hz
    gain = motor.resistance_ohm * (2.0 * np.pi * crossover_hz * period) \
        / (1.0 - pole)
    return (gain * np.array([1.0, -pole]), np.array([1.0, -1.0]), gain)


def design_notch(point: DesignPoint, sample_rate_hz: float = SAMPLE_RATE_HZ
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Resonance notch, from mpdsp's RBJ bandstop."""
    biquad = mpdsp.rbj_bandstop(
        sample_rate=sample_rate_hz, center_freq=point.notch_hz,
        bandwidth=point.notch_bandwidth_oct).coefficients()[0]
    b0, b1, b2, a1, a2 = biquad
    return np.array([b0, b1, b2]), np.array([1.0, a1, a2])


def notch_depth_db(num: np.ndarray, den: np.ndarray, freq_hz: float,
                   sample_rate_hz: float = SAMPLE_RATE_HZ) -> float:
    """Attenuation at the notch centre, in dB."""
    z = np.exp(2j * np.pi * freq_hz / sample_rate_hz)
    response = np.polyval(num, z) / np.polyval(den, z)
    return float(20.0 * np.log10(abs(response) + 1e-30))


def _quantize(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "reference":
        return np.asarray(values, dtype=np.float64)
    return np.asarray(mpdsp.project_onto(
        np.ascontiguousarray(values, dtype=np.float64), dtype))


def sweep(point: DesignPoint, dtypes: list[str], motor: Motor | None = None,
          sample_rate_hz: float = SAMPLE_RATE_HZ) -> list[SweepResult]:
    """Quantize controller + notch per dtype and re-close the loop."""
    motor = motor or Motor()
    plant = motor.discrete_plant(sample_rate_hz)
    c_num, c_den, _ = design_pi(motor, point.crossover_hz, sample_rate_hz)
    n_num, n_den = design_notch(point, sample_rate_hz)

    # A double-precision cascade of the same controller, so the library's own
    # pole_displacement can be quoted alongside the closed-loop numbers.
    controller_biquad = mpdsp.BiquadCoefficients()
    controller_biquad.b0, controller_biquad.b1, controller_biquad.b2 = n_num
    controller_biquad.a1, controller_biquad.a2 = n_den[1], n_den[2]
    notch_filter = mpdsp.IIRFilter.from_coefficients([controller_biquad])

    results = []
    for dtype in dtypes:
        cn, cd = _quantize(c_num, dtype), _quantize(c_den, dtype)
        nn, nd = _quantize(n_num, dtype), _quantize(n_den, dtype)

        forward = series((cn, cd), (nn, nd), plant)
        cl_num, cl_den = close_loop(*forward)

        step = step_response(cl_num, cl_den)
        results.append(SweepResult(
            dtype=dtype,
            design_point=point.name,
            max_pole=float(np.max(np.abs(closed_loop_poles(cl_den)))),
            notch_depth_db=notch_depth_db(nn, nd, point.notch_hz,
                                          sample_rate_hz),
            step=step,
            metrics=response_metrics(step, sample_rate_hz),
            pole_displacement=notch_filter.pole_displacement(dtype),
        ))
    return results


def write_summary_csv(path: Path, results: list[SweepResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["design_point", "dtype", "sample_bits", "status",
                         "max_pole", "notch_depth_db", "notch_pole_displacement",
                         "overshoot_pct", "rise_ms", "settle_ms",
                         "steady_state"])
        for r in results:
            writer.writerow([
                r.design_point, r.dtype, mpdsp.bits_of(r.dtype), r.status,
                f"{r.max_pole:.6f}", f"{r.notch_depth_db:.1f}",
                f"{r.pole_displacement:.3e}",
                f"{r.metrics.get('overshoot_pct', float('nan')):.1f}",
                f"{r.metrics.get('rise_ms', float('nan')):.3f}",
                f"{r.metrics.get('settle_ms', float('nan')):.3f}",
                f"{r.metrics.get('steady_state', float('nan')):.4f}"])


def write_step_csv(path: Path, results: list[SweepResult],
                   sample_rate_hz: float = SAMPLE_RATE_HZ) -> None:
    """Per-dtype step response, as the issue asks for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    length = max(len(r.step) for r in results)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_ms"] + [f"{r.design_point}:{r.dtype}"
                                       for r in results])
        for k in range(length):
            row = [f"{k / sample_rate_hz * 1e3:.4f}"]
            for r in results:
                row.append("" if k >= len(r.step) or not np.isfinite(r.step[k])
                           else f"{r.step[k]:.6f}")
            writer.writerow(row)


def plot_summary(path: Path, by_point: dict[str, list[SweepResult]],
                 motor: Motor, sample_rate_hz: float = SAMPLE_RATE_HZ
                 ) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Motor current loop — PI + resonance notch vs arithmetic",
                 y=0.98)

    # Open-loop Bode of plant, controller and the combination.
    ax = axes[0][0]
    freqs = np.logspace(0, np.log10(sample_rate_hz / 2 * 0.99), 800)
    z = np.exp(2j * np.pi * freqs / sample_rate_hz)
    point = DESIGN_POINTS["conservative"]
    plant = motor.discrete_plant(sample_rate_hz)
    c_num, c_den, _ = design_pi(motor, point.crossover_hz, sample_rate_hz)
    n_num, n_den = design_notch(point, sample_rate_hz)
    for label, (num, den) in (("plant", plant), ("PI", (c_num, c_den)),
                              ("notch", (n_num, n_den)),
                              ("open loop", series((c_num, c_den),
                                                   (n_num, n_den), plant))):
        mag = np.abs(np.polyval(num, z) / np.polyval(den, z))
        ax.semilogx(freqs, 20 * np.log10(mag + 1e-30), linewidth=1.1,
                    label=label)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axvline(point.crossover_hz, color="tab:red", linestyle=":",
               linewidth=0.9)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB")
    ax.set_ylim(-80, 80)
    ax.set_title("Open-loop Bode (conservative point)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # Step responses, one panel per design point.
    for ax, (name, results) in zip(axes[0][1:].tolist() + [axes[1][0]],
                                   by_point.items()):
        for r in results:
            time_ms = np.arange(len(r.step)) / sample_rate_hz * 1e3
            label = r.dtype + ("" if r.status == "ok" else f" [{r.status}]")
            ax.plot(time_ms, r.step, linewidth=1.1, alpha=0.85, label=label)
        ax.axhline(1.0, color="k", linestyle=":", linewidth=0.9)
        ax.set_xlabel("ms")
        ax.set_ylabel("current (A, normalized)")
        ax.set_ylim(-0.5, 2.0)
        ax.set_title(f"Step response — {name}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    # Closed-loop poles on the unit circle, both design points.
    ax = axes[1][1]
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="0.7", linewidth=1.0)
    markers = {"conservative": "o", "aggressive": "x"}
    for name, results in by_point.items():
        for r in results:
            plant = motor.discrete_plant(sample_rate_hz)
            colour = "tab:red" if not r.stable else None
            ax.scatter([r.max_pole], [0.0], marker=markers.get(name, "o"),
                       s=28, alpha=0.0)  # keeps autoscale sane
    for name, results in by_point.items():
        radii = [r.max_pole for r in results]
        labels = [r.dtype for r in results]
        ax.scatter(radii, np.zeros(len(radii)), marker=markers.get(name, "o"),
                   s=36, label=f"{name}: max|pole|")
        for radius, label in zip(radii, labels):
            if radius > 0.99:
                ax.annotate(label, (radius, 0.0), fontsize=7,
                            textcoords="offset points", xytext=(3, 6))
    ax.axvline(1.0, color="tab:red", linestyle="--", linewidth=0.9,
               label="stability boundary")
    ax.set_xlim(0.85, 1.05)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel("|pole|")
    ax.set_title("Dominant closed-loop pole radius")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-point", choices=list(DESIGN_POINTS) + ["both"],
                        default="both")
    parser.add_argument("--dtypes", nargs="+", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = (Path(args.outdir) if args.outdir
              else Path(__file__).parent / "artifacts")
    dtypes = list(args.dtypes) if args.dtypes else list(DEFAULT_DTYPES)
    names = (list(DESIGN_POINTS) if args.design_point == "both"
             else [args.design_point])

    motor = Motor()
    print(f"plant: R={motor.resistance_ohm*1e3:.0f} mOhm, "
          f"L={motor.inductance_h*1e6:.0f} uH, "
          f"tau={motor.electrical_tau_s*1e6:.0f} us, "
          f"loop rate {SAMPLE_RATE_HZ/1e3:.0f} kHz\n")

    by_point: dict[str, list[SweepResult]] = {}
    for name in names:
        point = DESIGN_POINTS[name]
        results = sweep(point, dtypes, motor)
        by_point[name] = results

        print(f"[{name}] crossover {point.crossover_hz:.0f} Hz, "
              f"notch {point.notch_hz:.0f} Hz "
              f"({point.notch_bandwidth_oct:.1f} oct) — {point.rationale}")
        print(f"  {'dtype':14s}{'bits':>5}{'max|pole|':>11}{'notch dB':>10}"
              f"{'over%':>8}{'t90 ms':>8}{'final':>8}   status")
        for r in results:
            m = r.metrics
            print(f"  {r.dtype:14s}{mpdsp.bits_of(r.dtype):>5}"
                  f"{r.max_pole:>11.5f}{r.notch_depth_db:>10.0f}"
                  f"{m.get('overshoot_pct', float('nan')):>8.1f}"
                  f"{m.get('rise_ms', float('nan')):>8.2f}"
                  f"{m.get('steady_state', float('nan')):>8.4f}"
                  f"   {r.status}")
        print()

    flat = [r for results in by_point.values() for r in results]
    write_summary_csv(outdir / "summary.csv", flat)
    write_step_csv(outdir / "step_responses.csv", flat)
    plotted = plot_summary(outdir / "summary.png", by_point, motor)

    print(f"artifacts -> {outdir}")
    if not plotted:
        print("  (summary.png skipped: matplotlib not installed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
