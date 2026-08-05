"""BLDC electrical plant and closed-loop simulation for the current loop.

The plant is the per-phase electrical model of a small racing BLDC — a
series R-L driven by the inverter, with back-EMF as a disturbance:

    V(s) - E(s) = (L s + R) I(s)      =>   I/V = 1 / (L s + R)

Back-EMF is treated as a disturbance rather than modelled dynamically. At
current-loop rates the electrical time constant is a few hundred
microseconds while the mechanical one is tens of milliseconds, so within any
one current-loop transient the speed — and therefore the back-EMF — is
effectively constant. That separation is what lets a current loop be tuned
without a mechanical model, and it is why this demo can stay in the
electrical domain.

    python demos/02_motor_servo/simulate.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

# Current-loop rate. 20 kHz is a common FOC choice: fast enough to control a
# ~500 us electrical time constant, slow enough to leave MCU headroom for the
# rest of the loop.
SAMPLE_RATE_HZ = 20000.0


@dataclass(frozen=True)
class Motor:
    """Per-phase electrical parameters of a small racing BLDC.

    Defaults are representative of a 2207-class outrunner: milliohms of
    winding resistance and tens of microhenries of inductance. The exact
    numbers matter less than their ratio — the electrical time constant
    L/R is what sets how fast the current loop can be.
    """

    resistance_ohm: float = 0.055
    inductance_h: float = 30e-6
    # Back-EMF constant, V per rad/s. Used only to report the disturbance
    # scale; the loop is tuned without it, per the module docstring.
    ke_v_per_rad_s: float = 0.0035

    @property
    def electrical_tau_s(self) -> float:
        return self.inductance_h / self.resistance_ohm

    def discrete_plant(self, sample_rate_hz: float = SAMPLE_RATE_HZ
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Zero-order-hold discretization of 1/(Ls + R).

        Returns (numerator, denominator) in ascending powers of z^-1.

            G(z) = (1/R)(1 - a) / (1 - a z^-1),   a = exp(-T / tau)

        ZOH rather than bilinear: the inverter genuinely holds its output
        constant across the PWM period, so ZOH is the physically faithful
        discretization here, and it avoids the bilinear frequency warp that
        would misplace the plant pole the controller is about to cancel.
        """
        period = 1.0 / sample_rate_hz
        pole = float(np.exp(-period / self.electrical_tau_s))
        numerator = np.array([(1.0 / self.resistance_ohm) * (1.0 - pole)])
        denominator = np.array([1.0, -pole])
        return numerator, denominator

    def plant_pole(self, sample_rate_hz: float = SAMPLE_RATE_HZ) -> float:
        return float(np.exp(-1.0 / (sample_rate_hz * self.electrical_tau_s)))


def series(*sections: tuple[np.ndarray, np.ndarray]
           ) -> tuple[np.ndarray, np.ndarray]:
    """Cascade transfer functions by polynomial convolution."""
    numerator = np.array([1.0])
    denominator = np.array([1.0])
    for num, den in sections:
        numerator = np.convolve(numerator, num)
        denominator = np.convolve(denominator, den)
    return numerator, denominator


def close_loop(forward_num: np.ndarray, forward_den: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Unity-feedback closed loop: T = L / (1 + L).

    The denominators are zero-padded to a common length before adding —
    getting that wrong silently shifts the polynomial by a sample and
    produces a plausible-looking but entirely different system.
    """
    width = max(len(forward_num), len(forward_den))
    num = np.pad(forward_num, (width - len(forward_num), 0))
    den = np.pad(forward_den, (width - len(forward_den), 0))
    return num, den + num


def step_response(num: np.ndarray, den: np.ndarray,
                  num_samples: int = 1200) -> np.ndarray:
    """Unit-step response of a z-domain transfer function.

    Direct-form difference equation. Bails out early once the output stops
    being finite so a divergent design produces a short trace rather than
    a wall of inf, and the caller can see where it left.
    """
    out = np.zeros(num_samples)
    # Divergence is an expected outcome here, not an error: an unstable
    # quantized design is exactly what the demo is looking for. Overflow on
    # the way to inf is part of that, so it is silenced rather than left to
    # print a warning that reads like a bug.
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(num_samples):
            acc = float(np.sum(num[: k + 1]))      # unit step input
            for j in range(1, min(len(den), k + 1)):
                acc -= den[j] * out[k - j]
            out[k] = acc / den[0]
            if not np.isfinite(out[k]):
                out[k:] = np.nan
                break
    return out


def closed_loop_poles(den: np.ndarray) -> np.ndarray:
    return np.roots(den)


def response_metrics(step: np.ndarray, sample_rate_hz: float,
                     target: float = 1.0) -> dict:
    """Overshoot, rise time, settling time and steady-state error."""
    if not np.all(np.isfinite(step)):
        return {"overshoot_pct": float("nan"), "rise_ms": float("nan"),
                "settle_ms": float("nan"), "steady_state": float("nan"),
                "diverged": True}

    peak = float(np.max(step))
    reached = np.nonzero(step >= 0.9 * target)[0]
    rise_ms = (reached[0] / sample_rate_hz * 1e3) if len(reached) else float("nan")

    # Settling: last excursion outside +/-2% of target.
    outside = np.nonzero(np.abs(step - target) > 0.02 * target)[0]
    settle_ms = ((outside[-1] + 1) / sample_rate_hz * 1e3
                 if len(outside) and outside[-1] + 1 < len(step)
                 else float("nan"))

    return {
        "overshoot_pct": (peak - target) / target * 100.0,
        "rise_ms": rise_ms,
        "settle_ms": settle_ms,
        "steady_state": float(step[-1]),
        "diverged": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=float, default=SAMPLE_RATE_HZ)
    args = parser.parse_args()

    motor = Motor()
    num, den = motor.discrete_plant(args.sample_rate)
    pole = motor.plant_pole(args.sample_rate)

    print(f"BLDC per-phase electrical plant @ {args.sample_rate/1e3:.0f} kHz")
    print(f"  R = {motor.resistance_ohm*1e3:.0f} mOhm, "
          f"L = {motor.inductance_h*1e6:.0f} uH, "
          f"Ke = {motor.ke_v_per_rad_s*1e3:.1f} mV/(rad/s)")
    print(f"  electrical tau = {motor.electrical_tau_s*1e6:.0f} us "
          f"= {motor.electrical_tau_s*args.sample_rate:.1f} samples")
    print(f"  discrete pole  = {pole:.5f}")
    print(f"  DC gain        = {num.sum()/den.sum():.2f} A/V "
          f"(1/R = {1/motor.resistance_ohm:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
