# Demo 2 — Motor current loop + resonance notch

A BLDC current-loop controller — PI plus a resonance notch — designed,
closed, and swept across seven arithmetic configurations, then exported as a
C header for a motor-control MCU.

```
i_ref ──(+)──> PI ──> notch ──> inverter ──> [ 1/(Ls+R) ] ──┬──> i_meas
        (−)                                                  │
         └───────────────────────────────────────────────────┘
```

```bash
pip install -e '.[plot]'
python demos/02_motor_servo/design.py
python demos/02_motor_servo/emit_c_header.py
```

Artifacts (gitignored): `summary.csv`, `step_responses.csv`, `summary.png`,
`current_loop.h`.

## The plant

Per-phase electrical model of a 2207-class racing outrunner: R = 55 mΩ,
L = 30 µH, so the electrical time constant is 545 µs — about 11 samples at a
20 kHz current loop. Discretized with ZOH rather than bilinear, because the
inverter genuinely holds its output constant across the PWM period, and ZOH
avoids the frequency warp that would misplace the very pole the controller is
about to cancel.

Back-EMF is a disturbance, not a state. At current-loop rates the electrical
time constant is sub-millisecond while the mechanical one is tens of
milliseconds, so speed is effectively constant within any current transient.
That separation is what lets a current loop be tuned without a mechanical
model.

The PI zero is placed on the plant pole, making the compensated loop a pure
integrator whose crossover is set by `Kp` alone. That choice is also what
gives this demo something to say: **the cancellation is only as good as the
precision of that coefficient.**

## Two design points, and why that is the result

| | crossover | notch | rationale |
|---|---:|---:|---|
| **conservative** | 1200 Hz | 2500 Hz, 1.0 oct | notch *above* crossover — a structural or PWM-related mode |
| **aggressive** | 1500 Hz | 1000 Hz, 0.5 oct | notch *inside* the loop — a low-frequency blade-pass mode |

### Conservative: precision costs notch depth, not stability

| dtype | bits | max&#124;pole&#124; | notch depth | overshoot | status |
|---|---:|---:|---:|---:|---|
| `reference` | 64 | 0.91241 | −600 dB | 19.6% | ok |
| `gpu_baseline` | 32 | 0.91241 | −137 dB | 19.6% | ok |
| `cf24` | 24 | 0.91241 | −116 dB | 19.6% | ok |
| `posit_full` | 16 | 0.91263 | −68 dB | 19.5% | ok |
| `half` | 16 | 0.91230 | −68 dB | 19.6% | ok |
| `fpga_fixed` | 16 | 0.91236 | −68 dB | 19.6% | ok |
| `posit_8_2` | 8 | 0.93693 | **−16 dB** | 14.8% | ok |

Every type is stable, including 8-bit. What degrades is the notch: from a
numerical null to 16 dB of attenuation. If the resonance you are notching is
20 dB above where you need it, 8-bit coefficients simply do not solve your
problem — but they do not destabilize the loop either.

### Aggressive: the same 8-bit type destabilizes it

| dtype | bits | max&#124;pole&#124; | notch depth | overshoot | status |
|---|---:|---:|---:|---:|---|
| `reference` | 64 | 0.98037 | −296 dB | 41.4% | ok |
| `gpu_baseline` | 32 | 0.98037 | −131 dB | 41.4% | ok |
| `cf24` | 24 | 0.98037 | −93 dB | 41.4% | ok |
| `posit_full` | 16 | 0.98041 | −66 dB | 41.4% | ok |
| `half` | 16 | 0.98052 | −51 dB | 41.5% | ok |
| `fpga_fixed` | 16 | 0.98035 | −66 dB | 41.4% | ok |
| `posit_8_2` | 8 | **1.00000** | −7 dB | — | **UNSTABLE** |

Same arithmetic, same plant — only the design margin differs. That is the
actionable question: not *"is 8 bits enough"* but *"how much margin does
8 bits cost me"*.

### The failure mechanism is specific

At `posit_8_2` the notch denominator `[1, −1.45309, 0.52786]` rounds to
`[1, −1.5, 0.5]`, which factors as poles at **z = 1.0 and z = 0.5**. The
numerator `[0.76393, −1.45309, 0.76393]` rounds to `0.75·[1, −2, 1]` — a
double zero at DC.

So the quantized "notch" is not a blunt notch. It is an **integrator**:
quantization moved a pole exactly onto the unit circle. Combined with the PI
controller's own integrator, the loop acquires a double integrator and the
step response collapses to zero current.

This is precisely what a coefficient-sensitivity number cannot tell you.
`pole_displacement` reports how far poles moved; it does not report that one
of them landed on the boundary and changed what the filter *is*.

## What is quantized, and what is not

Controller and notch **coefficients** are round-tripped through the target
type with `project_onto`; the simulation runs in double. That is the
deployment question — what does storing these coefficients in T do to my
closed loop — and it matches the model `IIRFilter.frequency_response(dtype=)`
and `pole_displacement(dtype)` use.

It does **not** model the arithmetic of the loop's own state updates, which
on a fixed-point MCU matters too. A full treatment needs the loop run
sample-by-sample at the target type, which `IIRFilter.process` cannot do
today — it builds fresh state per call. Stated rather than glossed, since a
fixed-point implementer will hit that second effect and should know this
demo did not measure it.

## Hardware port

`emit_c_header.py` writes the PI difference equation and the notch biquad as
C89-compatible constants with no includes — verified clean under
`-std=c89/c99` and C++ with `-Wall -Wextra -pedantic -Werror`.

```c
#include "current_loop.h"
/* once per current sample */
err = i_ref - i_meas;
u   = u_prev + PI_B0*err + PI_B1*err_prev;      /* PI  */
y   = NOTCH_B0*u + NOTCH_B1*u1 + NOTCH_B2*u2
    - NOTCH_A1*y1 - NOTCH_A2*y2;                 /* notch */
```

`PI_B1` is `−exp(−T/τ)` and therefore depends on the winding. **Re-run the
exporter if R or L change** — a controller tuned for a different motor does
not merely perform worse, it misplaces the cancellation and leaves a slow
pole-zero doublet in the response.

The exporter refuses to emit an unstable configuration:

```
$ python emit_c_header.py --design-point aggressive --dtype posit_8_2
error: closed loop is unstable at 'posit_8_2' for the aggressive design point
(max|pole| = 1.00000) — refusing to export. Quantizing the notch to this type
moves a pole onto the unit circle; shipping these coefficients would put an
oscillator in the current loop.
```

### Target devkits

| Target | Role | Cost | Notes |
|---|---|---|---|
| **TI LAUNCHXL-F28379D** + **BOOSTXL-DRV8323RS** | Single-motor BLDC with FOC | ~$100 | The C2000 has a hardware FPU; `gpu_baseline` is the honest simulation target and both design points are comfortable there. The header drops into Code Composer Studio unchanged. |
| **STM32 Nucleo-G431RB** + **X-NUCLEO-IHM16M1** | Racing-ESC analogue | ~$50 | Cortex-M4F, single-precision FPU — same `gpu_baseline` row. Closest thing here to what a real ESC runs. |

## Interceptor-drone applicability

The controller pattern carries over, with caveats below:

| Loop | Rate | mpdsp fit |
|---|---|---|
| Current-sense PI + anti-alias LP | 20–50 kHz | `IIRFilter` (Butterworth LP) |
| Notch at PWM carrier / blade-pass | same rate | `rbj_bandstop` |
| Back-EMF observer (hybrid sensing) | same rate | `KalmanFilter` |
| Velocity loop | 1–10 kHz | `IIRFilter` |

**The blade-pass notch is the aggressive design point, and that is the
point.** A 2-blade prop at 30 000 RPM puts blade-pass at 1 kHz — inside any
current loop worth having. That is exactly the case where 8-bit coefficients
destabilized the loop above. A PWM-carrier notch, by contrast, sits far above
crossover and is the conservative point.

### Honest caveats

- **This is not an ESC drop-in.** Real racing firmware (BLHeli lineage) runs
  on STM32F4/G4-class MCUs with hand-tuned assembly commutation, drives four
  motors, and does sensorless startup. This demo is single-motor,
  single-loop, and educational.
- **The C2000 target is an analogue, not the deployment part.** It is what
  you would prototype on, not what ships in a 5-inch quad.
- **Sensorless FOC back-EMF estimation wants an EKF.** When this issue was
  filed that was unavailable; `mpdsp.ExtendedKalmanFilter` and
  `UnscentedKalmanFilter` are bound now, so that extension is open to
  anyone who wants it. It is out of scope here — this demo stays on the
  linear current loop.
- **No thermal, no saturation, no dead-time.** All three matter on real
  hardware and none is modelled.

## Files

```
demos/02_motor_servo/
├── README.md
├── simulate.py        # BLDC electrical plant, ZOH discretization, closed loop
├── design.py          # PI tuning + notch + 7-dtype quantization sweep
├── emit_c_header.py   # controller + notch coefficients as C
└── artifacts/         # gitignored
```

Tests are in `tests/test_demos.py`.
