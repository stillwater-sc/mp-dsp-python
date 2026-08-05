# Demo 3 — Active vibration cancellation

A feedforward active-noise-control loop trained at every arithmetic
configuration, showing how precision affects an adaptive filter's ability to
converge, to hold, and to re-converge when the machine changes speed.

```
machine ──> reference accelerometer ──> x(n) ──> adaptive filter W(z) ──> y(n)
   │                                                                       │
   └──── primary path P(z) ────> error accelerometer ──> d(n)              │
                                          e(n) = d(n) − y(n)  <────────────┘
                                          (actuator emits −y)
```

The filter learns `P(z)`; whatever it fails to reproduce survives as residual
vibration. Run it:

```bash
pip install -e '.[plot]'
python demos/03_vibration_cancellation/design.py
python demos/03_vibration_cancellation/emit_c_header.py
```

Artifacts (gitignored): `summary.csv`, `summary.png`, `canceller.h`.

## Adaptive filters fail differently

A quantized fixed filter gets a slightly wrong frequency response. A quantized
*adaptive* filter can stop converging, converge to the wrong answer, or
diverge — and in a cancellation loop that last case does not mean "no
cancellation". It means the actuator is now driving the structure. A SQNR
number cannot express that difference; a residual trace can, which is why
this demo plots one and reports **AMPLIFIED** as a distinct outcome from
**DIVERGED**.

## Results

Sensor noise puts a 30 dB floor on achievable cancellation, so `reference`
converging to ~29.5 dB *is* the physical limit, and every other number should
be read as a shortfall against it.

| algorithm | dtype | bits | before speed change | after | status |
|---|---|---:|---:|---:|---|
| LMS | `reference` | 64 | 29.4 dB | 29.7 dB | ok |
| LMS | `gpu_baseline` | 32 | 29.4 | 29.7 | ok |
| LMS | `posit_full` | 16 | 29.4 | 29.7 | ok |
| LMS | `cf24` | 24 | 29.4 | 29.7 | ok |
| LMS | `half` | 16 | 29.4 | 29.8 | ok |
| LMS | `posit_8_2` | 8 | 8.6 | 9.5 | ok |
| NLMS | `posit_8_2` | 8 | 14.8 | 15.8 | ok |
| **RLS** | `reference` | 64 | 29.9 | 30.2 | ok |
| **RLS** | `gpu_baseline` | 32 | 19.1 | **−33.4** | **AMPLIFIED** |
| **RLS** | `posit_full` | 16 | 29.9 | **−34.8** | **AMPLIFIED** |
| **RLS** | `cf24` | 24 | — | — | **DIVERGED** at 3.9 s |
| **RLS** | `half` | 16 | — | — | **DIVERGED** at 0.35 s |
| **RLS** | `posit_8_2` | 8 | — | — | **DIVERGED** at 0.02 s |

Three things worth reading off this.

**The gradient-descent filters degrade gracefully; RLS does not.** LMS and
NLMS hold the full 29.7 dB down to 16-bit `half` and only fall off at 8-bit
posit — and even there they still cancel (9.5 dB and 15.8 dB), they just
cancel less. NLMS is notably tougher than LMS at 8 bits (15.8 vs 9.5), which
is the normalization earning its keep: dividing by the input power keeps the
effective step size in range when the representable dynamic range is tiny.

**The speed change is the trigger, not the precision alone.** `posit_full`
runs RLS at a flawless 29.9 dB for the first three seconds — indistinguishable
from double — and then amplifies by 34.8 dB once the machine speed steps.
`cf24` diverges at 3.9 s, just after the change at 3.0 s. Only `half` and
`posit_8_2` fail during steady state. The practical reading is uncomfortable:
**an RLS canceller can pass a bench test on a steady machine and destroy
itself the first time the machine changes speed.**

**32-bit float is not enough for RLS here.** `gpu_baseline` is already down to
19.1 dB before the change and amplifies after it. If your instinct is that
single precision is always plenty for control-rate DSP, this is the
counter-example.

The `notebooks/06_estimation` finding that RLS loses P-matrix symmetry at
narrow precision reproduces exactly, and the divergence-onset column above is
that finding turned into a measurement.

## A confound worth knowing about

The RLS forgetting factor here is **0.9995**, not the more usual 0.999, and
that is deliberate.

At 0.999 this problem suffers *covariance windup*: `P` gradually loses
positive-definiteness through accumulated rounding, and the filter diverges
**even in double** given enough samples — measured, `reference` amplifies by
30.9 dB over a 40 000-sample run, with a peak residual 430× the disturbance.
Presenting that as a precision result would be a lie, because the algorithm
is unstable at every precision.

At 0.9995 double is stable indefinitely (30.0 dB, peak residual 0.22×), which
leaves precision as the only variable and makes the failures above real ones.
The memory is still only ~1/(1−λ) = 2000 samples = 0.5 s, short enough to
track the speed change.

If you are porting RLS to hardware, the lesson is the same as the demo's:
check that your λ is stable in double for far longer than your test run
before blaming precision for anything.

## Hardware port

`emit_c_header.py` writes the converged taps plus the adaptation parameters
as a C89-compatible header with no includes — verified clean under
`-std=c89/c99` and C++ with `-Wall -Wextra -pedantic -Werror`.

```c
#include "canceller.h"
/* y[n] = sum w[k] * x[n-k];  drive the actuator with -y[n] */
```

The header carries the step size as well as the taps on purpose. The taps
alone are a *fixed* filter that cancels the disturbance it was trained on;
a real installation keeps adapting, so the taps are a warm start and the
parameters are how the target continues from there. Shipping only the taps
silently converts an adaptive canceller into a fixed one — which works right
up until the machine changes speed.

The exporter **refuses** to write a header from a diverged or amplifying run:

```
$ python emit_c_header.py --algorithm RLS --dtype posit_full
error: RLS at 'posit_full' amplified the disturbance by 34.8 dB — refusing
to export. These weights would drive the structure, not cancel it.
```

### Target devkits

| Target | Capability | Python on target | Cost | Notes |
|---|---|---|---|---|
| **ADI SHARC ADZS-SC589-MINI** | 1 GHz SHARC+, multi-channel codec | No — C/C++ | ~$500 | The realistic target. SHARC is natively 32/40-bit float, so simulate with `gpu_baseline` first — and note what that row says about RLS. For LMS/NLMS it is comfortably sufficient. |
| **TI C6000 + external codec** | Classic fixed/float DSP, aging | No | varies | Fixed-point parts want the `fpga_fixed` row as their proxy; add it with `--dtypes fpga_fixed`. |
| **Bela.io + BeagleBone Black** | Linux + Xenomai, sub-ms latency | **Yes** | ~$200 | The only "Python on target" path — `mpdsp` runs on it, so you can A/B this exact script against a C port on the same board. Start here if you are iterating on the design rather than shipping it. |

Only Bela runs Python on the target. For the other two the flow is: design
and validate here, export the header, port the update loop by hand, and use
the dtype rows above to predict what the target's arithmetic will do to it.

## Files

```
demos/03_vibration_cancellation/
├── README.md
├── simulate.py        # disturbance: 3 tones + broadband, primary path, speed change
├── design.py          # LMS / NLMS / RLS across dtypes, convergence + drift metrics
├── emit_c_header.py   # converged taps + adaptation parameters as C
└── artifacts/         # gitignored
```

Tests are in `tests/test_demos.py`.
