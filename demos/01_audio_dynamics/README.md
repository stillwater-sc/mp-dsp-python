# Demo 1 — Audio dynamics processor

A mastering-style chain — three-band EQ, compressor, peak limiter — run at
every arithmetic configuration `mpdsp` exposes, so you can **hear** what
precision loss does rather than only read it off a SQNR table.

```
in → lowshelf(120 Hz) → peaking(1 kHz) → highshelf(8 kHz)
   → compressor(−18 dB, 4:1, +10 dB makeup)
   → peak limiter(0.95 ceiling, 5 ms lookahead)
   → out
```

48 kHz stereo. Each channel is processed independently, which is what a real
two-channel chain does with linked-free dynamics.

## Run it

```bash
pip install -e '.[plot]'
python demos/01_audio_dynamics/design.py
```

Writes to `artifacts/` (gitignored):

| File | What |
|---|---|
| `input.wav` | the source clip, for A/B |
| `output_<dtype>.wav` | **the headline deliverable** — one listenable render per arithmetic type |
| `summary.csv` | SQNR, peak, clipped-sample count, max gain reduction |
| `summary.png` | input/output spectrograms, gain-reduction traces, SQNR vs. bit width |

Useful flags:

```bash
python design.py --input my_clip.wav      # process a real WAV instead
python design.py --all-dtypes             # all 18 configs, not the default 7
python design.py --coeff-dtype half       # sweep design-time precision instead
python design.py --duration 8 --no-wav    # longer clip, metrics only
python emit_c_header.py                   # export coefficients as C
```

## What the numbers say

A representative run (1 s synthesized clip, default settings):

| dtype | sample bits | SQNR vs reference | output peak | clipped | max gain reduction |
|---|---:|---:|---:|---:|---:|
| `reference` | 64 | — | 0.992 | 0 | −5.69 dB |
| `gpu_baseline` | 32 | 73.8 dB | 0.992 | 0 | −5.69 dB |
| `posit_full` | 16 | 74.1 dB | 0.992 | 0 | −5.69 dB |
| `cf24` | 24 | 48.3 dB | 0.992 | 0 | −5.70 dB |
| `ml_hw` | 16 | 12.2 dB | 0.999 | 0 | −5.62 dB |
| `half` | 16 | 0.4 dB | 1.031 | 12 | −4.32 dB |
| `posit_8_2` | 8 | −38.6 dB | 24.0 | 95421 | 0.00 dB |

Three things worth noticing, and they are the reason the demo exists:

**Bit width is not the whole story.** `posit_full` (16-bit sample path) scores
74 dB — better than `cf24` at 24 bits, and 74 dB better than `half` at the
same 16 bits. Posit's tapered accuracy sits where audio signals actually
live; IEEE half's uniform exponent spacing does not.

**Failure is not graceful.** `half` does not merely add noise — it degrades
the limiter enough that peaks escape the ceiling and 12 samples clip.
`posit_8_2` collapses entirely: gain reduction reads 0.00 dB because the
envelope arithmetic has lost the resolution to track anything, so the limiter
stops limiting and the output runs 24× over full scale. A SQNR number alone
would have called that "−38 dB"; the WAV tells you the chain is *broken*.

**Listen to the WAVs.** `output_ml_hw.wav` at 12 dB SQNR still sounds like
music with a noise floor. `output_half.wav` at 0.4 dB is audibly wrecked. The
gap between those two is the entire argument for choosing a number system
rather than a bit count.

## What runs at the selected dtype

The three EQ biquads, the compressor, and the limiter's envelope follower all
run their arithmetic at the chosen dtype.

Two steps do not, because no bound primitive covers them: the peaking band's
parallel mix (`x + (g−1)·bandpass(x)`) and the limiter's final gain multiply
(`x · gain`), both float64 NumPy. The filters and followers feeding them
*are* quantized, so the measurement is close — but a hardware port would run
these at the target type too, and the numbers above are optimistic by that
small margin. Stated rather than buried, because a demo about precision cost
should not quietly cheat on precision.

Coefficients are designed in `double`, the library default and the right
choice for a deployed EQ. `--coeff-dtype` sweeps that axis separately.

## Hardware port

`emit_c_header.py` writes `artifacts/chain.h` — three biquads plus the
compressor and limiter parameters, as C89-compatible constants with no
includes. It compiles clean under `-std=c89/c99/c11` and C++ with
`-Wall -Wextra -pedantic -Werror`.

```c
#include "chain.h"

/* y = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2], a0 = 1 */
for (i = 0; i < CHAIN_NUM_BIQUADS; ++i) { /* ... */ }
```

Mind the mid band: it is a **parallel** path,
`y = x + (CHAIN_MID_GAIN − 1)·biquad(mid, x)`. In series it would strip
everything outside the mid band.

### Target devkits

| Target | Capability | Python on-target | Cost | Port notes |
|---|---|---|---|---|
| **ADI SHARC ADSP-21593 EZ-KIT** | 1 GHz SHARC+, integrated codec | No — C/C++ | ~$600 | The closest match to what this demo models. SHARC is natively 32/40-bit float, so `gpu_baseline` is the honest simulation target and its 74 dB SQNR is what you should expect to measure. Drop `chain.h` in and implement the biquads in the CCES DSP library's direct-form-II. |
| **ADI SigmaDSP EVAL-ADAU1452** | Fixed-point DSP, SigmaStudio graphical flow | No | ~$400 | Fixed-point 28.0 internal. Simulate with `fpga_fixed` first — if that run clips or loses the limiter, the SigmaStudio build will too. The EQ ports as a SigmaStudio biquad block; the compressor/limiter map to its dynamics blocks with these parameters. |
| **Bela.io + BeagleBone Black** | Linux + Xenomai, sub-millisecond latency | **Yes** | ~$200 | The only target that runs `mpdsp` itself, so you can A/B Python against C on the same board. Best starting point if you want to iterate on the design rather than ship it. |

The precision question each target actually asks is different: SHARC asks
"is 32-bit float enough" (yes, 74 dB), SigmaDSP asks "does fixed-point hold
the limiter" (run `--dtypes fpga_fixed` and look at the clipped column), and
Bela asks nothing — it has the headroom to run `reference` and be done.

## Files

```
demos/01_audio_dynamics/
├── README.md          # this file
├── simulate.py        # synthesize a test clip, or load/save WAV
├── design.py          # design the chain, run it across dtypes, write artifacts
├── emit_c_header.py   # export coefficients + parameters as C
└── artifacts/         # gitignored — WAVs, PNG, CSV, chain.h
```

Tests live in `tests/test_demos.py`: the chain runs at every dtype, the
emitted header compiles, and the WAV round-trip holds.
