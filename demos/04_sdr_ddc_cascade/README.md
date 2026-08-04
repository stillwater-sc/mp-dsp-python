# Demo 4 — SDR digital down-conversion cascade

Design and validate a 4-stage DDC decimation cascade for a 5 GSPS
direct-sampling front end, sweep it against coefficient quantization, and
export the tables in the two formats an FPGA flow actually consumes.

```
RF in (5 GSPS)
  └─ NCO mix to baseband + stage 1 decimate /2      ← mpdsp.DDC
     └─ stage 2 /2 ─ stage 3 /2 ─ stage 4 /2 sharp  ← mpdsp.DecimationChain
        └─ I/Q baseband at 312.5 MSPS
```

## Where this runs, and where it targets

**It runs in simulation, on a few hundred thousand samples.** `mpdsp` does
not stream at 5 GSPS and this demo does not pretend otherwise. Nothing in the
design depends on the absolute rate — filter design, decimation ratios, and
normalized responses are all rate-independent — so a short buffer at a
nominal 5 GSPS proves exactly what a long one would. Sample rates appear in
Hz for labelling only.

**It targets RFSoC-class hardware at multi-GHz.** The claim the demo supports
is: *this cascade design is validated, and it will behave as measured on a
ZCU208 at 5 GSPS if implemented faithfully.* The coefficient files are the
handoff.

## Run it

```bash
pip install -e '.[plot]'
python demos/04_sdr_ddc_cascade/design.py                 # design + validate + sweep dtypes
python demos/04_sdr_ddc_cascade/quantize_coefficients.py  # pick the coefficient width
python demos/04_sdr_ddc_cascade/emit_coeffs_vivado.py     # .coe for the FIR Compiler
python demos/04_sdr_ddc_cascade/emit_coeffs_verilog.py    # .hex for $readmemh
```

Artifacts (gitignored): `summary.csv`, `summary.png`, `quantization.csv`,
`quantization.png`, `vivado/*.coe`, `verilog/*.hex`, `verilog/cascade_params.vh`.

## The cascade

| Stage | Taps | Cutoff (own rate) | Rate out |
|---|---:|---:|---:|
| 1 — mix + decimate | 51 | 0.25 | 2.5 GSPS |
| 2 — decimate | 35 | 0.25 | 1.25 GSPS |
| 3 — decimate | 27 | 0.25 | 625 MSPS |
| 4 — sharp | 95 | 0.23 | 312.5 MSPS |

208 taps total, decimation 16. Measured composite: **0.001 dB passband
ripple**, **89.9 dB alias rejection** — against a spec of 0.5 dB and 60 dB.

### Alias rejection is not stopband attenuation

The number that matters for a decimator is the worst level over the bands
that *fold onto* the passband at the output rate — the intervals
`k/16 ± f_pass` for k = 1…8. A response can look excellent between those
bands and still alias badly, so `design.py` measures them explicitly rather
than quoting a single stopband figure. The test signal includes an interferer
placed at exactly one output-rate step from the carrier for the same reason:
it is invisible near the wanted band in the input spectrum and lands directly
on top of the signal after decimation if the cascade under-performs.

## Precision results

Datapath arithmetic, SNR of the baseband output vs. the `reference` run:

| dtype | sample bits | output SNR |
|---|---:|---:|
| `posit_full` | 16 | **84.6 dB** |
| `gpu_baseline` | 32 | 60.5 dB |
| `half` | 16 | 57.2 dB |
| `fpga_fixed` | 16 | 54.5 dB |
| `cf24` | 24 | 24.5 dB |

Coefficient quantization, fixed point with a per-stage power-of-two scale:

| bits | alias rejection | ROM | spec |
|---:|---:|---:|:--|
| 8 | 35.7 dB | 208 B | fail |
| 12 | 54.2 dB | 312 B | fail |
| **14** | **68.1 dB** | 364 B | **pass** |
| 18 | 90.1 dB | 468 B | pass |
| 24 | 89.9 dB | 624 B | pass |

Three things worth reading off these tables:

**16-bit posit beats 32-bit float here.** `posit_full` scores 84.6 dB against
`gpu_baseline`'s 60.5 — at half the sample-path width. A decimating cascade
spends most of its time on small residuals after filtering, which is exactly
where posit's tapered accuracy concentrates precision and where a fixed
exponent field wastes it.

**24 bits is not automatically better than 16.** `cf24` lands last at 24.5 dB
— worse than every 16-bit option. `cfloat<24,5>` carries only 5 exponent
bits, and the cascade's dynamic range exceeds what that supports; the width
goes into mantissa that the signal never needs while the exponent clips.
Bit count is not a quality ordering.

**Coefficients quantize far better than the datapath.** 14-bit coefficients
already clear the spec, and past 18 bits the curve is flat — the design stops
improving because the unquantized cascade itself tops out near 90 dB. Widening
the coefficient ROM past that buys nothing; widening the datapath still does.

Note the gap between `fpga_fixed` (fails at 56.0 dB) and the plain 16-bit
fixed-point sweep (passes at 81.5 dB). Both are 16-bit fixed point. The
difference is scaling: `fpga_fixed`'s sample scalar is a fixed `fixpnt<16,12>`
Q-format, while the sweep fits a power-of-two scale per stage. That is why
real FPGA designs scale per stage, and it is visible here as 25 dB.

## Two upstream problems this demo ran into

Both are filed as **#117**; the demo works around them and says where.

**`design_halfband` and `remez_lowpass` do not meet spec.** The halfband
designer tops out near 21 dB of stopband attenuation and gets *worse* with
more taps (127 taps at `transition_width=0.15` measured −24.7 dB, i.e. the
stopband is above the passband). `remez_lowpass` carries a fixed ~2.4 dB
passband ripple and a 1.25 DC gain regardless of length. Both are Remez-based
and look like a convergence failure. The cascade therefore uses
`fir_lowpass` with a Kaiser window, which measures 88 dB stopband and 0.00 dB
ripple at 51 taps — the window method is fine.

**`NCO` / `DDC` overflow on absolute rates.** Both take frequency and sample
rate as the configuration's state scalar and divide only afterwards, so GHz
values overflow narrow state types before the division can bring them back
into range. At 1.2 GHz / 5 GSPS: `fpga_fixed` raises "sample_rate must be
positive", and `cf24` and `half` **silently return a NaN phase increment**
and carry on. This demo passes normalized rates (sample rate 1.0, carrier as
a fraction), which is well-defined for every dtype and means the same thing.

## FPGA handoff

`emit_coeffs_vivado.py` writes one `.coe` per stage for the Vivado FIR
Compiler or Block Memory Generator. `emit_coeffs_verilog.py` writes the same
coefficients as `$readmemh` hex plus a `cascade_params.vh` carrying widths,
tap counts, and scale exponents:

```verilog
`include "cascade_params.vh"

reg signed [`COEFF_WIDTH-1:0] stage1_taps [0:`STAGE1_TAPS-1];
initial $readmemh("stage1_decim2.hex", stage1_taps);
// accumulator >>> `STAGE1_FRAC to undo the coefficient scale
```

Default width is 18 bits — the native coefficient port on a DSP48E slice, so
each tap costs one slice. 14 bits is the narrowest that meets spec; use it
only if coefficient ROM is genuinely the binding constraint, since it gives
up 22 dB of margin to save 104 bytes.

Hex is written as explicit two's complement at the declared width.
`$readmemh` reads raw bit patterns, so a negative coefficient must appear as
its unsigned image or the filter's negative taps load as large positives — a
silent and confusing simulation failure.

### Target devkits

| Target | ADC rate | Role | Notes |
|---|---|---|---|
| **Xilinx RFSoC ZCU208** | 5 GSPS integrated | Production scale (~$7k) | What the cascade is dimensioned for. The `.coe` files drop into the FIR Compiler as-is. |
| **Xilinx RFSoC ZCU216** | 8 GSPS × 8 ch | High channel count (~$10k) | Same cascade per channel; the 8 GSPS front end wants one more decimation stage for the same output rate. |
| ADI AD9361 / AD9371 eval | ~245 MSPS | Educational | Not multi-GHz. The cascade shape holds, the rates do not. |
| **ADI ADALM-PLUTO** | 61 MSPS | **Python-accessible** (~$150) | The only board here you can drive from Python, via `pyadi-iio`. Run the same design at 61 MSPS to compare simulated against captured. |

The precision question each target asks differs: the RFSoC parts ask "how
many coefficient bits" (14 minimum, 18 free), while a soft-float or posit
core asks "which arithmetic" — and there the answer is that 16-bit posit
outperforms 32-bit float on this workload.

## Files

```
demos/04_sdr_ddc_cascade/
├── README.md
├── simulate.py                # narrowband RF in wideband noise + alias trap
├── design.py                  # cascade design, spec validation, dtype sweep
├── quantize_coefficients.py   # coefficient width and arithmetic sweeps
├── emit_coeffs_vivado.py      # Xilinx .coe
├── emit_coeffs_verilog.py     # $readmemh hex + cascade_params.vh
└── artifacts/                 # gitignored
```

Tests are in `tests/test_demos.py`.
