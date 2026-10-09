# Decimals of accuracy: choosing a number system by its precision profile

> This page reproduces Universal's tutorial
> [Decimals of accuracy](https://github.com/stillwater-sc/universal/blob/main/docs/tutorials/decimals-of-accuracy.md)
> in mp-dsp-python. The narrative, figures and tables are the same. What
> changes is the workflow: in Universal you build a C++ application, run
> it to write one CSV per number system, and then call a plotting script
> once per figure. Here the profiles come straight from
> `mpdsp.precision_profile()`, and one command regenerates everything.

Every number system spends its bits on two things: **range** (how large and how small a value it can hold) and **precision** (how finely it divides the values in between). A precision profile shows where each system puts them. It plots, at every magnitude, how many decimal digits the system keeps.

This tutorial uses the profile to compare five 16-bit number systems. It then zooms into the magnitudes a signal-processing pipeline actually uses, notices that the standard b-posit wastes most of its encodings outside them, and fits a b-posit to the job.

## The measure

For an encoding x whose next larger encoding is x+, the spacing is ulp(x) = x+ - x. Rounding a real number to the nearest encoding makes a relative error of at most ulp / (2|x|). Expressed in decimal digits, that is Gustafson's **decimals of accuracy** (*The End of Error*, 2015):

    decimals(x) = -log10( ulp(x) / (2 |x|) )

A system has 0 decimals at magnitudes it cannot represent, below its smallest positive value or above its largest. (Averaged over an ulp, the relative error is ulp / (4|x|): the same curve, shifted up by log10(2) = 0.30.)

The profiles come from Universal's `precision_profile_of<T>()` in [`universal/utility/precision_profile.hpp`](https://github.com/stillwater-sc/universal/blob/main/include/sw/universal/utility/precision_profile.hpp), bound as `mpdsp.precision_profile()`. For types of up to 16 bits it decodes every encoding, so each curve below has about 32,000 points. Wider types are sampled a few points per binade.

## Generating the data

A profile is one call. `mpdsp.available_profile_types()` lists the types: the tutorial set, plus the scalar behind every `dtype=` configuration.

```python
import mpdsp

fp16 = mpdsp.precision_profile("fp16")
fp16                     # PrecisionProfile('fp16', 31743 points, exhaustive)
fp16.decimals_at(0.5)    # 3.311...
fp16.log2_magnitude      # NumPy arrays: magnitude, log2_magnitude,
fp16.decimals            #   decimals, fraction_bits
```

`mpdsp.plot_precision_profiles()` overlays profiles in the style of Universal's plotting script. `mpdsp.precision` holds the reductions behind the tables. To regenerate every figure and table on this page:

```bash
python scripts/plot_decimals_of_accuracy.py --output-dir docs/img/precision-profiles --format png
```

`PrecisionProfile.write_csv()` writes Universal's CSV layout, so a profile can also be read by Universal's `tools/notebooks/plot_precision_profiles.py`.

## The full range

First, the whole picture: five 16-bit systems across everything they can represent.

```python
types = ["int16", "Q15", "fp16", "lns<16,10>", "bposit<16,6,5>"]
mpdsp.plot_precision_profiles(
    [mpdsp.precision_profile(t) for t in types],
    title="16-bit number systems: decimals of accuracy across the full range")
```

![16-bit number systems across the full range](img/precision-profiles/16bit-full-range.png)

Each system's shape is its design:

- **`int16`** starts at 1, its smallest positive value, and gains precision with magnitude: the spacing is always 1, so a larger value is relatively more precise. It peaks at 4.8 decimals at 2^15.
- **Q15**, `fixpnt<16,15>`, is the same line shifted down by 15 binades. It covers 2^-15 to just below 1.
- **fp16**, `cfloat<16,5>`, has a flat top: 10 fraction bits, 3.3 to 3.6 decimals, in every binade from 2^-14 to 2^15. Below that, its subnormals ramp down to 2^-24.
- **`lns<16,10>`** is perfectly flat at 3.47 decimals. A logarithmic system's relative spacing is the same everywhere, here across 2^-16 to 2^16.
- **`bposit<16,6,5>`**, the standard bounded posit, is a tent. It keeps 2.7 to 3.0 decimals near 1 and steps down by one fraction bit every 32 binades, out to 2^±192. Its regime is capped, so it never drops below 4 fraction bits (1.5 decimals) and stops sharply at its bounded range.

The picture shows the trade every 16-bit format makes. The b-posit spans 384 binades and the integer formats about 15. The formats with the widest range keep the fewest digits.

## Zooming into the region of interest

A digital signal processing (DSP) pipeline does not use 384 binades:

- **The signal band.** Samples are normalized to [-1, 1). The precision just below 1.0 sets the quantization noise floor, at about 6.02 dB of SQNR per bit.
- **The floor.** Small filter coefficients and the tail of a spectrum sit down to about 2^-15 (-90 dBFS).
- **FFT growth.** An N-point FFT grows magnitudes by up to log2(N) bits: 2^8, 2^10 and 2^12 for N = 256, 1024 and 4096.

So the **region of interest** is [2^-15, 2^12]. Below are the same five curves, restricted to it with `xmin` and `xmax`. `dsp=True` shades the signal band, marks the FFT growth and adds an SQNR axis:

```python
mpdsp.plot_precision_profiles(
    [mpdsp.precision_profile(t) for t in types],
    dsp=True, xmin=-32, xmax=24,
    title="16-bit number systems: the region of interest")
```

![16-bit number systems in the region of interest](img/precision-profiles/16bit-region-of-interest.png)

At this scale each binade's sawtooth is visible. Precision is best at the top of a binade and drops by log10(2) where the spacing doubles. In this region:

- **Q15** is the most precise in the signal band, 4.52 decimals in [0.5, 1). It has nothing above 1.0, so an FFT must rescale at every stage. Below the band it falls off a cliff: at 2^-15 it is down to its last bit.
- **`int16`** cannot hold the signal band at all without a scale factor.
- **fp16** holds 3.3 to 3.6 decimals across the region, except its lowest binade: 2^-15 is subnormal, at 3.0. **`lns<16,10>`** is flat at 3.47.
- **`bposit<16,6,5>`** holds 2.7 to 3.0, the least of the floating-point formats. Its range reaches some 180 binades beyond the region on each side, and it pays for that range here.

## The standard b-posit's range is wasted

A 16-bit format has 32,767 positive encodings. How many of them land in the region of interest? `mpdsp.share_in_roi()` counts them:

| type | range (log2) | positive encodings in [2^-15, 2^12] |
|---|---|---|
| `bposit<16,6,5>` | -192 .. 192 | **21.1 %** |
| fp16 | -24 .. 16 | 85.5 % |
| `lns<16,10>` | -16 .. 16 | 84.4 % |

Nearly four in five of the standard b-posit's encodings represent magnitudes between 2^-192 and 2^-15, or between 2^12 and 2^192. A DSP pipeline never produces them. Those encodings are not free. Every bit the regime spends reaching them is a fraction bit that the values in the region do not get.

## Fitting the range

A b-posit's range is set by its maximum regime size rS and its exponent size eS: it spans 2^±(rS 2^eS). The question is which configuration covers [2^-15, 2^12] with the fewest bits spent on range. `mpdsp.range_fit_table()` lists the candidates:

```python
fit = ["bposit<16,6,5>", "bposit<16,6,3>", "bposit<16,6,2>",
       "bposit<16,5,2>", "bposit<16,4,2>", "bposit<16,3,1>"]
rows = mpdsp.range_fit_table([mpdsp.precision_profile(t) for t in fit])
```

| type | range (log2) | encodings in the region | decimals in [0.5, 1) | worst in the region | floor (bits) |
|---|---|---|---|---|---|
| `bposit<16,6,5>` | -192 .. 192 | 21.1 % | 2.71 | 2.71 | 4 |
| `bposit<16,6,3>` | -48 .. 48 | 67.2 % | 3.31 | 3.01 | 6 |
| `bposit<16,6,2>` | -24 .. 24 | 89.8 % | 3.61 | 2.71 | 7 |
| `bposit<16,5,2>` | -20 .. 20 | 89.8 % | 3.61 | 2.71 | 8 |
| **`bposit<16,4,2>`** | **-16 .. 16** | **92.2 %** | **3.61** | **3.01** | **9** |
| `bposit<16,3,1>` | -6 .. 6 | 100.0 % | 3.91 | 0.00 | 11 |

### Reading the table: "worst in the region"

The region is the same for every row: the region of interest, [2^-15, 2^12]. "Worst in the region" is the fewest decimals of accuracy the type keeps at **any** magnitude inside it. It is a guarantee: whatever value the pipeline produces, the type rounds it with at least this much precision. "Floor (bits)" is the same kind of minimum, taken over the type's entire range rather than the region. Together the two columns separate the precision where the workload lives from the precision anywhere at all.

At the start of a binade, decimals and fraction bits are tied by decimals = log10(2) (fraction bits + 1). So the values in the table are a ladder of whole bits, 0.30 decimals (6 dB of SQNR) apart:

| decimals | fraction bits | worst-case relative rounding error |
|---|---|---|
| 2.71 | 8 | 2^-9 ≈ 0.20 % |
| 3.01 | 9 | 2^-10 ≈ 0.10 % |
| 3.31 | 10 | 2^-11 ≈ 0.05 % |
| 3.61 | 11 | 2^-12 ≈ 0.024 % |
| 3.91 | 12 | 2^-13 ≈ 0.012 % |

A worst of 2.71 reads "at least 8 fraction bits everywhere in the region", and 3.01 reads "at least 9".

`mpdsp.min_decimals_over()` computes the column by evaluating each binade start 2^k, for k = -15 .. 12. That is enough. Within a binade the precision is lowest at its start, where the spacing has just doubled, so the binade starts are the bottoms of the sawtooth. Over every encoding in the region, the exhaustive minimum is the same value for every type in the table.

Where each configuration reaches its worst case explains the ranking:

| type | worst | where in [2^-15, 2^12] |
|---|---|---|
| `bposit<16,6,5>` | 2.71 (8 bits) | **everywhere.** With eS = 5 the regime steps only every 32 binades, so the whole region sits on the 2-bit regime around 1.0: 16 - 1 (sign) - 2 (regime) - 5 (exponent) = 8 fraction bits, flat. |
| `bposit<16,6,3>` | 3.01 (9 bits) | the outer binades, 2^-15 .. 2^-9 and 2^8 .. 2^12 |
| `bposit<16,6,2>` | 2.71 (8 bits) | only the extreme edges, 2^-15 .. 2^-13 and 2^12, where the regime has grown to 5 bits |
| `bposit<16,5,2>` | 2.71 (8 bits) | the same edges: rS = 5 permits the same 5-bit regime there |
| `bposit<16,4,2>` | 3.01 (9 bits) | the outer binades, 2^-15 .. 2^-9 and 2^8 .. 2^12, where rS = 4 caps the regime at 4 bits: 16 - 1 - 4 - 2 = 9, never 8 |
| `bposit<16,3,1>` | 0.00 | 2^-15 .. 2^-6 and 2^6 .. 2^12, which lie outside its 2^±6 range |

So `bposit<16,4,2>` matches `bposit<16,5,2>` and `bposit<16,6,2>` near 1.0, at 3.61 decimals in [0.5, 1). Its capped regime then buys one more bit at the edges of the region, where small coefficients and the top of an FFT's growth land.

- **`bposit<16,4,2>` is the best fit.** Its range, 2^±16, just covers the region. It puts 92% of its encodings there and keeps at least 3.01 decimals across all of it. It never drops below 9 fraction bits, against 4 for the standard configuration.
- **`bposit<16,5,2>`** buys four more binades of headroom, up to 2^20 (million-point FFTs), for one bit of floor. It is the choice if the growth budget is uncertain.
- **`bposit<16,3,1>`** shows the limit. All its encodings fall inside the region, but its 2^±6 range does not cover it: 0 decimals at 2^-15 and at 2^10. A fit must cover the region, not just sit inside it.

The fitted b-posits against fp16, the standard b-posit and Q15:

```python
mpdsp.plot_precision_profiles(
    [mpdsp.precision_profile(t) for t in
     ["Q15", "fp16", "bposit<16,6,5>", "bposit<16,5,2>", "bposit<16,4,2>"]],
    dsp=True, xmin=-32, xmax=24,
    title="Fitting the b-posit range to the region of interest")
```

![Fitting the b-posit range to the region of interest](img/precision-profiles/16bit-bposit-fit.png)

The two fitted b-posits coincide from 2^-12 to 2^12, peaking at 3.91 decimals. They part only at the edges of the region, where `bposit<16,4,2>` keeps one more bit.

- **Against the standard `bposit<16,6,5>`**, the fitted configurations are one to three bits ahead at every magnitude in the region: up to 0.9 decimals, or 18 dB of SQNR, near 1.0.
- **Against fp16** the comparison is local, because a b-posit's precision is tapered:

  | magnitudes | `bposit<16,4,2>` against fp16 |
  |---|---|
  | 2^-4 to 2^4 | one bit ahead: 3.61 to 3.91 decimals against 3.31 to 3.61 |
  | 2^-8 to 2^-4, and 2^4 to 2^8 | the same |
  | 2^-15 to 2^-8, and 2^8 to 2^12 | one bit behind (tied at 2^-15, where fp16 is subnormal) |

  The b-posit concentrates its bits around 1.0, where a normalized signal's large samples, and most of its power, live. fp16 spreads them evenly.

That is the paper's point (*Closing the Gap Between Float and Posit Hardware Efficiency*, arXiv 2603.01615): a b-posit with a smaller eS suffices for signal processing, and the range it gives up is range the workload never uses.

## The same at 32 bits

The region of interest does not change with the width, so the same configuration fits:

```python
mpdsp.plot_precision_profiles(
    [mpdsp.precision_profile(t) for t in
     ["Q31", "float", "bposit<32,6,5>", "bposit<32,4,2>"]],
    dsp=True, xmin=-40, xmax=24, title="32-bit: the same fit")
```

![32-bit: the same fit](img/precision-profiles/32bit-bposit-fit.png)

| type | range (log2) | decimals in [0.5, 1) | worst in the region | floor (bits) |
|---|---|---|---|---|
| `float` | -149 .. 128 | 7.22 | 7.22 | 0 (subnormals) |
| `bposit<32,6,5>` | -192 .. 192 | 7.53 | 7.53 | 20 |
| `bposit<32,5,2>` | -20 .. 20 | 8.43 | 7.53 | 24 |
| **`bposit<32,4,2>`** | **-16 .. 16** | **8.43** | **7.83** | **25** |

`bposit<32,4,2>` beats float across the whole region, by 0.6 to 1.2 decimals (2 to 4 bits), and never drops below 25 fraction bits. Q31 is more precise in the signal band, but like Q15 it has no headroom and fades below it.

## Summary

- **Start with the full range.** The profile's shape shows how a system divides its bits between range and precision.
- **Then zoom in.** Restrict the plot to the magnitudes the computation actually produces: the region of interest.
- **Count where the encodings land.** Encodings outside the region are not just unused; the bits that reach them come out of the precision inside it.
- **Size the format to the region.** For a b-posit, choose rS and eS so that 2^±(rS 2^eS) just covers the region. Here `bposit<16,4,2>` is one to three bits ahead of the standard `bposit<16,6,5>` everywhere in the region, and one bit ahead of fp16 around 1.0.

The profile is a *static* view: it says what a type can represent, not what an algorithm achieves in it. Measuring an actual FFT in each of these types, with complex arithmetic, log2(N) growth and per-stage error, is the next step ([#134](https://github.com/stillwater-sc/mp-dsp-python/issues/134)).

Source: Universal's [decimals-of-accuracy tutorial](https://github.com/stillwater-sc/universal/blob/main/docs/tutorials/decimals-of-accuracy.md) and the application behind it, [`applications/mixed-precision/dsp/precision_profiles.cpp`](https://github.com/stillwater-sc/universal/blob/main/applications/mixed-precision/dsp/precision_profiles.cpp).
