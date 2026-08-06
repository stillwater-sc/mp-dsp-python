# Changelog

All notable changes to `mpdsp` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions track the `mixed-precision-dsp` C++ library in lockstep — the X.Y.Z
prefix always matches the upstream release the wheel was built against, and
Python-only patches ship as PEP 440 post-releases. See `docs/publishing.md`.

Releases before 0.9.0 were tagged without a changelog file; this file starts
with the 0.9.0 cycle.

## [Unreleased]

### Added

- **Acquisition — `DDC`** (#87). Digital down-converter: NCO mixing plus
  matched polyphase decimation of the I and Q streams. The decimator is fixed
  to `PolyphaseDecimator` and built in place from `(taps, decimation_factor)`;
  a `PyPolyphaseDecimator` cannot be adopted because it holds a type-erased
  `unique_ptr`, so the concrete `PolyphaseDecimator<T>` the constructor copies
  is unrecoverable. `process_block` returns a `(real, imag)` tuple, matching
  `NCO.mix_down`. Measured: a 6 kHz tone at 48 kHz with R=4 lands at baseband
  DC with mean magnitude 0.5001, and a +1 kHz offset lands within one 23 Hz
  FFT bin.

- **Acquisition — `DecimationChain` and `design_cic_compensator`** (#88).
  Upstream's chain is variadic with stages in a `std::tuple`, so its arity is
  a compile-time property while Python needs a runtime list. Bridged with an
  `ErasedStage<T>` value type satisfying the upstream stage contract plus a
  recursive arity dispatch over 1–6 stages, so no chain semantics are
  reimplemented here. Stages are given as prototypes and rebuilt at the
  chain's dtype — they cannot be adopted, since inter-stage samples must flow
  at `T` rather than the `double` the existing wrappers marshal.
  `design_cic_compensator` measured: 1.63 dB passband ripple → 0.04 dB
  (R=4, M=3), 6.83 → 0.33 (passband 0.4).

- **Types — analog prototypes and swept Bode** (#115). `PoleZeroPlot`, the
  five `*_prototype` factories, `lp_to_hp` / `lp_to_bp` / `lp_to_bs`,
  `apply_bilinear`, `BodeResult` and `sweep_bode`. The transforms return new
  plots rather than mutating, giving Python value semantics. `sweep_bode`
  measures the *realized* response by driving the filter with settled sines,
  so unlike `frequency_response` it registers sample-path quantization —
  measured 18.5 dB of deviation at `posit_8_2` on an order-4 Butterworth
  where the analytic form shows none.

- **`IIRFilter.frequency_response(freqs, dtype=)`** (#77). Quantizes the
  coefficients through the target type before evaluating — the dual of
  `pole_displacement(dtype)`, sharing one quantization table so the two
  cannot drift. Surfaced that `fpga_fixed` shifts a steep lowpass by 0.38 dB
  while displacing its poles by ~1e-7: the numerator quantizes worse than the
  denominator, which pole displacement alone cannot report.

- **`coeff_dtype=` on all seven RBJ designers** (#94). Runs the coefficient
  math (w0, cos/sin, alpha, a0 normalization) at the chosen precision and
  stores the result in `double` — lossless, since a `T`-designed coefficient
  is `T`-representable. Broadband agreement against reference: posit⟨32,2⟩
  2.2e-08, cfloat⟨24,5⟩ 9.5e-06, `half` 6.6e-03, posit⟨8,2⟩ 5.6e-01.
  Seven designers, not eight — upstream `sw::dsp::rbj` has no Peaking class.

- **Multirate — `Channelizer`, `FractionalDelay`, `channelizer_prototype_bank`.**
  Upstream's `sw::dsp::multirate`, previously unbound. `Channelizer` splits a
  wideband input into M complex baseband channels for about one
  prototype-filter evaluation per input sample; measured ~100 dB of
  out-of-band rejection at 16 taps per phase. `process_block` returns
  `(num_blocks, num_channels)` arrays and drops a trailing partial block
  rather than zero-padding it, since padding would inject a transient the
  caller did not ask for. `FractionalDelay` resamples at an arbitrary
  sub-sample offset — measured accurate to better than 0.01 samples at unity
  gain, with requests below the group-delay floor rounding up rather than
  failing silently.

  `FractionalDelay.taps_per_phase` defaults to **11**, not upstream's 12:
  that default is even and upstream's own validator rejects it, so the C++
  class throws when constructed with its documented defaults
  (`mixed-precision-dsp#208`).

- **`design_halfband(exact_dc_gain=)`** (upstream #206). A half-band
  satisfies A(0) + A(0.5) = 1 identically and A(0.5) is a stopband extremum,
  so unity DC gain and maximum stopband depth are mutually exclusive.
  Measured at 51 taps: 87.1 dB with DC 1.000044, or 81.1 dB with DC exactly 1.

- **`HalfBandFilter.taps` and `PolyphaseDecimator.taps`**. Upstream retains
  no retrievable copy — polyphase decomposes into sub-filters at construction
  — so `DecimationChain` could not rebuild a stage without them.

- **Dashboard: analog-prototype Bode pane** (#78). Tab 2, showing the
  pre-bilinear H(s) with the digital Nyquist marked, so the bilinear warp is
  visible as the difference between two tabs. Explains itself for RBJ (no
  analog prototype exists) and Legendre (no upstream factory).

- **Dashboard: per-dtype magnitude overlay** (#77). Replaces label-only
  legend entries with real curves. The SQNR annotation stays, because it is
  measured by running the signal through and therefore carries the
  sample-path effect the coefficient-quantized curve cannot.

- **Four reference demonstrations** (#62): audio dynamics (#58), motor
  current loop with resonance notch (#59), active vibration cancellation
  (#60), and an SDR down-conversion cascade (#61). Each runs end to end from
  one command and emits a C-compatible artifact — a C89 header, or `.coe`
  and `$readmemh` tables — verified compiling under `-std=c89/c99` and C++
  with `-Wall -Wextra -pedantic -Werror`.

- **`CLAUDE.md`**, documenting the build path, the floor-vs-pin distinction,
  the `ArithConfig` dispatch model, and the nanobind `take_ownership` trap.

### Fixed

- **`NCO` / `DDC` accept absolute RF rates at every dtype** (#117, upstream
  #207). Both used to hold `frequency` and `sample_rate` at the
  configuration's state scalar and divide only afterwards, so GHz values
  overflowed narrow types: at 1.2 GHz / 5 GSPS, `cf24` and `half`
  constructed successfully and then emitted NaN forever, and `fpga_fixed`
  could not hold the rate at all.

  Upstream now forms the ratio in double before converting. This package had
  to **stop casting to `T` at the binding boundary** for that to take effect
  — the premature cast overflowed before upstream ever saw the values, which
  defeated the fix entirely. A binding-side finiteness check remains as a
  backstop against a non-finite accumulator from any other cause.

- **`scripts/build_api_ref.py` runs again, and its output is enforced**
  (#116). It had not parsed on Python < 3.12 since April — a backslash inside
  an f-string expression, legal only from PEP 701, on a project declaring
  `requires-python >= 3.9`. Its `CATEGORIES`/`CLASSES` tables had meanwhile
  fallen 61 names behind, while `docs/api_reference.md` was hand-edited
  around it — so regenerating would have *deleted* hand-written sections.
  All 61 backfilled, the prose migrated into `INTROS`/`CLASS_INTROS`, and
  `tests/test_scripts.py` now asserts the script parses, that coverage is
  complete, and that the committed document matches its output byte for byte.

- **`release.yml` fires on every PEP 440 tag** (#73). The filter matched
  `vX.Y.Z` and `vX.Y.Z-*` only, so `.postN` — which has no `-` separator —
  matched neither and the whole release chain was skipped silently. Replaced
  with one broad glob. The pre-release flag moved out of the trigger into an
  explicit classification step: it was `contains(github.ref, '-')`, which
  calls `v1.2.3rc1` a full release, and `publish.yml` routes on that flag to
  choose TestPyPI versus PyPI. Broadening the glob without fixing the flag
  would have sent release candidates to real PyPI.

### Changed

- **Peer pinned to upstream v0.9.0**, which closes upstream #203–#206. Two of
  those were reported from this repo: the Parks-McClellan designers were not
  equiripple (`design_halfband` capped near 21 dB and got *worse* with more
  taps), and `lp_to_bp` / `lp_to_bs` produced constellations that were not
  bandpass and bandstop — `lp_to_bs` emitted no notch zeros at all, so its
  response peaked at the band centre. Both are fixed; the workarounds and
  their `KNOWN LIMITATION` docstrings are removed.

- **README and `CLAUDE.md` no longer claim coefficients are "always designed
  in `double`"** unconditionally. The classical IIR families are; the `rbj_*`
  and FIR/Remez designers expose `coeff_dtype=` to *measure* what design-time
  precision costs.
