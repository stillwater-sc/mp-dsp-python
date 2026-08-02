# Python Bindings Gap Analysis — 2026-08-02 (post-roadmap)

Successor to [`gap_analysis_2026-08-01.md`](./gap_analysis_2026-08-01.md).
That snapshot identified ~35% of the C++ v0.6.0 surface as unbound and
proposed a 5-phase roadmap to close it. **All 5 phases delivered
(2026-08-02).** This document is the refreshed coverage state — what's
now bound, what's intentionally not, and what's left as niche follow-up.

**Method.** Same as the prior snapshot: cross-reference
`../mixed-precision-dsp/include/sw/dsp/` against `src/*.cpp` +
`python/mpdsp/` + `git log`. Spot-checked with `grep` on the binding
sources to confirm what's present.

---

## 1. Executive summary

- **Roadmap delivery**: 18 sub-issues closed across 5 phases
  (`#96–#99`, `#101–#109`, `#110–#114`); **19 commits**; **+483 tests**
  (suite grew 770 → 1253 pass, +62.7%).
- **Coverage**: went from ~65% → **~93%** of the user-facing v0.6.0
  surface. Every P0 and P1 gap in the previous analysis is closed. The
  residual is small (see §4) and none of it is currently blocking a
  user workflow that we know about.
- **Upstream numerical issues surfaced during binding work**: two
  bugs filed against `stillwater-sc/mixed-precision-dsp`
  (`#200` dolph_chebyshev, `#201` design_fir_lowpass narrow-cfloat
  center-tap NaN). Documented and tracked; workarounds captured in
  test docstrings so future readers don't re-hit the same puzzles.

---

## 2. Module coverage matrix (post-roadmap)

| Module (C++) | Coverage | Notes |
|--------------|----------|-------|
| `types/` | **Strong** | `TransferFunction`, `ContinuousTransferFunction`, `project_onto`, plus the new structured types `ComplexPair`, `PoleZeroPair`, `BiquadCoefficients` (`#114`). `IIRFilter.from_coefficients(list)` factory now enables importing filters designed elsewhere. Missing: `FilterKind` enum, `FilterSpec` dataclass, `design_filter(...)` facade — deliberately scoped out per the API design discussion. `embed_into` still unbound (opposite of `project_onto`; niche). |
| `concepts/` | **N/A** | Compile-time C++ concepts, not runtime API. |
| `math/` | **Strong** | `evaluate_polynomial`, `multiply_polynomials`, `solve_quadratic{,_1,_2}`, `elliptic_K`, `RootFinder` all bound (`#113`). Intentionally skipped: `DenormalPrevention` (implementation detail), math constants (Python has `math.pi` etc.). |
| `signals/` | **Strong** | All generators bound including `multitone`, `ramp`, `upsample`, `downsample` (`#99`). Not bound: `Signal<T>` container class — Python idiomatically passes `(ndarray, sample_rate)` pairs rather than a wrapper type. |
| `windows/` | **Complete** | All 10 window functions bound (`#98` added `tukey`, `gaussian`, `dolph_chebyshev`, `bartlett_hann`). Not bound: `apply_window` / `windowed` free functions (users multiply arrays natively). |
| `filter/iir/` | **Strong** | All 6 filter families + LP/HP/BP/BS variants + RBJ suite. Missing: `ChannelFilter` (multi-channel), state-form selector (`DirectFormI`/`DirectFormII`/`TransposedDirectFormII` — currently hardcoded to `DirectFormI` in `process()`). Would matter for state-form numerical comparison studies. |
| `filter/fir/` | **Strong** | Window-method design + `filtfilt` (`#96`) + Remez (`#111`) + `OverlapAddConvolver` / `OverlapSaveConvolver` (`#111`). Complete. |
| `filter/generic/` | **N/A** | Only `filtfilt` lived here; now bound. |
| `quantization/` | **Complete** | Already fully bound before the roadmap. |
| `conditioning/` | **Complete** | Envelope, Compressor, AGC, plus `RationalResampler` (`#110`). |
| `spectral/` | **Strong** | FFT, IFFT, periodogram, spectrogram, PSD + Welch (`#97`). Missing: `Bluestein` (upstream WIP), direct `dft`/`idft` free functions (FFT covers the use case). Z-transform / Laplace exposed as free functions on `TransferFunction`. |
| `spectrum/` | **Complete** | Entire analyzer stack bound: `RealtimeSpectrum` + detectors (`#104`), `RBWFilter` + `VBWFilter` (`#105`), `SweptLO` + `CalibrationProfile` + `FrontEndCorrector` (`#106`), `TraceAverager` + `WaterfallBuffer` + `Marker`/`DeltaMarker` + `find_peaks` + `harmonic_markers` (`#107`). |
| `acquisition/` | **Strong** | NCO, CIC, halfband, polyphase all bound. NCO now has `.measure_sfdr_db()`; CIC has `.check_bit_growth()` (`#112`). Missing: `DDC` (composite NCO+CIC — users can compose), `DecimationChain` (bit-width-adaptive cascade). Both are convenience compositions, not new primitives. |
| `analysis/` | **Complete** | Coefficient sensitivity, condition number, plus `enob_from_snr_db`, `snr_db`, `AcquisitionPrecisionRow`, `write_acquisition_csv`, `CICBitGrowthReport` (`#112`). |
| `estimation/` | **Complete** | Linear Kalman + `ExtendedKalmanFilter` (`#108`) + `UnscentedKalmanFilter` (`#109`) + LMS/NLMS/RLS. |
| `image/` | **Complete** | Already fully bound before the roadmap. |
| `instrument/` | **Strong** | All 7 measurement free functions (`#101`) + `PeakDetectDecimator` (`#102`) + `TriggerRingBuffer` (`#103`). Missing: `SegmentedCapture` (multi-capture back-to-back), trigger primitives (level/edge/window/video), fractional delay, channel aligner, display envelope, standalone calibration primitives (though `CalibrationProfile` is exposed via the spectrum path). |
| `viz/` | **N/A** | ASCII plotting — intentionally superseded by `python/mpdsp/plotting.py` (matplotlib). |
| `io/` | **Strong** | WAV, PGM/PPM/BMP, CSV all bound. Missing: raw binary read/write (`read_raw`/`write_raw`) — niche. |

---

## 3. What was delivered

### Phase 1 — scipy parity (4 sub-issues, +56 tests)

| # | Item | Commit |
|---|------|--------|
| #96 | `filtfilt` — zero-phase forward-backward IIR | `0cd5d0c` |
| #97 | `welch` PSD estimator | `50aa317` |
| #98 | Missing windows: tukey, gaussian, dolph_chebyshev, bartlett_hann | `5f121fb` |
| #99 | Missing generators: multitone, ramp, upsample, downsample | `8dc5eac` |

### Phase 2 — instrument primitives (3 sub-issues, +78 tests)

| # | Item | Commit |
|---|------|--------|
| #101 | 7 measurement free functions | `8b8fce0` |
| #102 | `PeakDetectDecimator` (scope-style min/max) | `174876e` |
| #103 | `TriggerRingBuffer` (pre/post capture) | `375061e` |

### Phase 3 — spectrum analyzer stack (4 sub-issues, +130 tests)

| # | Item | Commit |
|---|------|--------|
| #104 | `RealtimeSpectrum` + 5 detector reducers | `973d445` |
| #105 | `RBWFilter` + `VBWFilter` | `6163e9b` |
| #106 | `SweptLO` + `CalibrationProfile` + `FrontEndCorrector` | `12253f8` |
| #107 | `TraceAverager` + `WaterfallBuffer` + Marker + `find_peaks` + `harmonic_markers` | `5d40614` |

### Phase 4 — nonlinear estimation (2 sub-issues, +36 tests)

| # | Item | Commit |
|---|------|--------|
| — | MTL5 CMake floor 5.2.1 → 5.7.0 (prep for UKF's `ldlt`) | `f450cf5` |
| #108 | `ExtendedKalmanFilter` (Python callbacks: f, F, h, H) | `1a23c37` |
| #109 | `UnscentedKalmanFilter` (Python callbacks: f, h) | `1f41ede` |

Introduced a **Python-callback marshaling pattern** — `nb::callable`
capture into `std::function` closures, `callback_result_to_{vec,mat}<T>`
helpers with shape/length validation, GIL held throughout. Reusable for
any future binding that needs user-supplied Python callbacks.

### Phase 5 — advanced / opportunistic (5 sub-issues, +123 tests)

| # | Item | Commit |
|---|------|--------|
| #110 | `RationalResampler` (polyphase L/M) | `320f7c3` |
| #111 | Remez + `OverlapAddConvolver`/`OverlapSaveConvolver` | `db6753c` |
| #112 | acquisition_precision (`enob`, `snr_db`, `CICBitGrowthReport`, `AcquisitionPrecisionRow`, `write_acquisition_csv`, `NCO.measure_sfdr_db`, `CICDecimator.check_bit_growth`) | `8cce32d` |
| #113 | math utilities (`RootFinder`, polynomial, quadratic, elliptic_K) | `ccbb8b0` |
| #114 | Structured types (`ComplexPair`, `PoleZeroPair`, `BiquadCoefficients`) + `IIRFilter.from_coefficients` | `1c9f9b8` |

---

## 4. Residual gaps

None currently blocking a user workflow. Grouped by why they're absent:

### Intentionally not bound

- **`viz/` ASCII plotting** — superseded by `python/mpdsp/plotting.py`.
- **`math/DenormalPrevention`** — implementation detail of the streaming
  IIR classes; users don't construct it directly.
- **`FilterKind` enum, `FilterSpec` dataclass, `design_filter(...)`
  facade** — API expansion, not gap-closure. Skipped per the API
  discussion for `#114`. Would add a second way to design filters
  alongside the existing per-family functions.
- **`Signal<T>` container class** — Python idiomatically uses
  `(ndarray, sample_rate)` pairs.
- **`apply_window` / `windowed` free functions** — trivial with numpy
  multiplication.
- **Math constants** — Python has `math.pi`, `math.tau` etc.

### Niche / low-priority

- **`ChannelFilter` (multi-channel IIR)** — users can loop over channels
  in Python. Would matter for hot-loop multi-channel processing.
- **State-form selector on IIRFilter** — `DirectFormI` is hardcoded in
  `process()`. Would matter for numerical-precision studies comparing
  the three state forms.
- **`SegmentedCapture` / trigger primitives / fractional delay /
  channel aligner / display envelope / standalone calibration** — extra
  instrument-module classes beyond the P0 items bound in Phase 2.
- **`DDC` / `DecimationChain`** — composite wrappers over already-bound
  primitives. Users can compose in Python.
- **Direct `dft` / `idft` free functions** — FFT covers the use case.
- **`Z-transform` / `Laplace` as first-class stateful objects** — free
  functions `ztransform`, `freqz`, `group_delay`, `laplace_freqs` are
  bound over `TransferFunction`, which covers the analytical use case.
- **`embed_into` (opposite of `project_onto`)** — narrow → wide type
  widening. Users can `astype(np.float64)` for the common case.
- **Raw binary I/O** — `read_raw` / `write_raw`. Users can `np.fromfile`.

### Upstream-blocked

- **`spectral/Bluestein`** — upstream marks as WIP in the header.
- **`filter/fir/remez` differentiator / hilbert modes** — bound but
  untested by us; the general `remez(type=...)` accepts them.

### Blocked by open upstream bugs

- **Narrow cfloat filter design at coefficient precision** — several
  filter design paths (`RationalResampler`, `remez_lowpass` etc.) will
  produce NaN taps at `half` / `cf24` CoeffScalar due to
  `mixed-precision-dsp#201`. Documented in test docstrings; will
  clear when upstream lands the fix (drop `T(1e12)` threshold or check
  `n == center` directly).
- **`dolph_chebyshev` window** — collapses to constant at all precisions
  (`mixed-precision-dsp#200`, not a narrow-type issue). Bound; test
  covers the invariants that hold; behavior test is skipped with a note.

---

## 5. New patterns established during the roadmap

Infrastructure that showed up during the work and is reusable for
future bindings:

- **Callback marshaling** — `nb::callable` captured into `std::function`
  closures, `callback_result_to_{vec,mat}<T>` helpers with clear
  shape/length error messages, GIL held during predict/update. First
  used by EKF/UKF (`#108/#109`), reusable for any future binding that
  needs user-supplied Python callbacks (custom design functions,
  adaptive-rule hooks, per-sample callbacks, etc.).
- **Class-template-scoped enum handling** — `SweptLO<T>::Sweep`,
  `TraceAverager<T>::Mode`, etc. are type-distinct across
  instantiations. Solution: pass a plain int / bool across the
  type-erasure boundary and reconstruct the correct-typed enum in each
  `Impl<T>::ctor`. Applied in `#106`, `#107`.
- **Shared window-name dispatcher** — `mpdsp::bindings::make_window_T<T>`
  in `_binding_helpers.hpp` (introduced in `#97`, extended by `#98`).
  Any binding that needs to design a window at the target dtype uses it.
- **Upstream-narrow-type-limitation escape hatch** — when a test would
  fail for a subset of dtypes due to a documented upstream numerical
  bug, split the parametrization into "wide dtypes (strict all-finite
  assertion)" and "narrow dtypes (weaker any-finite assertion)" with a
  test-class docstring explaining the limitation and linking the
  upstream issue. Applied in `#110` for `RationalResampler`.

---

## 6. Suggested next work

None currently prioritized. Watching for:

- **User feedback** — the roadmap was driven by the previous doc's
  P0/P1 tiers, which were the author's best guess. Real user feedback
  may surface a different priority ordering for the residual gaps in §4.
- **Upstream fixes** — `mixed-precision-dsp#200` and `#201` land the
  narrow-cfloat filter design paths become useful; the dolph_chebyshev
  window becomes usable. Neither needs code changes on our side.
- **Upstream v0.7.0 release** — a new upstream minor version will
  likely add API. File a new epic when that lands and repeat the
  inventory-diff exercise.
- **Version bump for mp-dsp-python** — the roadmap added significant
  API surface. A minor version bump (0.7.0) plus a `CHANGELOG.md`
  entry would signal to users that this happened.

---

*Regenerate by re-running the two-inventory diff (see the prior
`gap_analysis_2026-08-01.md` methodology section) against the
`../mixed-precision-dsp` head at the current release tag.*
