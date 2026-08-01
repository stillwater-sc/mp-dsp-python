# Python Bindings Gap Analysis — 2026-08-01

Compares the Python surface exposed by **mp-dsp-python** against the full C++
API of **mixed-precision-dsp v0.6.0**. Goal: identify missing modules,
missing primitives inside partially covered modules, and prioritize what to
bind next.

**Method.** Two independent inventories were compiled — one from
`../mixed-precision-dsp/include/sw/dsp/` (headers under 17 top-level
modules), one from `src/*.cpp` + `python/mpdsp/` in this repo — then
cross-referenced. Spot-checked with `grep` against the binding sources to
confirm absences.

---

## 1. Executive summary

The Python bindings cover **~65% of the C++ library's user-facing surface**.
Coverage is strong for the primitives most Python users reach for first
(filter design, FFT-based spectral analysis, quantization, image, adaptive
filters), but two entire modules and several high-value scipy-equivalents
are missing:

| Priority | Gap | Impact |
|----------|-----|--------|
| **P0** | `instrument/` — entire module unbound (measurements, triggering, ring buffer, scope decimator, fractional delay, channel aligner) | Blocks oscilloscope / test-equipment workflows in Python |
| **P0** | `spectrum/` — entire module unbound (`RealtimeSpectrum`, RBW/VBW, swept-LO, detectors, waterfall, markers) | Blocks spectrum-analyzer application (the capstone of the C++ demo suite) |
| **P0** | `filtfilt` (zero-phase forward-backward) | scipy parity gap; users expect `scipy.signal.filtfilt` equivalent |
| **P1** | `estimation/` — EKF and UKF missing (linear Kalman only) | Nonlinear tracking / sensor fusion cannot be done from Python |
| **P1** | `conditioning/RationalResampler` — polyphase L/M rate conversion | Only fixed-integer resamplers are exposed today (CIC, halfband, polyphase in acquisition) |
| **P1** | `spectral/welch` — Welch's PSD method | Current `psd()` binding is periodogram-only (verified in `src/spectral_bindings.cpp:239`); Welch is the standard estimator |
| **P2** | 4 window functions: `tukey`, `gaussian`, `dolph_chebyshev`, `bartlett_hann` | Users have to fall back to `scipy.signal.windows` for these |
| **P2** | Signal generators: `multitone`, `ramp`, `upsample`, `downsample` | Small but common primitives |
| **P2** | `filter/fir/remez` (Parks-McClellan) | Optimal linear-phase FIR design; scipy has this as `remez` |
| **P3** | Structured types: `FilterKind`, `FilterSpec`, `PoleZeroPair`, `BiquadCoefficients`, `ComplexPair` as first-class Python classes | Currently only `TransferFunction` and `IIRFilter` are exposed; users get tuples/lists for the rest |
| **P3** | `math/`: `RootFinder` (Laguerre), polynomial ops, `elliptic_K` | Useful for advanced users doing custom design |
| **P3** | `analysis/acquisition_precision` | The C++ demo has a matching test; Python cannot reproduce the analysis |
| **P3** | Multi-channel `ChannelFilter`, state-form selector (DirectFormI vs DirectFormII vs TransposedDirectFormII) | IIR filters currently hardcoded; some numerical studies need to compare forms |

---

## 2. Module coverage matrix

| Module (C++) | Coverage | Notes |
|--------------|----------|-------|
| `types/` | **Partial** | `TransferFunction`, `ContinuousTransferFunction`, `project_onto` bound. Missing: `FilterKind` enum, `FilterSpec`, `ComplexPair`, `PoleZeroPair`, `BiquadCoefficients` (exposed only as tuple list via `IIRFilter.coefficients()`), `BiquadPoleState`, `embed_into` |
| `concepts/` | **N/A** | Compile-time C++ concepts, not runtime API — correctly not bound |
| `math/` | **None** | `DenormalPrevention`, `solve_quadratic{,_1,_2}`, `evaluate_polynomial`, `multiply_polynomials`, `RootFinder<T, MaxDegree>` (Laguerre), `elliptic_K` all unbound |
| `signals/` | **Partial** | Most generators bound. Missing: `multitone`, `ramp`, `upsample`, `downsample`, `Signal<T>` class (Python uses raw ndarray + sample_rate) |
| `windows/` | **Partial** | 6 of 10 windows bound (rectangular, hamming, hanning, blackman, kaiser, flat_top). Missing: `tukey`, `gaussian`, `dolph_chebyshev`, `bartlett_hann`; also missing free functions `apply_window`/`windowed` |
| `filter/iir/` | **Strong** | All 6 filter families bound (Butterworth, Chebyshev I/II, Bessel, Legendre, Elliptic, RBJ) with LP/HP/BP/BS variants. Missing: `rbj_bandshelf`, `rbj_peaking`, `ChannelFilter` (multi-channel), state-form selector |
| `filter/fir/` | **Partial** | Window-method design + basic FIR processing bound. Missing: **`filtfilt` (zero-phase)** — high-impact scipy gap; `remez` (Parks-McClellan); OLA/OLS overlap methods |
| `quantization/` | **Complete** | ADC/DAC, SQNR, RPDF/TPDF dither, first-order noise shaper all bound |
| `conditioning/` | **Partial** | Envelope (peak/RMS), Compressor, AGC bound. Missing: **`RationalResampler`** (polyphase L/M resampler) |
| `spectral/` | **Partial** | FFT, IFFT, periodogram, spectrogram, PSD bound. Missing: **`welch`** (segmented averaged PSD — the standard estimator; current `psd()` is periodogram-only per `src/spectral_bindings.cpp:239`); direct `dft`/`idft`; Bluestein (upstream WIP); Z-transform / Laplace as first-class objects (only `evaluate`-style free functions bound) |
| `spectrum/` | **None** | **Entire module unbound.** Missing: `RealtimeSpectrum`, detector modes (peak/sample/average/rms/negative-peak), RBW filter, VBW filter, swept-LO, front-end corrector, trace averaging, waterfall buffer, markers |
| `acquisition/` | **Strong** | NCO, CIC decimator/interpolator, halfband, polyphase decimator/interpolator all bound. Missing: `DDC` (composite NCO+CIC), `DecimationChain` (bit-width-adaptive cascade) |
| `analysis/` | **Partial** | Method forms on `IIRFilter` (`.stability_margin`, `.condition_number`, `.worst_case_sensitivity`, `.pole_displacement`) + free functions `coefficient_sensitivity`, `biquad_condition_number`. Missing: free-function `biquad_poles`, `max_pole_radius`, `is_stable`; `acquisition_precision` sub-module |
| `estimation/` | **Partial** | Linear `KalmanFilter`, `LMSFilter`, `NLMSFilter`, `RLSFilter` bound. Missing: **`ExtendedKalmanFilter` (EKF)**, **`UnscentedKalmanFilter` (UKF)** — the nonlinear estimators |
| `image/` | **Complete** | 14 generators, convolution, separable filter, Gaussian/box blur, Sobel/Prewitt/Canny edge, morphology (dilate/erode/open/close/gradient/tophat/blackhat), PGM/PPM/BMP I/O all bound |
| `instrument/` | **None** | **Entire module unbound.** Missing: measurements (`peak_to_peak`, `mean`, `rms`, `frequency`, `period`, `rise_time`, `fall_time`), `TriggerRingBuffer`, `SegmentedCapture`, `PeakDetectDecimator`, trigger primitives, fractional delay, channel aligner, display envelope, calibration |
| `viz/` | **N/A** | ASCII plotting — intentionally superseded by `python/mpdsp/plotting.py` (matplotlib). Correctly not bound |
| `io/` | **Partial** | WAV, PGM/PPM/BMP bound. CSV handled at Python level. Missing: raw binary read/write |

---

## 3. Detailed gaps with rationale

### 3.1 `instrument/` — P0

The C++ library has a rich oscilloscope-primitives module built out for issues
#132 and Epic #134: pre/post-trigger ring buffer with a full state machine,
segmented capture, scope-style peak-detect decimation (preserves min/max
rather than averaging), fractional-delay Lagrange interpolators for
sub-sample alignment, multi-channel skew correction, and one-shot
measurements (frequency via zero-crossings, rise/fall time with configurable
percent thresholds). None of it is reachable from Python.

**Recommendation:** Bind measurements first (all stateless free functions on
`span<const T>` — a trivial dispatch pattern already used elsewhere in this
repo). Ring buffer and peak-detect decimator next; they're stateful classes
that map cleanly to the pattern used for `CICDecimator`.

### 3.2 `spectrum/` — P0

The spectrum-analyzer stack (RealtimeSpectrum streaming FFT engine, RBW/VBW
filters, swept-LO, detector modes, trace averaging, waterfall buffer,
markers) is the "capstone" of the C++ application layer — there's a full
`spectrum_analyzer_demo` in the C++ tree. Python has FFT and spectrogram
primitives but nothing that composes into an analyzer.

**Recommendation:** Bind `RealtimeSpectrum` (mirrors the existing
`class-with-process_block` pattern) plus detector-mode dispatch. RBW/VBW/
swept-LO can follow as separate PRs; each is a stateful class that plugs
into the same pattern.

### 3.3 `filtfilt` — P0

Zero-phase forward-backward IIR filtering with edge reflection. This is the
single most-requested scipy DSP primitive by users doing offline analysis
(scipy calls it `scipy.signal.filtfilt`). The C++ implementation exists at
`include/sw/dsp/filter/generic/filtfilt.hpp`. Binding is a straightforward
free function taking an `IIRFilter` and an ndarray.

### 3.4 `estimation/` EKF and UKF — P1

Linear Kalman is bound, but nonlinear estimation (extended Kalman with
Jacobian linearization, unscented Kalman with sigma points) is not.
Sensor-fusion notebooks are limited to linear systems as a result. UKF in
particular is popular for use cases where the user cannot easily provide a
Jacobian (e.g., trigonometric measurement models).

**Note:** UKF depends on MTL5's `ldlt` operation, which is why the
mixed-precision-dsp `README.md` pins MTL5 ≥ v5.3.0. The dependency floor in
`CMakeLists.txt` here is already v5.2.1; would need to bump to v5.3.0 to
bind UKF.

### 3.5 `RationalResampler` — P1

Polyphase L/M rational resampler (arbitrary rational sample-rate
conversion). Today Python users doing SRC have to compose CIC + halfband +
polyphase manually, or drop out to scipy's `resample_poly`. The C++ class
exposes `push()` / `pop_if_ready()` for streaming and `process()` for batch;
matches the pattern used by the acquisition-module bindings.

### 3.6 `spectral/welch` — P1

The current `psd(signal, sample_rate, dtype)` binding calls
`sw::dsp::spectral::periodogram<T>` under the hood (verified at
`src/spectral_bindings.cpp:239`). A single-shot periodogram has notoriously
high variance; Welch's segmented-and-averaged method is the standard
estimator every scipy user expects. The C++ signature is
`welch<T>(x, segment_size, overlap, window)`.

**Recommendation:** Add a new `welch()` binding rather than replacing the
existing `psd()`, to avoid breaking callers.

### 3.7 Missing windows — P2

Absent: `tukey` (cosine-tapered — widely used for spectral leakage
mitigation), `gaussian`, `dolph_chebyshev` (equiripple sidelobes, common in
radar/sonar), `bartlett_hann`. All are single-function additions to
`signal_bindings.cpp` following the pattern already used for `kaiser`.

### 3.8 Signal generators — P2

`multitone` (sum of sinusoids at specified frequencies/amplitudes/phases —
canonical for IMD / two-tone tests) is a notable omission. `ramp`,
`upsample`, `downsample` are small conveniences.

### 3.9 First-class structured types — P3

Currently Python sees biquad coefficients as `List[Tuple[float, float,
float, float, float]]` from `IIRFilter.coefficients()`. Binding
`BiquadCoefficients`, `PoleZeroPair`, `ComplexPair`, `FilterSpec`, and the
`FilterKind` enum as proper nanobind classes would let users construct
filters from raw coefficients (currently impossible from Python — you can
only get filters from design functions) and give IDE completion for the
structured-type fields.

**Note:** `FilterSpec`-driven design would enable a single generic
`design_filter(kind, spec)` entry point in Python, replacing the ~30
per-family design functions with something more discoverable.

### 3.10 `math/` utilities — P3

`RootFinder` (Laguerre's method for complex polynomial roots),
`evaluate_polynomial` / `multiply_polynomials`, `elliptic_K`. Useful for
users building custom filter designs in Python; low-priority since scipy /
numpy cover most of this, but binding them keeps mixed-precision types in
play (numpy will silently promote to double).

---

## 4. What's *already* well-covered — do not re-bind

For completeness, these areas are strong and should not be prioritized:

- Full IIR family (Butterworth / Chebyshev I / Chebyshev II / Bessel /
  Legendre / Elliptic / RBJ) — 30+ design functions across LP/HP/BP/BS
- FFT / IFFT / periodogram / spectrogram — the common spectral primitives
- Image processing — 14 generators, convolution, edge detection, morphology,
  I/O (PGM/PPM/BMP)
- Quantization — ADC/DAC/SQNR/dither/noise-shaping all present
- Adaptive filters — LMS / NLMS / RLS all present
- Acquisition — NCO, CIC decimator/interpolator, halfband, polyphase all
  present with both streaming (`push`) and batch (`process_block`) APIs
- Mixed-precision dtype dispatch — 18 pre-instantiated configurations
  across all data-processing paths; this is the *distinguishing feature*
  of the library and it's fully wired

---

## 5. Suggested roadmap

Grouped by binding effort, not just priority:

**Phase 1 — scipy parity (small, high user impact):**
1. `filtfilt` (free function)
2. `welch` (free function)
3. Missing windows: `tukey`, `gaussian`, `dolph_chebyshev`, `bartlett_hann`
4. Missing generators: `multitone`, `ramp`, `upsample`, `downsample`

**Phase 2 — instrument measurements (small, unlocks scope workflows):**
5. All 7 measurement free functions (`peak_to_peak`, `mean`, `rms`,
   `frequency`, `period`, `rise_time`, `fall_time`)
6. `PeakDetectDecimator` class
7. `TriggerRingBuffer` class

**Phase 3 — spectrum analyzer (medium, unlocks analyzer app):**
8. `RealtimeSpectrum` + detector-mode dispatch
9. RBW / VBW filter classes
10. Trace averaging / waterfall buffer / markers

**Phase 4 — nonlinear estimation (medium, requires MTL5 bump):**
11. `ExtendedKalmanFilter` — needs Python-side dynamics/measurement callbacks
12. `UnscentedKalmanFilter` — same callback machinery

**Phase 5 — advanced (large, low urgency):**
13. `RationalResampler`
14. FIR `remez` (waiting on upstream), OLA/OLS
15. First-class structured types + `design_filter(kind, spec)` facade
16. `math/` utilities

---

## 6. Verification methodology

- **C++ inventory:** Read every `.hpp` under `include/sw/dsp/` in the
  mixed-precision-dsp repo at v0.6.0 (commit visible in the parent repo's
  `git log`). Compiled a per-module catalog of public classes, structs, and
  free-function templates.
- **Python inventory:** Read every `src/*_bindings.cpp` file plus
  `python/mpdsp/*.py`, cataloging every `.def(`, `nb::class_<`, and
  `NB_MODULE` declaration.
- **Spot-check:** Ran
  `grep -rn -E "filtfilt|welch|multitone|RealtimeSpectrum|peak_to_peak|
  rise_time|TriggerRing|PeakDetectDecim|RationalResampler|ExtendedKalman|
  UnscentedKalman|dolph_chebyshev|tukey_window|gaussian_window|
  bartlett_hann|remez" src/ python/`
  → zero matches, confirming absence.
- **`psd` internals:** Read `src/spectral_bindings.cpp:229-240` directly
  to confirm current `psd()` binding is periodogram-based, not Welch.

---

*Filed at* `docs/gap_analysis_2026-08-01.md` *by an automated cross-repo
inventory. Regenerate by re-running the two-inventory diff against
`../mixed-precision-dsp` at the current release tag.*
