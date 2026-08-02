# mp-dsp-python

Python integration layer for the
[mixed-precision-dsp](https://github.com/stillwater-sc/mixed-precision-dsp)
C++ library, providing nanobind bindings, matplotlib visualizations,
and Jupyter notebooks for the full DSP domain.

## Why

The [mixed-precision-dsp](https://github.com/stillwater-sc/mixed-precision-dsp)
library is a C++20 header-only DSP library covering signals, windows,
quantization, IIR/FIR filtering, spectral analysis, signal conditioning,
estimation (Kalman/LMS/RLS), image processing, and numerical analysis
— all parameterized on arithmetic type for mixed-precision research.

DSP researchers work in Python. Jupyter notebooks, matplotlib, SciPy,
and NumPy are the standard tools for prototyping, analysis, and
publication-quality visualization. This repository bridges the gap:
**C++ does the mixed-precision math across the full DSP domain;
Python orchestrates experiments and presents results.**

Without this layer, every mixed-precision experiment requires writing
a C++ application, exporting CSV, and hand-crafting plotting scripts.
With `mp-dsp-python`, the entire `sw::dsp` library is accessible from
a single `import mpdsp` statement.

```python
import mpdsp
import numpy as np
import matplotlib.pyplot as plt

# Signal generation
signal = mpdsp.sine(length=2000, frequency=440, sample_rate=44100)
noise = mpdsp.gaussian_noise(length=2000, stddev=0.1)
noisy = signal + noise

# Windowing
window = mpdsp.hamming(2000)
windowed = noisy * window

# Spectral analysis
freqs, psd = mpdsp.psd(windowed, sample_rate=44100)
plt.semilogy(freqs, psd)

# IIR filtering with mixed precision
filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100, cutoff=1000)
ref    = filt.process(signal, dtype="reference")      # double/double/double
posit  = filt.process(signal, dtype="posit_full")      # double/posit<32,2>/posit<16,1>
print(f"SQNR: {mpdsp.sqnr_db(ref, posit):.1f} dB")

# Image processing
img = mpdsp.checkerboard(256, 256, block_size=8)
edges = mpdsp.canny(img, low_threshold=0.1, high_threshold=0.3, sigma=1.0)
mpdsp.write_pgm("edges.pgm", edges)

# Estimation
kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)
# ... configure and run

# Analysis
margin = filt.stability_margin()
poles = filt.poles()
sensitivity = filt.worst_case_sensitivity()
```

## What

### Full DSP Domain Coverage

`mp-dsp-python` exposes every module of the C++ library to Python. **The
2026-08-02 bindings-gap roadmap (epic #100) closed 18 sub-issues across
5 phases, bringing coverage from ~65% to ~93% of the v0.6.0 surface** —
see [`docs/gap_analysis_2026-08-02.md`](docs/gap_analysis_2026-08-02.md)
for the module-by-module state. For the complete enumeration of every
public name with signatures and one-line descriptions, see
[`docs/api_reference.md`](docs/api_reference.md).

| Module | C++ Headers | Python API | Description |
|--------|-------------|------------|-------------|
| **signals** | `generators.hpp`, `sampling.hpp` | `mpdsp.sine()`, `mpdsp.chirp()`, `mpdsp.impulse()`, `mpdsp.step()`, `mpdsp.ramp()`, `mpdsp.multitone()`, `mpdsp.white_noise()`, `mpdsp.gaussian_noise()`, `mpdsp.pink_noise()`, `mpdsp.upsample()`, `mpdsp.downsample()`, ... | Full signal generator suite returning NumPy arrays. Rate-conversion helpers (`upsample`/`downsample`) are zero-insert / naive decimation — for anti-aliasing pair them with FIR / halfband / polyphase. |
| **windows** | `hamming.hpp`, `hanning.hpp`, `blackman.hpp`, `kaiser.hpp`, `tukey.hpp`, `gaussian.hpp`, `dolph_chebyshev.hpp`, `bartlett_hann.hpp`, `flat_top.hpp`, `rectangular.hpp` | `mpdsp.hamming()`, `mpdsp.hanning()`, `mpdsp.blackman()`, `mpdsp.kaiser()`, `mpdsp.tukey()`, `mpdsp.gaussian()`, `mpdsp.dolph_chebyshev()`, `mpdsp.bartlett_hann()`, `mpdsp.flat_top()`, `mpdsp.rectangular()` | All 10 window functions bound. |
| **quantization** | `adc.hpp`, `dac.hpp`, `dither.hpp`, `noise_shaping.hpp`, `sqnr.hpp` | `mpdsp.adc()`, `mpdsp.dac()`, `mpdsp.sqnr_db()`, `mpdsp.measure_sqnr_db()`, `mpdsp.RPDFDither()`, `mpdsp.TPDFDither()`, `mpdsp.FirstOrderNoiseShaper()`, ... | ADC/DAC modeling with type dispatch. RPDF/TPDF dithering and first-order error-feedback noise shaping for quantization improvement. SQNR measurement — the core metric for mixed-precision evaluation. |
| **filter/iir** | `butterworth.hpp`, `chebyshev1.hpp`, `chebyshev2.hpp`, `elliptic.hpp`, `bessel.hpp`, `legendre.hpp`, `rbj.hpp` | `mpdsp.butterworth_lowpass()`, `mpdsp.chebyshev1_highpass()`, `mpdsp.elliptic_bandpass()`, `mpdsp.rbj_lowshelf()`, `IIRFilter.from_coefficients(list)`, ... | All 7 IIR families with LP/HP/BP/BS (and RBJ shelf/allpass) variants. Design in double, process with type dispatch. Filter objects expose `poles()`, `frequency_response()`, `stability_margin()`, `condition_number()`, `pole_displacement()`, `worst_case_sensitivity()` as methods. `IIRFilter.from_coefficients()` imports filters designed elsewhere (scipy, MATLAB, hand-cascaded). |
| **filter/fir** | `fir_filter.hpp`, `fir_design.hpp`, `remez.hpp`, `overlap.hpp`, `filtfilt.hpp` | `mpdsp.fir_lowpass()`, `mpdsp.fir_bandpass()`, `mpdsp.fir_filter()`, `mpdsp.remez()`, `mpdsp.remez_lowpass()`, `mpdsp.remez_bandpass()`, `mpdsp.filtfilt()`, `mpdsp.OverlapAddConvolver()`, `mpdsp.OverlapSaveConvolver()`, ... | FIR window-method design, Parks-McClellan (Remez) equiripple design, zero-phase forward-backward filtering (`filtfilt`, scipy analogue), block-FFT convolvers for long signals. |
| **spectral** | `fft.hpp`, `dft.hpp`, `psd.hpp`, `spectrogram.hpp`, `ztransform.hpp`, `laplace.hpp` | `mpdsp.fft()`, `mpdsp.ifft()`, `mpdsp.fft_magnitude_db()`, `mpdsp.psd()`, `mpdsp.periodogram()`, `mpdsp.welch()`, `mpdsp.spectrogram()`, `mpdsp.ztransform()`, `mpdsp.freqz()`, `mpdsp.group_delay()`, `mpdsp.laplace_freqs()` | FFT (Cooley-Tukey), power spectral density (single-shot `psd` and averaged `welch`), STFT/spectrogram, Z-transform and Laplace evaluation. All primitives accept `dtype=` for mixed-precision arithmetic. |
| **spectrum** | `realtime_spectrum.hpp`, `detectors.hpp`, `rbw_filter.hpp`, `vbw_filter.hpp`, `swept_lo.hpp`, `front_end_corrector.hpp`, `trace_averaging.hpp`, `waterfall_buffer.hpp`, `markers.hpp` | `mpdsp.RealtimeSpectrum()`, `mpdsp.detect_peak()` + `_sample`/`_average`/`_rms`/`_negative_peak`/`detect(mode)`, `mpdsp.RBWFilter()`, `mpdsp.VBWFilter()`, `mpdsp.SweptLO()`, `mpdsp.FrontEndCorrector()`, `mpdsp.CalibrationProfile()`, `mpdsp.TraceAverager()`, `mpdsp.WaterfallBuffer()`, `mpdsp.Marker`/`DeltaMarker`, `mpdsp.find_peaks()`, `mpdsp.harmonic_markers()` | Full spectrum-analyzer stack: streaming FFT engine + 5 detector reducers, resolution / video bandwidth filters, swept local oscillator, front-end equalization, cross-sweep trace averaging (5 modes), 2D waterfall memory, marker + peak-finder primitives. |
| **acquisition** | `nco.hpp`, `cic.hpp`, `halfband.hpp`, `polyphase_decimator.hpp` | `mpdsp.NCO()`, `mpdsp.CICDecimator()`, `mpdsp.CICInterpolator()`, `mpdsp.HalfBandFilter()`, `mpdsp.PolyphaseDecimator()`, `mpdsp.PolyphaseInterpolator()`, `mpdsp.design_halfband()`, `mpdsp.polyphase_decompose()`, `nco.measure_sfdr_db()`, `cic.check_bit_growth()` | High-rate acquisition pipeline (numerically-controlled oscillator, CIC decimator/interpolator, halfband/polyphase filters). NCO and CIC also carry precision-analysis methods (`measure_sfdr_db`, `check_bit_growth`). |
| **conditioning** | `envelope.hpp`, `compressor.hpp`, `agc.hpp`, `src.hpp` | `mpdsp.PeakEnvelope()`, `mpdsp.RMSEnvelope()`, `mpdsp.Compressor()`, `mpdsp.AGC()`, `mpdsp.RationalResampler()` | Envelope followers (peak, RMS). Dynamic range compressor with soft knee. Automatic gain control. Polyphase L/M rate conversion (scipy `resample_poly` analogue). |
| **estimation** | `kalman.hpp`, `ekf.hpp`, `ukf.hpp`, `lms.hpp`, `rls.hpp` | `mpdsp.KalmanFilter()`, `mpdsp.ExtendedKalmanFilter()`, `mpdsp.UnscentedKalmanFilter()`, `mpdsp.LMSFilter()`, `mpdsp.NLMSFilter()`, `mpdsp.RLSFilter()` | Linear Kalman + nonlinear EKF (Python callbacks for f, F, h, H) + UKF (Python callbacks for f, h — no Jacobians). LMS/NLMS adaptive filters. RLS with forgetting factor. State matrices as NumPy 2D arrays. |
| **image** | `image.hpp`, `convolve2d.hpp`, `separable.hpp`, `morphology.hpp`, `edge.hpp`, `generators.hpp` | `mpdsp.convolve2d()`, `mpdsp.gaussian_blur()`, `mpdsp.sobel_x()`, `mpdsp.canny()`, `mpdsp.dilate()`, `mpdsp.checkerboard()`, ... | 2D convolution, separable filters, Gaussian/box blur. Morphological operations (erode, dilate, open, close, gradient, tophat). Sobel, Prewitt, Canny edge detection. Image generators (checkerboard, zone plate, gradients, noise, blobs). |
| **instrument** | `measurements.hpp`, `peak_detect.hpp`, `ring_buffer.hpp` | `mpdsp.peak_to_peak()`, `mpdsp.instrument_mean()`, `mpdsp.instrument_rms()`, `mpdsp.rise_time()`, `mpdsp.fall_time()`, `mpdsp.period()`, `mpdsp.frequency()`, `mpdsp.PeakDetectDecimator()`, `mpdsp.TriggerRingBuffer()` | Oscilloscope-style stateless measurements (7 primitives), scope min/max-preserving decimator, pre/post-trigger capture with 4-state lifecycle. `mean`/`rms` prefixed with `instrument_` to avoid shadowing `numpy.mean`/`numpy.rms`. |
| **analysis** | `stability.hpp`, `sensitivity.hpp`, `condition.hpp`, `acquisition_precision.hpp` | `filt.stability_margin()`, `filt.condition_number()`, `filt.worst_case_sensitivity()`, `filt.pole_displacement(dtype)`, `mpdsp.coefficient_sensitivity()`, `mpdsp.biquad_condition_number()`, `mpdsp.enob_from_snr_db()`, `mpdsp.snr_db()`, `mpdsp.CICBitGrowthReport`, `mpdsp.AcquisitionPrecisionRow`, `mpdsp.write_acquisition_csv()` | Coefficient-level (free function) and cascade-level (filter method) stability / sensitivity / conditioning analysis. Acquisition-pipeline precision metrics (ENOB, SNR) plus a CSV writer schema-compatible with the C++ precision-sweep outputs. |
| **math** | `polynomial.hpp`, `quadratic.hpp`, `elliptic_integrals.hpp`, `root_finder.hpp` | `mpdsp.evaluate_polynomial()`, `mpdsp.multiply_polynomials()`, `mpdsp.solve_quadratic()` (+ `_1`, `_2`), `mpdsp.elliptic_K()`, `mpdsp.RootFinder()` | Numerical utilities for advanced filter design: Horner polynomial evaluation, polynomial multiplication (convolution), quadratic solver returning complex roots, complete elliptic integral (Cauer filter design), Laguerre polynomial root finder up to degree 32. |
| **types** | `projection.hpp`, `transfer_function.hpp`, `biquad_coefficients.hpp`, `pole_zero_pair.hpp`, `complex_pair.hpp` | `mpdsp.TransferFunction()`, `mpdsp.ContinuousTransferFunction()`, `mpdsp.project_onto()`, `mpdsp.projection_error()`, `mpdsp.BiquadCoefficients()`, `mpdsp.PoleZeroPair()`, `mpdsp.ComplexPair()`, `mpdsp.to_transfer_function(filt)` | Rational transfer function H(z) = B(z)/A(z) with complex-plane evaluation, frequency response, stability check, and cascade via `*`. Structured biquad-level types with read-write fields for constructing filters from raw coefficients. Type-projection round-trip for quantifying quantization loss outside the filter path. |
| **io** | `wav.hpp`, `csv.hpp`, `pgm.hpp`, `ppm.hpp`, `bmp.hpp` | `mpdsp.read_wav()`, `mpdsp.write_wav()`, `mpdsp.read_pgm()`, `mpdsp.write_pgm()`, `mpdsp.read_ppm()`, `mpdsp.write_ppm()`, `mpdsp.read_bmp()`, `mpdsp.write_bmp()`, CSV via `mpdsp.load_sweep()` | WAV audio (8/16/24/32-bit integer PCM read+write, 32-bit float PCM read). PGM/PPM/BMP image I/O. CSV signal I/O. All converting to/from NumPy arrays. |

### Mixed-Precision Type Dispatch

Every processing function that operates on data accepts a `dtype`
parameter selecting the arithmetic configuration. Python never sees
C++ template types — it passes a string key and gets back `float64`
NumPy arrays.

```python
# Same API, different arithmetic — IIR/FIR filters
result_f32  = filt.process(signal, dtype="gpu_baseline")    # float state+sample
result_p16  = filt.process(signal, dtype="posit_full")      # posit<32,2> / posit<16,1>
result_half = filt.process(signal, dtype="half")            # cfloat<16,5> throughout

# Image processing — convolve2d, separable_filter, gaussian_blur,
# box_blur, sobel_x/y, prewitt_x/y, gradient_magnitude, canny, rgb_to_gray
edges_ref = mpdsp.canny(img, 0.1, 0.3, dtype="reference")
edges_p8  = mpdsp.canny(img, 0.1, 0.3, dtype="tiny_posit")

# Quantization — adc, measure_sqnr_db
quantized = mpdsp.adc(signal, dtype="half")

# Conditioning — PeakEnvelope, RMSEnvelope, Compressor, AGC
comp = mpdsp.Compressor(sample_rate=44100, threshold_db=-12.0, ratio=4.0,
                        attack_ms=5.0, release_ms=50.0, dtype="posit_full")

# Estimation — KalmanFilter, LMSFilter, NLMSFilter, RLSFilter
kf = mpdsp.KalmanFilter(2, 1, dtype="cf24")
```

Spectral primitives (`fft`, `ifft`, `psd`, `welch`, `periodogram`,
`spectrogram`) all accept `dtype=`; inputs and outputs stay double/
complex128 at the Python layer while the internal arithmetic runs at the
selected precision. Signal generators are intentionally reference-
precision (they aren't part of a mixed-precision datapath). Window
functions accept `dtype=` for cases where the window itself is part of a
precision study.

#### Pre-Instantiated Configurations

| Config | CoeffScalar | StateScalar | SampleScalar | Target |
|--------|-------------|-------------|--------------|--------|
| `reference` | double | double | double | Ground truth |
| `gpu_baseline` | double | float | float | GPU / embedded CPU |
| `ml_hw` | double | float | cfloat<16,5> (IEEE half) | ML accelerator |
| `posit_full` | double | posit<32,2> | posit<16,1> | Mixed-precision posit pipeline |
| `cf24` | double | cfloat<24,5> | cfloat<24,5> | Custom 24-bit float research |
| `half` | double | cfloat<16,5> | cfloat<16,5> | IEEE half throughout |
| `sensor_8bit` | double | double | integer<8> | Standard 8-bit sensor ADC |
| `sensor_6bit` | double | double | integer<6> | Noise-limited sensor |
| `fpga_fixed` | double | fixpnt<32,24> | fixpnt<16,12> | FPGA fixed-point datapath |

**Posit taxonomy grid** — `posit<N, es>` single-type configs for N ∈ {8, 16, 32},
es ∈ {0, 1, 2}. All three scalars (coefficient, state, sample) use the same
posit type, so these cells cleanly compare ES-vs-precision tradeoff at fixed
bit width:

| Config | Posit type | Notes |
|--------|-----------|-------|
| `posit_8_0` / `posit_8_1` / `posit_8_2` | `posit<8, 0/1/2>` | `posit_8_2` is canonical for 8-bit; `tiny_posit` is a legacy alias |
| `posit_16_0` / `posit_16_1` / `posit_16_2` | `posit<16, 0/1/2>` | `posit_16_1` is the standard 16-bit posit (also used as posit_full's sample) |
| `posit_32_0` / `posit_32_1` / `posit_32_2` | `posit<32, 0/1/2>` | `posit_32_2` is the standard 32-bit posit (also used as posit_full's state) |

Query the live set at runtime with `mpdsp.available_dtypes()` (18 entries).
Sample-scalar bit width per config is available via `mpdsp.bits_of(dtype)` —
useful for labeling the x-axis of precision-vs-cost plots. For posit grid
cells the ES dimension doesn't affect bit width, so every `posit_N_*` reports
N; plotting a full sweep gives 3 points stacked vertically at each width
showing ES's effect on SQNR.

Coefficients are always designed in `double` — design-time precision is
non-negotiable for IIR filters (see the
[educational guide](https://github.com/stillwater-sc/mixed-precision-dsp/blob/main/docs/topics/mixed-precision-iir-filter-design.md)).
For algorithms that don't have a design/runtime split (FFT, convolution,
Kalman), all three scalars use the target configuration.

### Visualization Toolkit

Beyond bindings, `mp-dsp-python` provides matplotlib helpers and
Jupyter notebooks tailored to mixed-precision DSP research:

| Visualization | Description |
|---------------|-------------|
| **Magnitude/phase response** | Filter frequency response overlaid across arithmetic types |
| **Impulse response** | Time-domain comparison of filter outputs |
| **SQNR heatmap** | Filter family × arithmetic type, colored by SQNR (dB) |
| **SQNR bar chart** | Grouped bars per filter family |
| **Pole-zero diagram** | Unit circle with reference vs. displaced poles |
| **Spectrogram** | Time-frequency display from STFT |
| **PSD comparison** | Power spectral density across arithmetic types |
| **Image pipeline** | Side-by-side: original → noisy → filtered → edges |
| **Sensor noise analysis** | SQNR vs. bit-width for image processing |
| **Precision-cost frontier** | SQNR vs. bits-per-sample Pareto plot |
| **Kalman tracking** | State estimation convergence across types |

### Interactive Filter Designer

A Streamlit dashboard at `scripts/plot_dashboard.py` exposes every IIR
family (Butterworth, Chebyshev I/II, Bessel, Legendre, Elliptic, RBJ
biquads) with live magnitude/phase plots, pole-zero diagrams, impulse
and step response, and a side-by-side mixed-precision comparison across
all 7 arithmetic configurations — modeled on Vinnie Falco's classic
DSPFilters demo, with the mixed-precision angle that is the whole point
of this library.

```bash
pip install mpdsp[dashboard]
streamlit run scripts/plot_dashboard.py
```

Full walkthrough (install paths for local / SSH-tunnel / LAN, tab-by-tab
tour, mixed-precision interpretation guide, export conventions) in
[`docs/dashboard.md`](docs/dashboard.md).

## How

### Repository Structure

```
mp-dsp-python/
├── CMakeLists.txt                  # nanobind + sw::dsp + Universal + MTL5
├── src/
│   ├── bindings.cpp                # nanobind module definition
│   ├── types.hpp                   # ArithConfig enum + dispatch table
│   ├── types_bindings.cpp          # TransferFunction, structured biquad types
│   ├── _binding_helpers.hpp        # Shared marshalling + dispatch helpers
│   ├── BINDING_PATTERNS.md         # Contributor notes on binding conventions
│   ├── signal_bindings.cpp         # signals + windows + WAV I/O
│   ├── filter_bindings.cpp         # IIR/FIR design, filtfilt, remez, overlap
│   ├── spectral_bindings.cpp      # FFT, PSD, welch, spectrogram
│   ├── spectrum_bindings.cpp      # analyzer stack (RealtimeSpectrum, RBW/VBW, ...)
│   ├── conditioning_bindings.cpp  # envelope, compressor, AGC, RationalResampler
│   ├── estimation_bindings.cpp    # Kalman + EKF + UKF + LMS/NLMS/RLS
│   ├── acquisition_bindings.cpp   # NCO, CIC, halfband, polyphase
│   ├── instrument_bindings.cpp    # scope measurements, PeakDetectDecimator, TriggerRingBuffer
│   ├── image_bindings.cpp         # 2D convolution, morphology, edge
│   ├── quantization_bindings.cpp  # ADC/DAC, dither, SQNR
│   ├── analysis_bindings.cpp      # stability, sensitivity, condition, acquisition-precision
│   └── math_bindings.cpp          # polynomial, quadratic, elliptic_K, RootFinder
├── python/
│   └── mpdsp/
│       ├── __init__.py             # Public API surface
│       ├── filters.py              # Pythonic filter wrapper classes
│       ├── estimation.py           # Kalman/adaptive filter wrappers
│       ├── image.py                # Image processing helpers
│       ├── analysis.py             # Analysis helpers
│       ├── plotting.py             # matplotlib convenience functions
│       └── io.py                   # File I/O + CSV import
├── notebooks/
│   ├── 02_iir_precision.ipynb          # Mixed-precision IIR comparison
│   ├── 03_fir_and_windows.ipynb        # FIR design, window functions
│   ├── 04_interactive_precision.ipynb  # Interactive precision sweep
│   ├── 05_conditioning.ipynb           # Envelope, compression, AGC
│   ├── 06_estimation.ipynb             # Kalman tracking, LMS adaptive
│   ├── 07_image_processing.ipynb       # 2D filtering, edge detection
│   ├── 08_sensor_noise.ipynb           # Sensor noise precision analysis
│   └── 09_numerical_analysis.ipynb     # Stability, sensitivity, condition
├── scripts/
│   ├── plot_precision.py           # Magnitude/phase from CSV
│   ├── plot_heatmap.py             # SQNR heatmap from CSV
│   ├── plot_pole_zero.py           # Pole-zero on unit circle
│   └── plot_dashboard.py           # Streamlit interactive dashboard
├── tests/                          # 16 test files, ~1250 tests
│   ├── test_signals.py             # generators + windows (bundled)
│   ├── test_filters.py             # IIR + FIR + Remez + Overlap + filtfilt
│   ├── test_spectral.py            # FFT, PSD, welch, spectrogram
│   ├── test_spectrum.py            # analyzer stack (RealtimeSpectrum, RBW/VBW, ...)
│   ├── test_conditioning.py        # envelope, AGC, RationalResampler
│   ├── test_estimation.py          # Kalman + EKF + UKF + adaptive
│   ├── test_acquisition.py         # NCO, CIC, halfband, polyphase
│   ├── test_instrument.py          # scope measurements + capture primitives
│   ├── test_analysis.py            # stability, sensitivity, acquisition-precision
│   ├── test_math.py                # polynomial, quadratic, RootFinder, elliptic_K
│   ├── test_types.py               # TransferFunction + structured biquad types
│   ├── test_image.py               # image processing
│   ├── test_quantization.py        # ADC/DAC, dither, SQNR
│   ├── test_io.py                  # WAV/PGM/PPM/BMP round-trips
│   ├── test_scripts.py             # CSV-plotting script smoke tests
│   └── test_version.py             # version lockstep check
├── docs/
│   ├── api_reference.md
│   ├── dashboard.md
│   ├── publishing.md
│   ├── gap_analysis_2026-08-01.md  # Pre-roadmap coverage snapshot
│   └── gap_analysis_2026-08-02.md  # Post-roadmap coverage snapshot
└── README.md
```

### Build

```bash
# Prerequisites: Python 3.9+, CMake 3.22+, C++20 compiler
pip install nanobind numpy matplotlib

# Build the C++ extension module
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Install in development mode
pip install -e .
```

The build system resolves `mixed-precision-dsp`, Universal, and MTL5 in this
order:

1. **Sibling clone** — if a checkout exists at `../mixed-precision-dsp`,
   `../universal`, or `../mtl5`, it is used directly. This is the recommended
   workflow when iterating across the C++ stack and the Python bindings
   together.
2. **`find_package`** — MTL5 only; checks for an installed package config.
3. **`FetchContent`** — pulled from GitHub at the pin tags below. Used by
   `cibuildwheel` and any other environment without local siblings.

Minimum versions enforced at configure time on the sibling-clone path
(stale checkouts abort with a clear error and the `git checkout` command
needed to fix them):

| Peer | Floor (sibling-path) | FetchContent pin |
|---|---|---|
| `mixed-precision-dsp` | ≥ 0.6.0 | `v0.6.0` |
| `universal` | ≥ 4.6.11 | `v4.6.11` |
| `mtl5` | ≥ 5.7.0 | `v5.7.0` |

_Note: the MTL5 floor was bumped from 5.2.1 → 5.7.0 in 2026-08-02 as
prep for `UnscentedKalmanFilter`, which uses `mtl::ldlt_factor` (landed
in MTL5 v5.3.0). Jumping straight to the latest 5.x release keeps the
project on current upstream._

Only the **DSP pin** is constrained to lag the floor during a development
cycle: it moves in lockstep with `project(VERSION)` (see
`tests/test_version.py::test_lockstep_prefix`) and only advances at
release time. The universal and mtl5 pins are free to track the floor —
keeping them current avoids CI building in a configuration strictly
weaker than what sibling-path devs require.

Override at configure time with `-DMPDSP_REQUIRED_DSP_VERSION=...` (lower the
floor for experimentation) or `-DMPDSP_DSP_PIN=main` (build against an
unreleased upstream).

### Quick Start: CSV Plotting (No Build Required)

The plotting scripts work immediately with CSV output from the C++
precision sweep, without building any nanobind module:

```bash
# In the mixed-precision-dsp repo:
cd build && ./applications/mp_comparison/iir_precision_sweep /tmp/csv_output

# In this repo:
python scripts/plot_precision.py /tmp/csv_output
python scripts/plot_heatmap.py /tmp/csv_output
python scripts/plot_pole_zero.py /tmp/csv_output
```

### Quick Start: Full Python API

```python
import mpdsp
import numpy as np
import matplotlib.pyplot as plt

# --- Signal Processing ---
# Generate and analyze signals
signal = mpdsp.sine(2000, frequency=440, sample_rate=44100)
window = mpdsp.blackman(2000)
freqs, psd = mpdsp.psd(signal * window, sample_rate=44100)

# --- Filtering ---
# Design and compare IIR filters across arithmetic types
filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100, cutoff=1000)
results = {}
for dtype in ["reference", "gpu_baseline", "posit_full", "half"]:
    results[dtype] = filt.process(signal, dtype=dtype)
    if dtype != "reference":
        sqnr = mpdsp.sqnr_db(results["reference"], results[dtype])
        print(f"  {dtype:20s}  SQNR = {sqnr:.1f} dB")

# --- Spectral Analysis ---
# fft / ifft / psd / periodogram / spectrogram all accept dtype=.
# Returned tuple is (real, imag).
real, imag = mpdsp.fft(signal, dtype="posit_full")

# --- Image Processing ---
# Full image pipeline
img = mpdsp.checkerboard(256, 256, block_size=16)
noisy = mpdsp.add_noise(img, stddev=0.1)
denoised = mpdsp.gaussian_blur(noisy, sigma=1.5)
edges = mpdsp.canny(denoised, low_threshold=0.1, high_threshold=0.3)

# Compare edge detection across arithmetic types
edges_ref = mpdsp.canny(denoised, 0.1, 0.3, dtype="reference")
edges_p8  = mpdsp.canny(denoised, 0.1, 0.3, dtype="tiny_posit")
agreement = np.mean(edges_ref == edges_p8)
print(f"  Edge agreement (posit<8,2>): {agreement:.1%}")

# --- Estimation ---
# Kalman filter tracking
kf = mpdsp.KalmanFilter(state_dim=4, meas_dim=2)
# configure F, H, Q, R matrices as NumPy arrays
# kf.predict(); kf.update(measurement)

# --- Analysis ---
# Numerical quality tools
print(f"  Stability margin: {filt.stability_margin():.4f}")
print(f"  Condition number: {filt.condition_number():.2e}")
print(f"  Worst sensitivity: {filt.worst_case_sensitivity():.4f}")
```

## Relationship to mixed-precision-dsp

This repository is the **Python integration layer** for the full
[stillwater-sc/mixed-precision-dsp](https://github.com/stillwater-sc/mixed-precision-dsp)
C++ library. The C++ library implements 17 DSP modules with
mixed-precision arithmetic; this repo makes essentially all of them
accessible to Python researchers (~93% of the v0.6.0 surface after
the 2026-08-02 bindings-gap roadmap; see
[`docs/gap_analysis_2026-08-02.md`](docs/gap_analysis_2026-08-02.md)
for the current coverage state and residual gaps).

### Design Documents

- [Python integration architecture](https://github.com/stillwater-sc/mixed-precision-dsp/blob/main/docs/designs/python-integration.md) — dispatch mechanism, pre-instantiated configs
- [Projection/embedding generalization](https://github.com/stillwater-sc/mixed-precision-dsp/blob/main/docs/designs/projection-embedding-generalization.md) — type conversion across domains
- [Mixed-precision IIR guide](https://github.com/stillwater-sc/mixed-precision-dsp/blob/main/docs/topics/mixed-precision-iir-filter-design.md) — numerical sensitivity primer
- [OpenCV API comparison](https://github.com/stillwater-sc/mixed-precision-dsp/blob/main/docs/assessments/image-api-opencv-comparison.md) — image processing design rationale

## Dependencies

| Library | Purpose | Repository |
|---------|---------|------------|
| [mixed-precision-dsp](https://github.com/stillwater-sc/mixed-precision-dsp) | C++ DSP algorithms (all 12 modules) | `stillwater-sc/mixed-precision-dsp` |
| [Universal](https://github.com/stillwater-sc/universal) | Number type arithmetic (posit, cfloat, fixpnt, ...) | `stillwater-sc/universal` |
| [MTL5](https://github.com/stillwater-sc/mtl5) | Dense/sparse linear algebra | `stillwater-sc/mtl5` |
| [nanobind](https://github.com/wjakob/nanobind) | C++ ↔ Python bindings | `wjakob/nanobind` |
| [NumPy](https://numpy.org/) | Array interop (all data passes through NumPy) | — |
| [matplotlib](https://matplotlib.org/) | 2D visualization | — |
| [Streamlit](https://streamlit.io/) | Interactive dashboard (optional) | — |

## License

MIT License. Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
