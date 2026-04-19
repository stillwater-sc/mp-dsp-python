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

`mp-dsp-python` exposes every module of the C++ library to Python:

Bindings marked ✓ are available today; ⏳ are planned for `0.5.0` (see issue
tracker for the per-module roadmap). For the complete enumeration of every
public name with signatures and one-line descriptions, see
[`docs/api_reference.md`](docs/api_reference.md).

| Module | C++ Headers | Python API | Description |
|--------|-------------|------------|-------------|
| **signals** ✓ | `generators.hpp`, `signal.hpp`, `sampling.hpp` | `mpdsp.sine()`, `mpdsp.chirp()`, `mpdsp.impulse()`, `mpdsp.white_noise()`, `mpdsp.gaussian_noise()`, `mpdsp.pink_noise()`, ... | Signal generators returning NumPy arrays. |
| **windows** ✓ | `hamming.hpp`, `hanning.hpp`, `blackman.hpp`, `kaiser.hpp`, ... | `mpdsp.hamming()`, `mpdsp.kaiser()`, ... | Window functions returning NumPy arrays. |
| **quantization** ✓ | `adc.hpp`, `dac.hpp`, `dither.hpp`, `noise_shaping.hpp`, `sqnr.hpp` | `mpdsp.adc()`, `mpdsp.dac()`, `mpdsp.sqnr_db()`, `mpdsp.measure_sqnr_db()`, `mpdsp.RPDFDither()`, `mpdsp.TPDFDither()`, `mpdsp.FirstOrderNoiseShaper()`, ... | ADC/DAC modeling with type dispatch. RPDF/TPDF dithering and first-order error-feedback noise shaping for quantization improvement. SQNR measurement — the core metric for mixed-precision evaluation. |
| **filter/iir** ✓ | `butterworth.hpp`, `chebyshev1.hpp`, `chebyshev2.hpp`, `elliptic.hpp`, `bessel.hpp`, `legendre.hpp`, `rbj.hpp` | `mpdsp.butterworth_lowpass()`, `mpdsp.chebyshev1_highpass()`, `mpdsp.elliptic_bandpass()`, `mpdsp.rbj_lowshelf()`, ... | All 7 IIR families with LP/HP/BP/BS (and RBJ shelf/allpass) variants. Design in double, process with type dispatch. Filter objects expose `poles()`, `frequency_response()`, `stability_margin()`, `condition_number()`, `pole_displacement()`, `worst_case_sensitivity()` as methods. |
| **filter/fir** ✓ | `fir_filter.hpp`, `fir_design.hpp` | `mpdsp.fir_lowpass()`, `mpdsp.fir_bandpass()`, `mpdsp.fir_filter()`, ... | FIR filter design (window method). Direct convolution. |
| **spectral** ✓ partial | `fft.hpp`, `dft.hpp`, `psd.hpp`, `spectrogram.hpp`, `ztransform.hpp`, `laplace.hpp` | `mpdsp.fft()`, `mpdsp.ifft()`, `mpdsp.fft_magnitude_db()`, `mpdsp.psd()`, `mpdsp.periodogram()`, `mpdsp.spectrogram()` (⏳ `ztransform`, `laplace`) | FFT (Cooley-Tukey), power spectral density, STFT/spectrogram. |
| **conditioning** ✓ | `envelope.hpp`, `compressor.hpp`, `agc.hpp` | `mpdsp.PeakEnvelope()`, `mpdsp.RMSEnvelope()`, `mpdsp.Compressor()`, `mpdsp.AGC()` | Envelope followers (peak, RMS). Dynamic range compressor with soft knee. Automatic gain control. |
| **estimation** ✓ | `kalman.hpp`, `lms.hpp`, `rls.hpp` | `mpdsp.KalmanFilter()`, `mpdsp.LMSFilter()`, `mpdsp.NLMSFilter()`, `mpdsp.RLSFilter()` | Linear Kalman filter with predict/update. LMS/NLMS adaptive filters. RLS with forgetting factor. State matrices as NumPy 2D arrays. |
| **image** ✓ | `image.hpp`, `convolve2d.hpp`, `separable.hpp`, `morphology.hpp`, `edge.hpp`, `generators.hpp` | `mpdsp.convolve2d()`, `mpdsp.gaussian_blur()`, `mpdsp.sobel_x()`, `mpdsp.canny()`, `mpdsp.dilate()`, `mpdsp.checkerboard()`, ... | 2D convolution, separable filters, Gaussian/box blur. Morphological operations (erode, dilate, open, close, gradient, tophat). Sobel, Prewitt, Canny edge detection. Image generators (checkerboard, zone plate, gradients, noise, blobs). |
| **io** ✓ | `wav.hpp`, `csv.hpp`, `pgm.hpp`, `ppm.hpp`, `bmp.hpp` | `mpdsp.read_wav()`, `mpdsp.write_wav()`, `mpdsp.read_pgm()`, `mpdsp.write_pgm()`, `mpdsp.read_ppm()`, `mpdsp.write_ppm()`, `mpdsp.read_bmp()`, `mpdsp.write_bmp()`, CSV via `mpdsp.load_sweep()` | WAV audio (8/16/24/32-bit integer PCM read+write, 32-bit float PCM read). PGM/PPM/BMP image I/O. CSV signal I/O. All converting to/from NumPy arrays. |
| **analysis** ✓ via filter methods | `stability.hpp`, `sensitivity.hpp`, `condition.hpp` | `filt.stability_margin()`, `filt.condition_number()`, `filt.worst_case_sensitivity()`, `filt.pole_displacement(dtype)` | Stability/sensitivity/conditioning analysis exposed as methods on filter objects rather than free functions. `pole_displacement` takes a dtype string and reports how far poles drift under that arithmetic. |
| **types** ✓ | `projection.hpp`, `transfer_function.hpp` | `mpdsp.TransferFunction()`, `mpdsp.project_onto()`, `mpdsp.projection_error()`, `mpdsp.to_transfer_function(filt)` | Rational transfer function H(z) = B(z)/A(z) with complex-plane evaluation, frequency response, stability check, and cascade via `*`. Type-projection round-trip primitives for quantifying quantization loss outside the filter path. |

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

Spectral primitives (`fft`, `psd`, `spectrogram`, ...) are pure double-precision
in `0.4.x` — mixed-precision dispatch on transforms is planned for `0.5.0`.
Signal generators and window functions are intentionally reference-precision
(they are not part of a mixed-precision datapath).

#### Pre-Instantiated Configurations

| Config | CoeffScalar | StateScalar | SampleScalar | Target |
|--------|-------------|-------------|--------------|--------|
| `reference` | double | double | double | Ground truth |
| `gpu_baseline` | double | float | float | GPU / embedded CPU |
| `ml_hw` | double | float | cfloat<16,5> (IEEE half) | ML accelerator |
| `posit_full` | double | posit<32,2> | posit<16,1> | Posit arithmetic research |
| `tiny_posit` | double | posit<8,2> | posit<8,2> | Ultra-low-power edge |
| `cf24` | double | cfloat<24,5> | cfloat<24,5> | Custom 24-bit float research |
| `half` | double | cfloat<16,5> | cfloat<16,5> | IEEE half throughout |

Query the live set at runtime with `mpdsp.available_dtypes()`. Fixed-point
and integer configurations (sensor ADCs, FPGA datapaths) are planned for
`0.5.0` alongside the remaining upstream bindings.

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
├── CMakeLists.txt                  # nanobind + sw::dsp + Universal
├── src/
│   ├── bindings.cpp                # nanobind module definition
│   ├── types.hpp                   # ArithConfig enum + dispatch table
│   ├── signal_bindings.cpp         # signals + windows → NumPy
│   ├── filter_bindings.cpp         # IIR/FIR design + process
│   ├── spectral_bindings.cpp       # FFT, PSD, spectrogram
│   ├── conditioning_bindings.cpp   # envelope, compressor, AGC
│   ├── estimation_bindings.cpp     # Kalman, LMS, RLS
│   ├── image_bindings.cpp          # 2D convolution, morphology, edge
│   ├── quantization_bindings.cpp   # ADC/DAC, dither, SQNR
│   ├── analysis_bindings.cpp       # stability, sensitivity, condition
│   └── io_bindings.cpp             # WAV, PGM, PPM, BMP, CSV
├── python/
│   └── mpdsp/
│       ├── __init__.py             # Public API surface
│       ├── filters.py              # Pythonic filter wrapper classes
│       ├── spectral.py             # Spectral analysis helpers
│       ├── estimation.py           # Kalman/adaptive filter wrappers
│       ├── image.py                # Image processing helpers
│       ├── plotting.py             # matplotlib convenience functions
│       └── io.py                   # File I/O + CSV import
├── notebooks/
│   ├── 01_signals_and_spectra.ipynb    # Signal generation, FFT, PSD
│   ├── 02_iir_precision.ipynb          # Mixed-precision IIR comparison
│   ├── 03_fir_and_windows.ipynb        # FIR design, window functions
│   ├── 04_quantization.ipynb           # ADC/DAC, dithering, SQNR
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
├── tests/
│   ├── test_signals.py
│   ├── test_filters.py
│   ├── test_spectral.py
│   ├── test_image.py
│   └── test_estimation.py
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

The build system finds `mixed-precision-dsp`, Universal, and MTL5 via
CMake `find_package` or `FetchContent`.

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
# FFT is pure double-precision in 0.4.x (mixed-precision dispatch on
# transforms is planned for 0.5.0). The returned tuple is (real, imag).
real, imag = mpdsp.fft(signal)

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
C++ library. The C++ library implements 12 DSP modules with
mixed-precision arithmetic; this repo makes all of them accessible
to Python researchers.

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
| [Streamlit](https://streamlit.io/) | Interactive dashboard (Phase 7) | — |

## License

MIT License. Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
