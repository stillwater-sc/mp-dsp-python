# mp-dsp-python

Python integration layer for
[mixed-precision-dsp](https://github.com/stillwater-sc/mixed-precision-dsp),
providing nanobind bindings, matplotlib visualizations, and Jupyter notebooks
for mixed-precision DSP research.

## Why

Digital signal processing researchers work in Python. Jupyter notebooks,
matplotlib, and SciPy are the standard tools for prototyping, analysis,
and publication-quality visualization. The mixed-precision DSP library
(`sw::dsp`) is a C++20 header-only library where the core value
proposition — demonstrating that narrower arithmetic types can be
*sufficient* for real workloads — requires quantitative evidence
presented as SQNR tables, frequency response overlays, and pole-zero
diagrams.

The C++ library produces the data. Python presents it.

Without Python bindings, a researcher would need to:
1. Write a C++ application for each experiment
2. Export CSV manually
3. Write custom Python scripts to parse and plot

With `mp-dsp-python`, the workflow becomes:

```python
import mpdsp
import matplotlib.pyplot as plt

filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100, cutoff=1000)
signal = mpdsp.sine(length=2000, frequency=440, sample_rate=44100)

ref    = filt.process(signal, dtype="reference")
posit  = filt.process(signal, dtype="posit_full")
sensor = filt.process(signal, dtype="sensor_6bit")

print(f"posit SQNR:  {mpdsp.sqnr_db(ref, posit):.1f} dB")
print(f"6-bit SQNR:  {mpdsp.sqnr_db(ref, sensor):.1f} dB")

plt.plot(ref, label="double")
plt.plot(posit, label="posit<32,2>")
plt.legend()
plt.show()
```

## What

### Architecture: Python Orchestrates, C++ Crunches

The C++ library has three-scalar parameterization:
`ButterworthLowPass<Order, CoeffScalar, StateScalar, SampleScalar>`.
Naively exposing all type combinations to Python would require 6^3 = 216
instantiations per filter family — tens of thousands total. Build times
would be hours and the binary enormous.

Instead, we pre-instantiate a fixed set of **arithmetic configurations**
that represent the research-relevant type combinations. Python selects
a configuration by name; C++ dispatches to the correct template
instantiation internally. Results always come back as `float64` NumPy
arrays.

### Pre-Instantiated Configurations

| Config | CoeffScalar | StateScalar | SampleScalar | Research Question |
|--------|-------------|-------------|--------------|-------------------|
| `reference` | double | double | double | Ground truth |
| `gpu_baseline` | double | float | float | GPU/embedded baseline |
| `ml_hw` | double | float | bfloat16 | ML accelerator |
| `sensor_8bit` | double | double | integer<8> | Standard ADC |
| `sensor_6bit` | double | double | integer<6> | Noise-limited sensor |
| `posit_full` | double | posit<32,2> | posit<16,1> | Posit research |
| `fpga_fixed` | double | fixpnt<32,24> | fixpnt<16,12> | FPGA datapath |
| `tiny_posit` | double | posit<8,2> | posit<8,2> | Ultra-low power |

Coefficients are always designed in `double` — design-time precision is
non-negotiable (see the
[educational guide](https://github.com/stillwater-sc/mixed-precision-dsp/blob/main/docs/topics/mixed-precision-iir-filter-design.md)
for why). The mixed precision happens in state variables and sample
processing.

### Components

| Component | Description |
|-----------|-------------|
| **C++ bindings** (`src/`) | nanobind module wrapping `sw::dsp` with runtime type dispatch |
| **Python package** (`python/mpdsp/`) | Pythonic API, matplotlib helpers, convenience wrappers |
| **Notebooks** (`notebooks/`) | Interactive Jupyter notebooks for each research question |
| **Scripts** (`scripts/`) | Standalone plotting scripts consuming CSV from the C++ sweep |

## How

### Repository Structure

```
mp-dsp-python/
├── CMakeLists.txt                # nanobind + sw::dsp + Universal
├── src/
│   ├── bindings.cpp              # nanobind module definition
│   ├── types.hpp                 # ArithConfig enum + dispatch
│   ├── signal_bindings.cpp       # generators → NumPy
│   ├── filter_bindings.cpp       # IIR/FIR design + process
│   ├── image_bindings.cpp        # image ops → NumPy 2D
│   └── analysis_bindings.cpp     # stability, sensitivity, SQNR
├── python/
│   └── mpdsp/
│       ├── __init__.py
│       ├── filters.py            # Pythonic filter wrapper classes
│       ├── plotting.py           # matplotlib convenience functions
│       └── io.py                 # CSV import from C++ sweep
├── notebooks/
│   ├── 01_iir_precision.ipynb    # Mixed-precision IIR comparison
│   ├── 02_sensor_noise.ipynb     # Sensor noise image processing
│   ├── 03_sqnr_comparison.ipynb  # SQNR across filter families
│   └── 04_pole_zero.ipynb        # Pole-zero displacement visualization
├── scripts/
│   ├── plot_precision.py         # Magnitude/phase/impulse from CSV
│   ├── plot_heatmap.py           # Filter × type SQNR heatmap
│   ├── plot_pole_zero.py         # Pole-zero on unit circle
│   └── plot_dashboard.py         # Streamlit interactive dashboard
├── tests/
│   └── test_basic.py
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

Before building the nanobind module, you can use the Python scripts
directly with CSV output from the C++ precision sweep:

```bash
# In the mixed-precision-dsp repo:
cd build && ./applications/mp_comparison/iir_precision_sweep /tmp/csv_output

# In this repo:
python scripts/plot_precision.py /tmp/csv_output
python scripts/plot_heatmap.py /tmp/csv_output
python scripts/plot_pole_zero.py /tmp/csv_output
```

This two-step workflow (C++ generates CSV, Python plots) works
immediately without building any bindings.

### Quick Start: Nanobind Module

```python
import mpdsp

# Design a filter (always in double)
filt = mpdsp.butterworth_lowpass(order=4, sample_rate=44100, cutoff=1000)

# Generate test signal
signal = mpdsp.sine(length=2000, frequency=440, sample_rate=44100)

# Compare arithmetic types
configs = ["reference", "gpu_baseline", "posit_full", "sensor_6bit"]
results = {c: filt.process(signal, dtype=c) for c in configs}

# SQNR comparison
ref = results["reference"]
for name, result in results.items():
    if name != "reference":
        sqnr = mpdsp.sqnr_db(ref, result)
        print(f"  {name:20s}  SQNR = {sqnr:.1f} dB")

# Analysis
print(f"  Stability margin: {filt.stability_margin():.4f}")
print(f"  Worst-case sensitivity: {filt.worst_case_sensitivity():.4f}")
poles = filt.poles()  # list of complex numbers
```

## Phased Implementation

### Phase 1: CSV Visualization Scripts
**No C++ build required.** Python scripts that consume the three CSV
files from `iir_precision_sweep`:

- `iir_precision_sweep.csv` — summary metrics (30 rows)
- `frequency_response.csv` — magnitude/phase at 200 frequencies (6000 rows)
- `pole_positions.csv` — complex pole locations (120 rows)

Deliverables:
- `scripts/plot_precision.py` — magnitude response overlay per filter family
- `scripts/plot_heatmap.py` — filter × type SQNR heatmap
- `scripts/plot_pole_zero.py` — poles on unit circle with displacement

### Phase 2: Signal + SQNR Bindings
nanobind module with signal generators and quantization analysis:
- Signal generators (sine, chirp, noise) → NumPy arrays
- ADC/DAC quantization with type dispatch
- SQNR measurement

### Phase 3: IIR Filter Bindings
Filter design + mixed-precision processing:
- All 7 IIR families (Butterworth through RBJ)
- `process()` with type dispatch across 8 configs
- Frequency response, pole-zero access, stability analysis

### Phase 4: Image Processing Bindings
Image generators, convolution, edge detection:
- Generators → NumPy 2D arrays
- Sobel, Canny, morphology
- PGM/PPM/BMP I/O

### Phase 5: Interactive Dashboard
Streamlit or Panel web application for parameter sweeping:
- Select filter family, order, cutoff
- Compare arithmetic types in real time
- Publication-quality figure export

## Relationship to mixed-precision-dsp

This repository is the **visualization and orchestration layer** for
[stillwater-sc/mixed-precision-dsp](https://github.com/stillwater-sc/mixed-precision-dsp).
The C++ library does the mixed-precision math; this repo makes the
results accessible to Python researchers.

### Cross-Repository Issues

**C++ library issues that produce data for Python visualization:**

| C++ Issue | Status | What it produces |
|-----------|--------|-----------------|
| [dsp#22](https://github.com/stillwater-sc/mixed-precision-dsp/issues/22) | Epic | Mixed-precision IIR comparison (parent) |
| [dsp#23](https://github.com/stillwater-sc/mixed-precision-dsp/issues/23) | Merged | Precision sweep app (console tables) |
| [dsp#24](https://github.com/stillwater-sc/mixed-precision-dsp/issues/24) | Merged | CSV export (3 files for Python) |
| [dsp#41](https://github.com/stillwater-sc/mixed-precision-dsp/issues/41) | Open | Sensor noise image precision demo |
| [dsp#46](https://github.com/stillwater-sc/mixed-precision-dsp/issues/46) | Open | nanobind architecture design |
| [dsp#47](https://github.com/stillwater-sc/mixed-precision-dsp/issues/47) | Merged | project_onto/embed_into operators |

**Python visualization issues (tracked in dsp repo, implemented here):**

| Issue | Description | Phase |
|-------|-------------|-------|
| [dsp#25](https://github.com/stillwater-sc/mixed-precision-dsp/issues/25) | Magnitude, phase, impulse plots from CSV | 1 |
| [dsp#26](https://github.com/stillwater-sc/mixed-precision-dsp/issues/26) | Heatmap and SQNR bar chart | 1 |
| [dsp#27](https://github.com/stillwater-sc/mixed-precision-dsp/issues/27) | Pole-zero displacement visualization | 1 |
| [dsp#28](https://github.com/stillwater-sc/mixed-precision-dsp/issues/28) | Jupyter notebook for interactive exploration | 3 |
| [dsp#29](https://github.com/stillwater-sc/mixed-precision-dsp/issues/29) | Web dashboard for parameter sweeping | 5 |

### Design Documents

- [Python integration architecture](https://github.com/stillwater-sc/mixed-precision-dsp/blob/main/docs/designs/python-integration.md) — dispatch mechanism, pre-instantiated configs, build system
- [Projection/embedding generalization](https://github.com/stillwater-sc/mixed-precision-dsp/blob/main/docs/designs/projection-embedding-generalization.md) — type conversion operators across domains
- [Mixed-precision IIR filter design guide](https://github.com/stillwater-sc/mixed-precision-dsp/blob/main/docs/topics/mixed-precision-iir-filter-design.md) — educational primer on numerical sensitivity

## Dependencies

| Library | Purpose | Repository |
|---------|---------|------------|
| [mixed-precision-dsp](https://github.com/stillwater-sc/mixed-precision-dsp) | C++ DSP algorithms | `stillwater-sc/mixed-precision-dsp` |
| [Universal](https://github.com/stillwater-sc/universal) | Number type arithmetic | `stillwater-sc/universal` |
| [MTL5](https://github.com/stillwater-sc/mtl5) | Linear algebra | `stillwater-sc/mtl5` |
| [nanobind](https://github.com/wjakob/nanobind) | C++ ↔ Python bindings | `wjakob/nanobind` |
| [NumPy](https://numpy.org/) | Array interop | — |
| [matplotlib](https://matplotlib.org/) | Visualization | — |

## License

MIT License. Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
