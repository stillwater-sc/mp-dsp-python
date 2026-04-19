# `mpdsp` API reference

Complete enumeration of every public name in the `mpdsp` package, grouped
by subsystem. Generated from `0.4.1.post2` (upstream `sw::dsp
0.4.1`) via `inspect` and the nanobind-attached
`__doc__` strings. Keep this in sync by re-running the generator — see
the note at the bottom.

---

## Contents

- [Arithmetic configurations](#arithmetic-configurations)
- [Module attributes](#module-attributes)
- [Signal generators](#signal-generators)
- [Window functions](#window-functions)
- [Quantization](#quantization)
- [Spectral analysis](#spectral-analysis)
- [IIR filter design — classical families](#iir-filter-design--classical-families)
- [IIR filter design — RBJ biquads](#iir-filter-design--rbj-biquads)
- [FIR filter design](#fir-filter-design)
- [Image — generators](#image--generators)
- [Image — processing](#image--processing)
- [Image — morphology](#image--morphology)
- [Image — file I/O](#image--file-io)
- [Audio — WAV file I/O](#audio--wav-file-io)
- [Types — transfer function and type projection](#types--transfer-function-and-type-projection)
- [Numerical-analysis helpers (pure Python)](#numerical-analysis-helpers-pure-python)
- [Mixed-precision helpers](#mixed-precision-helpers)
- [CSV + image-pipeline helpers (pure Python)](#csv--image-pipeline-helpers-pure-python)
- [Matplotlib plotting helpers](#matplotlib-plotting-helpers)
- [Classes](#classes)
  - [`IIRFilter`](#iirfilter)
  - [`FIRFilter`](#firfilter)
  - [`RPDFDither`](#rpdfdither)
  - [`TPDFDither`](#tpdfdither)
  - [`FirstOrderNoiseShaper`](#firstordernoiseshaper)
  - [`PeakEnvelope`](#peakenvelope)
  - [`RMSEnvelope`](#rmsenvelope)
  - [`Compressor`](#compressor)
  - [`AGC`](#agc)
  - [`KalmanFilter`](#kalmanfilter)
  - [`LMSFilter`](#lmsfilter)
  - [`NLMSFilter`](#nlmsfilter)
  - [`RLSFilter`](#rlsfilter)
  - [`TransferFunction`](#transferfunction)

---

## Arithmetic configurations

Every `dtype=` parameter across the API (on `filt.process`, `canny`,
`adc`, the conditioning/estimation constructors, etc.) accepts one of
these string keys. Query the live set at runtime with
`mpdsp.available_dtypes()`.

| Key | CoeffScalar | StateScalar | SampleScalar | Target |
|-----|-------------|-------------|--------------|--------|
| `reference` | double | double | double | Ground truth |
| `gpu_baseline` | double | float | float | GPU / embedded CPU |
| `ml_hw` | double | float | cfloat<16,5> (half) | ML accelerator |
| `posit_full` | double | posit<32,2> | posit<16,1> | Posit research |
| `tiny_posit` | double | posit<8,2> | posit<8,2> | Ultra-low-power edge |
| `cf24` | double | cfloat<24,5> | cfloat<24,5> | Custom 24-bit float |
| `half` | double | cfloat<16,5> | cfloat<16,5> | IEEE half throughout |

Planned for `0.5.0` (issue #40): fixed-point / integer configs
(`sensor_8bit`, `fpga_fixed`, ...) and dtype dispatch on the spectral
transforms.

## Module attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mpdsp.__version__` | `str` | The installed wheel version (PEP 440). Current: `"0.4.1.post2"`. |
| `mpdsp.__dsp_version__` | `str` | The upstream `sw::dsp` C++ library version the wheel was built against. Current: `"0.4.1"`. |
| `mpdsp.__dsp_version_info__` | `tuple` | `(major, minor, patch)` tuple of ints for `__dsp_version__`. |
| `mpdsp.HAS_CORE` | `bool` | `True` when the nanobind extension imported cleanly. `False` in unbuilt source checkouts, and (pre-0.4.1.post1) indicated a packaging bug before we hardened the import. |
| `mpdsp.HAS_PLOT` | `bool` | `True` when matplotlib is importable — gates the `plot_*` helpers. |
| `mpdsp.__core_import_error__` | `NoneType` | `None` if `HAS_CORE`; otherwise the exception raised when `_core` failed to import. |

## Signal generators

Return a 1D float64 NumPy array. All generators except the noise family accept deterministic parameters; `white_noise`, `gaussian_noise`, and `pink_noise` additionally take a `seed` argument (default 0 → nondeterministic from `std::random_device`).

| Name | Signature | Description |
|------|-----------|-------------|
| `sine` | `(length: int, frequency: float, sample_rate: float, amplitude: float = 1.0, phase: float = 0.0) -> ndarray` | Generate a sine wave. Returns NumPy float64 array. |
| `cosine` | `(length: int, frequency: float, sample_rate: float, amplitude: float = 1.0, phase: float = 0.0) -> ndarray` | Generate a cosine wave. |
| `chirp` | `(length: int, f_start: float, f_end: float, sample_rate: float, amplitude: float = 1.0) -> ndarray` | Generate a linear chirp (frequency sweep). |
| `square` | `(length: int, frequency: float, sample_rate: float, amplitude: float = 1.0) -> ndarray` | Generate a square wave. |
| `triangle` | `(length: int, frequency: float, sample_rate: float, amplitude: float = 1.0) -> ndarray` | Generate a triangle wave. |
| `sawtooth` | `(length: int, frequency: float, sample_rate: float, amplitude: float = 1.0) -> ndarray` | Generate a sawtooth wave. |
| `impulse` | `(length: int, position: int = 0) -> ndarray` | Generate an impulse (single 1.0 at position, rest 0). |
| `step` | `(length: int, position: int = 0) -> ndarray` | Generate a unit step (0 before position, 1 from position onward). |
| `white_noise` | `(length: int, amplitude: float = 1.0, seed: int = 0) -> ndarray` | Generate white noise (uniform in [-amplitude, amplitude]). |
| `gaussian_noise` | `(length: int, stddev: float = 1.0, seed: int = 0) -> ndarray` | Generate Gaussian white noise (mean=0, normal distribution with given stddev). |
| `pink_noise` | `(length: int, amplitude: float = 1.0, seed: int = 0) -> ndarray` | Generate pink noise (1/f spectrum, Voss-McCartney algorithm). |

## Window functions

Return a length-N float64 NumPy array. Apply by element-wise multiplication against a signal before spectral analysis. `kaiser` additionally takes a shape parameter `beta`.

| Name | Signature | Description |
|------|-----------|-------------|
| `hamming` | `(N: int) -> ndarray` | Hamming window of length N. |
| `hanning` | `(N: int) -> ndarray` | Hanning (Hann) window of length N. |
| `blackman` | `(N: int) -> ndarray` | Blackman window of length N. |
| `kaiser` | `(N: int, beta: float = 5.0) -> ndarray` | Kaiser window of length N with shape parameter beta. |
| `rectangular` | `(N: int) -> ndarray` | Rectangular (boxcar) window of length N. |
| `flat_top` | `(N: int) -> ndarray` | Flat-top window of length N. |

## Quantization

`adc` / `dac` round-trip a signal through the target precision — ADC models the quantization step, DAC the reconstruction step (in Python, both sides are float64, so they're mechanically symmetric but serve different roles in pipeline code). `RPDFDither`, `TPDFDither` (stateful classes in the Classes section below) add decorrelating noise before quantization; `FirstOrderNoiseShaper` pushes quantization-noise energy out of the signal band via error feedback. The remaining primitives measure how far a quantized signal drifted from its reference.

| Name | Signature | Description |
|------|-----------|-------------|
| `adc` | `(signal: ndarray1d[ro], dtype: str = 'reference') -> ndarray` | Quantize signal through target type (double -> T -> double). |
| `dac` | `(quantized: ndarray1d[ro], dtype: str = 'reference') -> ndarray` | Reconstruct a quantized signal through target type (T -> double). Companion to adc(): in Python both sides are float64 so the call is mechanically symmetric to adc, but dac models the DAC reconstruction step of a full ADC->DAC pipeline explicitly. |
| `sqnr_db` | `(reference: ndarray1d[ro], quantized: ndarray1d[ro]) -> float` | Compute SQNR (dB) between reference and quantized signals. |
| `measure_sqnr_db` | `(signal: ndarray1d[ro], dtype: str) -> float` | Measure SQNR of a signal after ADC round-trip through target type. |
| `max_absolute_error` | `(reference: ndarray1d[ro], test: ndarray1d[ro]) -> float` | Maximum absolute error between two signals. |
| `max_relative_error` | `(reference: ndarray1d[ro], test: ndarray1d[ro]) -> float` | Maximum relative error between two signals. |

## Spectral analysis

All spectral primitives in 0.4.x operate in double precision — dtype dispatch on FFT/PSD/spectrogram is planned for 0.5.0 (see issue #40). Signal inputs must be 1D float64.

| Name | Signature | Description |
|------|-----------|-------------|
| `fft` | `(signal: ndarray1d[ro]) -> tuple` | Compute FFT of a real signal. Returns (real, imag) tuple of NumPy arrays. |
| `ifft` | `(real: ndarray1d[ro], imag: ndarray1d[ro]) -> ndarray` | Compute inverse FFT from (real, imag) arrays. Returns real signal. |
| `fft_magnitude_db` | `(signal: ndarray1d[ro]) -> ndarray` | Compute FFT magnitude spectrum in dB. |
| `psd` | `(signal: ndarray1d[ro], sample_rate: float) -> tuple` | Compute PSD with frequency axis. Returns (freqs_hz, power) tuple. |
| `periodogram` | `(signal: ndarray1d[ro]) -> ndarray` | Compute periodogram power spectral density estimate. |
| `spectrogram` | `(signal: ndarray1d[ro], sample_rate: float, window_size: int = 1024, hop_size: int = 256) -> tuple` | Compute spectrogram. Returns (times, freqs, magnitude_db) tuple. magnitude_db is a 2D array [n_frames x n_freqs]. |

## IIR filter design — classical families

Each function designs the filter in double precision and returns an `IIRFilter` object whose `.process(signal, dtype=...)` method dispatches through the target arithmetic. Chebyshev I, Chebyshev II, and Elliptic take additional passband-ripple / stopband-attenuation parameters.

| Name | Signature | Description |
|------|-----------|-------------|
| `butterworth_lowpass` | `(order: int, sample_rate: float, cutoff: float) -> mpdsp._core.IIRFilter` | Design a Butterworth lowpass filter. order in [1, 16]. |
| `butterworth_highpass` | `(order: int, sample_rate: float, cutoff: float) -> mpdsp._core.IIRFilter` | Design a Butterworth highpass filter. order in [1, 16]. |
| `butterworth_bandpass` | `(order: int, sample_rate: float, center_freq: float, width_freq: float) -> mpdsp._core.IIRFilter` | Design a Butterworth bandpass filter. order in [1, 8] (the bandpass transform doubles the internal order). |
| `butterworth_bandstop` | `(order: int, sample_rate: float, center_freq: float, width_freq: float) -> mpdsp._core.IIRFilter` | Design a Butterworth bandstop filter. order in [1, 8]. |
| `chebyshev1_lowpass` | `(order: int, sample_rate: float, cutoff: float, ripple_db: float) -> mpdsp._core.IIRFilter` | Design a Chebyshev Type I lowpass filter with equiripple passband. |
| `chebyshev1_highpass` | `(order: int, sample_rate: float, cutoff: float, ripple_db: float) -> mpdsp._core.IIRFilter` | Design a Chebyshev Type I highpass filter with equiripple passband. |
| `chebyshev1_bandpass` | `(order: int, sample_rate: float, center_freq: float, width_freq: float, ripple_db: float) -> mpdsp._core.IIRFilter` | Design a Chebyshev Type I bandpass filter. |
| `chebyshev1_bandstop` | `(order: int, sample_rate: float, center_freq: float, width_freq: float, ripple_db: float) -> mpdsp._core.IIRFilter` | Design a Chebyshev Type I bandstop filter. |
| `chebyshev2_lowpass` | `(order: int, sample_rate: float, cutoff: float, stopband_db: float) -> mpdsp._core.IIRFilter` | Design an inverse Chebyshev (Type II) lowpass filter with equiripple stopband. |
| `chebyshev2_highpass` | `(order: int, sample_rate: float, cutoff: float, stopband_db: float) -> mpdsp._core.IIRFilter` | Design an inverse Chebyshev (Type II) highpass filter. |
| `chebyshev2_bandpass` | `(order: int, sample_rate: float, center_freq: float, width_freq: float, stopband_db: float) -> mpdsp._core.IIRFilter` | Design an inverse Chebyshev (Type II) bandpass filter. |
| `chebyshev2_bandstop` | `(order: int, sample_rate: float, center_freq: float, width_freq: float, stopband_db: float) -> mpdsp._core.IIRFilter` | Design an inverse Chebyshev (Type II) bandstop filter. |
| `bessel_lowpass` | `(order: int, sample_rate: float, cutoff: float) -> mpdsp._core.IIRFilter` | Design a Bessel (Thomson) lowpass filter — maximally flat group delay. |
| `bessel_highpass` | `(order: int, sample_rate: float, cutoff: float) -> mpdsp._core.IIRFilter` | Design a Bessel highpass filter. |
| `bessel_bandpass` | `(order: int, sample_rate: float, center_freq: float, width_freq: float) -> mpdsp._core.IIRFilter` | Design a Bessel bandpass filter. |
| `bessel_bandstop` | `(order: int, sample_rate: float, center_freq: float, width_freq: float) -> mpdsp._core.IIRFilter` | Design a Bessel bandstop filter. |
| `legendre_lowpass` | `(order: int, sample_rate: float, cutoff: float) -> mpdsp._core.IIRFilter` | Design a Legendre (Papoulis) lowpass filter — steepest monotonic passband response. |
| `legendre_highpass` | `(order: int, sample_rate: float, cutoff: float) -> mpdsp._core.IIRFilter` | Design a Legendre highpass filter. |
| `legendre_bandpass` | `(order: int, sample_rate: float, center_freq: float, width_freq: float) -> mpdsp._core.IIRFilter` | Design a Legendre bandpass filter. |
| `legendre_bandstop` | `(order: int, sample_rate: float, center_freq: float, width_freq: float) -> mpdsp._core.IIRFilter` | Design a Legendre bandstop filter. |
| `elliptic_lowpass` | `(order: int, sample_rate: float, cutoff: float, ripple_db: float, rolloff: float = 1.0) -> mpdsp._core.IIRFilter` | Design an Elliptic (Cauer) lowpass filter — equiripple in both passband and stopband. rolloff in [0.1, 5.0] controls transition selectivity (higher = steeper). |
| `elliptic_highpass` | `(order: int, sample_rate: float, cutoff: float, ripple_db: float, rolloff: float = 1.0) -> mpdsp._core.IIRFilter` | Design an Elliptic highpass filter. rolloff in [0.1, 5.0]. |
| `elliptic_bandpass` | `(order: int, sample_rate: float, center_freq: float, width_freq: float, ripple_db: float, rolloff: float = 1.0) -> mpdsp._core.IIRFilter` | Design an Elliptic bandpass filter. |
| `elliptic_bandstop` | `(order: int, sample_rate: float, center_freq: float, width_freq: float, ripple_db: float, rolloff: float = 1.0) -> mpdsp._core.IIRFilter` | Design an Elliptic bandstop filter. |

## IIR filter design — RBJ biquads

Robert Bristow-Johnson audio-EQ biquads. Always 2nd-order (no `order` parameter). Include shelf and allpass topologies not present in the classical families. Parameterized by `q` (quality factor) or `bandwidth` (for BP/BS); shelves take `gain_db`.

| Name | Signature | Description |
|------|-----------|-------------|
| `rbj_lowpass` | `(sample_rate: float, cutoff: float, q: float = 0.7071) -> mpdsp._core.IIRFilter` | RBJ biquad lowpass. q ~ 0.7071 gives a Butterworth-like response. |
| `rbj_highpass` | `(sample_rate: float, cutoff: float, q: float = 0.7071) -> mpdsp._core.IIRFilter` | RBJ biquad highpass. |
| `rbj_bandpass` | `(sample_rate: float, center_freq: float, bandwidth: float = 1.0) -> mpdsp._core.IIRFilter` | RBJ biquad bandpass. bandwidth is in octaves. |
| `rbj_bandstop` | `(sample_rate: float, center_freq: float, bandwidth: float = 1.0) -> mpdsp._core.IIRFilter` | RBJ biquad bandstop (notch). bandwidth is in octaves. |
| `rbj_allpass` | `(sample_rate: float, center_freq: float, q: float = 0.7071) -> mpdsp._core.IIRFilter` | RBJ biquad allpass — unit magnitude, phase shift only. |
| `rbj_lowshelf` | `(sample_rate: float, cutoff: float, gain_db: float, slope: float = 1.0) -> mpdsp._core.IIRFilter` | RBJ biquad low shelf. gain_db is the low-frequency shelf gain. |
| `rbj_highshelf` | `(sample_rate: float, cutoff: float, gain_db: float, slope: float = 1.0) -> mpdsp._core.IIRFilter` | RBJ biquad high shelf. gain_db is the high-frequency shelf gain. |

## FIR filter design

Window-method designs returning an `FIRFilter`. `fir_filter` constructs directly from a coefficient array when you need a custom design.

| Name | Signature | Description |
|------|-----------|-------------|
| `fir_lowpass` | `(num_taps: int, sample_rate: float, cutoff: float, window: str = 'hamming', kaiser_beta: float = 8.6) -> mpdsp._core.FIRFilter` | Design an FIR lowpass filter via the window method. |
| `fir_highpass` | `(num_taps: int, sample_rate: float, cutoff: float, window: str = 'hamming', kaiser_beta: float = 8.6) -> mpdsp._core.FIRFilter` | Design an FIR highpass filter via spectral inversion of a lowpass. |
| `fir_bandpass` | `(num_taps: int, sample_rate: float, f_low: float, f_high: float, window: str = 'hamming', kaiser_beta: float = 8.6) -> mpdsp._core.FIRFilter` | Design an FIR bandpass filter. |
| `fir_bandstop` | `(num_taps: int, sample_rate: float, f_low: float, f_high: float, window: str = 'hamming', kaiser_beta: float = 8.6) -> mpdsp._core.FIRFilter` | Design an FIR bandstop (notch) filter via spectral inversion. |
| `fir_filter` | `(coefficients: ndarray1d[ro]) -> mpdsp._core.FIRFilter` | Construct an FIR filter from explicit tap coefficients. |

## Image — generators

All return `(rows, cols)` float64 2D NumPy arrays. The `*_noise*` and `salt_and_pepper` generators accept a `seed`. `threshold` is both a generator (arguments like `threshold(image, value)`) and a pipeline primitive — consult the signature.

| Name | Signature | Description |
|------|-----------|-------------|
| `checkerboard` | `(rows: int, cols: int, block_size: int, low: float = 0.0, high: float = 1.0) -> ndarray2d` | Checkerboard of alternating `low` / `high` blocks, `block_size` pixels per square. |
| `stripes_horizontal` | `(rows: int, cols: int, stripe_width: int, low: float = 0.0, high: float = 1.0) -> ndarray2d` | Alternating horizontal stripes of `stripe_width` rows each. |
| `stripes_vertical` | `(rows: int, cols: int, stripe_width: int, low: float = 0.0, high: float = 1.0) -> ndarray2d` | Alternating vertical stripes of `stripe_width` columns each. |
| `grid` | `(rows: int, cols: int, spacing: int, background: float = 0.0, line: float = 1.0) -> ndarray2d` | Thin grid lines at every `spacing` pixels against a uniform background. |
| `gradient_horizontal` | `(rows: int, cols: int, start: float = 0.0, end: float = 1.0) -> ndarray2d` | Linear horizontal gradient from `start` (left) to `end` (right). |
| `gradient_vertical` | `(rows: int, cols: int, start: float = 0.0, end: float = 1.0) -> ndarray2d` | Linear vertical gradient from `start` (top) to `end` (bottom). |
| `gradient_radial` | `(rows: int, cols: int, center_val: float = 1.0, edge_val: float = 0.0) -> ndarray2d` | Radial gradient: `center_val` at the image center linearly interpolated to `edge_val` at the corners. |
| `gaussian_blob` | `(rows: int, cols: int, sigma: float, amplitude: float = 1.0) -> ndarray2d` | 2D Gaussian centred on the image with standard deviation `sigma`. |
| `circle` | `(rows: int, cols: int, radius: int, foreground: float = 1.0, background: float = 0.0) -> ndarray2d` | Filled circle of `radius` pixels centred on the image. |
| `rectangle` | `(rows: int, cols: int, y: int, x: int, h: int, w: int, foreground: float = 1.0, background: float = 0.0) -> ndarray2d` | Filled rectangle with top-left corner at (y, x) and dimensions (h, w). Pixels outside the rectangle get `background`. |
| `zone_plate` | `(rows: int, cols: int, max_freq: float = 0.0) -> ndarray2d` | Zone plate (chirp image) — radial frequency that sweeps from 0 at the center to `max_freq` (cycles/pixel) at the corners. `max_freq = 0` (default) auto-selects half-Nyquist. |
| `uniform_noise_image` | `(rows: int, cols: int, low: float = 0.0, high: float = 1.0, seed: int = 42) -> ndarray2d` | Uniform-distribution noise in [low, high]. |
| `gaussian_noise_image` | `(rows: int, cols: int, mean: float = 0.0, stddev: float = 1.0, seed: int = 42) -> ndarray2d` | Gaussian-distribution noise with the given mean and stddev. |
| `salt_and_pepper` | `(rows: int, cols: int, density: float = 0.05, low: float = 0.0, high: float = 1.0, seed: int = 42) -> ndarray2d` | Salt-and-pepper noise: `density` fraction of pixels randomly flipped to `low` (pepper) or `high` (salt); the rest stay at the midpoint (low+high)/2. |
| `add_noise` | `(image: ndarray2d[ro], stddev: float, seed: int = 42) -> ndarray2d` | Return `image` with i.i.d. Gaussian noise of the given stddev added to each pixel. |
| `threshold` | `(image: ndarray2d[ro], thresh: float, low: float = 0.0, high: float = 1.0) -> ndarray2d` | Binary threshold: pixels greater than or equal to `thresh` become `high`; pixels strictly below become `low`. |

## Image — processing

All take and return `(rows, cols)` float64 2D arrays. Almost every processing function accepts a `dtype=` parameter for mixed-precision dispatch on the internal arithmetic. `border=` (`"reflect_101"` by default) controls the boundary handling for convolution-based operations.

| Name | Signature | Description |
|------|-----------|-------------|
| `convolve2d` | `(image: ndarray2d[ro], kernel: ndarray2d[ro], border: str = 'reflect_101', pad: float = 0.0, dtype: str = 'reference') -> ndarray2d` | 2D spatial correlation. `border` is one of constant, replicate, reflect, reflect_101, or wrap; `pad` is the fill value for border='constant'. `dtype` selects the internal arithmetic — see available_dtypes(). |
| `separable_filter` | `(image: ndarray2d[ro], row_kernel: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], col_kernel: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], border: str = 'reflect_101', pad: float = 0.0, dtype: str = 'reference') -> ndarray2d` | Apply a row kernel then a column kernel (separable 2D filter). Equivalent to convolve2d with an outer-product kernel but cheaper for a KxL kernel: O(K+L) per pixel instead of O(KL). |
| `gaussian_blur` | `(image: ndarray2d[ro], sigma: float, radius: int = 0, border: str = 'reflect_101', dtype: str = 'reference') -> ndarray2d` | Separable Gaussian blur. `radius=0` auto-selects a radius that captures most of the Gaussian tail (usually ceil(3*sigma)). |
| `box_blur` | `(image: ndarray2d[ro], size: int, border: str = 'reflect_101', dtype: str = 'reference') -> ndarray2d` | Box-average blur with an `size x size` uniform kernel. |
| `sobel_x` | `(image: ndarray2d[ro], border: str = 'reflect_101', dtype: str = 'reference') -> ndarray2d` | — |
| `sobel_y` | `(image: ndarray2d[ro], border: str = 'reflect_101', dtype: str = 'reference') -> ndarray2d` | — |
| `prewitt_x` | `(image: ndarray2d[ro], border: str = 'reflect_101', dtype: str = 'reference') -> ndarray2d` | — |
| `prewitt_y` | `(image: ndarray2d[ro], border: str = 'reflect_101', dtype: str = 'reference') -> ndarray2d` | — |
| `gradient_magnitude` | `(gx: ndarray2d[ro], gy: ndarray2d[ro], dtype: str = 'reference') -> ndarray2d` | Pixel-wise sqrt(gx^2 + gy^2). Typically fed Sobel or Prewitt gradient outputs. |
| `canny` | `(image: ndarray2d[ro], low_threshold: float, high_threshold: float, sigma: float = 1.0, dtype: str = 'reference') -> ndarray2d` | Canny edge detector: Gaussian smooth, Sobel gradients, non-maximum suppression, hysteresis thresholding. Returns a binary edge map (0.0 for non-edge, 1.0 for edge). |
| `rgb_to_gray` | `(r: ndarray2d[ro], g: ndarray2d[ro], b: ndarray2d[ro], dtype: str = 'reference') -> ndarray2d` | Convert an RGB image (three NumPy 2D arrays) to grayscale using ITU-R BT.601 weights: Y = 0.299*R + 0.587*G + 0.114*B. |

## Image — morphology

The `make_*_element` helpers construct structuring elements (boolean 2D arrays) for `dilate`/`erode` and the higher-level compositions (open, close, gradient, tophat, blackhat). All accept `dtype=` for mixed-precision arithmetic on the max-reduction.

| Name | Signature | Description |
|------|-----------|-------------|
| `make_rect_element` | `(rows: int, cols: int) -> ndarray2d[bool]` | Rectangular structuring element of shape (rows, cols), all True. |
| `make_cross_element` | `(size: int) -> ndarray2d[bool]` | Cross-shaped structuring element of size `size`x`size`: True along the center row and center column, False elsewhere. |
| `make_ellipse_element` | `(size: int) -> ndarray2d[bool]` | Elliptical (disk-like) structuring element of size `size`x`size`. |
| `dilate` | `(image: ndarray2d[ro], element: ndarray2d[bool, ro], dtype: str = 'reference') -> ndarray2d` | — |
| `erode` | `(image: ndarray2d[ro], element: ndarray2d[bool, ro], dtype: str = 'reference') -> ndarray2d` | — |
| `morphological_open` | `(image: ndarray2d[ro], element: ndarray2d[bool, ro], dtype: str = 'reference') -> ndarray2d` | — |
| `morphological_close` | `(image: ndarray2d[ro], element: ndarray2d[bool, ro], dtype: str = 'reference') -> ndarray2d` | — |
| `morphological_gradient` | `(image: ndarray2d[ro], element: ndarray2d[bool, ro], dtype: str = 'reference') -> ndarray2d` | — |
| `tophat` | `(image: ndarray2d[ro], element: ndarray2d[bool, ro], dtype: str = 'reference') -> ndarray2d` | — |
| `blackhat` | `(image: ndarray2d[ro], element: ndarray2d[bool, ro], dtype: str = 'reference') -> ndarray2d` | — |

## Image — file I/O

PGM (grayscale 8/16-bit), PPM (RGB 8-bit), and BMP (8-bit grayscale + RGB). Reads return float64 arrays normalized to `[0.0, 1.0]`; writes expect the same range.

| Name | Signature | Description |
|------|-----------|-------------|
| `read_pgm` | `(path: str) -> ndarray2d` | Read a PGM file. Returns a 2D NumPy float64 array normalized to [0, 1]. |
| `write_pgm` | `(path: str, image: ndarray2d[ro], max_val: int = 255) -> None` | Write a grayscale image to a PGM file. Values are clamped to [0, max_val] during quantization. |
| `read_ppm` | `(path: str) -> tuple[ndarray2d, ndarray2d, ndarray2d]` | Read a PPM file. Returns a (r, g, b) tuple of NumPy float64 arrays normalized to [0, 1]. |
| `write_ppm` | `(path: str, r: ndarray2d[ro], g: ndarray2d[ro], b: ndarray2d[ro], max_val: int = 255) -> None` | Write an RGB image to a PPM file (P6 binary format). |
| `read_bmp` | `(path: str) -> tuple[ndarray2d, ndarray2d, ndarray2d, bool]` | Read a BMP file (8-bit palette or 24-bit RGB). Returns (r, g, b, is_grayscale) — channels normalized to [0, 1]. |
| `write_bmp` | `(path: str, image: ndarray2d[ro]) -> None` | Write a grayscale image to a 24-bit BMP file (R=G=B=image). |
| `write_bmp_rgb` | `(path: str, r: ndarray2d[ro], g: ndarray2d[ro], b: ndarray2d[ro]) -> None` | Write an RGB image to a 24-bit BMP file. |

## Audio — WAV file I/O

8/16/24/32-bit integer PCM (read + write) and 32-bit float PCM (read only — upstream doesn't write float PCM even though it reads it). Samples normalized to `[-1, 1]`. `read_wav` returns 1D for mono files, 2D `(N, channels)` for multi-channel — same convention as `scipy.io.wavfile`.

| Name | Signature | Description |
|------|-----------|-------------|
| `read_wav` | `(path: str) -> tuple` | Read a WAV file. Returns (data, sample_rate): data is a float64 ndarray normalized to [-1, 1] — shape (N,) for mono files, shape (N, channels) for multi-channel. Supports 8/16/24/32-bit integer PCM and 32-bit float PCM. |
| `write_wav` | `(path: str, data: ndarray[], sample_rate: int, bits_per_sample: int = 16) -> None` | Write a WAV file. `data` is a float64 ndarray — 1D for mono or 2D (N, channels) for multi-channel. Values outside [-1, 1] are clipped. bits_per_sample must be 8, 16, 24, or 32 (integer PCM only — float32-PCM write is not supported by upstream even though float32-PCM read is). |

## Types — transfer function and type projection

`TransferFunction` is bound on double in 0.5.0 and represents the rational H(z) = B(z)/A(z) directly (as opposed to `IIRFilter`'s cascade-of-biquads form). Use `to_transfer_function(filt)` to fold an IIR cascade into a single TF for evaluation, cascade composition, or handing to the upcoming `ztransform` (Phase 5 / #54). `project_onto` / `projection_error` are the round-trip primitives underlying `measure_sqnr_db` — use them when you want the quantized samples or the raw error magnitude rather than the SQNR number.

| Name | Signature | Description |
|------|-----------|-------------|
| `project_onto` | `(data: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str) -> ndarray` | Project data through the sample scalar of `dtype` and back to float64. The round-trip surfaces the quantization error you'd see feeding a signal through an ADC at that precision — it's the underlying mechanic of `measure_sqnr_db`, exposed directly for when you want the quantized samples rather than just the SQNR. |
| `projection_error` | `(data: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str) -> float` | Max absolute error between data and its round-trip through `dtype`. Equivalent to max(abs(data - project_onto(data, dtype))) but computed without allocating the intermediate ndarray. |
| `to_transfer_function` | `(filt)` | Fold an `IIRFilter` cascade into a single `TransferFunction`. |

## Numerical-analysis helpers (pure Python)

Thin layer over already-bound `IIRFilter` methods. `biquad_poles` is a standalone quadratic solver that takes a 5-tuple of coefficients. See `IIRFilter.stability_margin()`, `.condition_number()`, `.worst_case_sensitivity()`, and `.pole_displacement(dtype)` for the per-filter metrics.

| Name | Signature | Description |
|------|-----------|-------------|
| `biquad_poles` | `(b0: 'float', b1: 'float', b2: 'float', a1: 'float', a2: 'float') -> 'list[complex]'` | Two poles of a single biquad section. |
| `max_pole_radius` | `(filt) -> 'float'` | Largest ``\|pole\|`` in the filter's z-plane. |
| `is_stable` | `(filt, tol: 'float' = 0.0) -> 'bool'` | True iff all poles are strictly inside the unit circle. |

## Mixed-precision helpers

`available_dtypes()` is the runtime-queryable source of truth for the string keys accepted by every `dtype=` parameter throughout the API. `compare_filters(filt, signal, dtypes=...)` is the one-call way to sweep SQNR / max-abs-error across all dtypes.

| Name | Signature | Description |
|------|-----------|-------------|
| `available_dtypes` | `() -> list[str]` | List available arithmetic configuration names. |
| `compare_filters` | `(filt, signal, dtypes=None)` | Process `signal` through `filt` at multiple dtypes and report error metrics. |

## CSV + image-pipeline helpers (pure Python)

`load_sweep` reads the CSV emitted by upstream `iir_precision_sweep`. `apply_per_channel` maps a single-channel function across a multi-channel image. `collect_adaptive_weights` drives an `LMSFilter` / `NLMSFilter` / `RLSFilter` and returns the weight trajectory.

| Name | Signature | Description |
|------|-----------|-------------|
| `load_sweep` | `(directory: str) -> dict` | Load all CSV files from an iir_precision_sweep output directory. |
| `apply_per_channel` | `(r: numpy.ndarray, g: numpy.ndarray, b: numpy.ndarray, func: Callable[[numpy.ndarray], numpy.ndarray]) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]` | Run a single-channel image function across three RGB planes. |
| `collect_adaptive_weights` | `(adaptive_filter, inputs, desireds, record_every=1)` | Run an adaptive filter over (inputs, desireds) and record the tap weights every `record_every` samples. |

## Matplotlib plotting helpers

All optional — require `mpdsp[plot]`. Return `matplotlib.figure.Figure` objects so the caller can `fig.savefig(...)` or further customize. None of these are callable in a headless environment without a matplotlib backend set to `Agg` first.

| Name | Signature | Description |
|------|-----------|-------------|
| `plot_signal` | `(signal, sample_rate=1.0, title='Signal', ax=None, **kwargs)` | Plot a time-domain signal. |
| `plot_spectrum` | `(signal, sample_rate=1.0, title='Spectrum', ax=None, db=True, **kwargs)` | Plot the magnitude spectrum of a signal. |
| `plot_signal_and_spectrum` | `(signal, sample_rate=1.0, title='')` | Plot signal in time domain and frequency domain side by side. |
| `plot_quantization_comparison` | `(signal, dtypes, sample_rate=1.0, title='Quantization Comparison')` | Plot a signal quantized through multiple arithmetic types. |
| `plot_sqnr_comparison` | `(signal, dtypes=None, title='SQNR Comparison')` | Bar chart comparing SQNR across arithmetic types. |
| `plot_window_comparison` | `(window_funcs, N=256, title='Window Comparison')` | Plot multiple windows and their frequency responses. |
| `plot_spectrogram` | `(times, freqs, magnitude_db, title='Spectrogram', vmin=-80, vmax=0, ax=None)` | Plot a spectrogram from mpdsp.spectrogram() output. |
| `plot_psd` | `(freqs, power, title='Power Spectral Density', ax=None, **kwargs)` | Plot power spectral density from mpdsp.psd() output. |
| `plot_filter_comparison` | `(filt, dtypes=None, num_freqs=512, signal=None, sample_rate=1.0, title=None, figsize=(12, 4))` | Plot magnitude, phase, and pole locations for a filter. |
| `plot_kalman_tracking` | `(truth, measurements, estimates, covariances=None, dt=1.0, title='Kalman filter tracking', figsize=(10, 4))` | Plot a Kalman filter's state estimate against the true trajectory. |
| `plot_adaptive_convergence` | `(weight_traces, true_weights=None, labels=None, dt=1, title='Adaptive-filter weight convergence', figsize=(11, 5))` | Plot weight trajectories of one or more adaptive filters over time. |
| `plot_image` | `(img: numpy.ndarray, title: str = '', ax=None, cmap: str = 'gray', vmin: Optional[float] = None, vmax: Optional[float] = None, colorbar: bool = True, figsize=(6, 5))` | Display a 2D grayscale image with an optional colorbar. |
| `plot_image_grid` | `(images: Sequence[numpy.ndarray], titles: Optional[Sequence[str]] = None, ncols: int = 4, cmap: str = 'gray', figsize: Optional[Tuple[float, float]] = None, colorbar: bool = False, suptitle: Optional[str] = None)` | Display a sequence of images in a grid layout. |
| `plot_pipeline` | `(stages: Sequence[numpy.ndarray], titles: Optional[Sequence[str]] = None, cmap: str = 'gray', figsize: Optional[Tuple[float, float]] = None, suptitle: Optional[str] = None)` | Display a pipeline's successive stages in a single row. |

## Classes
Stateful objects. All carry a `.dtype` string attribute reflecting the arithmetic they were constructed with, and a `.reset()` method where meaningful. Process methods come in per-sample (`.process(x)`) and block (`.process_block(signal)`) variants except on the filter classes, which are block-only.

### `IIRFilter`

Returned by every `*_lowpass` / `*_highpass` / `*_bandpass` / `*_bandstop` / `rbj_*` designer. Coefficients are always designed in double; processing, analysis, and pole placement happen per the dtype passed to each method.

> Cascade-of-biquads IIR filter.

| Member | Signature / description |
|--------|-------------------------|
| `.coefficients` | `(self) -> list[tuple[float, float, float, float, float]]` — List of (b0, b1, b2, a1, a2) tuples, one per stage. |
| `.condition_number` | `(self, num_freqs: int = 256) -> float` — Worst-case relative change in \|H\| per coefficient perturbation across stages. Higher = more sensitive to coefficient quantization. |
| `.frequency_response` | `(self, normalized_freqs: numpy.ndarray[dtype=float64, shape=(*), writable=False]) -> numpy.ndarray[dtype=complex128]` — Evaluate H(e^{j2*pi*f}) at each normalized frequency (f/fs). Returns complex128. |
| `.num_stages` | `(self) -> int` — Number of active biquad sections. |
| `.pole_displacement` | `(self, dtype: str) -> float` — Max pole displacement when coefficients are quantized through the target dtype (see available_dtypes). Returns 0 for 'reference'. |
| `.poles` | `(self) -> list[complex]` — List of complex pole locations in the z-plane. |
| `.process` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), writable=False], dtype: str = 'reference') -> numpy.ndarray[dtype=float64]` — Filter a signal. dtype selects arithmetic for state and samples (see available_dtypes()). Returns NumPy float64. |
| `.stability_margin` | `(self) -> float` — 1 - max(\|pole\|). Positive = stable, 0 = marginal, < 0 = unstable. |
| `.worst_case_sensitivity` | `(self, epsilon: float = 1e-08) -> float` — Worst-case \|d(max_pole_radius)/d(coeff)\| across stages, computed by finite differences. |

### `FIRFilter`

Returned by `fir_lowpass` / `fir_highpass` / `fir_bandpass` / `fir_bandstop` / `fir_filter`. Direct-form convolution; coefficients in double, processing dispatches via `dtype=`.

> Finite-impulse-response filter with a double-precision tap vector.

| Member | Signature / description |
|--------|-------------------------|
| `.coefficients` | `(self) -> numpy.ndarray[dtype=float64]` — Taps as a NumPy float64 array. |
| `.frequency_response` | `(self, normalized_freqs: numpy.ndarray[dtype=float64, shape=(*), writable=False]) -> numpy.ndarray[dtype=complex128]` — Evaluate H(e^{j2*pi*f}) at each normalized frequency (f/fs). Returns complex128. |
| `.impulse_response` | `(self, length: int) -> numpy.ndarray[dtype=float64]` — Impulse response — the taps, padded or truncated to `length`. |
| `.num_taps` | `(self) -> int` — Number of tap coefficients. |
| `.process` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), writable=False], dtype: str = 'reference') -> numpy.ndarray[dtype=float64]` — Filter a signal. dtype selects arithmetic for taps, state, and samples (see available_dtypes()). Returns NumPy float64. |

### `RPDFDither`

Rectangular-PDF (uniform) dither generator. Produces noise in `[-amplitude, +amplitude]`. Use before quantization to decorrelate error from the signal, at the cost of a flat noise floor. Stateful because it carries a `std::mt19937` internally.

> Rectangular-PDF dither generator.

| Member | Signature / description |
|--------|-------------------------|
| `.amplitude` | (self) -> float |
| `.apply` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), writable=False]) -> numpy.ndarray[dtype=float64]` — Dither `signal` (float64 ndarray). Returns a new float64 ndarray. |
| `.dtype` | Arithmetic configuration selected at construction. |
| `.sample` | `(self) -> float` — Draw a single dither sample as a Python float. |

### `TPDFDither`

Triangular-PDF dither generator — sum of two RPDF draws. Eliminates the noise-modulation artifact that RPDF leaves on low-level signals, at a +3 dB noise-power cost. Generally preferred over RPDF when the added noise power is tolerable.

> Triangular-PDF dither generator.

| Member | Signature / description |
|--------|-------------------------|
| `.amplitude` | (self) -> float |
| `.apply` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), writable=False]) -> numpy.ndarray[dtype=float64]` — Dither `signal` (float64 ndarray). Returns a new float64 ndarray. |
| `.dtype` | Arithmetic configuration selected at construction. |
| `.sample` | `(self) -> float` — Draw a single dither sample as a Python float. |

### `FirstOrderNoiseShaper`

First-order error-feedback noise shaper. Quantizes `double → dtype → double` while feeding the quantization error back (negated) onto the next input. First-order shaping is a high-pass on the noise floor — most useful upstream of a lowpass reconstruction that rejects the shifted noise.

> First-order error-feedback noise shaper.

| Member | Signature / description |
|--------|-------------------------|
| `.dtype` | Arithmetic configuration selected at construction. |
| `.process` | `(self, input: float) -> float` — Process a single sample. Returns the shaped+quantized output. |
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), writable=False]) -> numpy.ndarray[dtype=float64]` — Process a float64 ndarray signal. Returns a new float64 ndarray with the shaped+quantized output. |
| `.reset` | `(self) -> None` — Clear the error-feedback state to zero. |

### `PeakEnvelope`

Peak envelope follower with configurable attack/release. The `.value` property exposes the current envelope state.

> Peak envelope follower with exponential attack and release.

| Member | Signature / description |
|--------|-------------------------|
| `.dtype` | The arithmetic configuration selected at construction. |
| `.process` | `(self, input: float) -> float` — Process a single sample. Returns the updated envelope value. |
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Process a 1D NumPy float64 signal. Returns the envelope trace (same length as the input). The per-sample loop releases the GIL internally so |
| `.reset` | `(self) -> None` — Clear the internal envelope state to zero. |
| `.value` | `(self) -> float` — Current envelope value without consuming a sample. |

### `RMSEnvelope`

RMS envelope follower. Same interface shape as `PeakEnvelope`; tracks the signal's moving root-mean-square.

> RMS envelope follower.

| Member | Signature / description |
|--------|-------------------------|
| `.dtype` | The arithmetic configuration selected at construction. |
| `.process` | `(self, input: float) -> float` — Process a single sample. Returns the updated RMS level. |
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Process a 1D NumPy float64 signal. Returns the RMS envelope trace (same length as the input). The per-sample loop releases the GIL. |
| `.reset` | `(self) -> None` — Clear the internal mean-square state to zero. |
| `.value` | `(self) -> float` — Current RMS value without consuming a sample. |

### `Compressor`

Dynamic-range compressor. Threshold, ratio, attack/release, and optional makeup gain + soft-knee. Internal envelope follower is peak-based.

> Dynamic-range compressor with soft-knee option.

| Member | Signature / description |
|--------|-------------------------|
| `.dtype` | The arithmetic configuration selected at construction. |
| `.process` | `(self, input: float) -> float` — Process a single sample. Returns the compressed output. |
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Process a 1D NumPy float64 signal. Returns the compressed signal (same length as the input). The per-sample loop releases the GIL. |
| `.reset` | `(self) -> None` — Clear the internal envelope state. |

### `AGC`

Automatic gain control: drives the signal toward a target level using a configurable attack/release time constant.

> Automatic Gain Control.

| Member | Signature / description |
|--------|-------------------------|
| `.dtype` | The arithmetic configuration selected at construction. |
| `.process` | `(self, input: float) -> float` — Process a single sample. Returns the gain-adjusted output. |
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Process a 1D NumPy float64 signal. Returns the gain-adjusted signal (same length as the input). The per-sample loop releases the GIL. |
| `.reset` | `(self) -> None` — Clear the internal RMS envelope state. |

### `KalmanFilter`

Linear Kalman filter. State/measurement/control dimensions set at construction. `F`, `H`, `Q`, `R`, `P`, `B` are writeable NumPy 2D array properties; `state` is the 1D state vector. Call `.predict()` then `.update(measurement)` each step.

> Linear Kalman filter for state estimation.

| Member | Signature / description |
|--------|-------------------------|
| `.B` | Control-input matrix (state_dim x ctrl_dim). |
| `.F` | State transition matrix (state_dim x state_dim). |
| `.H` | Observation matrix (meas_dim x state_dim). |
| `.P` | Estimation-error covariance (state_dim x state_dim). |
| `.Q` | Process-noise covariance (state_dim x state_dim). |
| `.R` | Measurement-noise covariance (meas_dim x meas_dim). |
| `.ctrl_dim` | (self) -> int |
| `.dtype` | Arithmetic configuration selected at construction. |
| `.meas_dim` | (self) -> int |
| `.predict` | `(self) -> None` — Predict step without control input. |
| `.state` | Current state estimate (length state_dim). |
| `.state_dim` | (self) -> int |
| `.update` | `(self, z: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> None` — Update step with a measurement vector of length meas_dim. |

### `LMSFilter`

Least-mean-squares adaptive filter. Coefficients adapt online via the LMS update. `.weights` exposes the current tap vector.

> Least-mean-squares adaptive FIR filter.

| Member | Signature / description |
|--------|-------------------------|
| `.dtype` | Arithmetic configuration selected at construction. |
| `.last_error` | Error residual from the most recent process() call. |
| `.num_taps` | (self) -> int |
| `.process` | `(self, input: float, desired: float) -> tuple[float, float]` — Process one sample with adaptation. Returns a (output, error) tuple where output is y[n] = w^T x[n] and error is d[n] - y[n]. |
| `.process_block` | `(self, inputs: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], desireds: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> tuple[numpy.ndarray[dtype=float64], numpy.ndarray[dtype=float64]]` — Process two equal-length NumPy float64 signals (input, desired) and return a (outputs, errors) tuple of float64 arrays. The per-sample loop  |
| `.reset` | `(self) -> None` — Zero the weights and delay line. |
| `.weights` | Current tap weights as a 1D NumPy float64 array (read-only copy). |

### `NLMSFilter`

Normalized LMS — divides the step size by the input power for tunability that's robust across signal levels.

> Normalized LMS adaptive filter — scales the step size by input power to stay stable across varying signal levels.

| Member | Signature / description |
|--------|-------------------------|
| `.dtype` | Arithmetic configuration selected at construction. |
| `.last_error` | Error residual from the most recent process() call. |
| `.num_taps` | (self) -> int |
| `.process` | `(self, input: float, desired: float) -> tuple[float, float]` — Process one sample with adaptation. Returns a (output, error) tuple where output is y[n] = w^T x[n] and error is d[n] - y[n]. |
| `.process_block` | `(self, inputs: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], desireds: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> tuple[numpy.ndarray[dtype=float64], numpy.ndarray[dtype=float64]]` — Process two equal-length NumPy float64 signals (input, desired) and return a (outputs, errors) tuple of float64 arrays. The per-sample loop  |
| `.reset` | `(self) -> None` — Zero the weights and delay line. |
| `.weights` | Current tap weights as a 1D NumPy float64 array (read-only copy). |

### `RLSFilter`

Recursive least-squares adaptive filter. Faster convergence than LMS/NLMS at the cost of O(N²) memory for the P matrix. Known to diverge under reduced precision when P loses symmetry — see `notebooks/06_estimation.ipynb`.

> Recursive least-squares adaptive filter. Faster convergence than LMS at O(N^2) per sample cost. forgetting_factor in (0, 1] controls tracking of non-stationary signals (1.0 = no forgetting).

| Member | Signature / description |
|--------|-------------------------|
| `.dtype` | Arithmetic configuration selected at construction. |
| `.last_error` | Error residual from the most recent process() call. |
| `.num_taps` | (self) -> int |
| `.process` | `(self, input: float, desired: float) -> tuple[float, float]` — Process one sample with adaptation. Returns a (output, error) tuple where output is y[n] = w^T x[n] and error is d[n] - y[n]. |
| `.process_block` | `(self, inputs: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], desireds: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> tuple[numpy.ndarray[dtype=float64], numpy.ndarray[dtype=float64]]` — Process two equal-length NumPy float64 signals (input, desired) and return a (outputs, errors) tuple of float64 arrays. The per-sample loop  |
| `.reset` | `(self) -> None` — Zero the weights, delay line, and reset P to delta*I. |
| `.weights` | Current tap weights as a 1D NumPy float64 array (read-only copy). |

### `TransferFunction`

Rational H(z) = B(z)/A(z) with double-precision coefficients. Construct from numerator + denominator ndarrays; the leading `1` in the denominator is implicit (don't pass `a0`). Cascade via `*`. The `to_transfer_function(filt)` helper folds an IIRFilter cascade into one of these, useful when evaluating the full filter's H(z) directly rather than staging by stage.

> Rational transfer function H(z) = B(z) / A(z).

| Member | Signature / description |
|--------|-------------------------|
| `.denominator` | Denominator coefficients a1, a2, ... as a float64 ndarray (a0 = 1 implicit). |
| `.evaluate` | `(self, z: complex) -> complex` — Evaluate H(z) at a single complex point. Returns complex128. |
| `.evaluate_many` | `(self, z: numpy.ndarray[dtype=complex128, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=complex128]` — Evaluate H(z) at each point in a complex128 ndarray. Returns a complex128 ndarray of the same length. |
| `.frequency_response` | `(self, f: float) -> complex` — Evaluate H(e^{j 2*pi*f}) at normalized frequency f in [0, 0.5]. |
| `.frequency_response_many` | `(self, freqs: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=complex128]` — Vectorized frequency_response(...) over a float64 ndarray of normalized frequencies. Returns complex128. |
| `.is_stable` | `(self) -> bool` — Check stability via a 360-angle sampling of the denominator on the unit circle. False if any sample is within 1e-6 of zero. |
| `.numerator` | Numerator coefficients b0, b1, b2, ... as a float64 ndarray. |

---

## Regenerating this document

This file was generated from an installed `mpdsp` package. Re-run
after landing new bindings:

```bash
pip install -e .
python scripts/build_api_ref.py
```

Edit the `CATEGORIES`, `INTROS`, and `CLASS_INTROS` tables in
`scripts/build_api_ref.py` to add new bindings or revise prose. The
function-table signatures come from nanobind's attached `__doc__` and
don't need manual editing — they regenerate from the installed
extension.
