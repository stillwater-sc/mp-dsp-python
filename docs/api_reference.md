# `mpdsp` API reference

Complete enumeration of every public name in the `mpdsp` package, grouped
by subsystem. Generated from `0.8.0.dev0` (upstream `sw::dsp
0.8.0`) via `inspect` and the nanobind-attached
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
- [Instrument — oscilloscope-style measurement](#instrument--oscilloscope-style-measurement)
- [Spectrum analyzer — detectors, peaks, markers](#spectrum-analyzer--detectors-peaks-markers)
- [Numerical utilities — polynomial, quadratic, elliptic](#numerical-utilities--polynomial-quadratic-elliptic)
- [Analog prototypes — s-plane pole/zero constellations](#analog-prototypes--s-plane-polezero-constellations)
- [Acquisition — high-rate ADC → baseband pipeline](#acquisition--high-rate-adc--baseband-pipeline)
- [Image — generators](#image--generators)
- [Image — processing](#image--processing)
- [Image — morphology](#image--morphology)
- [Image — file I/O](#image--file-io)
- [Audio — WAV file I/O](#audio--wav-file-io)
- [Types — transfer function and type projection](#types--transfer-function-and-type-projection)
- [Numerical analysis — pure-Python helpers](#numerical-analysis--pure-python-helpers)
- [Numerical analysis — free-function primitives (bound)](#numerical-analysis--free-function-primitives-bound)
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
  - [`NCO`](#nco)
  - [`CICDecimator`](#cicdecimator)
  - [`CICInterpolator`](#cicinterpolator)
  - [`HalfBandFilter`](#halfbandfilter)
  - [`PolyphaseDecimator`](#polyphasedecimator)
  - [`PolyphaseInterpolator`](#polyphaseinterpolator)
  - [`DDC`](#ddc)
  - [`DecimationChain`](#decimationchain)
  - [`PoleZeroPlot`](#polezeroplot)
  - [`BodeResult`](#boderesult)
  - [`KalmanFilter`](#kalmanfilter)
  - [`ExtendedKalmanFilter`](#extendedkalmanfilter)
  - [`UnscentedKalmanFilter`](#unscentedkalmanfilter)
  - [`LMSFilter`](#lmsfilter)
  - [`NLMSFilter`](#nlmsfilter)
  - [`RLSFilter`](#rlsfilter)
  - [`RationalResampler`](#rationalresampler)
  - [`OverlapAddConvolver`](#overlapaddconvolver)
  - [`OverlapSaveConvolver`](#overlapsaveconvolver)
  - [`PeakDetectDecimator`](#peakdetectdecimator)
  - [`TriggerRingBuffer`](#triggerringbuffer)
  - [`RealtimeSpectrum`](#realtimespectrum)
  - [`RBWFilter`](#rbwfilter)
  - [`VBWFilter`](#vbwfilter)
  - [`SweptLO`](#sweptlo)
  - [`CalibrationProfile`](#calibrationprofile)
  - [`FrontEndCorrector`](#frontendcorrector)
  - [`TraceAverager`](#traceaverager)
  - [`WaterfallBuffer`](#waterfallbuffer)
  - [`Marker`](#marker)
  - [`DeltaMarker`](#deltamarker)
  - [`RootFinder`](#rootfinder)
  - [`CICBitGrowthReport`](#cicbitgrowthreport)
  - [`AcquisitionPrecisionRow`](#acquisitionprecisionrow)
  - [`ComplexPair`](#complexpair)
  - [`PoleZeroPair`](#polezeropair)
  - [`BiquadCoefficients`](#biquadcoefficients)
  - [`TransferFunction`](#transferfunction)
  - [`ContinuousTransferFunction`](#continuoustransferfunction)

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
| `posit_full` | double | posit<32,2> | posit<16,1> | Mixed-precision posit pipeline |
| `cf24` | double | cfloat<24,5> | cfloat<24,5> | Custom 24-bit float |
| `half` | double | cfloat<16,5> | cfloat<16,5> | IEEE half throughout |
| `sensor_8bit` | double | double | integer<8> | Standard 8-bit sensor ADC |
| `sensor_6bit` | double | double | integer<6> | Noise-limited sensor |
| `fpga_fixed` | double | fixpnt<32,24> | fixpnt<16,12> | FPGA fixed-point datapath |

**Posit taxonomy grid** — single-type `posit<N, es>` configs for every
combination of N ∈ {8, 16, 32} and es ∈ {0, 1, 2}. Coefficient = state =
sample, so ES-vs-precision tradeoff is readable directly from a fixed-N
sweep:

| Key | Posit type |
|-----|-----------|
| `posit_8_0`, `posit_8_1`, `posit_8_2` | `posit<8, 0/1/2>` |
| `posit_16_0`, `posit_16_1`, `posit_16_2` | `posit<16, 0/1/2>` |
| `posit_32_0`, `posit_32_1`, `posit_32_2` | `posit<32, 0/1/2>` |

`posit_8_2` is the canonical 8-bit posit (accepts the legacy `tiny_posit`
alias via `parse_config`); `posit_16_1` and `posit_32_2` are the
single-type standalones for the types `posit_full` mixes together.

Use `mpdsp.bits_of(dtype)` to query the sample-scalar bit width for any
config (useful for precision-vs-cost plots). The sensor configs keep
coefficient and state at double; only the ADC sample path quantizes
through `integer<N>`, which is the common case for ingesting real-world
ADC streams without re-architecting the downstream filter. For posit
grid cells the ES dimension doesn't affect bit width — every `posit_N_*`
reports N, so a sweep produces 3 stacked points per width on the
precision-cost frontier.

## Module attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mpdsp.__version__` | `str` | The installed wheel version (PEP 440). Current: `"0.8.0.dev0"`. |
| `mpdsp.__dsp_version__` | `str` | The upstream `sw::dsp` C++ library version the wheel was built against. Current: `"0.8.0"`. |
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
| `ramp` | `(length: int, slope: float = 1.0) -> ndarray` | Linear ramp: x[n] = slope * n, starting at 0. Add an offset in NumPy if you need to shift the starting value. |
| `multitone` | `(length: int, frequencies: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], sample_rate: float, amplitude: float = 1.0) -> ndarray` | Generate a sum of sinusoids at the given frequencies (Hz). All tones share the same amplitude, scaled so the summed peak amplitude matches the `amplitude` argument (per-tone contribution is amplitude / len(frequencies)). Useful for filter passband/stopband demos and two-tone IMD tests. |
| `white_noise` | `(length: int, amplitude: float = 1.0, seed: int = 0) -> ndarray` | Generate white noise (uniform in [-amplitude, amplitude]). |
| `gaussian_noise` | `(length: int, stddev: float = 1.0, seed: int = 0) -> ndarray` | Generate Gaussian white noise (mean=0, normal distribution with given stddev). |
| `pink_noise` | `(length: int, amplitude: float = 1.0, seed: int = 0) -> ndarray` | Generate pink noise (1/f spectrum, Voss-McCartney algorithm). |
| `upsample` | `(input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], factor: int) -> ndarray` | Upsample by an integer factor via zero insertion. Output length is input.size() * factor. This is zero-insertion only — apply a lowpass interpolator (e.g. FIR, halfband, polyphase) afterwards to remove the imaging spectral replicas. |
| `downsample` | `(input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], factor: int) -> ndarray` | Downsample by an integer factor by keeping every factor-th sample. Output length is input.size() // factor. This is a naive decimator — apply a lowpass anti-aliasing filter beforehand to avoid aliasing. |

## Window functions

Return a length-N float64 NumPy array. Apply by element-wise multiplication against a signal before spectral analysis. `kaiser` additionally takes a shape parameter `beta`.

| Name | Signature | Description |
|------|-----------|-------------|
| `hamming` | `(N: int, dtype: str = 'reference') -> ndarray` | Hamming window of length N. dtype controls the internal compute precision; result is always NumPy float64. |
| `hanning` | `(N: int, dtype: str = 'reference') -> ndarray` | Hanning (Hann) window of length N. |
| `blackman` | `(N: int, dtype: str = 'reference') -> ndarray` | Blackman window of length N. |
| `kaiser` | `(N: int, beta: float = 5.0, dtype: str = 'reference') -> ndarray` | Kaiser window of length N with shape parameter beta. |
| `rectangular` | `(N: int, dtype: str = 'reference') -> ndarray` | Rectangular (boxcar) window of length N. |
| `flat_top` | `(N: int, dtype: str = 'reference') -> ndarray` | Flat-top window of length N. |
| `tukey` | `(N: int, alpha: float = 0.5, dtype: str = 'reference') -> ndarray` | Tukey (cosine-tapered) window of length N. alpha in [0, 1] controls the fraction of the window that is cosine-tapered (0 = rectangular, 1 = Hann). |
| `gaussian` | `(N: int, sigma: float = 0.4, dtype: str = 'reference') -> ndarray` | Gaussian window of length N. sigma is expressed relative to N/2 (scipy convention): smaller sigma gives a narrower, more peaked window. |
| `dolph_chebyshev` | `(N: int, attenuation_db: float = 100.0, dtype: str = 'reference') -> ndarray` | Dolph-Chebyshev window of length N with equiripple sidelobes at the given attenuation (dB, positive value). Common in radar/sonar; attenuation_db must be > 0. |
| `bartlett_hann` | `(N: int, dtype: str = 'reference') -> ndarray` | Bartlett-Hann window of length N — hybrid of Bartlett (triangular) and Hann (raised-cosine). |

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

All five primitives accept a `dtype=` parameter selecting the internal arithmetic (see `mpdsp.available_dtypes()`). Inputs and outputs at the Python layer remain double/complex128; only the C++ computation runs at the target precision. For rational-transfer-function evaluation see `ztransform`, `freqz`, `group_delay`, `laplace_freqs` in the Types section.

| Name | Signature | Description |
|------|-----------|-------------|
| `fft` | `(signal: ndarray1d[ro], dtype: str = 'reference') -> tuple` | Compute FFT of a real signal. Returns (real, imag) tuple of NumPy arrays. `dtype` selects the internal arithmetic (see `mpdsp.available_dtypes()`). |
| `ifft` | `(real: ndarray1d[ro], imag: ndarray1d[ro], dtype: str = 'reference') -> ndarray` | Compute inverse FFT from (real, imag) arrays. Returns real signal. `dtype` selects the internal arithmetic (see `mpdsp.available_dtypes()`). |
| `fft_magnitude_db` | `(signal: ndarray1d[ro], dtype: str = 'reference') -> ndarray` | Compute FFT magnitude spectrum in dB. `dtype` selects the internal arithmetic (see `mpdsp.available_dtypes()`). |
| `psd` | `(signal: ndarray1d[ro], sample_rate: float, dtype: str = 'reference') -> tuple` | Compute PSD with frequency axis. Returns (freqs_hz, power) tuple. `dtype` selects the internal arithmetic (see `mpdsp.available_dtypes()`). |
| `periodogram` | `(signal: ndarray1d[ro], dtype: str = 'reference') -> ndarray` | Compute periodogram power spectral density estimate. `dtype` selects the internal arithmetic (see `mpdsp.available_dtypes()`). |
| `welch` | `(signal: ndarray1d[ro], sample_rate: float, segment_size: int, overlap: int = -1, window: str = 'hamming', dtype: str = 'reference') -> tuple` | Welch's method: segmented, windowed, averaged periodogram PSD estimate. |
| `spectrogram` | `(signal: ndarray1d[ro], sample_rate: float, window_size: int = 1024, hop_size: int = 256, dtype: str = 'reference') -> tuple` | Compute spectrogram. Returns (times, freqs, magnitude_db) tuple. magnitude_db is a 2D array [n_frames x n_freqs]. `dtype` selects the internal arithmetic (see `mpdsp.available_dtypes()`). |

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

Every designer takes `coeff_dtype=` (default `'reference'`), which selects the arithmetic used to **compute** the coefficients — the `w0` scaling, `cos`/`sin`, the `alpha` divide, and the `a0` normalization. The finished coefficients are stored in `double` either way, losslessly: a biquad designed in `T` yields `T`-representable values, and every `T` in the dispatch table is narrower than `double`. This is the dual of `IIRFilter.pole_displacement(dtype)`, which quantizes coefficients that were already computed in `double` — `coeff_dtype` asks what computing them in `T` costs, `pole_displacement` asks what storing them in `T` costs.

`sensor_8bit` / `sensor_6bit` route their compute path to `double`, so they design identically to `reference`. There is no `rbj_peaking`: upstream `sw::dsp::rbj` has no Peaking class.

| Name | Signature | Description |
|------|-----------|-------------|
| `rbj_lowpass` | `(sample_rate: float, cutoff: float, q: float = 0.7071, coeff_dtype: str = 'reference') -> mpdsp._core.IIRFilter` | RBJ biquad lowpass. q ~ 0.7071 gives a Butterworth-like response. |
| `rbj_highpass` | `(sample_rate: float, cutoff: float, q: float = 0.7071, coeff_dtype: str = 'reference') -> mpdsp._core.IIRFilter` | RBJ biquad highpass. |
| `rbj_bandpass` | `(sample_rate: float, center_freq: float, bandwidth: float = 1.0, coeff_dtype: str = 'reference') -> mpdsp._core.IIRFilter` | RBJ biquad bandpass. bandwidth is in octaves. |
| `rbj_bandstop` | `(sample_rate: float, center_freq: float, bandwidth: float = 1.0, coeff_dtype: str = 'reference') -> mpdsp._core.IIRFilter` | RBJ biquad bandstop (notch). bandwidth is in octaves. |
| `rbj_allpass` | `(sample_rate: float, center_freq: float, q: float = 0.7071, coeff_dtype: str = 'reference') -> mpdsp._core.IIRFilter` | RBJ biquad allpass — unit magnitude, phase shift only. |
| `rbj_lowshelf` | `(sample_rate: float, cutoff: float, gain_db: float, slope: float = 1.0, coeff_dtype: str = 'reference') -> mpdsp._core.IIRFilter` | RBJ biquad low shelf. gain_db is the low-frequency shelf gain. |
| `rbj_highshelf` | `(sample_rate: float, cutoff: float, gain_db: float, slope: float = 1.0, coeff_dtype: str = 'reference') -> mpdsp._core.IIRFilter` | RBJ biquad high shelf. gain_db is the high-frequency shelf gain. |

## FIR filter design

Window-method designs returning an `FIRFilter`. `fir_filter` constructs directly from a coefficient array when you need a custom design.

| Name | Signature | Description |
|------|-----------|-------------|
| `fir_lowpass` | `(num_taps: int, sample_rate: float, cutoff: float, window: str = 'hamming', kaiser_beta: float = 8.6, coeff_dtype: str = 'reference') -> mpdsp._core.FIRFilter` | Design an FIR lowpass filter via the window method. coeff_dtype controls the precision of the design-time math; the resulting taps are stored as float64 in the returned filter. |
| `fir_highpass` | `(num_taps: int, sample_rate: float, cutoff: float, window: str = 'hamming', kaiser_beta: float = 8.6, coeff_dtype: str = 'reference') -> mpdsp._core.FIRFilter` | Design an FIR highpass filter via spectral inversion of a lowpass. |
| `fir_bandpass` | `(num_taps: int, sample_rate: float, f_low: float, f_high: float, window: str = 'hamming', kaiser_beta: float = 8.6, coeff_dtype: str = 'reference') -> mpdsp._core.FIRFilter` | Design an FIR bandpass filter. |
| `fir_bandstop` | `(num_taps: int, sample_rate: float, f_low: float, f_high: float, window: str = 'hamming', kaiser_beta: float = 8.6, coeff_dtype: str = 'reference') -> mpdsp._core.FIRFilter` | Design an FIR bandstop (notch) filter via spectral inversion. |
| `fir_filter` | `(coefficients: ndarray1d[ro]) -> mpdsp._core.FIRFilter` | Construct an FIR filter from explicit tap coefficients. |
| `remez` | `(num_taps: int, bands: ndarray1d[ro], desired: ndarray1d[ro], weights: ndarray1d[ro], type: str = 'bandpass', max_iterations: int = 40, grid_density: int = 16, coeff_dtype: str = 'reference') -> mpdsp._core.FIRFilter` | General Parks-McClellan equiripple FIR design. bands is a flat list of band edges in normalized frequency [0, 0.5], length 2N for N bands; desired has one value per band edge; weights has one per band. type is 'bandpass' (default; symmetric taps), 'differentiator', or 'hilbert' (both antisymmetric). |
| `remez_lowpass` | `(num_taps: int, sample_rate: float, passband_edge_hz: float, stopband_edge_hz: float, passband_weight: float = 1.0, stopband_weight: float = 1.0, coeff_dtype: str = 'reference') -> mpdsp._core.FIRFilter` | Equiripple lowpass FIR via Parks-McClellan (Remez exchange). passband_edge_hz and stopband_edge_hz define the transition band; weights control the passband-vs-stopband trade-off (larger stopband weight -> deeper stopband). |
| `remez_bandpass` | `(num_taps: int, sample_rate: float, stop1_hz: float, pass1_hz: float, pass2_hz: float, stop2_hz: float, stopband_weight: float = 1.0, passband_weight: float = 1.0, coeff_dtype: str = 'reference') -> mpdsp._core.FIRFilter` | Equiripple bandpass FIR via Parks-McClellan. Requires stop1 < pass1 < pass2 < stop2, all in Hz. Symmetric stopband weights on both sides. |
| `filtfilt` | `(iir_filter: mpdsp._core.IIRFilter, signal: ndarray1d[ro], dtype: str = 'reference') -> ndarray` | Zero-phase IIR filtering via forward-backward biquad cascade processing. |

## Instrument — oscilloscope-style measurement

Stateless measurements over a captured buffer, in the idiom a bench scope presents. `mean` and `rms` carry an `instrument_` prefix so they do not shadow `numpy.mean` / `numpy.rms` when the module is star-imported.

| Name | Signature | Description |
|------|-----------|-------------|
| `peak_to_peak` | `(signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str = 'reference') -> float` | Peak-to-peak amplitude of the segment: max(signal) - min(signal). For a unit-amplitude sine returns 2.0. |
| `instrument_mean` | `(signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str = 'reference') -> float` | Arithmetic mean (DC level) of the segment. Sum is accumulated in double regardless of dtype. Prefixed to avoid shadowing numpy.mean when users do `from mpdsp import *`. |
| `instrument_rms` | `(signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str = 'reference') -> float` | Root-mean-square of the segment. For a unit-amplitude sine returns 1/sqrt(2). Sum-of-squares is accumulated in double. |
| `rise_time` | `(signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], low_pct: float = 0.1, high_pct: float = 0.9, dtype: str = 'reference') -> float` | Rise time in SAMPLES between low_pct and high_pct of the segment's peak-to-peak range, on the first rising transition. Returns NaN if no transition spans both thresholds. Divide by sample_rate for seconds. Sub-sample crossings via linear interpolation. |
| `fall_time` | `(signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], low_pct: float = 0.1, high_pct: float = 0.9, dtype: str = 'reference') -> float` | Fall time in SAMPLES: mirror of rise_time for the first falling transition from high_pct down to low_pct. Returns NaN if no such transition. Divide by sample_rate for seconds. |
| `period` | `(signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], threshold: float = 0.0, dtype: str = 'reference') -> float` | Period in SAMPLES: average distance between consecutive rising threshold-crossings. Threshold defaults to 0 (zero-crossing, appropriate for AC-coupled signals). Returns NaN if fewer than two rising crossings occur. |
| `frequency` | `(signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], sample_rate: float, threshold: float = 0.0, dtype: str = 'reference') -> float` | Fundamental frequency in Hz: sample_rate / period_samples. Returns NaN if the period cannot be measured (see `period`). |

## Spectrum analyzer — detectors, peaks, markers

Detector reducers collapse each FFT bin group to one displayed point the way a swept analyzer does — `detect(mode)` dispatches on a string, the `detect_*` variants are the direct forms. The peak and marker helpers operate on an already-computed trace.

| Name | Signature | Description |
|------|-----------|-------------|
| `detect` | `(mode: str, bin: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str = 'reference') -> float` | Runtime-dispatch detector. mode is one of: 'peak', 'sample', 'average', 'rms', 'negative_peak'. For a compile-time-known mode prefer the named detect_* functions (one less string parse and switch branch). |
| `detect_peak` | `(bin: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str = 'reference') -> float` | Peak detector: max(bin). The standard scope/analyzer 'peak' mode. |
| `detect_negative_peak` | `(bin: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str = 'reference') -> float` | Negative-peak detector: min(bin). Finds the deepest notch or the noise floor. |
| `detect_sample` | `(bin: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str = 'reference') -> float` | Sample detector: returns the FIRST sample in the bin. Conceptually a 'no-detector' mode — picks one representative time instant per bin, matching the CISPR/Keysight sample-detector convention. |
| `detect_average` | `(bin: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str = 'reference') -> float` | Average detector: arithmetic mean of the bin samples (linear). Sum accumulated in double regardless of dtype. |
| `detect_rms` | `(bin: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str = 'reference') -> float` | RMS (energy) detector: sqrt(mean(bin**2)). For a unit-amplitude sine returns 1/sqrt(2). |
| `find_peaks` | `(trace: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], bin_freq_step_hz: float, top_n: int, min_separation_bins: int = 3, dtype: str = 'reference') -> list[mpdsp._core.Marker]` | Find the top-N strongest peaks in a trace with a minimum-separation greedy selection. Returns a list of Marker objects in descending amplitude order. Sub-bin frequency position is recovered via parabolic interpolation across the three bins around each peak; edge bins skip interpolation. |
| `harmonic_markers` | `(trace: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], bin_freq_step_hz: float, fundamental_hz: float, harmonics: int, dtype: str = 'reference') -> list[mpdsp._core.Marker]` | Markers at bins nearest k * fundamental_hz for k = 2..harmonics+1. Returns a list of Marker objects; harmonics past the trace's frequency range are silently omitted. Combine with find_peaks() and a small neighborhood search to peak-snap each harmonic. |
| `make_delta_marker` | `(a: mpdsp._core.Marker, b: mpdsp._core.Marker) -> mpdsp._core.DeltaMarker` | Compute a DeltaMarker from two Markers: delta_freq_hz and delta_amplitude are b - a. |

## Numerical utilities — polynomial, quadratic, elliptic

Building blocks for advanced filter design: Horner evaluation, polynomial multiplication (convolution), a quadratic solver returning complex roots, and the complete elliptic integral K used by Cauer design.

| Name | Signature | Description |
|------|-----------|-------------|
| `evaluate_polynomial` | `(coeffs: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], x: float) -> float` | evaluate_polynomial(coeffs: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], x: complex) -> complex |
| `multiply_polynomials` | `(a: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], b: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> ndarray` | Multiply two polynomials — coefficient-vector convolution. Result has degree = deg(a) + deg(b). Either input empty gives an empty result. Equivalent to numpy.convolve(a, b, mode='full') for the coefficient-in-ascending-order convention. |
| `solve_quadratic` | `(a: float, b: float, c: float) -> tuple[complex, complex]` | Return both roots of a*x^2 + b*x + c = 0 as a tuple of complex numbers. Real roots are returned with zero imaginary part. |
| `solve_quadratic_1` | `(a: float, b: float, c: float) -> complex` | Root with the positive discriminant sign: (-b + sqrt(b^2 - 4ac)) / (2a). |
| `solve_quadratic_2` | `(a: float, b: float, c: float) -> complex` | Root with the negative discriminant sign: (-b - sqrt(b^2 - 4ac)) / (2a). |
| `elliptic_K` | `(k: float) -> float` | Complete elliptic integral of the first kind K(k) via the arithmetic-geometric mean (AGM) iteration. Modulus k must be in [0, 1). Peak error less than 2e-16. Used by Elliptic (Cauer) filter design. |

## Analog prototypes — s-plane pole/zero constellations

The pre-bilinear view a designed `IIRFilter` hides. Every classical IIR family is designed as an analog prototype in the s-plane and then bilinear-transformed to a digital cascade; the digital response bakes in the resulting frequency warp, while these functions expose the prototype itself. Useful for teaching bilinear warping (plot both and compare) and for reading a family's signature without warp artifacts — Bessel's flat group delay, for instance, is an omega-space property that the digital form only approximates near DC.

Plain `double` throughout: no `dtype=` dispatch, because a prototype is a constellation of exact pole/zero locations rather than a datapath.

The transforms return a **new** `PoleZeroPlot` rather than mutating in place (upstream mutates), so a prototype can feed several transforms without being consumed. Chaining reads left to right:

```python
plot = mpdsp.apply_bilinear(
    mpdsp.lp_to_bp(mpdsp.butterworth_prototype(4, 1.0), 300.0, 3000.0),
    48000.0)
```

| Name | Signature | Description |
|------|-----------|-------------|
| `butterworth_prototype` | `(order: int, cutoff_hz: float = 1.0) -> mpdsp._core.PoleZeroPlot` | Butterworth analog prototype: `order` poles evenly spaced on the left half of a circle of radius 2*pi*cutoff_hz. All-pole — s_zeros is empty. |
| `chebyshev1_prototype` | `(order: int, cutoff_hz: float = 1.0, ripple_db: float = 1.0) -> mpdsp._core.PoleZeroPlot` | Chebyshev I analog prototype: poles on an ellipse, giving equiripple passband at the cost of a less flat response. ripple_db must be > 0. All-pole. |
| `chebyshev2_prototype` | `(order: int, cutoff_hz: float = 1.0, stopband_db: float = 40.0) -> mpdsp._core.PoleZeroPlot` | Chebyshev II (inverse Chebyshev) analog prototype: flat passband, equiripple stopband. Carries finite s_zeros on the jw axis, which is what produces the stopband nulls. stopband_db must be > 0. |
| `bessel_prototype` | `(order: int, cutoff_hz: float = 1.0) -> mpdsp._core.PoleZeroPlot` | Bessel analog prototype: maximally flat group delay. All-pole. The flat-delay signature is an omega-space property, which is why it reads clearly here and only approximately in a bilinear-warped digital response. |
| `elliptic_prototype` | `(order: int, cutoff_hz: float = 1.0, ripple_db: float = 1.0, selectivity_k: float = 0.9) -> mpdsp._core.PoleZeroPlot` | Elliptic (Cauer) analog prototype: equiripple in both bands, the steepest transition for a given order. Carries finite s_zeros. selectivity_k in (0, 1) sets the modulus of the elliptic functions — higher is more selective. order <= 12. |
| `lp_to_hp` | `(plot: mpdsp._core.PoleZeroPlot, cutoff_hz: float) -> mpdsp._core.PoleZeroPlot` | Lowpass -> highpass frequency transformation. Returns a new plot; the input is left unchanged. Pole count is preserved and zeros move to the origin. |
| `lp_to_bp` | `(plot: mpdsp._core.PoleZeroPlot, low_hz: float, high_hz: float) -> mpdsp._core.PoleZeroPlot` | Lowpass -> bandpass frequency transformation. Returns a new plot; the input is left unchanged. Each prototype pole splits into two, so the resulting order is doubled. Requires 0 < low_hz < high_hz. |
| `lp_to_bs` | `(plot: mpdsp._core.PoleZeroPlot, low_hz: float, high_hz: float) -> mpdsp._core.PoleZeroPlot` | Lowpass -> bandstop frequency transformation. Returns a new plot; the input is left unchanged. Order doubles, as with lp_to_bp. Requires 0 < low_hz < high_hz. |
| `apply_bilinear` | `(plot: mpdsp._core.PoleZeroPlot, sample_rate_hz: float) -> mpdsp._core.PoleZeroPlot` | Map the s-plane constellation to the z-plane via the bilinear transform, populating z_poles / z_zeros and sample_rate_hz. Returns a new plot; the input is left unchanged. Every stable analog pole (Re < 0) maps inside the unit circle. |
| `sweep_bode` | `(filt: mpdsp._core.IIRFilter, sample_rate: float, freq_min_hz: float, freq_max_hz: float, num_points: int = 200, settle_samples: int = 512, target_cycles: float = 32.0, max_measure_samples: int = 32768, dtype: str = 'reference') -> mpdsp._core.BodeResult` | sweep_bode(filt: mpdsp._core.FIRFilter, sample_rate: float, freq_min_hz: float, freq_max_hz: float, num_points: int = 200, settle_samples: int = 512, target_cycles: float = 32.0, max_measure_samples: int = 32768, dtype: str = 'reference') -> mpdsp._core.BodeResult |

## Acquisition — high-rate ADC → baseband pipeline

Multirate primitives for the high-rate data-acquisition pipeline (CIC → half-band → polyphase FIR → baseband). The class entries live in the [Classes](#classes) section; the free-function design helpers are listed here.

**Known limitation — `design_halfband` ([#117](https://github.com/stillwater-sc/mp-dsp-python/issues/117)).** The upstream Remez exchange does not converge correctly, so this designer tops out near 21 dB of stopband attenuation and gets *worse* with more taps — 127 taps at `transition_width=0.15` measures −24.7 dB, meaning the stopband sits above the passband. Where real selectivity matters, use `fir_lowpass` with a Kaiser or Blackman window, which reaches 88 dB at 51 taps.

**`NCO` and `DDC` hold `frequency` and `sample_rate` at the configuration's state precision** and divide only afterwards, so absolute rates at RF scale overflow narrow types. Passing normalized rates — `sample_rate=1.0` with the frequency as a fraction of it — is well defined for every dtype and means the same thing, since an oscillator only uses the ratio. Rates that would produce a non-finite phase increment now raise rather than yielding silent NaN.

| Name | Signature | Description |
|------|-----------|-------------|
| `design_halfband` | `(num_taps: int, transition_width: float = 0.1, dtype: str = 'reference') -> ndarray` | Design an equiripple half-band lowpass filter via Remez exchange. num_taps must be of the form 4K+3 (e.g., 7, 11, 15, 19, ...). Returns NumPy float64 taps; dtype controls internal design precision. |
| `polyphase_decompose` | `(taps: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], factor: int, dtype: str = 'reference') -> list[ndarray]` | Decompose an FIR prototype into `factor` polyphase sub-filters. Returns a list of NumPy float64 arrays of length ceil(N/factor). |
| `design_cic_compensator` | `(num_taps: int, cic_stages: int, cic_ratio: int, passband: float, differential_delay: int = 1, dtype: str = 'reference') -> ndarray` | Design an FIR that inverts a CIC decimator's passband droop, to be run at the CIC's output rate. Frequency-sampling design: samples 1/\|H_cic(f)\| across [0, passband], rolls off smoothly to Nyquist, IDFTs, applies a Hamming window, and normalizes to unit DC gain. |

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

`TransferFunction` (discrete-time H(z)) and `ContinuousTransferFunction` (analog H(s)) are the rational-function classes, bound on double. `to_transfer_function(filt)` folds an IIRFilter cascade into a single TF; the spectral-analysis free functions (`ztransform`, `freqz`, `group_delay`, `laplace_freqs`) operate on those classes. `project_onto` / `projection_error` are the round-trip primitives underlying `measure_sqnr_db` — use them when you want the quantized samples or the raw error magnitude rather than the SQNR number.

| Name | Signature | Description |
|------|-----------|-------------|
| `project_onto` | `(data: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str) -> ndarray` | Project data through the sample scalar of `dtype` and back to float64. The round-trip surfaces the quantization error you'd see feeding a signal through an ADC at that precision — it's the underlying mechanic of `measure_sqnr_db`, exposed directly for when you want the quantized samples rather than just the SQNR. |
| `projection_error` | `(data: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], dtype: str) -> float` | Max absolute error between data and its round-trip through `dtype`. Equivalent to max(abs(data - project_onto(data, dtype))) but computed without allocating the intermediate ndarray. |
| `to_transfer_function` | `(filt)` | Fold an `IIRFilter` cascade into a single `TransferFunction`. |
| `ztransform` | `(tf: mpdsp._core.TransferFunction, z: numpy.ndarray[dtype=complex128, shape=(*), order='C', writable=False]) -> ndarray[complex]` | Evaluate H(z) at each z-plane point. Free-function spelling of `tf.evaluate_many(z)`. Returns complex128 ndarray. |
| `freqz` | `(tf: mpdsp._core.TransferFunction, num_points: int = 512) -> ndarray[complex]` | Evaluate H(e^{j 2*pi*f}) at `num_points` uniformly spaced normalized frequencies in [0, 0.5). Returns complex128 ndarray. |
| `group_delay` | `(tf: mpdsp._core.TransferFunction, num_points: int = 512) -> ndarray` | Group delay at `num_points` uniformly spaced normalized frequencies in [0, 0.5). Returns float64 ndarray (samples of -d(phase)/d(omega)). |
| `laplace_freqs` | `(tf: mpdsp._core.ContinuousTransferFunction, omega_max: float, num_points: int = 512) -> ndarray[complex]` | Evaluate H(j*omega) at `num_points` uniformly spaced angular frequencies in [0, omega_max). Returns complex128 ndarray. |

## Numerical analysis — pure-Python helpers

Thin layer over already-bound `IIRFilter` methods. `biquad_poles` is a standalone quadratic solver that takes a 5-tuple of coefficients. `cascade_condition_number(filt, num_freqs)` is the free-function companion to `filt.condition_number(num_freqs)` — identical upstream call, just different calling convention. See `IIRFilter.stability_margin()`, `.condition_number()`, `.worst_case_sensitivity()`, and `.pole_displacement(dtype)` for the per-filter metrics.

| Name | Signature | Description |
|------|-----------|-------------|
| `biquad_poles` | `(b0: 'float', b1: 'float', b2: 'float', a1: 'float', a2: 'float') -> 'list[complex]'` | Two poles of a single biquad section. |
| `max_pole_radius` | `(filt) -> 'float'` | Largest ``\|pole\|`` in the filter's z-plane. |
| `is_stable` | `(filt, tol: 'float' = 0.0) -> 'bool'` | True iff all poles are strictly inside the unit circle. |
| `cascade_condition_number` | `(filt, num_freqs: 'int' = 512) -> 'float'` | Condition number of an entire IIR cascade. |

## Numerical analysis — free-function primitives (bound)

Coefficient-level analysis that doesn't require a constructed IIRFilter — useful for design-time coefficient sweeps. `coefficient_sensitivity` returns the finite-difference pole-radius sensitivities `(dp_da1, dp_da2)`; `biquad_condition_number` returns the max relative response change per unit coefficient perturbation over a frequency sweep. Both bound on double only — mixed-precision analysis of the full filter lives on the IIRFilter methods, which dispatch through `ArithConfig`.

| Name | Signature | Description |
|------|-----------|-------------|
| `coefficient_sensitivity` | `(b0: float, b1: float, b2: float, a1: float, a2: float, epsilon: float = 1e-08) -> tuple` | Coefficient sensitivity of a biquad, as a (dp_da1, dp_da2) tuple of doubles. |
| `biquad_condition_number` | `(b0: float, b1: float, b2: float, a1: float, a2: float, num_freqs: int = 512) -> float` | Condition number of a single biquad section. |
| `enob_from_snr_db` | `(snr_db: float) -> float` | Effective number of bits from SNR (dB) using the standard formula ENOB = (SNR_dB - 1.76) / 6.02. Assumes a sinusoidal full-scale input with quantization-noise-dominated error. |
| `snr_db` | `(reference: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], test: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> float` | Signal-to-noise ratio in dB of `test` against `reference`. Both must be 1D NumPy float64 arrays of equal length. Returns +300 dB (effectively infinite) for a bit-identical match. To assess narrow-precision effects, quantize inputs via mpdsp.adc(x, dtype=...) first, then compute snr_db on the results. |
| `write_acquisition_csv` | `(path: str, rows: collections.abc.Sequence[mpdsp._core.AcquisitionPrecisionRow]) -> None` | Write a list of AcquisitionPrecisionRow to CSV at the given path. Header row is emitted first; column layout matches applications/precision_sweep/precision_sweep.csv for cross-tool compatibility with the existing plot_precision / plot_heatmap scripts. |

## Mixed-precision helpers

`available_dtypes()` is the runtime-queryable source of truth for the string keys accepted by every `dtype=` parameter throughout the API. `compare_filters(filt, signal, dtypes=...)` is the one-call way to sweep SQNR / max-abs-error across all dtypes.

| Name | Signature | Description |
|------|-----------|-------------|
| `available_dtypes` | `() -> list[str]` | List available arithmetic configuration names. |
| `bits_of` | `(dtype: str) -> int` | Return the sample-scalar bit width for `dtype`. Use this to label a precision-vs-cost axis instead of hardcoding the mapping. Raises ValueError for unknown dtype strings. |
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
| `.frequency_response` | `(self, normalized_freqs: numpy.ndarray[dtype=float64, shape=(*), writable=False], dtype: str = 'reference') -> numpy.ndarray[dtype=complex128]` — Evaluate H(e^{j2*pi*f}) at each normalized frequency (f/fs). Returns complex128. |
| `.from_coefficients` | `(biquads: collections.abc.Sequence[mpdsp._core.BiquadCoefficients]) -> mpdsp._core.IIRFilter` — Construct an IIRFilter from a list of BiquadCoefficients. Length must be in [1, 8] (compile-time cascade bound). Each element populates one biquad section in order. Enables importing coefficients designed elsewhere (scipy, MATLAB, hand-designed cascades) into the mpdsp… |
| `.num_stages` | `(self) -> int` — Number of active biquad sections. |
| `.pole_displacement` | `(self, dtype: str) -> float` — Max pole displacement when coefficients are quantized through the target dtype (see available_dtypes). Returns 0 for 'reference'. |
| `.poles` | `(self) -> list[complex]` — List of complex pole locations in the z-plane. |
| `.process` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), writable=False], dtype: str = 'reference') -> numpy.ndarray[dtype=float64]` — Filter a signal. dtype selects arithmetic for state and samples (see available_dtypes()). Returns NumPy float64. |
| `.stability_margin` | `(self) -> float` — 1 - max(\|pole\|). Positive = stable, 0 = marginal, < 0 = unstable. |
| `.worst_case_sensitivity` | `(self, epsilon: float = 1e-08) -> float` — Worst-case \|d(max_pole_radius)/d(coeff)\| across stages, computed by finite differences. |
| `.zeros` | `(self) -> list[complex]` — List of complex zero locations in the z-plane. For all-pole families (Butterworth / Chebyshev I / Bessel / Legendre), all finite zeros map to z = -1, so expect an N-fold cluster there. Chebyshev II and Elliptic distribute zeros on the unit circle. |

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
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Process a 1D NumPy float64 signal. Returns the envelope trace (same length as the input). The per-sample loop releases the GIL internally so other Python threads can run. |
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

### `NCO`

Numerically Controlled Oscillator. Generates complex sinusoids (I/Q) for digital mixing in the acquisition pipeline. Phase-accumulator precision determines spurious-free dynamic range (SFDR ~ 6.02 * W dB for a W-bit accumulator). Phase is in normalized cycles in [0, 1), not radians.

> Numerically Controlled Oscillator. Generates complex sinusoids (I/Q) for digital mixing. Phase accumulator precision determines SFDR.

| Member | Signature / description |
|--------|-------------------------|
| `.generate_block` | `(self, length: int) -> tuple` — Generate a block of complex samples. Returns (real, imag) tuple. |
| `.generate_block_real` | `(self, length: int) -> numpy.ndarray[dtype=float64]` — Generate a block of real-valued samples (cos). |
| `.generate_real` | `(self) -> float` — Generate one real-valued sample (cos only) and advance the phase. |
| `.generate_sample` | `(self) -> tuple[float, float]` — Generate one (real, imag) I/Q sample and advance the phase. |
| `.measure_sfdr_db` | `(self, fft_size: int, guard_bins: int = 2) -> float` — Measure spurious-free dynamic range: generate fft_size samples, FFT them (zero-padded to next power of 2), find the largest spur outside `guard_bins` around the tuned peak, and return 20*log10(peak / spur) in dB. **Mutates the NCO's phase** — call reset() before/after for a… |
| `.mix_down` | `(self, input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> tuple` — Multiply real input by conj(NCO output). Returns (real, imag) tuple of the resulting complex baseband signal. |
| `.phase` | (self) -> float |
| `.phase_increment` | (self) -> float |
| `.reset` | `(self) -> None` |
| `.set_frequency` | `(self, frequency: float, sample_rate: float) -> None` |
| `.set_phase_offset` | `(self, offset: float) -> None` |

### `CICDecimator`

Cascaded Integrator-Comb decimation filter. Multiplier-free; the canonical first decimation stage after a high-rate ADC. Bit growth is `M * ceil(log2(R * D))`.

> Cascaded Integrator-Comb decimation filter. Multiplier-free; ideal for the first decimation stage after a high-rate ADC.

| Member | Signature / description |
|--------|-------------------------|
| `.check_bit_growth` | `(self, input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> mpdsp._core.CICBitGrowthReport` — Run `input` through the CIC and record the peak absolute output. Returns a CICBitGrowthReport comparing observed vs. theoretical (Hogenauer M*ceil(log2(R*D))) bit growth. **Mutates the CIC state** (same as calling process_block); reset() before/after if you need a clean run. |
| `.decimation_ratio` | (self) -> int |
| `.differential_delay` | (self) -> int |
| `.num_stages` | (self) -> int |
| `.output` | Most recent decimated output (valid after push() emits). |
| `.process_block` | `(self, input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Decimate a block; returns the decimated outputs. |
| `.push` | `(self, input: float) -> tuple[bool, float]` — Feed one input sample. Returns (emit, output) — emit is True when the decimated output is valid this call. |
| `.reset` | `(self) -> None` |

### `CICInterpolator`

Cascaded Integrator-Comb interpolation filter — the dual of `CICDecimator`. Multiplier-free upsampling.

> Cascaded Integrator-Comb interpolation filter (the dual of CICDecimator). Multiplier-free upsampling.

| Member | Signature / description |
|--------|-------------------------|
| `.differential_delay` | (self) -> int |
| `.interpolation_ratio` | (self) -> int |
| `.num_stages` | (self) -> int |
| `.output` | `(self) -> float` |
| `.process_block` | `(self, input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Interpolate a block; returns ratio*N output samples. |
| `.push` | `(self, input: float) -> None` |
| `.reset` | `(self) -> None` |

### `HalfBandFilter`

Half-band FIR filter. `process_decimate` / `process_block_decimate` exploit the alternating-zero tap structure to skip ~half the multiplies — typically ~4x faster than naive filter-then-decimate at 2:1.

> Half-band FIR filter. Use process_decimate() / process_block_decimate() for efficient 2x decimation that skips zero-valued tap multiplies.

| Member | Signature / description |
|--------|-------------------------|
| `.num_nonzero_taps` | (self) -> int |
| `.num_taps` | (self) -> int |
| `.process` | `(self, input: float) -> float` — Full-rate process: one input -> one output. |
| `.process_block` | `(self, input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` |
| `.process_block_decimate` | `(self, input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Decimate a block; returns floor(N/2) output samples. |
| `.process_decimate` | `(self, input: float) -> tuple[bool, float]` — 2x decimation: feed one input, returns (emit, output) where emit alternates True/False. |
| `.reset` | `(self) -> None` |
| `.taps` | The design taps this filter was constructed with, as float64. |

### `PolyphaseDecimator`

M-factor polyphase FIR decimator. Decomposes the prototype into M sub-filters; each advances once per output sample, so the multiplier cost is ~N/output instead of ~N*M for naive filter-then-downsample.

> M-factor polyphase FIR decimator. Decomposes the prototype into M sub-filters; each advances once per output sample, so the cost is ~N mults per output instead of ~N*M for naive filter+downsample.

| Member | Signature / description |
|--------|-------------------------|
| `.factor` | (self) -> int |
| `.process` | `(self, input: float) -> tuple[bool, float]` — Feed one input. Returns (emit, output). |
| `.process_block` | `(self, input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` |
| `.reset` | `(self) -> None` |
| `.taps` | The full-rate prototype taps this decimator was constructed with, as float64 (not the decomposed sub-filters — use polyphase_decompose for those). |

### `PolyphaseInterpolator`

L-factor polyphase FIR interpolator. Each input produces L outputs.

| Member | Signature / description |
|--------|-------------------------|
| `.factor` | (self) -> int |
| `.process` | `(self, input: float) -> numpy.ndarray[dtype=float64]` — Feed one input, returns array of `factor` upsampled outputs. |
| `.process_block` | `(self, input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` |
| `.reset` | `(self) -> None` |

### `DDC`

Digital down-converter: mixes a real input band down to complex baseband with an NCO, then decimates the I and Q streams through matched polyphase FIR decimators running in lockstep. A real tone at `center_frequency` lands at baseband DC with magnitude 0.5 — a real cosine is two conjugate half-amplitude exponentials and the mixer keeps one.

The decimator is fixed to `PolyphaseDecimator` and built from `taps` / `decimation_factor`; richer decimator composition is the job of `DecimationChain`. Design `taps` as a lowpass with cutoff below `0.5 / decimation_factor` (normalized to the input rate) to suppress aliasing — `fir_lowpass(...).coefficients()` is the usual source. `process_block` returns a `(real, imag)` tuple, matching `NCO.mix_down`.

> Digital Down-Converter: mixes a real input band down to complex baseband with an NCO, then decimates the I and Q streams through matched polyphase FIR decimators.

| Member | Signature / description |
|--------|-------------------------|
| `.center_frequency` | (self) -> float |
| `.decimation_factor` | (self) -> int |
| `.nco_phase` | Current phase of the internal NCO, in normalized cycles in [0, 1) — multiply by 2*pi for radians. Exposed as a read-only scalar rather than an NCO handle: the DDC owns its oscillator, and handing out a live reference through the type-erased impl would outlive-alias it. |
| `.nco_phase_increment` | Per-sample phase step of the internal NCO, in normalized cycles — equal to center_frequency / sample_rate. |
| `.process` | `(self, input: float) -> tuple[bool, complex]` — Feed one real input sample. Returns (emit, value) where `value` is the complex baseband sample, valid only when emit is True (once per decimation_factor inputs). On non-emit cycles the value is 0j. |
| `.process_block` | `(self, input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> tuple` — Down-convert a block of real samples. Returns a (real, imag) tuple of float64 arrays holding the ~len(input)/decimation_factor complex baseband samples produced during the block — matching the convention used by NCO.mix_down() and NCO.generate_block(). Combine with `real +… |
| `.reset` | `(self) -> None` — Clear the NCO phase and both decimator delay lines. |
| `.sample_rate` | (self) -> float |
| `.set_center_frequency` | `(self, frequency: float) -> None` — Retune the local oscillator. The decimator state is left untouched; call reset() first for a clean retune. |

### `DecimationChain`

Multi-stage decimation cascade — `ADC -> CIC -> half-band -> ... -> baseband`. Large decimation ratios are cheapest as a cascade of small ones, since each stage runs at the progressively lower rate its predecessor emits.

`stages` is a list of `CICDecimator` / `HalfBandFilter` / `PolyphaseDecimator` instances used as **prototypes**: the chain reads their design parameters and rebuilds equivalent stages at the chain's own dtype. Prototypes are neither mutated nor aliased, and their individual dtypes are ignored — upstream threads a single sample type between stages, so the chain's `dtype` governs throughout. `PolyphaseInterpolator` is rejected: it upsamples.

Recommended compositions, input order mattering (bulk reduction first, sharpest filter last): `[CIC]` alone for multiplier-free bulk reduction, paired with `design_cic_compensator` downstream to undo passband droop; `[CIC, HalfBand]` as the common two-stage front end; `[CIC, HalfBand, Polyphase]` to add a shaped FIR at the lowest rate where taps are cheapest; `[CIC, HalfBand, HalfBand, Polyphase]` for a deep SDR cascade.

At most **6 stages** — each additional arity is a separate template instantiation per dtype, so the cap is a compile-time budget rather than an algorithmic limit.

> Multi-stage decimation cascade: ADC -> CIC -> half-band -> ... -> baseband. Large decimation ratios are cheapest as a cascade of small ones, each stage running at the (progressively lower) rate its predecessor emits.

| Member | Signature / description |
|--------|-------------------------|
| `.input_rate` | (self) -> float |
| `.num_stages` | (self) -> int |
| `.output_rate` | input_rate / total_decimation. |
| `.process` | `(self, input: float) -> tuple[bool, float]` — Feed one input sample. Returns (emit, output); emit is True only on the cycle where the *final* stage produces a sample, i.e. once per total_decimation inputs. |
| `.process_block` | `(self, input: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Decimate a block; returns the ~len(input)/total_decimation samples emitted by the final stage. |
| `.reset` | `(self) -> None` — Reset every stage's internal state. |
| `.stage_rates` | `(self) -> list[float]` — Sample rate at the *output* of each stage, in input order. The last element equals output_rate. |
| `.stage_ratios` | `(self) -> list[int]` — Per-stage decimation ratios, in input order. HalfBandFilter reports 2 (it is structurally fixed at 2:1). |
| `.total_decimation` | Product of the per-stage decimation ratios. |

### `PoleZeroPlot`

Analog (s-plane) prototype pole/zero constellation, optionally carrying its bilinear-transformed z-plane counterpart. Produced by the `*_prototype` factories, reshaped by `lp_to_hp` / `lp_to_bp` / `lp_to_bs`, mapped to discrete time by `apply_bilinear`. `z_poles` / `z_zeros` are empty until `apply_bilinear` has been applied.

An immutable value type here: the transforms return new plots rather than mutating, so one prototype can feed several.

> Analog (s-plane) prototype pole/zero constellation, optionally carrying its bilinear-transformed z-plane counterpart.

| Member | Signature / description |
|--------|-------------------------|
| `.cutoff_hz` | (self) -> float |
| `.design` | Family name — 'butterworth', 'chebyshev1', ... |
| `.high_hz` | Upper band edge; set by lp_to_bp / lp_to_bs. |
| `.kind` | 'lowpass', 'highpass', 'bandpass', or 'bandstop'. |
| `.low_hz` | Lower band edge; set by lp_to_bp / lp_to_bs. |
| `.order` | (self) -> int |
| `.ripple_db` | Passband ripple, for the families that use one. |
| `.s_poles` | Continuous-time poles, as a list of complex. |
| `.s_zeros` | Continuous-time zeros. All-pole families return an empty list; elliptic and Chebyshev II carry finite jw-axis zeros. |
| `.sample_rate_hz` | 0.0 until apply_bilinear. |
| `.stopband_db` | Stopband attenuation, for the families that use one. |
| `.z_poles` | Discrete-time poles. Empty until apply_bilinear. |
| `.z_zeros` | Discrete-time zeros. Empty until apply_bilinear. |

### `BodeResult`

Result of a swept Bode measurement — one entry per frequency, as three parallel float64 arrays. Returned by `sweep_bode`. `len(result)` gives the point count.

> Result of a swept Bode measurement: one entry per frequency.

| Member | Signature / description |
|--------|-------------------------|
| `.freqs_hz` | Log-spaced sweep frequencies, in Hz. |
| `.magnitudes_db` | Measured \|H\| in dB. Floored at -300 dB. |
| `.phases_rad` | Measured phase in radians, wrapped to (-pi, pi]. |

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

### `ExtendedKalmanFilter`

> Nonlinear Kalman filter that linearizes the state-transition f(x) and observation h(x) via their Jacobians F(x), H(x) at each step. Users supply FOUR Python callbacks — f, F, h, H — that take a 1D NumPy state vector and return either a 1D vector (for f, h) or a 2D matrix (for F, H).

| Member | Signature / description |
|--------|-------------------------|
| `.P` | State-estimation covariance (state_dim x state_dim). Initialized to the identity; overwrite for informative priors. |
| `.Q` | Process-noise covariance (state_dim x state_dim). |
| `.R` | Measurement-noise covariance (meas_dim x meas_dim). |
| `.dtype` | Arithmetic configuration selected at construction. |
| `.meas_dim` | (self) -> int |
| `.predict` | `(self) -> None` — Propagate the state through f and the covariance through F. Raises if the state function pair hasn't been set. |
| `.set_observation_function` | `(self, h: collections.abc.Callable, H: collections.abc.Callable) -> None` — Register the nonlinear observation h(x) -> vector[meas_dim] and its Jacobian H(x) -> matrix[meas_dim, state_dim]. |
| `.set_state_function` | `(self, f: collections.abc.Callable, F: collections.abc.Callable) -> None` — Register the nonlinear state transition f(x) -> vector[state_dim] and its Jacobian F(x) -> matrix[state_dim, state_dim]. Both must be Python callables returning float64 ndarrays. |
| `.state` | Current state estimate (length state_dim). |
| `.state_dim` | (self) -> int |
| `.update` | `(self, z: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> None` — Apply a measurement of length meas_dim. Raises if the observation function pair hasn't been set. |

### `UnscentedKalmanFilter`

> Nonlinear Kalman filter that propagates a set of 2n+1 sigma points through the state-transition f(x) and observation h(x) functions, then reconstructs the predicted mean and covariance from the propagated points. Unlike the EKF, no Jacobians are required — users supply only two callbacks.

| Member | Signature / description |
|--------|-------------------------|
| `.P` | State-estimation covariance (state_dim x state_dim). |
| `.Q` | Process-noise covariance (state_dim x state_dim). |
| `.R` | Measurement-noise covariance (meas_dim x meas_dim). |
| `.dtype` | Arithmetic configuration selected at construction. |
| `.meas_dim` | (self) -> int |
| `.predict` | `(self) -> None` — Sigma-point predict step. Raises if the state function hasn't been set. |
| `.set_observation_function` | `(self, h: collections.abc.Callable) -> None` — Register the nonlinear observation h(x) -> vector[meas_dim]. |
| `.set_state_function` | `(self, f: collections.abc.Callable) -> None` — Register the nonlinear state transition f(x) -> vector[state_dim]. Must be a Python callable returning a float64 ndarray. No Jacobian needed. |
| `.state` | Current state estimate (length state_dim). |
| `.state_dim` | (self) -> int |
| `.update` | `(self, z: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> None` — Sigma-point update step with measurement of length meas_dim. Raises if the observation function hasn't been set. |

### `LMSFilter`

Least-mean-squares adaptive filter. Coefficients adapt online via the LMS update. `.weights` exposes the current tap vector.

> Least-mean-squares adaptive FIR filter.

| Member | Signature / description |
|--------|-------------------------|
| `.dtype` | Arithmetic configuration selected at construction. |
| `.last_error` | Error residual from the most recent process() call. |
| `.num_taps` | (self) -> int |
| `.process` | `(self, input: float, desired: float) -> tuple[float, float]` — Process one sample with adaptation. Returns a (output, error) tuple where output is y[n] = w^T x[n] and error is d[n] - y[n]. |
| `.process_block` | `(self, inputs: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], desireds: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> tuple[numpy.ndarray[dtype=float64], numpy.ndarray[dtype=float64]]` — Process two equal-length NumPy float64 signals (input, desired) and return a (outputs, errors) tuple of float64 arrays. The per-sample loop releases the GIL. |
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
| `.process_block` | `(self, inputs: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], desireds: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> tuple[numpy.ndarray[dtype=float64], numpy.ndarray[dtype=float64]]` — Process two equal-length NumPy float64 signals (input, desired) and return a (outputs, errors) tuple of float64 arrays. The per-sample loop releases the GIL. |
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
| `.process_block` | `(self, inputs: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], desireds: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> tuple[numpy.ndarray[dtype=float64], numpy.ndarray[dtype=float64]]` — Process two equal-length NumPy float64 signals (input, desired) and return a (outputs, errors) tuple of float64 arrays. The per-sample loop releases the GIL. |
| `.reset` | `(self) -> None` — Zero the weights, delay line, and reset P to delta*I. |
| `.weights` | Current tap weights as a 1D NumPy float64 array (read-only copy). |

### `RationalResampler`

> Polyphase L/M rate conversion — the missing scipy-parity primitive (parallels scipy.signal.resample_poly). A Kaiser-windowed sinc lowpass at cutoff 0.5 / max(L, M) is designed at construction and decomposed into L polyphase sub-filters. (L, M) are reduced by their GCD upstream, so mpdsp.RationalResampler(6, 4) and mpdsp.RationalResampler(3, 2) give identical filters and identical output.

| Member | Signature / description |
|--------|-------------------------|
| `.decim_factor` | Decimation factor M (after GCD reduction). Read-only. |
| `.dtype` | The arithmetic configuration selected at construction. |
| `.interp_factor` | Interpolation factor L (after GCD reduction). Read-only. |
| `.process` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Resample a 1D NumPy float64 signal. Returns a fresh output array; length is ~ len(signal) * L / M plus up to L extra depending on streaming state. |
| `.ratio` | L / M as a float. Read-only. |
| `.reset` | `(self) -> None` — Clear the delay line and time register. Coefficients are preserved. |

### `OverlapAddConvolver`

> Block-based fast FIR convolution via the overlap-add method. Feed exactly block_size samples per process_block() call; each call returns block_size output samples. Call flush() once after the final process_block() to retrieve the trailing M-1 convolution tail (needed to recover the complete linear convolution).

| Member | Signature / description |
|--------|-------------------------|
| `.block_size` | (self) -> int |
| `.dtype` | (self) -> str |
| `.fft_size` | (self) -> int |
| `.filter_length` | (self) -> int |
| `.flush` | `(self) -> numpy.ndarray[dtype=float64]` — Emit the trailing M-1 convolution tail. Call once after the final process_block(); returns an empty array if no tail remains. |
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), writable=False]) -> numpy.ndarray[dtype=float64]` — Process exactly block_size samples; return block_size samples. |
| `.reset` | `(self) -> None` — Clear the internal tail state. Coefficients and sizes are preserved. |

### `OverlapSaveConvolver`

> Block-based fast FIR convolution via the overlap-save method. Feed exactly block_size samples per process_block() call; each call returns block_size output samples. No flush() needed — overlap-save keeps its history in a running buffer and never emits a tail past the last block.

| Member | Signature / description |
|--------|-------------------------|
| `.block_size` | (self) -> int |
| `.dtype` | (self) -> str |
| `.fft_size` | (self) -> int |
| `.filter_length` | (self) -> int |
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), writable=False]) -> numpy.ndarray[dtype=float64]` — Process exactly block_size samples; return block_size samples. |
| `.reset` | `(self) -> None` — Clear the internal history. Coefficients and sizes are preserved. |

### `PeakDetectDecimator`

> Scope-style decimator that emits one (min, max) pair per R input samples. Unlike a generic averaging decimator, a glitch shorter than the decimation interval still shows up in the output because both extremes are preserved.

| Member | Signature / description |
|--------|-------------------------|
| `.decimation_factor` | Decimation factor R (samples per output pair). Read-only. |
| `.dtype` | Sample-scalar dtype fixed at construction. Read-only. |
| `.process` | `(self, sample: float) -> tuple[float, float] \| None` — Push one sample. Returns None while accumulating within a decimation window; returns (min, max) as a tuple of floats on the sample that completes the current window. |
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> tuple` — Push a block of samples. Returns (mins, maxs) as a pair of NumPy arrays. Length of each output = (samples_in_window + len(signal)) // decimation_factor. Partial trailing windows carry over as internal state; call process() or another process_block() to keep going. |
| `.process_block_max` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Same as process_block() but returns only the upper envelope (the maxs array). |
| `.process_block_min` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Same as process_block() but returns only the lower envelope (the mins array). Convenience for callers building single-envelope views. |
| `.reset` | `(self) -> None` — Drop any partial window in progress and re-arm the decimator. |
| `.samples_in_window` | Number of samples pushed into the current incomplete window. Reaches decimation_factor - 1 just before the next output pair is emitted, then wraps back to 0. |

### `TriggerRingBuffer`

> Pre/post-trigger ring buffer for scope-style capture.

| Member | Signature / description |
|--------|-------------------------|
| `.capture_complete` | True once the post-trigger region has been filled (or immediately after push_trigger when post_trigger_samples is zero). Read-only. |
| `.captured_segment` | `(self) -> numpy.ndarray[dtype=float64]` — Return the captured pre + trigger + post window as a NumPy float64 array. Returns an empty array if capture is not yet complete. The output is a fresh copy — safe to hold across rearm(). |
| `.dtype` | Sample-scalar dtype fixed at construction. Read-only. |
| `.post_trigger_capacity` | Configured post-trigger sample count. Read-only. |
| `.pre_trigger_capacity` | Configured pre-trigger sample capacity. Read-only. |
| `.push` | `(self, sample: float) -> None` — Feed one non-trigger sample. Rotates through the pre-trigger ring during PreFill/Armed; extends the capture during Capturing; silently dropped in Complete (rearm first). |
| `.push_trigger` | `(self, sample: float) -> None` — Feed the sample that fires the trigger. Starts a capture using whatever pre-context has accumulated so far — if the pre-trigger ring isn't full yet, the resulting captured segment will be correspondingly shorter. Silently ignored if a capture is already in progress or complete. |
| `.rearm` | `(self) -> None` — Discard the captured segment and resume waiting for the next trigger. The pre-trigger ring retains its content, so a trigger arriving immediately still gets full pre-context. |
| `.reset` | `(self) -> None` — Wipe both the ring and any captured segment; return to a fresh PreFill state. |

### `RealtimeSpectrum`

> Streaming FFT engine that maintains a circular sample ring and produces an FFT every `hop_size` input samples once the initial `fft_size` samples have accumulated. Non-overlapping analysis uses hop_size == fft_size; the conventional 50%-overlap Hann analysis uses hop_size == fft_size // 2.

| Member | Signature / description |
|--------|-------------------------|
| `.dtype` | Scalar dtype fixed at construction. Read-only. |
| `.fft_size` | Configured FFT length. Read-only. |
| `.first_fft_ready` | True once at least one FFT has been produced (equivalent to `total_ffts > 0`). Read-only. |
| `.hop_size` | Configured hop size. Read-only. |
| `.latest_complex` | `(self) -> tuple` — Return the most recent FFT as (real, imag) — two NumPy float64 arrays of length fft_size. Both arrays are empty until `first_fft_ready` is True. |
| `.latest_magnitude_db` | `(self) -> numpy.ndarray[dtype=float64]` — Return the most recent magnitude spectrum in dB with a -200 dB floor. Empty array until `first_fft_ready` is True. |
| `.push` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> int` — Feed a block of samples. Returns the number of complete FFTs produced by this call (0 while still accumulating the initial fft_size samples, then one per hop_size samples). |
| `.reset` | `(self) -> None` — Clear the sample ring and counters; configuration (fft_size, hop_size, window) is preserved. Use between independent stream segments. |
| `.total_ffts` | Number of FFTs produced since construction or last reset(). Read-only. |

### `RBWFilter`

> Resolution-bandwidth filter for a spectrum analyzer: an N-stage synchronously-tuned cascade of RBJ-style bandpass biquads. Sits between the mixer and the detector; selects a narrow window around a center frequency. Higher order tightens the shape factor (60 dB / 3 dB bandwidth ratio) — order=5 gives ~10x shape factor, comparable to a Gaussian for analyzer use.

| Member | Signature / description |
|--------|-------------------------|
| `.bandwidth_hz` | Currently configured -3 dB bandwidth. Read-only. |
| `.center_freq_hz` | Currently tuned center frequency. Read-only. |
| `.dtype` | Scalar dtype fixed at construction. Read-only. |
| `.order` | Number of biquad stages (fixed at construction). Read-only. |
| `.process` | `(self, sample: float) -> float` — Filter one sample; returns the filtered scalar. |
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Filter a block of samples; returns a new NumPy array of the same length. |
| `.reset` | `(self) -> None` — Clear biquad delay-line state; coefficients and order retained. |
| `.retune` | `(self, center_freq_hz: float, bandwidth_hz: float) -> None` — Redesign coefficients around the new (center, bandwidth). State is preserved (bumpless). |
| `.sample_rate_hz` | Streaming sample rate. Read-only. |
| `.shape_factor` | Closed-form analytical 60 dB / 3 dB bandwidth ratio for the current order. Doesn't depend on tuning. Read-only. |

### `VBWFilter`

> Video-bandwidth filter: a single-pole leaky-integrator LPF that smooths detector output before the trace memory. Lower cutoff = more averaging = lower noise floor at the cost of slower response; higher cutoff = faster response but noisier trace. The standard analyzer noise-vs-speed knob.

| Member | Signature / description |
|--------|-------------------------|
| `.cutoff_hz` | Currently configured -3 dB cutoff. Read-only. |
| `.dtype` | Scalar dtype fixed at construction. Read-only. |
| `.process` | `(self, sample: float) -> float` — Filter one sample; returns the filtered scalar. |
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Filter a block of samples; returns a new NumPy array of the same length. |
| `.reset` | `(self) -> None` — Clear the running state y_prev to zero; cutoff and sample rate are preserved. |
| `.sample_rate_hz` | Streaming sample rate. Read-only. |
| `.set_cutoff` | `(self, cutoff_hz: float) -> None` — Redesign alpha for the new cutoff. y_prev is preserved (bumpless). |

### `SweptLO`

> Phase-coherent chirp generator that walks a frequency schedule from f_start to f_stop over a configurable duration, then restarts. The phase accumulator is continuous across the sweep boundary — no glitch at restart. Linear and logarithmic schedules are supported.

| Member | Signature / description |
|--------|-------------------------|
| `.current_frequency_hz` | Instantaneous frequency in Hz, derived from the current phase increment. Read-only. |
| `.dtype` | Scalar dtype fixed at construction. Read-only. |
| `.f_start_hz` | (self) -> float |
| `.f_stop_hz` | (self) -> float |
| `.generate_block` | `(self, n: int) -> tuple` — Advance n samples; returns (cos_array, sin_array) as a tuple of NumPy float64 arrays. |
| `.mode` | 'linear' or 'logarithmic'. Fixed at construction. Read-only. |
| `.num_sweep_samples` | Samples per sweep = floor(sweep_duration_s * sample_rate_hz). Read-only. |
| `.process` | `(self) -> tuple[float, float]` — Advance one sample; returns (cos, sin) as a tuple of floats. |
| `.reset` | `(self) -> None` — Restart the sweep at f_start with phase = 0. Coefficients (delta_inc / ratio_inc) are preserved. |
| `.sample_rate_hz` | (self) -> float |
| `.sweep_complete` | True iff the MOST RECENT process() call wrapped a sweep boundary. One-shot per sweep — the next process() clears it. Read-only. |
| `.sweep_duration_s` | (self) -> float |
| `.total_sweeps` | Monotone count of sweep boundaries crossed since construction or the last reset(). Read-only. |

### `CalibrationProfile`

> Tabulated frequency-response correction for a spectrum-analyzer or scope front end. Stores (frequency_hz, gain_dB, phase_rad) triples; the interpolants linearly interpolate between tabulated points and clamp outside the calibrated band. Fed to FrontEndCorrector to design an inverse-response equalizer.

| Member | Signature / description |
|--------|-------------------------|
| `.freq_max` | (self) -> float |
| `.freq_min` | (self) -> float |
| `.from_csv` | `(path: str) -> mpdsp._core.CalibrationProfile` — Load a profile from CSV. Format: one row per frequency, columns freq_hz, gain_dB, phase_rad. Header row is optional; lines starting with '#' are treated as comments. |
| `.gain_dB` | `(self, freq_hz: float) -> float` — Interpolated gain (dB) at the query frequency. Clamps to the endpoint values below freq_min / above freq_max. |
| `.phase_rad` | `(self, freq_hz: float) -> float` — Interpolated phase (radians) at the query frequency. |
| `.size` | (self) -> int |

### `FrontEndCorrector`

> Front-end equalizer for the analyzer input path: an FIR filter whose magnitude/phase response cancels a CalibrationProfile. Design uses frequency-sampling with a Hamming window; the inverse magnitude is clamped to `max_gain_dB` to avoid amplifying noise where the profile has deep nulls.

| Member | Signature / description |
|--------|-------------------------|
| `.dtype` | Scalar dtype fixed at construction. Read-only. |
| `.num_taps` | Length of the designed FIR (fixed at construction). Read-only. |
| `.process` | `(self, sample: float) -> float` — Filter one sample; returns the equalized scalar. |
| `.process_block` | `(self, signal: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=float64]` — Filter a block of samples; returns a new NumPy array of the same length. |

### `TraceAverager`

> Cross-sweep trace averaging with five commercial-analyzer modes:   linear      — cumulative unweighted mean of all sweeps.   exponential — single-pole IIR y = alpha*x + (1-alpha)*y_prev.                 config is alpha in (0, 1].   max_hold    — element-wise max across all sweeps.   min_hold    — element-wise min across all sweeps.   max_hold_n  — element-wise max over the last N sweeps.                 config is the window N >= 1 (integer-valued).

| Member | Signature / description |
|--------|-------------------------|
| `.accept_sweep` | `(self, trace: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> None` — Push a new sweep. Length must equal trace_length. |
| `.current_trace` | `(self) -> numpy.ndarray[dtype=float64]` — Return the current accumulated trace as a NumPy array. Value is meaningful only after at least one accept_sweep(). |
| `.dtype` | Scalar dtype fixed at construction. Read-only. |
| `.mode` | 'linear' / 'exponential' / 'max_hold' / 'min_hold' / 'max_hold_n'. Read-only. |
| `.reset` | `(self) -> None` — Discard accumulated state; mode and config are preserved. |
| `.sweeps_accumulated` | Number of sweeps accepted since construction or last reset(). Read-only. |
| `.trace_length` | Fixed bin count per sweep. Read-only. |

### `WaterfallBuffer`

> Circular buffer storing the last num_frames FFT magnitude frames from a streaming spectrum processor. Each frame has num_bins samples. When the ring is full, push_frame overwrites the oldest frame.

| Member | Signature / description |
|--------|-------------------------|
| `.clear` | `(self) -> None` — Discard all stored frames; capacity preserved. |
| `.dtype` | (self) -> str |
| `.frame_at` | `(self, idx_from_oldest: int) -> numpy.ndarray[dtype=float64]` — Return the chronologically-indexed frame (0 = oldest, num_frames_filled - 1 = newest) as a NumPy 1D array. Fresh copy — safe to hold across further push_frame calls. |
| `.last_frames` | `(self, count: int) -> numpy.ndarray[dtype=float64]` — Return the most recent `count` frames as a 2D NumPy array shape (available, num_bins), oldest first. count is clamped to num_frames_filled — fewer-than-requested frames are returned when the buffer hasn't filled yet. |
| `.num_bins` | (self) -> int |
| `.num_frames_capacity` | (self) -> int |
| `.num_frames_filled` | (self) -> int |
| `.push_frame` | `(self, magnitude: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> None` — Append one frame. Length must equal num_bins. |

### `Marker`

> A single marker on a spectrum trace: bin index, sub-bin-interpolated frequency (Hz), and amplitude. Returned by find_peaks() and harmonic_markers(); consumed by make_delta_marker().

| Member | Signature / description |
|--------|-------------------------|
| `.amplitude` | (self) -> float |
| `.bin_index` | (self) -> int |
| `.frequency_hz` | (self) -> float |

### `DeltaMarker`

> Two-marker delta measurement. delta_freq_hz and delta_amplitude are `b` minus `a`, matching the convention of every commercial analyzer's delta-marker mode.

| Member | Signature / description |
|--------|-------------------------|
| `.a` | (self) -> mpdsp._core.Marker |
| `.b` | (self) -> mpdsp._core.Marker |
| `.delta_amplitude` | (self) -> float |
| `.delta_freq_hz` | (self) -> float |

### `RootFinder`

> Complex polynomial root finder via Laguerre's method with deflation and optional polishing. Supports polynomials up to degree 32 (compile-time bound; passing a longer coefficient array raises).

| Member | Signature / description |
|--------|-------------------------|
| `.degree` | Current polynomial degree (0 before set_coefficients()). |
| `.max_degree` | Compile-time bound on the polynomial degree (32). |
| `.roots` | `(self) -> numpy.ndarray[dtype=complex128]` — Return the `degree` roots as a NumPy complex128 array. Requires solve() to have been called after set_coefficients(). |
| `.set_coefficients` | `(self, coeffs: numpy.ndarray[dtype=complex128, shape=(*), order='C', writable=False]) -> None` — Set polynomial coefficients from a NumPy complex128 array. Length is degree+1; element i is the coefficient of x^i (ascending order). Degree is inferred from array length. Maximum degree is 32. |
| `.solve` | `(self, polish: bool = True, sort: bool = True) -> None` — Find all roots of the polynomial set via set_coefficients(). polish=True (default) refines each root using the original (un-deflated) polynomial for accuracy. sort=True (default) orders roots by descending imaginary part. |

### `CICBitGrowthReport`

> Result of check_cic_bit_growth: theoretical vs. observed bit growth of a CIC decimator's output. `within_theory` is True when observed <= theoretical (the normal case for well-behaved inputs).

| Member | Signature / description |
|--------|-------------------------|
| `.headroom_bits` | theoretical - observed, both as floats (positive means headroom remaining). |
| `.max_abs_output` | Raw measured peak of \|output\|. |
| `.observed_bits` | ceil(log2(max \|output\|)) for the test input. |
| `.theoretical_bits` | M * ceil(log2(R*D)) — Hogenauer's formula. |
| `.within_theory` | True when observed <= theoretical. |

### `AcquisitionPrecisionRow`

> Schema-compatible Pareto-row record for the acquisition-pipeline precision sweeps. Written by write_acquisition_csv into the same column layout as applications/precision_sweep/precision_sweep.csv so the existing plot_precision and plot_heatmap scripts can read either file.

| Member | Signature / description |
|--------|-------------------------|
| `.cic_overflow_margin_bits` | Set to -1 when not applicable to the row's pipeline. |
| `.coeff_type` | String repr of CoeffScalar (e.g. 'double', 'posit<32,2>'). |
| `.config_name` | Human-readable configuration label. |
| `.nco_sfdr_db` | Set to -1 when not applicable to the row's pipeline. |
| `.output_enob` | (self) -> float |
| `.output_snr_db` | (self) -> float |
| `.pipeline` | Pipeline identifier: 'ddc' / 'decim_chain' / 'nco' / etc. |
| `.sample_type` | (self) -> str |
| `.state_type` | (self) -> str |
| `.total_bits` | Sum of bit-widths across the three scalars. |

### `ComplexPair`

> A pair of complex numbers — the building block for pole/zero representations that map directly to second-order sections. Typically holds either a conjugate pair or a pair of real values.

| Member | Signature / description |
|--------|-------------------------|
| `.first` | (self) -> complex |
| `.is_conjugate` | `(self) -> bool` — True if second == conj(first). |
| `.is_matched_pair` | `(self) -> bool` — True if this is either a conjugate pair or a pair of real values where neither is zero. |
| `.is_nan` | `(self) -> bool` — True if any real or imaginary component is NaN. |
| `.is_real` | `(self) -> bool` — True if both entries have zero imaginary part. |
| `.second` | (self) -> complex |

### `PoleZeroPair`

> Poles + zeros for a single second-order section (biquad). For a first-order section, the `.second` complex value in each ComplexPair is zero (see is_single_pole()).

| Member | Signature / description |
|--------|-------------------------|
| `.is_nan` | `(self) -> bool` |
| `.is_single_pole` | `(self) -> bool` — True if this represents a first-order section (second entries of both pole and zero pairs are zero). |
| `.poles` | (self) -> mpdsp._core.ComplexPair |
| `.zeros` | (self) -> mpdsp._core.ComplexPair |

### `BiquadCoefficients`

> Coefficients for a second-order (biquad) IIR section:   H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)

| Member | Signature / description |
|--------|-------------------------|
| `.a1` | (self) -> float |
| `.a2` | (self) -> float |
| `.apply_scale` | `(self, scale: float) -> None` — Multiply the numerator coefficients (b0, b1, b2) by a gain scale factor. |
| `.b0` | (self) -> float |
| `.b1` | (self) -> float |
| `.b2` | (self) -> float |
| `.response` | `(self, normalized_freq: float) -> complex` — Evaluate H(e^{j*2*pi*f}) at the normalized frequency f in [0, 0.5], where f = frequency / sample_rate. Returns complex. |
| `.set_from_pole_zero_pair` | `(self, pz: mpdsp._core.PoleZeroPair) -> None` — Set from a PoleZeroPair. Dispatches to set_one_pole or set_two_pole based on pz.is_single_pole(). |
| `.set_identity` | `(self) -> None` — Reset to the pass-through filter H(z) = 1 (b0=1, all others zero). |
| `.set_one_pole` | `(self, pole: complex, zero: complex) -> None` — Set from a first-order section (single pole, single zero). |
| `.set_two_pole` | `(self, pole1: complex, zero1: complex, pole2: complex, zero2: complex) -> None` — Set from a conjugate pair of poles and zeros (second-order section). |

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

### `ContinuousTransferFunction`

Analog (continuous-time) rational H(s) = N(s)/D(s) with coefficients in ascending powers of s. Use with `laplace_freqs(tf, omega_max, N)` to evaluate frequency response at uniformly spaced angular frequencies — the natural path for analyzing analog prototype filters before bilinear transformation to the digital domain.

> Continuous-time (analog) rational transfer function H(s) = N(s) / D(s).

| Member | Signature / description |
|--------|-------------------------|
| `.denominator` | Denominator coefficients in ascending powers of s. |
| `.evaluate` | `(self, s: complex) -> complex` — Evaluate H(s) at a single complex s-plane point. |
| `.evaluate_many` | `(self, s: numpy.ndarray[dtype=complex128, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=complex128]` — Evaluate H(s) at each point in a complex128 ndarray. Returns a complex128 ndarray of the same length. |
| `.frequency_response` | `(self, omega: float) -> complex` — Evaluate H(j*omega) at angular frequency omega (rad/s). |
| `.frequency_response_many` | `(self, omegas: numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]) -> numpy.ndarray[dtype=complex128]` — Vectorized frequency_response(...) over a float64 ndarray of angular frequencies. Returns complex128. |
| `.numerator` | Numerator coefficients in ascending powers of s. |

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
