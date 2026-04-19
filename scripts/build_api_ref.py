"""Generate docs/api_reference.md from an installed mpdsp.

Harvests nanobind's type-annotated first-line signatures and any prose
following in each binding's __doc__. For pure-Python helpers, uses
inspect.signature + the first paragraph of __doc__.

Run from the repo root against an editable install of mpdsp:

    pip install -e .
    python scripts/build_api_ref.py

When adding a new binding, add its name to the right `CATEGORIES`
entry (or `CLASSES` for a stateful class) and, if the category is
new, write a short intro in `INTROS` / `CLASS_INTROS` explaining the
precision invariants or dispatch model for that subsystem.
"""

from __future__ import annotations

import inspect
import re
import textwrap
from typing import Iterable

import mpdsp


def slugify(title: str) -> str:
    """Match GitHub's Markdown heading-anchor slug algorithm.

    GFM drops every character that isn't a letter, digit, whitespace,
    underscore, or hyphen — so `/`, `+`, `(`, `)`, em-dash all go to
    nothing. Then each remaining whitespace character becomes one
    hyphen (so two spaces produced by stripping an em-dash plus its
    surrounding whitespace yield a double hyphen — matching GitHub,
    which is the behavior CodeRabbit's MD051 check pins).
    """
    s = title.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s", "-", s)
    return s


def sig_and_blurb(name: str, obj) -> tuple[str, str]:
    """Return (signature_str, one_line_blurb) for a binding.

    nanobind exposes the signature as the first line of __doc__ (since
    inspect.signature returns *args, **kwargs for the C++-defined
    callable). Pure Python functions have a real inspect.signature.
    """
    doc = obj.__doc__ or ""
    lines = [ln for ln in doc.split("\n")]
    first = lines[0].strip() if lines else ""

    # Heuristic: nanobind's first line always starts with `name(` and looks
    # like a type-annotated signature. Pure Python callables don't.
    if first.startswith(f"{name}("):
        sig = first
        # First non-blank line after the signature, if any, as the blurb.
        blurb = ""
        for ln in lines[1:]:
            s = ln.strip()
            if s:
                blurb = s
                break
    else:
        try:
            sig = f"{name}{inspect.signature(obj)}"
        except (TypeError, ValueError):
            sig = f"{name}(...)"
        # First paragraph of the Python docstring.
        paras = [p.strip() for p in doc.split("\n\n") if p.strip()]
        blurb = paras[0].replace("\n", " ") if paras else ""

    blurb = " ".join(blurb.split())
    return sig, blurb


def class_methods(cls) -> list[tuple[str, str, str]]:
    """List (method_name, signature, blurb) for public methods of a class.

    Skips dunder methods and properties that look like constant state
    (dtype, num_taps, etc. — listed in the class's prose instead).
    """
    rows = []
    for mname in sorted(m for m in dir(cls) if not m.startswith("_")):
        member = getattr(cls, mname, None)
        if member is None:
            continue
        doc = getattr(member, "__doc__", "") or ""
        first = (doc.split("\n")[0] if doc else "").strip()
        # Skip obvious noise — empty-doc properties just get listed in prose.
        sig, blurb = sig_and_blurb(mname, member)
        # Prefix with dot so it's clearly a method in the rendered table.
        rows.append((f".{mname}", sig, blurb))
    return rows


# --- Category definitions ---
# Each category is (title, [list of names]). The order of names within a
# category is deliberate and the order of categories follows the README's
# module table.

CATEGORIES = [
    ("Signal generators", [
        "sine", "cosine", "chirp", "square", "triangle", "sawtooth",
        "impulse", "step", "white_noise", "gaussian_noise", "pink_noise",
    ]),
    ("Window functions", [
        "hamming", "hanning", "blackman", "kaiser", "rectangular", "flat_top",
    ]),
    ("Quantization", [
        "adc", "dac", "sqnr_db", "measure_sqnr_db",
        "max_absolute_error", "max_relative_error",
    ]),
    ("Spectral analysis", [
        "fft", "ifft", "fft_magnitude_db", "psd", "periodogram", "spectrogram",
    ]),
    ("IIR filter design — classical families", [
        "butterworth_lowpass", "butterworth_highpass",
        "butterworth_bandpass", "butterworth_bandstop",
        "chebyshev1_lowpass", "chebyshev1_highpass",
        "chebyshev1_bandpass", "chebyshev1_bandstop",
        "chebyshev2_lowpass", "chebyshev2_highpass",
        "chebyshev2_bandpass", "chebyshev2_bandstop",
        "bessel_lowpass", "bessel_highpass",
        "bessel_bandpass", "bessel_bandstop",
        "legendre_lowpass", "legendre_highpass",
        "legendre_bandpass", "legendre_bandstop",
        "elliptic_lowpass", "elliptic_highpass",
        "elliptic_bandpass", "elliptic_bandstop",
    ]),
    ("IIR filter design — RBJ biquads", [
        "rbj_lowpass", "rbj_highpass",
        "rbj_bandpass", "rbj_bandstop",
        "rbj_allpass", "rbj_lowshelf", "rbj_highshelf",
    ]),
    ("FIR filter design", [
        "fir_lowpass", "fir_highpass", "fir_bandpass", "fir_bandstop",
        "fir_filter",
    ]),
    ("Image — generators", [
        "checkerboard", "stripes_horizontal", "stripes_vertical", "grid",
        "gradient_horizontal", "gradient_vertical", "gradient_radial",
        "gaussian_blob", "circle", "rectangle", "zone_plate",
        "uniform_noise_image", "gaussian_noise_image", "salt_and_pepper",
        "add_noise", "threshold",
    ]),
    ("Image — processing", [
        "convolve2d", "separable_filter", "gaussian_blur", "box_blur",
        "sobel_x", "sobel_y", "prewitt_x", "prewitt_y",
        "gradient_magnitude", "canny", "rgb_to_gray",
    ]),
    ("Image — morphology", [
        "make_rect_element", "make_cross_element", "make_ellipse_element",
        "dilate", "erode",
        "morphological_open", "morphological_close", "morphological_gradient",
        "tophat", "blackhat",
    ]),
    ("Image — file I/O", [
        "read_pgm", "write_pgm", "read_ppm", "write_ppm",
        "read_bmp", "write_bmp", "write_bmp_rgb",
    ]),
    ("Types — transfer function and type projection", [
        "project_onto", "projection_error", "to_transfer_function",
    ]),
    ("Numerical-analysis helpers (pure Python)", [
        "biquad_poles", "max_pole_radius", "is_stable",
    ]),
    ("Mixed-precision helpers", [
        "available_dtypes", "compare_filters",
    ]),
    ("CSV + image-pipeline helpers (pure Python)", [
        "load_sweep", "apply_per_channel", "collect_adaptive_weights",
    ]),
    ("Matplotlib plotting helpers", [
        "plot_signal", "plot_spectrum", "plot_signal_and_spectrum",
        "plot_quantization_comparison", "plot_sqnr_comparison",
        "plot_window_comparison", "plot_spectrogram", "plot_psd",
        "plot_filter_comparison",
        "plot_kalman_tracking", "plot_adaptive_convergence",
        "plot_image", "plot_image_grid", "plot_pipeline",
    ]),
]


CLASSES = [
    "IIRFilter", "FIRFilter",
    "RPDFDither", "TPDFDither", "FirstOrderNoiseShaper",
    "PeakEnvelope", "RMSEnvelope", "Compressor", "AGC",
    "KalmanFilter", "LMSFilter", "NLMSFilter", "RLSFilter",
    "TransferFunction",
]


# --- Category prose intros ---

INTROS = {
    "Signal generators": (
        "Return a 1D float64 NumPy array. All generators except the noise "
        "family accept deterministic parameters; `white_noise`, "
        "`gaussian_noise`, and `pink_noise` additionally take a `seed` "
        "argument (default 0 → nondeterministic from `std::random_device`)."
    ),
    "Window functions": (
        "Return a length-N float64 NumPy array. Apply by element-wise "
        "multiplication against a signal before spectral analysis. "
        "`kaiser` additionally takes a shape parameter `beta`."
    ),
    "Quantization": (
        "`adc` / `dac` round-trip a signal through the target precision — "
        "ADC models the quantization step, DAC the reconstruction step "
        "(in Python, both sides are float64, so they're mechanically "
        "symmetric but serve different roles in pipeline code). "
        "`RPDFDither`, `TPDFDither` (stateful classes in the Classes "
        "section below) add decorrelating noise before quantization; "
        "`FirstOrderNoiseShaper` pushes quantization-noise energy out of "
        "the signal band via error feedback. The remaining primitives "
        "measure how far a quantized signal drifted from its reference."
    ),
    "Spectral analysis": (
        "All spectral primitives in 0.4.x operate in double precision — "
        "dtype dispatch on FFT/PSD/spectrogram is planned for 0.5.0 "
        "(see issue #40). Signal inputs must be 1D float64."
    ),
    "IIR filter design — classical families": (
        "Each function designs the filter in double precision and returns "
        "an `IIRFilter` object whose `.process(signal, dtype=...)` method "
        "dispatches through the target arithmetic. Chebyshev I, Chebyshev "
        "II, and Elliptic take additional passband-ripple / stopband-"
        "attenuation parameters."
    ),
    "IIR filter design — RBJ biquads": (
        "Robert Bristow-Johnson audio-EQ biquads. Always 2nd-order (no "
        "`order` parameter). Include shelf and allpass topologies not "
        "present in the classical families. Parameterized by `q` "
        "(quality factor) or `bandwidth` (for BP/BS); shelves take `gain_db`."
    ),
    "FIR filter design": (
        "Window-method designs returning an `FIRFilter`. `fir_filter` "
        "constructs directly from a coefficient array when you need a "
        "custom design."
    ),
    "Image — generators": (
        "All return `(rows, cols)` float64 2D NumPy arrays. The `*_noise*` "
        "and `salt_and_pepper` generators accept a `seed`. `threshold` is "
        "both a generator (arguments like `threshold(image, value)`) and "
        "a pipeline primitive — consult the signature."
    ),
    "Image — processing": (
        "All take and return `(rows, cols)` float64 2D arrays. Almost every "
        "processing function accepts a `dtype=` parameter for mixed-"
        "precision dispatch on the internal arithmetic. `border=` "
        "(`\"reflect_101\"` by default) controls the boundary handling for "
        "convolution-based operations."
    ),
    "Image — morphology": (
        "The `make_*_element` helpers construct structuring elements "
        "(boolean 2D arrays) for `dilate`/`erode` and the higher-level "
        "compositions (open, close, gradient, tophat, blackhat). All "
        "accept `dtype=` for mixed-precision arithmetic on the "
        "max-reduction."
    ),
    "Image — file I/O": (
        "PGM (grayscale 8/16-bit), PPM (RGB 8-bit), and BMP (8-bit "
        "grayscale + RGB). Reads return float64 arrays normalized to "
        "`[0.0, 1.0]`; writes expect the same range. WAV support is "
        "planned for 0.5.0 (see issue #40)."
    ),
    "Types — transfer function and type projection": (
        "`TransferFunction` is bound on double in 0.5.0 and represents the "
        "rational H(z) = B(z)/A(z) directly (as opposed to `IIRFilter`'s "
        "cascade-of-biquads form). Use `to_transfer_function(filt)` to "
        "fold an IIR cascade into a single TF for evaluation, cascade "
        "composition, or handing to the upcoming `ztransform` (Phase 5 / "
        "#54). `project_onto` / `projection_error` are the round-trip "
        "primitives underlying `measure_sqnr_db` — use them when you want "
        "the quantized samples or the raw error magnitude rather than the "
        "SQNR number."
    ),
    "Numerical-analysis helpers (pure Python)": (
        "Thin layer over already-bound `IIRFilter` methods. "
        "`biquad_poles` is a standalone quadratic solver that takes a "
        "5-tuple of coefficients. See `IIRFilter.stability_margin()`, "
        "`.condition_number()`, `.worst_case_sensitivity()`, and "
        "`.pole_displacement(dtype)` for the per-filter metrics."
    ),
    "Mixed-precision helpers": (
        "`available_dtypes()` is the runtime-queryable source of truth for "
        "the string keys accepted by every `dtype=` parameter throughout "
        "the API. `compare_filters(filt, signal, dtypes=...)` is the "
        "one-call way to sweep SQNR / max-abs-error across all dtypes."
    ),
    "CSV + image-pipeline helpers (pure Python)": (
        "`load_sweep` reads the CSV emitted by upstream `iir_precision_sweep`. "
        "`apply_per_channel` maps a single-channel function across a "
        "multi-channel image. `collect_adaptive_weights` drives an "
        "`LMSFilter` / `NLMSFilter` / `RLSFilter` and returns the weight "
        "trajectory."
    ),
    "Matplotlib plotting helpers": (
        "All optional — require `mpdsp[plot]`. Return `matplotlib.figure.Figure` "
        "objects so the caller can `fig.savefig(...)` or further customize. "
        "None of these are callable in a headless environment without a "
        "matplotlib backend set to `Agg` first."
    ),
}


CLASS_INTROS = {
    "IIRFilter": (
        "Returned by every `*_lowpass` / `*_highpass` / `*_bandpass` / "
        "`*_bandstop` / `rbj_*` designer. Coefficients are always designed "
        "in double; processing, analysis, and pole placement happen per "
        "the dtype passed to each method."
    ),
    "FIRFilter": (
        "Returned by `fir_lowpass` / `fir_highpass` / `fir_bandpass` / "
        "`fir_bandstop` / `fir_filter`. Direct-form convolution; coefficients "
        "in double, processing dispatches via `dtype=`."
    ),
    "PeakEnvelope": (
        "Peak envelope follower with configurable attack/release. The "
        "`.value` property exposes the current envelope state."
    ),
    "RMSEnvelope": (
        "RMS envelope follower. Same interface shape as `PeakEnvelope`; "
        "tracks the signal's moving root-mean-square."
    ),
    "Compressor": (
        "Dynamic-range compressor. Threshold, ratio, attack/release, and "
        "optional makeup gain + soft-knee. Internal envelope follower is "
        "peak-based."
    ),
    "AGC": (
        "Automatic gain control: drives the signal toward a target level "
        "using a configurable attack/release time constant."
    ),
    "KalmanFilter": (
        "Linear Kalman filter. State/measurement/control dimensions set at "
        "construction. `F`, `H`, `Q`, `R`, `P`, `B` are writeable NumPy "
        "2D array properties; `state` is the 1D state vector. Call "
        "`.predict()` then `.update(measurement)` each step."
    ),
    "LMSFilter": (
        "Least-mean-squares adaptive filter. Coefficients adapt online "
        "via the LMS update. `.weights` exposes the current tap vector."
    ),
    "NLMSFilter": (
        "Normalized LMS — divides the step size by the input power for "
        "tunability that's robust across signal levels."
    ),
    "RLSFilter": (
        "Recursive least-squares adaptive filter. Faster convergence than "
        "LMS/NLMS at the cost of O(N²) memory for the P matrix. Known to "
        "diverge under reduced precision when P loses symmetry — see "
        "`notebooks/06_estimation.ipynb`."
    ),
    "RPDFDither": (
        "Rectangular-PDF (uniform) dither generator. Produces noise in "
        "`[-amplitude, +amplitude]`. Use before quantization to decorrelate "
        "error from the signal, at the cost of a flat noise floor. "
        "Stateful because it carries a `std::mt19937` internally."
    ),
    "TPDFDither": (
        "Triangular-PDF dither generator — sum of two RPDF draws. "
        "Eliminates the noise-modulation artifact that RPDF leaves on "
        "low-level signals, at a +3 dB noise-power cost. Generally "
        "preferred over RPDF when the added noise power is tolerable."
    ),
    "FirstOrderNoiseShaper": (
        "First-order error-feedback noise shaper. Quantizes `double → "
        "dtype → double` while feeding the quantization error back "
        "(negated) onto the next input. First-order shaping is a high-"
        "pass on the noise floor — most useful upstream of a lowpass "
        "reconstruction that rejects the shifted noise."
    ),
    "TransferFunction": (
        "Rational H(z) = B(z)/A(z) with double-precision coefficients. "
        "Construct from numerator + denominator ndarrays; the leading `1` "
        "in the denominator is implicit (don't pass `a0`). Cascade via "
        "`*`. The `to_transfer_function(filt)` helper folds an IIRFilter "
        "cascade into one of these, useful when evaluating the full "
        "filter's H(z) directly rather than staging by stage."
    ),
}


# --- Render ---

def render_function_table(names: Iterable[str]) -> str:
    rows = ["| Name | Signature | Description |",
            "|------|-----------|-------------|"]
    for n in names:
        obj = getattr(mpdsp, n, None)
        if obj is None:
            rows.append(f"| `{n}` | *(not exported)* | |")
            continue
        sig, blurb = sig_and_blurb(n, obj)
        # Indent-wrap the signature into a single cell; markdown tables
        # don't like multi-line cells so collapse to one line.
        sig_compact = " ".join(sig.split())
        # Shorten verbose numpy ndarray type annotations for readability.
        sig_compact = sig_compact.replace(
            "numpy.ndarray[dtype=float64, shape=(*, *), order='C', writable=False]",
            "ndarray2d[ro]")
        sig_compact = sig_compact.replace(
            "numpy.ndarray[dtype=float64, shape=(*, *)]", "ndarray2d")
        sig_compact = sig_compact.replace(
            "numpy.ndarray[dtype=float64, shape=(*), writable=False]",
            "ndarray1d[ro]")
        sig_compact = sig_compact.replace(
            "numpy.ndarray[dtype=float64, shape=(*)]", "ndarray1d")
        sig_compact = sig_compact.replace(
            "numpy.ndarray[dtype=float64]", "ndarray")
        sig_compact = sig_compact.replace(
            "numpy.ndarray[dtype=complex128]", "ndarray[complex]")
        sig_compact = sig_compact.replace(
            "numpy.ndarray[dtype=bool, shape=(*, *), order='C', writable=False]",
            "ndarray2d[bool, ro]")
        sig_compact = sig_compact.replace(
            "numpy.ndarray[dtype=bool, shape=(*, *)]", "ndarray2d[bool]")
        # Drop leading `name(` and trailing `)` so the name column carries
        # that, leaving signature as just the args + return annotation.
        prefix = f"{n}("
        if sig_compact.startswith(prefix):
            sig_compact = "(" + sig_compact[len(prefix):]
        # Wrap inline code and escape pipes that would break table cells.
        sig_escaped = sig_compact.replace("|", "\\|")
        blurb_escaped = blurb.replace("|", "\\|") if blurb else "—"
        rows.append(f"| `{n}` | `{sig_escaped}` | {blurb_escaped} |")
    return "\n".join(rows)


def render_class(name: str) -> str:
    cls = getattr(mpdsp, name)
    intro = CLASS_INTROS.get(name, "")
    doc = cls.__doc__ or ""
    class_prose = doc.strip().split("\n\n")[0] if doc.strip() else ""

    lines = [f"### `{name}`", ""]
    if intro:
        lines.append(intro)
        lines.append("")
    if class_prose and class_prose != intro:
        lines.append("> " + class_prose.replace("\n", " "))
        lines.append("")

    # Attributes + methods. We emit a table of public names; nanobind
    # property signatures aren't in inspect.signature but __doc__ has them.
    lines.append("| Member | Signature / description |")
    lines.append("|--------|-------------------------|")
    for mname in sorted(m for m in dir(cls) if not m.startswith("_")):
        member = getattr(cls, mname)
        mdoc = (getattr(member, "__doc__", "") or "").strip()
        first = mdoc.split("\n")[0].strip() if mdoc else ""
        # Render either the signature or "(property)" + first-line blurb.
        if first.startswith(f"{mname}("):
            sig = " ".join(first.split())
            prefix = f"{mname}("
            sig_args = "(" + sig[len(prefix):] if sig.startswith(prefix) else sig
            # Remaining doc content after the first line (if any). Drop the
            # "Overloaded function." banner and its enumerated re-listing
            # of signatures — those would just repeat the signature above
            # in a broken-table way. Take the first non-signature paragraph.
            tail = mdoc.split("\n", 1)[1] if "\n" in mdoc else ""
            paras = [p.strip() for p in tail.split("\n\n") if p.strip()]
            rest = ""
            for p in paras:
                if "Overloaded function" in p:
                    continue
                if p.startswith(("1.", "2.")) or p.startswith(f"{mname}("):
                    continue
                rest = " ".join(p.split())[:140]
                break
            cell = f"`{sig_args}`" + (f" — {rest}" if rest else "")
        else:
            cell = first if first else "*(property)*"
            cell = cell[:160]
        lines.append(f"| `.{mname}` | {cell.replace('|', '\\|')} |")
    return "\n".join(lines)


def render_attributes_section() -> str:
    rows = ["| Attribute | Type | Description |",
            "|-----------|------|-------------|"]
    for name in ["__version__", "__dsp_version__", "__dsp_version_info__",
                 "HAS_CORE", "HAS_PLOT", "__core_import_error__"]:
        val = getattr(mpdsp, name, None)
        t = type(val).__name__
        desc = {
            "__version__": "The installed wheel version (PEP 440).",
            "__dsp_version__": ("The upstream `sw::dsp` C++ library version "
                                 "the wheel was built against."),
            "__dsp_version_info__": ("`(major, minor, patch)` tuple of ints "
                                      "for `__dsp_version__`."),
            "HAS_CORE": ("`True` when the nanobind extension imported "
                          "cleanly. `False` in unbuilt source checkouts, and "
                          "(pre-0.4.1.post1) indicated a packaging bug "
                          "before we hardened the import."),
            "HAS_PLOT": ("`True` when matplotlib is importable — gates the "
                          "`plot_*` helpers."),
            "__core_import_error__": ("`None` if `HAS_CORE`; otherwise the "
                                        "exception raised when `_core` "
                                        "failed to import."),
        }.get(name, "")
        if name in ("__version__", "__dsp_version__") and val is not None:
            desc += f" Current: `\"{val}\"`."
        rows.append(f"| `mpdsp.{name}` | `{t}` | {desc} |")
    return "\n".join(rows)


def main():
    parts = []
    parts.append(f"""# `mpdsp` API reference

Complete enumeration of every public name in the `mpdsp` package, grouped
by subsystem. Generated from `{mpdsp.__version__}` (upstream `sw::dsp
{mpdsp.__dsp_version__}`) via `inspect` and the nanobind-attached
`__doc__` strings. Keep this in sync by re-running the generator — see
the note at the bottom.

---

## Contents

- [Arithmetic configurations](#arithmetic-configurations)
- [Module attributes](#module-attributes)
""")
    for title, _ in CATEGORIES:
        slug = slugify(title)
        parts[-1] += f"- [{title}](#{slug})\n"
    parts[-1] += "- [Classes](#classes)\n"
    for c in CLASSES:
        parts[-1] += f"  - [`{c}`](#{c.lower()})\n"

    parts.append("""
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

""")
    parts[-1] += render_attributes_section() + "\n"

    for title, names in CATEGORIES:
        slug = slugify(title)
        parts.append(f"\n## {title}\n")
        intro = INTROS.get(title)
        if intro:
            parts[-1] += f"\n{intro}\n"
        parts[-1] += f"\n{render_function_table(names)}\n"

    parts.append("\n## Classes\n")
    parts.append("Stateful objects. All carry a `.dtype` string attribute "
                  "reflecting the arithmetic they were constructed with, and "
                  "a `.reset()` method where meaningful. Process methods "
                  "come in per-sample (`.process(x)`) and block "
                  "(`.process_block(signal)`) variants except on the filter "
                  "classes, which are block-only.\n")
    for c in CLASSES:
        parts.append("\n" + render_class(c) + "\n")

    parts.append("""
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
""")

    out = "".join(parts)
    with open("docs/api_reference.md", "w") as fh:
        fh.write(out)
    print(f"wrote docs/api_reference.md ({len(out)} bytes, "
           f"{out.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
