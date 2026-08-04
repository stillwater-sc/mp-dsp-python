# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`mpdsp` is the Python integration layer for the C++20 header-only
[mixed-precision-dsp](https://github.com/stillwater-sc/mixed-precision-dsp) library
(`sw::dsp`). It is bindings + pure-Python helpers only — **no DSP algorithms live
here**. If an algorithm is missing, it is either unbound (add a binding) or absent
upstream (fix it in the `mixed-precision-dsp` sibling clone, not here).

## Build & test

The build is driven by scikit-build-core → CMake → nanobind. `pip install -e .` is
the normal path; the raw `cmake -B build` path only produces `_core.so` inside
`build/` and isn't importable without `PYTHONPATH` fiddling.

```bash
pip install -e .                          # configure + build + install (re-run after ANY C++ change)
pip install -e . --no-build-isolation     # faster iteration once deps are present
pytest tests/                             # full suite (~1250 tests); -v/--tb=short come from pyproject addopts
pytest tests/test_filters.py              # one module
pytest tests/test_filters.py::TestButterworthDesign::test_lowpass_returns_filter_with_stages
pytest tests/ -k "posit and not image"    # by expression
MPDSP_REQUIRE_CORE=1 pytest tests/test_version.py   # fail (not skip) if _core didn't load
```

There is no editable-rebuild hook configured — editing `src/*.cpp` and re-running
`pytest` will silently test the *previous* extension. Always `pip install -e .` first.

There is no linter or formatter configured. Match surrounding style: tabs in C++,
4-space PEP 8 in Python, 72–79 column comment wrapping throughout.

Commits and PR titles must be Conventional Commits (`feat(scope): …`, enforced by
`.github/workflows/conventional-commits.yml`). Binding work is tracked by GitHub
issue and commits close them explicitly (`feat(spectrum): … (closes #104)`).

### Peer dependency resolution

CMake resolves `mixed-precision-dsp`, `universal`, and `mtl5` in this order:
sibling clone (`../mixed-precision-dsp`, `../universal`, `../mtl5` — all present in
this working tree and the recommended dev path) → `find_package` (MTL5 only) →
`FetchContent` at a pinned tag (used by CI/cibuildwheel).

Two decoupled knobs in `CMakeLists.txt`, easy to confuse:
- `MPDSP_REQUIRED_*_VERSION` — configure-time **floor** checked only on the
  sibling-clone path; a stale sibling aborts configure with the `git checkout` fix.
- `MPDSP_*_PIN` — git ref used by **FetchContent**. Only the DSP pin is constrained:
  it moves in lockstep with `project(VERSION)` and advances at release time, so
  during a dev cycle the pin legitimately lags the floor.

Escape hatches: `-DMPDSP_REQUIRED_DSP_VERSION=0.5.0` (lower a floor),
`-DMPDSP_DSP_PIN=main` (build against unreleased upstream).

### Versioning

`project(mp-dsp-python VERSION X.Y.Z)` in `CMakeLists.txt` is the **single source of
truth**; scikit-build-core's regex provider reads it, and `[tool.scikit-build.metadata.version].result`
in `pyproject.toml` appends the PEP 440 suffix (`.devN` mid-cycle → bare at release →
`.postN` for Python-only patches). Never hardcode a version elsewhere.
`tests/test_version.py::test_lockstep_prefix` pins `mpdsp.__version__`'s X.Y.Z prefix
to `mpdsp.__dsp_version__` (the upstream C++ version compiled in via `sw/dsp/version.hpp`).

## Architecture

Three layers, bottom to top:

1. **Upstream `sw::dsp` headers** (sibling clone) — all algorithms, templated on
   arithmetic scalar type.
2. **`src/*_bindings.cpp`** — one nanobind translation unit per DSP module. Each
   exposes a single `void bind_<module>(nb::module_&)` that `src/bindings.cpp`
   forward-declares and calls inside `NB_MODULE(_core, m)`. Adding a TU means
   editing three places: the file itself, the forward-decl + call in `bindings.cpp`,
   and the `nanobind_add_module(_core ...)` source list in `CMakeLists.txt`.
3. **`python/mpdsp/`** — `__init__.py` re-exports the entire `_core` surface by
   explicit name (so a new binding is invisible to users until listed there), plus
   pure-Python helpers: `filters.py`, `estimation.py`, `image.py`, `analysis.py`,
   `plotting.py` (matplotlib), `io.py` (CSV sweep loader, works with no `_core`).

`__init__.py` guards the `_core` import with `try/except ImportError` → `HAS_CORE`,
stashing the exception on `__core_import_error__`. This exists so pure-Python paths
work in an unbuilt checkout — but a failed import in an *installed wheel* is a
packaging bug (historically caused by `install(TARGETS _core DESTINATION mpdsp)`
double-nesting to `mpdsp/mpdsp/_core.so`; `DESTINATION .` is correct because
`wheel.install-dir` already prefixes `mpdsp/`).

### The mixed-precision dispatch model

This is the core idea of the whole repo. Python never sees C++ template types: it
passes a `dtype=` **string key**, C++ maps it to a compile-time instantiation. That
keeps the instantiation count at N configs rather than N³ scalar combinations.

- `src/types.hpp` — `ArithConfig` enum, `parse_config()` (string → enum, accepts
  legacy aliases like `"tiny_posit"`, `"double"`, `"float"`), `available_configs()`
  (canonical names only — aliases deliberately omitted), `bits_of()` (sample-path
  bit width, used for precision-vs-cost plots).
- `src/_binding_helpers.hpp` — `make_impl_for_dtype<Impl, Base>(config, ...)` for
  stateful classes with an `Impl<T>` template, and `dispatch_dtype_fn(config, name, lambda)`
  for free function templates. Both `switch` over every enumerator with no `default:`,
  so a new config is a compile error rather than a silent fall-through to `double`.

**Adding an arithmetic config touches five places**: the enum, `parse_config`,
`available_configs`, `bits_of` (all in `types.hpp`), and both switches in
`_binding_helpers.hpp`.

Semantics worth knowing before touching dispatch code:
- **Coefficients are designed in `double` by default.** For the classical IIR
  families this is unconditional — `dtype=` on those selects the processing path
  only, and that's a numerical requirement, not an implementation shortcut. The
  `rbj_*` biquads and the FIR/Remez designers additionally take `coeff_dtype=`,
  which runs the *design* math in `T` and narrows the result back to `double` for
  storage (lossless — a `T`-designed coefficient is `T`-representable, and every
  `T` in the table is narrower than `double`). That knob exists to quantify what
  design-time precision costs; its dual is `IIRFilter.pole_displacement(dtype)`,
  which quantizes an already-designed cascade. Algorithms with no design/runtime
  split (FFT, convolution, Kalman) use the target type for all three.
- **`sensor_8bit` / `sensor_6bit` dispatch to `double` state.** Their 8/6-bit
  character surfaces only through the sample path (`adc` / projection dispatchers),
  by design.
- **`integer<N>` sample scalars need scale–quantize–unscale**, not a plain cast — a
  raw `static_cast` truncates |x|<1 to zero and annihilates audio-range signals. See
  `quantize_sample_in/out` in `src/filter_bindings.cpp` and mirror that pattern.
- Signal generators are intentionally reference-precision (not part of a
  mixed-precision datapath); windows accept `dtype=` for window-precision studies.

### Binding conventions

**Read `src/BINDING_PATTERNS.md` before writing bindings.** Its headline rule: a
property getter returning an `nb::ndarray` built with a capsule (`make_f64_array`,
`make_f64_2d_array`, `make_bool_2d_array`) needs an explicit
`nb::rv_policy::take_ownership` on `def_prop_rw`/`def_prop_ro`. nanobind's default
`reference_internal` throws — **at runtime, on first attribute access**, so it
compiles fine and only a test that actually *reads* the attribute catches it.

Other conventions:
- All array marshalling goes through `src/_binding_helpers.hpp` (`np_f64_ro`,
  `mat_to_numpy`, `numpy_to_mat_fresh`, …). NumPy is `float64` at the boundary in
  both directions regardless of internal arithmetic; `c_contig` on the read-only
  typedefs makes nanobind transparently copy non-contiguous inputs.
- Python-facing names that would shadow NumPy get prefixed — `instrument_mean`,
  `instrument_rms`.
- `docs/api_reference.md` is **generated output — never hand-edit it**. Run
  `python scripts/build_api_ref.py` against an editable install and commit the
  result. New bindings must be added to that script's `CATEGORIES` / `CLASSES`
  tables; the run fails if any public `mpdsp` name is in no table. Prose belongs
  in the script's `INTROS` / `CLASS_INTROS`, per-binding descriptions in the C++
  docstrings. `tests/test_scripts.py::TestBuildApiRef` enforces that the script
  runs and that the committed document matches its output.

### Test conventions

Every test module that needs the extension opens with:

```python
mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)
```

matplotlib-dependent tests use `pytest.importorskip("matplotlib")` inside the test
body. `tests/test_scripts.py` runs the plotting scripts via subprocess (hence seaborn
in CI deps). Test files mirror binding files 1:1.

## Docs map

- `docs/api_reference.md` — generated full API surface with signatures.
- `docs/gap_analysis_2026-08-02.md` — module-by-module bindings coverage vs upstream
  v0.6.0 (~93%) and residual gaps; the place to look before asking "is X bound?".
- `docs/dashboard.md` — Streamlit filter designer (`streamlit run scripts/plot_dashboard.py`).
- `docs/publishing.md` — PyPI / trusted-publishing setup and release steps.
- `src/BINDING_PATTERNS.md` — nanobind gotchas (see above).
