# Binding patterns for mp-dsp-python

Project-specific gotchas and conventions for writing nanobind bindings
in this repository. Read this before adding new bindings, especially
any that expose a NumPy-backed matrix or vector as a Python property.

---

## Return-value policy for ndarray-returning properties

### The rule

**A property getter that returns a freshly-allocated `nb::ndarray`
owned by its own capsule needs an explicit
`nb::rv_policy::take_ownership` override on the `def_prop_rw` /
`def_prop_ro` call.**

### Why

nanobind's default return-value policy for `def_prop_rw` /
`def_prop_ro` is `rv_policy::reference_internal`. For most property
types that's correct: the getter returns a reference to a field
already owned by the parent C++ object, and nanobind keeps a
keep-alive tie to the parent so Python sees a live view.

An `nb::ndarray` built with its own capsule (see the
`make_f64_array` / `make_f64_2d_array` helpers in
`_binding_helpers.hpp`) is already **owned** — the capsule owns the
buffer. `reference_internal` tries to attach a keep-alive anyway
and nanobind's runtime refuses.

### The symptom

Failure is **runtime-only**, raised the first time the property is
accessed from Python:

```
RuntimeError: nanobind::detail::ndarray_export(): reference_internal
policy cannot be applied (ndarray already has an owner)
```

Because the build succeeds and the error fires only on attribute
access, property getters without the override slip past anything
short of a unit test that actually reads the attribute.

### The fix

Add `nb::rv_policy::take_ownership` as the second-to-last argument
to `def_prop_rw` / `def_prop_ro`:

```cpp
// Every getter builds a fresh NumPy array with its own capsule, so
// the default reference_internal policy doesn't apply — the returned
// ndarray already has an owner. Use take_ownership to hand the buffer
// off to Python cleanly.
.def_prop_rw("F", &PyKalmanFilter::get_F, &PyKalmanFilter::set_F,
             nb::rv_policy::take_ownership,
             "State transition matrix (state_dim x state_dim).")
```

(Live example: `src/estimation_bindings.cpp`, KalmanFilter matrices
`F`, `H`, `Q`, `R`, `P`, `B`, and `state`, plus the `weights`
property on all three adaptive filters.)

### When it does NOT apply

Don't blanket-add `take_ownership` everywhere. The override is only
needed for properties that build an owned `nb::ndarray` on every
call. Other property shapes use the default policy:

| Return type | Policy |
|-------------|--------|
| `std::string`, `int`, `float`, `bool`, `size_t` (scalars) | Default (copy) |
| `std::vector<int>`, `std::vector<double>` (STL containers — nanobind copies) | Default |
| `std::complex<double>` returned by value | Default |
| A raw `nb::ndarray` built with its own capsule | **`take_ownership`** |

Properties that return scalars or simple STL containers — e.g.
`num_taps`, `state_dim`, `dtype` on the conditioning and estimation
classes — deliberately don't include the override.

### Decision tree

When adding a new property:

1. **Does the getter return an `nb::ndarray`?** No → use default.
2. **Does the ndarray carry its own capsule (built with
   `make_f64_array` / `make_f64_2d_array` / `make_bool_2d_array`, or
   a manually-owned `nb::capsule`)?** No → the getter probably
   returns a reference-into-the-C++-object, which may have other
   lifetime issues; think carefully. Yes → **add
   `nb::rv_policy::take_ownership`**.

### Testing

Any property binding should have at least one pytest case that reads
the attribute back in Python — the rv_policy failure is runtime-only
and a build-only change won't catch it:

```python
def test_kalman_matrices_are_readable():
    kf = mpdsp.KalmanFilter(state_dim=2, meas_dim=1)
    _ = kf.F       # these reads trigger the ndarray export path
    _ = kf.H
    _ = kf.state
```

This pattern already lives in `tests/test_estimation.py`.

---

## See also

- `src/_binding_helpers.hpp` — the `make_*_array` helpers that build
  the capsule-owned `nb::ndarray` instances, and the
  `make_impl_for_dtype` / `dispatch_dtype_fn` utilities for the
  type-erased arithmetic dispatch pattern.
- `src/estimation_bindings.cpp` — reference implementation for
  stateful classes with ndarray-property surface area.
- [nanobind docs on return-value policies](https://nanobind.readthedocs.io/en/latest/functions.html#return-value-policies)
