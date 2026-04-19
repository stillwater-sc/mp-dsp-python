"""Pure-Python analysis helpers layered on top of ``mpdsp._core``.

These helpers give the free-function analysis surface that issue #8's
original scope listed — but for the primitives that are just a few
lines of math over the already-bound ``IIRFilter`` methods, a Python
implementation is lighter than a C++ binding and stays equally in
lockstep with upstream ``sw::dsp`` (the math is the math).

Genuinely numerical primitives (``coefficient_sensitivity``,
``biquad_condition_number``) still need proper C++ bindings — those
are tracked in the 0.5.0 sweep at #40.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def biquad_poles(b0: float, b1: float, b2: float,
                 a1: float, a2: float) -> list[complex]:
    """Two poles of a single biquad section.

    The biquad transfer function is::

        H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²)

    The poles are the roots of ``z² + a1 z + a2 = 0``. The numerator
    coefficients are accepted for signature symmetry with the C++
    upstream (and so callers can unpack the 5-tuple returned by
    ``IIRFilter.coefficients()`` directly) but don't affect the
    result — the poles depend only on the denominator.
    """
    del b0, b1, b2  # unused; accepted for signature symmetry
    return [complex(r) for r in np.roots([1.0, a1, a2])]


def max_pole_radius(filt) -> float:
    """Largest ``|pole|`` in the filter's z-plane.

    Stability requires ``max_pole_radius < 1``. Returns 0.0 for
    degenerate filters with no poles (e.g. an FIR filter, which has
    only zeros) so callers can safely chain without ``None``-guards.
    """
    poles = filt.poles()
    if not poles:
        return 0.0
    return float(max(abs(p) for p in poles))


def is_stable(filt, tol: float = 0.0) -> bool:
    """True iff all poles are strictly inside the unit circle.

    Filters returned by the family constructors (``butterworth_*``,
    ``chebyshev1_*``, etc.) are stable by construction, so this helper
    is primarily for filters whose coefficients have been mutated —
    e.g., after a quantization round-trip, or a filter loaded from
    foreign coefficient data.

    ``tol`` tightens the boundary inward. With ``tol=0`` (the default)
    a pole exactly on the unit circle is rejected; with a small positive
    tolerance you can insist on a numerical safety margin — useful when
    deciding whether to deploy under reduced-precision arithmetic, where
    ``max|pole|`` can drift outward by a quantization-dependent amount.
    """
    return max_pole_radius(filt) < (1.0 - tol)


def cascade_condition_number(filt, num_freqs: int = 256) -> float:
    """Condition number of an entire IIR cascade.

    Free-function companion to the per-biquad ``biquad_condition_number``
    (bound in C++). Equivalent to ``filt.condition_number(num_freqs)`` —
    the upstream ``sw::dsp::cascade_condition_number`` is exactly what the
    ``IIRFilter.condition_number`` method already wraps. This Python
    wrapper exists to surface the free-function spelling (useful when
    writing design-time sweeps that accept a filter plus a custom
    ``num_freqs`` as separate arguments) without duplicating the C++
    side.

    Parameters
    ----------
    filt : mpdsp.IIRFilter
        A designed IIR filter.
    num_freqs : int
        Number of frequency points sampled on [0, 0.5]. Larger values give
        a more accurate condition-number estimate at proportional cost.
    """
    return float(filt.condition_number(num_freqs))
