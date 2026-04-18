"""Multi-channel convenience helpers for image processing.

The C++ bindings in `mpdsp._core` operate on single-channel NumPy 2D
float64 arrays. RGB / RGBA processing is handled at the Python level by
unpacking the channels, running the per-channel pipeline, and re-packing
the results — `apply_per_channel` is that three-liner made explicit.

Pipeline / grid plotting helpers will land in a follow-up PR alongside
the image-processing notebooks.
"""

from typing import Callable, Tuple

import numpy as np


def apply_per_channel(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                      func: Callable[[np.ndarray], np.ndarray]
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single-channel image function across three RGB planes.

    Parameters
    ----------
    r, g, b : numpy.ndarray
        Three 2D float64 arrays of the same shape (red, green, blue planes).
    func : callable
        A function that takes a single 2D NumPy array and returns one.
        Typical callers pass a lambda that calls one of the `mpdsp` image
        processors, e.g. ``lambda plane: mpdsp.gaussian_blur(plane, sigma=1)``.

    Returns
    -------
    tuple of (r_out, g_out, b_out)
        The three independently-processed planes. Shape checking is on the
        caller — `func` is trusted to preserve whatever shape contract it
        documents.

    Raises
    ------
    ValueError
        If the three input planes don't have identical shapes.
    """
    r = np.asarray(r)
    g = np.asarray(g)
    b = np.asarray(b)
    if not (r.shape == g.shape == b.shape):
        raise ValueError(
            f"apply_per_channel: r/g/b must have the same shape; got "
            f"{r.shape}, {g.shape}, {b.shape}")
    return func(r), func(g), func(b)
