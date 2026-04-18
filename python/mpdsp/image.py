"""Multi-channel convenience helpers and plotting for image processing.

The C++ bindings in `mpdsp._core` operate on single-channel NumPy 2D
float64 arrays. RGB / RGBA processing is handled at the Python level by
unpacking the channels, running the per-channel pipeline, and re-packing
the results — `apply_per_channel` is that three-liner made explicit.

Three matplotlib helpers are also provided for the notebook walkthroughs:

- `plot_image(img, title, ax, cmap, ...)` — single grayscale image with colorbar
- `plot_image_grid(images, titles, ncols, ...)` — side-by-side comparison
- `plot_pipeline(stages, titles, ...)` — sequential pipeline stages
"""

from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_matplotlib() -> None:
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for mpdsp.image plotting helpers. "
            "Install with: pip install matplotlib")


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


# ---------------------------------------------------------------------------
# Matplotlib plotting helpers.
# ---------------------------------------------------------------------------


def plot_image(img: np.ndarray, title: str = "",
               ax=None, cmap: str = "gray",
               vmin: Optional[float] = None, vmax: Optional[float] = None,
               colorbar: bool = True, figsize=(6, 5)):
    """Display a 2D grayscale image with an optional colorbar.

    Parameters
    ----------
    img : numpy.ndarray
        2D image array.
    title : str
        Title placed above the axes.
    ax : matplotlib.axes.Axes, optional
        Plot onto this axes. Creates a new figure if omitted.
    cmap : str
        matplotlib colormap name (default "gray").
    vmin, vmax : float, optional
        Display-range limits. `None` auto-scales.
    colorbar : bool
        Attach a colorbar (default True).
    figsize : tuple
        Figure size, used only when `ax` is None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_image_grid(images: Sequence[np.ndarray],
                    titles: Optional[Sequence[str]] = None,
                    ncols: int = 4,
                    cmap: str = "gray",
                    figsize: Optional[Tuple[float, float]] = None,
                    colorbar: bool = False,
                    suptitle: Optional[str] = None):
    """Display a sequence of images in a grid layout.

    Useful for comparing a signal across arithmetic dtypes, kernel sizes,
    or pipeline variants side by side.

    Parameters
    ----------
    images : sequence of 2D arrays
        Images to display.
    titles : sequence of str, optional
        One title per image; uses index "[i]" if omitted.
    ncols : int
        Number of columns in the grid.
    cmap : str
        Shared colormap.
    figsize : tuple, optional
        Figure size; auto-computed from grid shape if omitted.
    colorbar : bool
        Attach a colorbar to each image (default False — usually too busy
        in a grid).
    suptitle : str, optional
        Figure-level title placed above the grid.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    n = len(images)
    if n == 0:
        raise ValueError("plot_image_grid: images must not be empty")
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (3.0 * ncols, 3.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    # Normalize axes to a flat list regardless of subplot layout
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.asarray(axes).ravel().tolist()
    for i, img in enumerate(images):
        title = (titles[i] if titles is not None and i < len(titles)
                 else f"[{i}]")
        plot_image(img, title=title, ax=axes_flat[i], cmap=cmap,
                   colorbar=colorbar)
    # Blank out unused axes in the last row.
    for j in range(n, nrows * ncols):
        axes_flat[j].axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_pipeline(stages: Sequence[np.ndarray],
                  titles: Optional[Sequence[str]] = None,
                  cmap: str = "gray",
                  figsize: Optional[Tuple[float, float]] = None,
                  suptitle: Optional[str] = None):
    """Display a pipeline's successive stages in a single row.

    Convenience wrapper for `plot_image_grid(..., ncols=len(stages))`.
    Typical use: show raw → blurred → gradient → threshold → morphed in
    one horizontal strip with labeled stages.

    Parameters
    ----------
    stages : sequence of 2D arrays
        Each array is one output of the pipeline.
    titles : sequence of str, optional
        Labels for each stage (default falls back to "stage {i}").
    cmap : str
        Colormap name.
    figsize : tuple, optional
        Figure size; auto-computed as (3*N, 3.5) if omitted.
    suptitle : str, optional
        Top-level title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(stages)
    if n == 0:
        raise ValueError("plot_pipeline: stages must not be empty")
    if titles is None:
        titles = [f"stage {i}" for i in range(n)]
    if figsize is None:
        figsize = (3.0 * n, 3.5)
    return plot_image_grid(stages, titles=titles, ncols=n, cmap=cmap,
                           figsize=figsize, suptitle=suptitle)
