"""Visualization helpers for the state-estimation bindings.

Built on top of the C++ bindings in `mpdsp._core` (KalmanFilter,
LMSFilter, NLMSFilter, RLSFilter). Two plotting helpers:

- `plot_kalman_tracking`: state-estimate vs truth with 1-sigma confidence
  bands drawn from the filter's covariance P.
- `plot_adaptive_convergence`: weight trajectories over time for one or
  more adaptive filters, optionally with a reference "true" weight set
  drawn as dashed lines.

Both are deliberately non-interactive — they return the figure so the
caller can customize or save it. The notebooks in `notebooks/` show
end-to-end examples.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_matplotlib():
    if not HAS_MPL:
        raise ImportError("matplotlib is required for estimation plotting. "
                          "Install with: pip install matplotlib")


def plot_kalman_tracking(truth, measurements, estimates, covariances=None,
                         dt=1.0, title="Kalman filter tracking",
                         figsize=(10, 4)):
    """Plot a Kalman filter's state estimate against the true trajectory.

    Parameters
    ----------
    truth : array_like, shape (T,) or (T, state_dim)
        True state values. If 2D, only the first column is plotted
        (typically position in a [position, velocity] setup).
    measurements : array_like, shape (T,) or (T, meas_dim)
        Observation time series. If 2D, only the first column is plotted.
    estimates : array_like, shape (T, state_dim)
        Kalman state estimates at each step.
    covariances : array_like, shape (T, state_dim, state_dim), optional
        Per-step covariance matrices. When provided, a 1-sigma band around
        the first state component is drawn.
    dt : float
        Time step used for the x axis.
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    truth = np.atleast_2d(np.asarray(truth))
    if truth.shape[0] == 1:
        truth = truth.T  # treat 1D input as (T, 1)
    measurements = np.atleast_2d(np.asarray(measurements))
    if measurements.shape[0] == 1:
        measurements = measurements.T
    estimates = np.asarray(estimates)
    if estimates.ndim == 1:
        estimates = estimates[:, None]

    n_steps = estimates.shape[0]
    t = np.arange(n_steps) * dt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, measurements[:n_steps, 0], "o", color="0.7",
            markersize=3, label="measurements", zorder=1)
    ax.plot(t, truth[:n_steps, 0], "--", color="C2",
            linewidth=1.5, label="truth", zorder=2)
    ax.plot(t, estimates[:, 0], "-", color="C0",
            linewidth=1.5, label="estimate", zorder=3)

    if covariances is not None:
        covariances = np.asarray(covariances)
        if covariances.shape != (n_steps, estimates.shape[1], estimates.shape[1]):
            raise ValueError(
                "covariances must have shape (T, state_dim, state_dim); "
                f"got {covariances.shape}")
        sigma = np.sqrt(np.maximum(covariances[:, 0, 0], 0.0))
        ax.fill_between(t, estimates[:, 0] - sigma, estimates[:, 0] + sigma,
                        alpha=0.2, color="C0", label="±1σ", zorder=0)

    ax.set_xlabel("Time")
    ax.set_ylabel("State[0]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_adaptive_convergence(weight_traces, true_weights=None,
                              labels=None, dt=1,
                              title="Adaptive-filter weight convergence",
                              figsize=(11, 5)):
    """Plot weight trajectories of one or more adaptive filters over time.

    Parameters
    ----------
    weight_traces : array_like or list of array_like
        Either a single (T, num_taps) array for one filter, or a list of
        such arrays (one per filter) for overlay.
    true_weights : array_like, shape (num_taps,), optional
        If provided, drawn as dashed horizontal lines — the asymptote that
        successfully-converging filters should approach.
    labels : list of str, optional
        One label per entry in `weight_traces`. Defaults to "filter 0",
        "filter 1", ...
    dt : float or int
        Sample step for the x axis.
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()

    # Accept either a single (T, N) trace or a list of them.
    if isinstance(weight_traces, np.ndarray) and weight_traces.ndim == 2:
        traces = [weight_traces]
    elif isinstance(weight_traces, (list, tuple)):
        traces = [np.asarray(t) for t in weight_traces]
    else:
        raise ValueError("weight_traces must be a 2D array or a list of 2D arrays")

    if not traces:
        raise ValueError("weight_traces must not be empty")
    num_taps = traces[0].shape[1]
    for i, tr in enumerate(traces):
        if tr.ndim != 2 or tr.shape[1] != num_taps:
            raise ValueError(
                f"all weight_traces must have shape (T, {num_taps}); "
                f"entry {i} has shape {tr.shape}")

    if labels is None:
        labels = [f"filter {i}" for i in range(len(traces))]
    elif len(labels) != len(traces):
        raise ValueError(
            f"labels ({len(labels)}) must match weight_traces ({len(traces)})")

    fig, ax = plt.subplots(figsize=figsize)

    # One color per filter; one linestyle per tap so overlays stay readable.
    linestyles = ["-", "--", "-.", ":"]
    for fi, (trace, label) in enumerate(zip(traces, labels)):
        t = np.arange(trace.shape[0]) * dt
        color = f"C{fi}"
        for k in range(num_taps):
            ls = linestyles[k % len(linestyles)]
            legend_label = f"{label} w[{k}]" if len(traces) > 1 else f"w[{k}]"
            ax.plot(t, trace[:, k], color=color, linestyle=ls,
                    linewidth=1.2, label=legend_label)

    if true_weights is not None:
        true_weights = np.asarray(true_weights)
        if true_weights.shape != (num_taps,):
            raise ValueError(
                f"true_weights must have shape ({num_taps},); "
                f"got {true_weights.shape}")
        for k, w in enumerate(true_weights):
            ax.axhline(float(w), color="0.5", linestyle="--", linewidth=0.8,
                       alpha=0.7)

    ax.set_xlabel("Sample")
    ax.set_ylabel("Weight value")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best", fontsize="small", ncol=max(1, len(traces)))
    fig.tight_layout()
    return fig


def collect_adaptive_weights(adaptive_filter, inputs, desireds,
                             record_every=1):
    """Run an adaptive filter over (inputs, desireds) and record the tap
    weights every `record_every` samples.

    Convenience wrapper for building a weight trace suitable for
    `plot_adaptive_convergence`. Returns a NumPy array of shape
    (num_records, num_taps).
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    desireds = np.asarray(desireds, dtype=np.float64)
    if inputs.shape != desireds.shape:
        raise ValueError("inputs and desireds must have the same shape")
    if inputs.ndim != 1:
        raise ValueError("inputs must be 1D")

    num_taps = adaptive_filter.num_taps
    records = []
    for i, (x, d) in enumerate(zip(inputs, desireds)):
        adaptive_filter.process(float(x), float(d))
        if (i + 1) % record_every == 0:
            records.append(np.asarray(adaptive_filter.weights))
    # Always include the final state.
    if not records or (len(inputs) % record_every) != 0:
        records.append(np.asarray(adaptive_filter.weights))
    return np.asarray(records)
