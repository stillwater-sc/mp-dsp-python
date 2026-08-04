"""Audio dynamics processor: 3-band EQ -> compressor -> peak limiter.

Runs the whole chain at each arithmetic configuration and writes a listenable
WAV per dtype, so precision loss can be *heard* rather than only read off a
SQNR number.

    python demos/01_audio_dynamics/design.py
    python demos/01_audio_dynamics/design.py --input my_clip.wav --all-dtypes

Signal flow, per channel:

    in -> lowshelf(120 Hz) -> peaking(1 kHz) -> highshelf(8 kHz)
       -> compressor -> peak limiter -> out

What runs at the selected dtype, and what does not
--------------------------------------------------
The three EQ biquads (`IIRFilter.process(dtype=)`), the compressor, and the
limiter's envelope follower all run their arithmetic at the chosen dtype.

Two steps do not, because no bound primitive covers them:

* the peaking band's parallel mix, `x + (g - 1) * bandpass(x)` — one scale
  and one add per sample;
* the limiter's final gain multiply, `x * gain` — one multiply per sample.

Both are float64 NumPy. The filters and followers feeding them *are*
quantized, so the measurement is close, but a hardware port would run these
at the target type too and the numbers here are correspondingly optimistic
by a small margin. Called out rather than buried because the point of the
demo is honesty about what precision costs.

Coefficients are designed in double throughout, which is the library's
default and the right choice for a deployed EQ. `--coeff-dtype` sweeps
design-time precision instead, if you want to see that axis.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mpdsp

from simulate import SAMPLE_RATE, load_wav, synthesize, write_wav

# The seven configurations the demo sweeps by default: the reference, the
# three float-ish targets, and three posit/custom points. `--all-dtypes`
# widens this to every entry of mpdsp.available_dtypes() (18 as of 0.8.0),
# which includes the sensor and FPGA fixed-point paths.
DEFAULT_DTYPES = [
    "reference",
    "gpu_baseline",
    "ml_hw",
    "cf24",
    "half",
    "posit_full",
    "posit_8_2",
]


@dataclass
class ChainConfig:
    """Every knob the chain and the emitted C header share."""

    sample_rate: float = SAMPLE_RATE

    # Three-band EQ. Shelves take a slope, the mid band a bandwidth in octaves.
    low_shelf_hz: float = 120.0
    low_shelf_gain_db: float = 4.5
    low_shelf_slope: float = 0.9
    mid_hz: float = 1000.0
    mid_bandwidth_oct: float = 1.4
    mid_gain_db: float = -5.0
    high_shelf_hz: float = 8000.0
    high_shelf_gain_db: float = 3.0
    high_shelf_slope: float = 0.9

    # Compressor.
    threshold_db: float = -18.0
    ratio: float = 4.0
    attack_ms: float = 8.0
    release_ms: float = 120.0
    makeup_db: float = 10.0
    knee_db: float = 6.0

    # Peak limiter (envelope follower + gain computer).
    # Makeup gain deliberately pushes the compressor output past the
    # ceiling so the limiter has real work to do — the standard mastering
    # topology, and what makes the limiter's precision behaviour visible.
    ceiling: float = 0.95
    limiter_attack_ms: float = 0.8
    limiter_release_ms: float = 60.0
    # Lookahead must span several follower time constants, not one: at
    # 0.8 ms of lookahead the gain smoother reaches only ~63% of its target
    # inside the window and a 1.83 peak survived a 0.95 ceiling. Measured
    # output peak against this material: 1.24 at 0.8 ms, 1.02 at 2 ms,
    # 0.99 at 5 ms. Past ~8 ms it degrades again as the held reduction
    # starts fighting the release.
    limiter_lookahead_ms: float = 5.0

    # Coefficient design precision. Kept at the library default; see the
    # module docstring.
    coeff_dtype: str = "reference"


@dataclass
class ChainResult:
    dtype: str
    audio: np.ndarray                     # (channels, N)
    gain_reduction_db: np.ndarray         # (N,) from channel 0
    sqnr_db: float | None = None          # vs the reference run
    peak: float = 0.0
    clipped_samples: int = 0
    stage_names: list[str] = field(default_factory=list)


def design_eq(config: ChainConfig) -> dict:
    """The three EQ biquads.

    RBJ rather than a classical family: shelves are exactly what an audio EQ
    wants, and each band is a single biquad, which keeps the emitted C header
    trivial to port.

    The mid band needs a *peaking* response — a bell that boosts or cuts
    around its center and leaves the rest alone. Upstream `sw::dsp::rbj` has
    no Peaking class (see mp-dsp-python#94), so it is synthesized the
    classical way, from a bandpass in a parallel path:

        y = x + (g - 1) * bandpass(x)

    At the center frequency the bandpass has unity gain, so y = g*x; far from
    it the bandpass output vanishes and y = x. Putting the bandpass in series
    instead would be a different filter entirely — it would strip everything
    outside the mid band, which is what an EQ must not do.
    """
    return {
        "low": mpdsp.rbj_lowshelf(
            sample_rate=config.sample_rate, cutoff=config.low_shelf_hz,
            gain_db=config.low_shelf_gain_db, slope=config.low_shelf_slope,
            coeff_dtype=config.coeff_dtype),
        "mid": mpdsp.rbj_bandpass(
            sample_rate=config.sample_rate, center_freq=config.mid_hz,
            bandwidth=config.mid_bandwidth_oct,
            coeff_dtype=config.coeff_dtype),
        "high": mpdsp.rbj_highshelf(
            sample_rate=config.sample_rate, cutoff=config.high_shelf_hz,
            gain_db=config.high_shelf_gain_db, slope=config.high_shelf_slope,
            coeff_dtype=config.coeff_dtype),
    }


def apply_eq(x: np.ndarray, eq: dict, config: ChainConfig,
             dtype: str) -> np.ndarray:
    """Low shelf -> peaking mid -> high shelf, all filtering at `dtype`."""
    x = eq["low"].process(np.ascontiguousarray(x), dtype=dtype)

    mid_gain = 10.0 ** (config.mid_gain_db / 20.0)
    band = eq["mid"].process(np.ascontiguousarray(x), dtype=dtype)
    x = x + (mid_gain - 1.0) * band

    return eq["high"].process(np.ascontiguousarray(x), dtype=dtype)


def _limit(signal: np.ndarray, config: ChainConfig, dtype: str
           ) -> tuple[np.ndarray, np.ndarray]:
    """Feed-forward peak limiter with lookahead.

    Returns (limited, gain_reduction_db).

    PeakEnvelope tracks the signal's peak at `dtype`; the gain computer asks
    for whatever attenuation keeps that envelope under the ceiling.

    The structure is the textbook brickwall one, and each piece is load
    bearing:

    1. **Sliding maximum** over the lookahead window sets the gain target.
       Feeding the gain computer an envelope follower instead does not hold
       a ceiling: a follower with a finite attack reaches a transient's peak
       only after it has passed. Measured on this material, a 0.8 ms
       follower read 0.835 where the signal was at 1.832 — barely half —
       so `ceiling / envelope` asked for far too little gain reduction
       exactly where it mattered, and a 1.83 peak survived a 0.95 ceiling.
    2. **PeakEnvelope smooths the gain reduction**, which is what a peak
       follower is genuinely right for: fast attack as reduction deepens,
       slow release as it recovers. Because step 1 already holds the target
       flat across the whole lookahead window, the follower has time to
       reach it. Applying the raw target instead would step the gain
       discontinuously and audibly distort.
    3. **The audio is delayed** by the lookahead, so the reduction is fully
       in place by the time the transient arrives. Real limiters pay this
       same price in latency.

    The result is a near-brickwall, not a hard one: because step 2 smooths
    with an exponential follower, the ceiling is held to within a fraction
    of a dB rather than absolutely. At the shipped settings the reference
    run peaks at 0.99 against a 0.95 ceiling — under full scale, so nothing
    clips — while the coarse dtypes overshoot badly, which is the point.

    The sliding max and the final multiply are float64 NumPy — no bound
    primitive covers either. See the module docstring.
    """
    lookahead = max(1, int(round(config.limiter_lookahead_ms
                                 * config.sample_rate / 1000.0)))

    # Sliding max over [n, n + lookahead]. `lookahead` is small (~38 samples
    # at 48 kHz), so folding that many shifted maxima is cheaper and clearer
    # than a monotonic-deque implementation.
    magnitude = np.abs(signal)
    padded = np.concatenate([magnitude, np.zeros(lookahead)])
    window_peak = magnitude.copy()
    for k in range(1, lookahead + 1):
        window_peak = np.maximum(window_peak, padded[k:k + len(magnitude)])

    target_gain = np.minimum(1.0, config.ceiling
                             / np.maximum(window_peak, 1e-12))

    # Smooth the *reduction* (1 - gain), which rises and falls like a peak
    # signal, so PeakEnvelope's attack/release apply in the right direction.
    follower = mpdsp.PeakEnvelope(
        sample_rate=config.sample_rate,
        attack_ms=config.limiter_attack_ms,
        release_ms=config.limiter_release_ms,
        dtype=dtype)
    reduction = follower.process_block(
        np.ascontiguousarray(1.0 - target_gain))
    gain = np.clip(1.0 - reduction, 0.0, 1.0)

    delayed = np.concatenate([np.zeros(lookahead), signal])[:len(signal)]
    return delayed * gain, 20.0 * np.log10(np.maximum(gain, 1e-12))


def run_chain(clip: np.ndarray, config: ChainConfig, dtype: str
              ) -> ChainResult:
    """Run one channel-major clip through the full chain at `dtype`."""
    eq = design_eq(config)
    channels = []
    gain_traces = []

    for channel in np.atleast_2d(clip):
        x = apply_eq(np.ascontiguousarray(channel, dtype=np.float64),
                     eq, config, dtype)

        compressor = mpdsp.Compressor(
            sample_rate=config.sample_rate,
            threshold_db=config.threshold_db,
            ratio=config.ratio,
            attack_ms=config.attack_ms,
            release_ms=config.release_ms,
            makeup_db=config.makeup_db,
            knee_db=config.knee_db,
            dtype=dtype)
        x = compressor.process_block(np.ascontiguousarray(x))

        x, gain_db = _limit(np.ascontiguousarray(x), config, dtype)
        channels.append(x)
        gain_traces.append(gain_db)

    audio = np.vstack(channels)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return ChainResult(
        dtype=dtype,
        audio=audio,
        gain_reduction_db=gain_traces[0],
        peak=peak,
        clipped_samples=int(np.count_nonzero(np.abs(audio) > 1.0)),
        stage_names=["lowshelf", "peaking", "highshelf",
                     "compressor", "limiter"],
    )


def run_all(clip: np.ndarray, config: ChainConfig, dtypes: list[str]
            ) -> list[ChainResult]:
    """Run every dtype and score each against the reference run.

    `reference` is forced to the front so it exists before anything is scored
    against it.
    """
    ordered = (["reference"] + [d for d in dtypes if d != "reference"]
               if "reference" in dtypes else list(dtypes))

    results = []
    reference = None
    for dtype in ordered:
        result = run_chain(clip, config, dtype)
        if dtype == "reference":
            reference = result
        elif reference is not None:
            # Score channel 0; the two channels are decorrelated but carry
            # the same programme, so one is representative.
            result.sqnr_db = mpdsp.sqnr_db(
                np.ascontiguousarray(reference.audio[0]),
                np.ascontiguousarray(result.audio[0]))
        results.append(result)
    return results


def write_summary_csv(path: Path, results: list[ChainResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dtype", "sample_bits", "sqnr_db", "peak",
                         "clipped_samples", "max_gain_reduction_db"])
        for r in results:
            writer.writerow([
                r.dtype,
                mpdsp.bits_of(r.dtype),
                "" if r.sqnr_db is None else f"{r.sqnr_db:.2f}",
                f"{r.peak:.4f}",
                r.clipped_samples,
                f"{float(np.min(r.gain_reduction_db)):.2f}",
            ])


def plot_summary(path: Path, clip: np.ndarray, results: list[ChainResult],
                 config: ChainConfig) -> bool:
    """Write the summary PNG. Returns False if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    reference = next((r for r in results if r.dtype == "reference"), results[0])
    scored = [r for r in results if r.sqnr_db is not None]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Audio dynamics chain — EQ → compressor → limiter", y=0.98)

    # Input vs reference output spectrogram.
    for ax, (data, title) in zip(
            axes[0],
            [(clip[0], "Input (ch 0)"),
             (reference.audio[0], "Output at `reference` (ch 0)")]):
        times, freqs, mag_db = mpdsp.spectrogram(
            np.ascontiguousarray(data), config.sample_rate,
            window_size=1024, hop_size=256)
        ax.pcolormesh(times, freqs, np.asarray(mag_db).T, shading="auto",
                      vmin=-120, vmax=0)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Hz")
        ax.set_ylim(0, config.sample_rate / 2)

    # Gain reduction traces.
    ax = axes[1][0]
    t = np.arange(len(reference.gain_reduction_db)) / config.sample_rate
    for r in results:
        ax.plot(t, r.gain_reduction_db, linewidth=0.9, label=r.dtype)
    ax.set_title("Limiter gain reduction (ch 0)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dB")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # SQNR vs sample bit width.
    ax = axes[1][1]
    if scored:
        bits = [mpdsp.bits_of(r.dtype) for r in scored]
        sqnr = [r.sqnr_db for r in scored]
        ax.scatter(bits, sqnr, s=45)
        for r, b, s in zip(scored, bits, sqnr):
            ax.annotate(r.dtype, (b, s), fontsize=7,
                        textcoords="offset points", xytext=(5, 3))
        ax.set_xlabel("sample-path bits")
        ax.set_ylabel("SQNR vs reference (dB)")
    ax.set_title("Precision vs. chain output SQNR")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the audio dynamics chain across arithmetic types.")
    parser.add_argument("--input", help="WAV to process (default: synthesize)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Synthesized clip length in seconds")
    parser.add_argument("--outdir", default=None,
                        help="Artifact directory (default: ./artifacts "
                             "next to this script)")
    parser.add_argument("--dtypes", nargs="+", default=None,
                        help="Explicit dtype list")
    parser.add_argument("--all-dtypes", action="store_true",
                        help="Sweep every mpdsp.available_dtypes() entry")
    parser.add_argument("--coeff-dtype", default="reference",
                        help="Coefficient design precision (default: "
                             "reference — see the module docstring)")
    parser.add_argument("--no-wav", action="store_true",
                        help="Skip writing per-dtype WAVs")
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(__file__).parent / "artifacts"

    if args.input:
        clip, sample_rate = load_wav(args.input)
        print(f"loaded {args.input}: {clip.shape[1]} frames x "
              f"{clip.shape[0]} ch @ {sample_rate:.0f} Hz")
    else:
        sample_rate = SAMPLE_RATE
        clip = synthesize(args.duration, sample_rate)
        print(f"synthesized {clip.shape[1]} frames x {clip.shape[0]} ch "
              f"@ {sample_rate:.0f} Hz")

    if args.all_dtypes:
        dtypes = list(mpdsp.available_dtypes())
    elif args.dtypes:
        dtypes = list(args.dtypes)
    else:
        dtypes = list(DEFAULT_DTYPES)

    config = ChainConfig(sample_rate=sample_rate, coeff_dtype=args.coeff_dtype)
    results = run_all(clip, config, dtypes)

    write_wav(outdir / "input.wav", clip, sample_rate)
    if not args.no_wav:
        for r in results:
            write_wav(outdir / f"output_{r.dtype}.wav", r.audio, sample_rate)

    write_summary_csv(outdir / "summary.csv", results)
    plotted = plot_summary(outdir / "summary.png", clip, results, config)

    print(f"\n{'dtype':<14}{'bits':>5}{'SQNR dB':>10}{'peak':>8}"
          f"{'clipped':>9}{'max GR dB':>11}")
    print("-" * 57)
    for r in results:
        sqnr = "  (ref)" if r.sqnr_db is None else f"{r.sqnr_db:9.2f}"
        print(f"{r.dtype:<14}{mpdsp.bits_of(r.dtype):>5}{sqnr:>10}"
              f"{r.peak:>8.4f}{r.clipped_samples:>9}"
              f"{float(np.min(r.gain_reduction_db)):>11.2f}")

    print(f"\nartifacts -> {outdir}")
    if not plotted:
        print("  (summary.png skipped: matplotlib not installed — "
              "pip install 'mpdsp[plot]')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
