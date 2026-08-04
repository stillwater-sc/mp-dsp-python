"""Test-signal sourcing for the audio dynamics demo.

Either synthesizes a stereo programme clip or loads a WAV from disk. The
synthesized clip is deliberately dynamic — sustained tonal material
punctuated by transient hits — because a compressor and limiter only show
their character on material whose level actually moves.

Run standalone to write a reference clip:

    python demos/01_audio_dynamics/simulate.py --out artifacts/input.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mpdsp

SAMPLE_RATE = 48000.0

# Fundamental plus a few harmonics, at decreasing weight. Enough spectral
# content that the three EQ bands each have something to act on.
_PARTIALS = [
    (110.0, 0.50),   # low — the lowshelf band
    (220.0, 0.28),
    (440.0, 0.20),
    (1000.0, 0.16),  # mid — the peaking band
    (2200.0, 0.10),
    (6500.0, 0.07),  # high — the highshelf band
    (9000.0, 0.05),
]


def _tone_bed(num_samples: int, sample_rate: float, detune: float = 0.0
              ) -> np.ndarray:
    """Sum the harmonic series into one buffer.

    `detune` shifts every partial by a fixed ratio so the two stereo
    channels are decorrelated without being independent noise.
    """
    out = np.zeros(num_samples, dtype=np.float64)
    for freq, amp in _PARTIALS:
        f = freq * (1.0 + detune)
        if f >= sample_rate / 2.0:
            continue
        out += mpdsp.sine(num_samples, frequency=f, sample_rate=sample_rate,
                          amplitude=amp)
    return out


def _dynamics_envelope(num_samples: int, sample_rate: float,
                       seed: int = 7) -> np.ndarray:
    """A level contour with quiet passages and hard transient hits.

    Shaped rather than random: a slow swell sets the bed level, and four
    fast-attack / slow-decay hits ride on top. The hits are what drive the
    compressor into gain reduction and what the limiter has to catch.
    """
    t = np.arange(num_samples) / sample_rate
    duration = num_samples / sample_rate

    # Slow swell between roughly -18 and -6 dBFS.
    swell = 0.18 + 0.30 * (0.5 - 0.5 * np.cos(2.0 * np.pi * t / duration))

    rng = np.random.default_rng(seed)
    env = swell.copy()
    for hit_time in np.linspace(0.12, 0.88, 4) * duration:
        start = int(hit_time * sample_rate)
        if start >= num_samples:
            continue
        decay_len = min(int(0.28 * sample_rate), num_samples - start)
        decay = np.exp(-np.arange(decay_len) / (0.055 * sample_rate))
        gain = 0.85 + 0.25 * rng.random()
        env[start:start + decay_len] += gain * decay
    return env


def synthesize(duration_s: float = 3.0, sample_rate: float = SAMPLE_RATE,
               seed: int = 7) -> np.ndarray:
    """Return a (2, N) float64 stereo clip in roughly [-1, 1].

    Peaks are left a little under full scale so the chain's own makeup gain
    has somewhere to go before the limiter engages.
    """
    num_samples = int(round(duration_s * sample_rate))
    env = _dynamics_envelope(num_samples, sample_rate, seed=seed)

    channels = []
    for index, detune in enumerate((0.0, 0.0015)):
        bed = _tone_bed(num_samples, sample_rate, detune=detune)
        # A little noise so the spectrogram has a floor to look at, and so
        # the quantized dtypes have broadband content to degrade.
        noise = mpdsp.gaussian_noise(num_samples, stddev=0.015,
                                     seed=seed + index)
        channels.append((bed + noise) * env)

    clip = np.vstack(channels)
    peak = np.max(np.abs(clip))
    if peak > 0.0:
        clip *= 0.89 / peak
    return clip


def load_wav(path: str | Path) -> tuple[np.ndarray, float]:
    """Load a WAV as a (channels, N) float64 array plus its sample rate.

    mpdsp.read_wav hands back `(N,)` for mono and `(N, channels)` otherwise;
    both are transposed to the channel-major layout the rest of the demo
    uses, so callers never need a mono special case.
    """
    data, sample_rate = mpdsp.read_wav(str(path))
    frames = np.asarray(data, dtype=np.float64)
    if frames.ndim == 1:
        frames = frames[:, np.newaxis]
    return frames.T.copy(), float(sample_rate)


def write_wav(path: str | Path, clip: np.ndarray, sample_rate: float,
              bits_per_sample: int = 16) -> None:
    """Write a (channels, N) float64 clip.

    mpdsp.write_wav takes frame-major `(N, channels)`, so this transposes.
    Samples are clipped to [-1, 1] first — upstream would clip them anyway,
    but doing it here makes the intent explicit: the demo expects some dtypes
    to overshoot, and clipping keeps that audible rather than letting an
    integer PCM writer wrap it into a full-scale glitch.
    """
    frames = np.clip(np.atleast_2d(clip), -1.0, 1.0).T
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mpdsp.write_wav(str(path), np.ascontiguousarray(frames),
                    int(sample_rate), bits_per_sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/input.wav")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=float, default=SAMPLE_RATE)
    args = parser.parse_args()

    clip = synthesize(args.duration, args.sample_rate)
    write_wav(args.out, clip, args.sample_rate)
    print(f"wrote {args.out} "
          f"({clip.shape[1]} frames x {clip.shape[0]} ch @ "
          f"{args.sample_rate:.0f} Hz, peak {np.max(np.abs(clip)):.3f})")


if __name__ == "__main__":
    main()
