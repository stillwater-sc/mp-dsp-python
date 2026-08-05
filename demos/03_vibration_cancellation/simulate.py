"""Disturbance signals for the active vibration cancellation demo.

Models a feedforward active-noise-control layout:

    machine ──> reference accelerometer ──> x(n)   (what the filter sees)
       │
       └─ primary path P(z) ──> error accelerometer ──> d(n)  (what we cancel)

The adaptive filter's job is to learn P(z) so its output reproduces d(n);
the actuator then emits the negative of that, and whatever survives is the
residual vibration.

Two properties of the synthetic signal are deliberate:

* **A sensor-noise floor on d(n).** Without it the identification is exact
  and every reasonable dtype reports 300+ dB of cancellation, which is
  meaningless. With a 30 dB floor, the reference run converges to ~29.5 dB —
  the physical limit — and a dtype's shortfall against that limit is the
  number worth reading.
* **A speed change partway through.** The machine tones shift, so the filter
  has to re-converge on a moving target. Precision costs show up twice: in
  the steady-state floor, and in whether the filter can track at all.

    python demos/03_vibration_cancellation/simulate.py
"""

from __future__ import annotations

import argparse

import numpy as np

import mpdsp

SAMPLE_RATE = 4000.0

# Machine order tones: fundamental plus two harmonics of a rotating shaft.
BASE_TONES = ((50.0, 1.0), (120.0, 0.6), (185.0, 0.4))

# Shaft speed steps up by this ratio at the halfway point.
SPEED_CHANGE = 1.18

# Primary path: the structural transfer function from reference sensor to
# error sensor. Short, non-minimum-phase, with a leading delay — the delay is
# what makes this a causal identification problem rather than a trivial one.
PRIMARY_PATH = np.array([0.0, 0.5, -0.35, 0.22, 0.0, -0.12, 0.06])

# Error-sensor noise floor. Caps achievable cancellation, which is what makes
# the per-dtype comparison physically meaningful.
SENSOR_SNR_DB = 30.0


def _tonal_bed(num_samples: int, sample_rate: float, speed: float,
               start_phase: float = 0.0) -> np.ndarray:
    out = np.zeros(num_samples, dtype=np.float64)
    for freq, amplitude in BASE_TONES:
        shifted = freq * speed
        if shifted >= sample_rate / 2.0:
            continue
        out += mpdsp.sine(num_samples, frequency=shifted,
                          sample_rate=sample_rate, amplitude=amplitude,
                          phase=start_phase)
    return out


def synthesize(num_samples: int = 24000, sample_rate: float = SAMPLE_RATE,
               sensor_snr_db: float = SENSOR_SNR_DB,
               speed_change: float = SPEED_CHANGE,
               seed: int = 3) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (reference, primary, metadata).

    `reference` is what the upstream accelerometer sees; `primary` is what
    reaches the error sensor after the structural path, plus sensor noise.
    """
    split = num_samples // 2

    # Broadband structural excitation under the tones — without it the input
    # is purely narrowband, the correlation matrix is rank-deficient at the
    # filter length, and the adaptive filter has no unique solution.
    broadband = mpdsp.gaussian_noise(num_samples, stddev=0.35, seed=seed)

    reference = np.concatenate([
        _tonal_bed(split, sample_rate, speed=1.0),
        _tonal_bed(num_samples - split, sample_rate, speed=speed_change),
    ]) + broadband

    clean = np.convolve(reference, PRIMARY_PATH)[:num_samples]

    noise_stddev = np.sqrt(np.mean(clean ** 2)
                           / (10.0 ** (sensor_snr_db / 10.0)))
    primary = clean + mpdsp.gaussian_noise(num_samples, stddev=noise_stddev,
                                           seed=seed + 100)

    return reference, primary, {
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "speed_change_at": split,
        "speed_change": speed_change,
        "sensor_snr_db": sensor_snr_db,
        "primary_path": PRIMARY_PATH.copy(),
        "tones_hz": [f for f, _ in BASE_TONES],
        # The cancellation any filter can reach is bounded by the sensor
        # floor, so quote it alongside every measured reduction.
        "achievable_reduction_db": sensor_snr_db,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=24000)
    parser.add_argument("--sensor-snr-db", type=float, default=SENSOR_SNR_DB)
    args = parser.parse_args()

    reference, primary, meta = synthesize(
        num_samples=args.num_samples, sensor_snr_db=args.sensor_snr_db)

    print(f"{meta['num_samples']} samples @ {meta['sample_rate']:.0f} Hz "
          f"({meta['num_samples'] / meta['sample_rate']:.1f} s)")
    print(f"  tones        {meta['tones_hz']} Hz, stepping to "
          f"{[round(f * meta['speed_change'], 1) for f in meta['tones_hz']]} "
          f"Hz at sample {meta['speed_change_at']}")
    print(f"  primary path {len(meta['primary_path'])} taps")
    print(f"  sensor floor {meta['sensor_snr_db']:.0f} dB SNR "
          f"→ cancellation cannot exceed ~{meta['sensor_snr_db']:.0f} dB")
    print(f"  reference    rms {np.sqrt(np.mean(reference ** 2)):.3f}")
    print(f"  primary      rms {np.sqrt(np.mean(primary ** 2)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
