"""Test-signal sourcing for the SDR down-conversion demo.

Synthesizes a narrowband signal sitting on an RF carrier, buried in wideband
noise — the situation a direct-sampling receiver is in before any filtering.

The sample rate is *labelled* 5 GSPS but only a few hundred thousand samples
are ever generated. Nothing here depends on the absolute rate: filter design,
decimation ratios, and normalized responses are all rate-independent, so a
short buffer at a nominal 5 GSPS says exactly what a long one would. See the
README for why the demo does not pretend to stream at multi-GHz.

    python demos/04_sdr_ddc_cascade/simulate.py --seconds 2e-5
"""

from __future__ import annotations

import argparse

import numpy as np

import mpdsp

# Nominal front-end rate: a 5 GSPS direct-sampling ADC (RFSoC ZCU208 class).
SAMPLE_RATE_HZ = 5.0e9

# Carrier the receiver is tuned to, and the modulation bandwidth around it.
CARRIER_HZ = 1.2e9
SIGNAL_BANDWIDTH_HZ = 40.0e6


def synthesize(num_samples: int = 262144,
               sample_rate_hz: float = SAMPLE_RATE_HZ,
               carrier_hz: float = CARRIER_HZ,
               bandwidth_hz: float = SIGNAL_BANDWIDTH_HZ,
               snr_db: float = 20.0,
               seed: int = 11) -> tuple[np.ndarray, dict]:
    """Return (signal, metadata) for a narrowband emitter in wideband noise.

    The wanted signal is a small comb of tones spanning `bandwidth_hz` around
    `carrier_hz` — enough structure to see the passband survive the cascade
    intact, without needing a modulation scheme the demo would then have to
    demodulate.

    Two interferers are placed deliberately: one just outside the wanted
    band, which the cascade must reject, and one at a frequency that folds
    onto the wanted band when decimated by 16. The second is the one that
    catches an under-designed cascade — it is invisible in the input
    spectrum's neighbourhood and only appears on top of the signal after
    decimation.
    """
    rng = np.random.default_rng(seed)
    t_index = np.arange(num_samples)

    signal = np.zeros(num_samples, dtype=np.float64)
    tone_offsets = np.linspace(-0.4, 0.4, 5) * bandwidth_hz
    for offset in tone_offsets:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += mpdsp.sine(num_samples, frequency=carrier_hz + offset,
                             sample_rate=sample_rate_hz,
                             amplitude=1.0 / len(tone_offsets), phase=phase)

    # Adjacent-channel interferer: outside the wanted band, inside the first
    # decimation stage's passband. Ordinary selectivity handles it.
    adjacent_hz = carrier_hz + 3.0 * bandwidth_hz
    interferer = mpdsp.sine(num_samples, frequency=adjacent_hz,
                            sample_rate=sample_rate_hz, amplitude=0.5)

    # Alias-trap interferer: offset by exactly one output-rate step, so
    # decimation by 16 folds it straight onto the wanted band unless the
    # cascade's stopband is doing its job.
    output_rate_hz = sample_rate_hz / 16.0
    alias_hz = carrier_hz + output_rate_hz
    alias_tone = mpdsp.sine(num_samples, frequency=alias_hz,
                            sample_rate=sample_rate_hz, amplitude=0.5)

    signal_power = float(np.mean(signal ** 2))
    noise_stddev = np.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
    noise = mpdsp.gaussian_noise(num_samples, stddev=noise_stddev, seed=seed)

    composite = signal + interferer + alias_tone + noise
    peak = float(np.max(np.abs(composite)))
    if peak > 0.0:
        composite = composite * (0.9 / peak)

    return composite, {
        "sample_rate_hz": sample_rate_hz,
        "carrier_hz": carrier_hz,
        "bandwidth_hz": bandwidth_hz,
        "adjacent_hz": adjacent_hz,
        "alias_hz": alias_hz,
        "snr_db": snr_db,
        "num_samples": num_samples,
    }


def wanted_band(meta: dict) -> tuple[float, float]:
    """(low, high) Hz of the wanted signal, in RF terms."""
    half = meta["bandwidth_hz"] / 2.0
    return meta["carrier_hz"] - half, meta["carrier_hz"] + half


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=262144)
    parser.add_argument("--snr-db", type=float, default=20.0)
    args = parser.parse_args()

    signal, meta = synthesize(num_samples=args.num_samples,
                              snr_db=args.snr_db)
    low, high = wanted_band(meta)
    print(f"{meta['num_samples']} samples @ "
          f"{meta['sample_rate_hz'] / 1e9:.2f} GSPS")
    print(f"  wanted      {low / 1e9:.4f} - {high / 1e9:.4f} GHz "
          f"(carrier {meta['carrier_hz'] / 1e9:.3f} GHz)")
    print(f"  adjacent    {meta['adjacent_hz'] / 1e9:.4f} GHz")
    print(f"  alias trap  {meta['alias_hz'] / 1e9:.4f} GHz "
          f"(folds onto the carrier at /16)")
    print(f"  input SNR   {meta['snr_db']:.1f} dB, peak "
          f"{np.max(np.abs(signal)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
