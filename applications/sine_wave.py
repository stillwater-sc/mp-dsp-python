import mpdsp
import matplotlib.pyplot as plt

# Generate a signal
sig = mpdsp.sine(2000, frequency=440.0, sample_rate=44100.0)

# Time + spectrum side by side
mpdsp.plot_signal_and_spectrum(sig, sample_rate=44100.0, title="440 Hz Sine")

# SQNR bar chart across all arithmetic types
mpdsp.plot_sqnr_comparison(sig, title="SQNR: 440 Hz Sine")

# Quantization error traces
mpdsp.plot_quantization_comparison(sig,
    ["gpu_baseline", "half", "posit_full", "tiny_posit"],
    sample_rate=44100.0)

# Window comparison (time + frequency response)
mpdsp.plot_window_comparison({
    "Hamming": mpdsp.hamming,
    "Blackman": mpdsp.blackman,
    "Kaiser (8)": lambda N: mpdsp.kaiser(N, beta=8.0),
})

plt.show()

