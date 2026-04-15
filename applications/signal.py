#!/usr/bin/env python3
import mpdsp
import numpy as np

sig = mpdsp.sine(10000, frequency=440.0, sample_rate=44100.0)

for dtype in mpdsp.available_dtypes():
    sqnr = mpdsp.measure_sqnr_db(sig, dtype)
    print(f"  {dtype:20s}  SQNR = {sqnr:.1f} dB")

