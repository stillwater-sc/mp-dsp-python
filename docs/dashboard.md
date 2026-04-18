# Interactive Filter Designer

A Streamlit dashboard for designing IIR filters and evaluating them
under mixed-precision arithmetic. Modeled on Vinnie Falco's classic
[DSPFilters](https://github.com/vinniefalco/DSPFilters) demo, with the
side-by-side arithmetic comparison that is the whole point of `mpdsp`.

The dashboard lives at `scripts/plot_dashboard.py` in the repo and
ships with `mpdsp[dashboard]` via its Streamlit + matplotlib extras.

---

## Contents

- [Install and launch](#install-and-launch)
- [Tour of the tabs](#tour-of-the-tabs)
- [Interpreting the mixed-precision comparison](#interpreting-the-mixed-precision-comparison)
- [Exports](#exports)
- [Extending the dashboard](#extending-the-dashboard)

---

## Install and launch

The dashboard depends on Streamlit and matplotlib, wrapped in the
`dashboard` extra:

```bash
pip install mpdsp[dashboard]
```

Then from a checkout of this repository (or any directory containing
`scripts/plot_dashboard.py`):

```bash
streamlit run scripts/plot_dashboard.py
```

Streamlit will print three URLs; open the **Local URL** in a browser
and the app appears. Streamlit listens on `localhost:8501` by default.

### Running on a remote machine

Three common setups, safest first.

#### SSH tunnel (recommended)

Forward the Streamlit port over SSH. No firewall changes, no public
exposure, and the dashboard survives your LAN's IP plan changing.

On the remote box:

```bash
streamlit run scripts/plot_dashboard.py
```

From your laptop:

```bash
ssh -L 8501:localhost:8501 user@remote
```

Then browse to `http://localhost:8501` on the laptop. The tunnel lives
as long as the SSH session.

#### LAN access (open the port, scope by subnet)

If multiple people on your LAN need access and you don't want to hand
out SSH keys:

```bash
# On the remote box — open the firewall to the LAN only
sudo ufw allow from 192.168.1.0/24 to any port 8501 proto tcp \
    comment 'Streamlit mpdsp dashboard'

# Bind Streamlit to all interfaces (it listens on loopback by default)
streamlit run scripts/plot_dashboard.py --server.address 0.0.0.0
```

Adjust `192.168.1.0/24` to whatever CIDR your LAN actually uses. Skip
the subnet scope (`sudo ufw allow 8501/tcp`) only if you understand
that Streamlit has no authentication.

#### Public exposure

**Don't.** Streamlit ships with no authentication. If you genuinely
need a public dashboard, put it behind a reverse proxy with TLS and
basic auth, or use
[`streamlit_authenticator`](https://github.com/mkhorasani/Streamlit-Authenticator)
— scope well beyond this document.

---

## Tour of the tabs

The sidebar drives what the whole dashboard renders. The four tabs in
the main panel are views of the same filter — changing any sidebar
control re-renders all of them.

### Sidebar: filter design

- **Family** — Butterworth, Chebyshev I, Chebyshev II, Bessel,
  Legendre, Elliptic, RBJ. The first six are classical "order-based"
  designs; RBJ biquads are parameter-based and add shelving and
  allpass topologies.
- **Topology** — which variant of the family (lowpass / highpass /
  bandpass / bandstop, plus RBJ's allpass and lowshelf/highshelf).
- **Order** — only for the six classical families; 1 to 8. RBJ biquads
  are fixed 2nd-order.
- **Sample rate (Hz)** — default 44 100.
- **Cutoff / Center + Bandwidth / Width** — position and width of the
  passband/stopband. Upper bound is always Nyquist (sample_rate / 2)
  minus a 100 Hz margin to avoid edge artifacts.
- **Family-specific extras** — the sidebar grows the appropriate
  controls as you switch families:
  - Chebyshev I: `ripple_db` (passband ripple, in dB).
  - Chebyshev II: `stopband_db` (stopband attenuation, in dB).
  - Elliptic: `ripple_db` + `rolloff` (the second controls how fast
    the transition between passband and stopband is).
  - RBJ LP/HP/AP: `q` (0.7071 is critically damped).
  - RBJ BP/BS: `bandwidth` (in octaves).
  - RBJ shelves: `gain_db` + `q`.

### Sidebar: mixed precision

A multiselect over the 7 pre-instantiated arithmetic configurations
(`mpdsp.available_dtypes()`):

`reference`, `gpu_baseline`, `ml_hw`, `posit_full`, `tiny_posit`,
`cf24`, `half`.

The selection drives the **Time domain** and **Mixed-precision
comparison** tabs, plus the SQNR annotations overlaid on the main
magnitude plot.

### Sidebar: test signal

The signal the filter is exercised on for SQNR measurements:

- **Shape** — `sine` (a pure tone, easy interpretation), `chirp` (full
  band sweep, catches transition-band artifacts), or `white_noise`
  (flat spectrum, stresses numerical error at every frequency).
- **Length** — 128 to 8192 samples.
- **Frequency** — only for `sine`.

### Tab 1: Frequency response

The canonical filter-designer view: magnitude (in dB, log scale) and
unwrapped phase (in radians) over the full `[0, fs/2]` band.

- The **reference curve** is the double-precision frequency response
  computed from the designed biquad coefficients.
- If you have any non-reference dtypes selected, they appear as
  **annotations in the legend** showing the SQNR between that dtype's
  processed-signal output and the reference. Genuine per-dtype
  magnitude response requires spectral `dtype=` dispatch, tracked in
  #40.

### Tab 2: Pole / zero

- The **unit circle** is drawn for reference — stability requires all
  poles strictly inside it.
- **Red crosses** mark the pole locations in the z-plane.
- The title shows the **stability margin** (`1 - max|pole|`): larger
  is safer.
- Below the plot, a **per-stage biquad coefficient table** lists
  `(b0, b1, b2, a1, a2)` for every stage. Always in double — these
  are the design-time coefficients before any quantization.

### Tab 3: Time domain

Impulse and step response for every selected dtype, overlaid on the
same pair of axes. Useful for:

- Spotting **instability under quantization** — a reference-stable
  filter whose `tiny_posit` version blows up exponentially shows up
  immediately here.
- Comparing **transient envelope accuracy** — how closely a 6-bit
  posit tracks the reference ripple on a step response.
- Seeing **ringing bound** — the peak overshoot on a step.

### Tab 4: Mixed-precision comparison

The payoff tab, and the reason this dashboard exists rather than
`scipy.signal`.

- **DataFrame** — one row per selected dtype. Columns:
  - `dtype` — the arithmetic config key.
  - `sqnr_db` — Signal-to-Quantization-Noise Ratio, in dB, against
    `reference`. Higher is better. `inf` means exact match.
  - `max_abs_error` — the largest absolute difference per sample.
  - `max_rel_error` — the largest relative difference per sample.
  - `error` — the exception message if a dtype raised (e.g., an
    unsupported operation), else `None`.
- **Pole-displacement bar chart** — how far each dtype's quantized
  coefficients drift the pole locations from reference. Computed via
  `IIRFilter.pole_displacement(dtype)`.

### Tab 5: Compare A vs B

Dedicated two-type side-by-side, driven by two dtype dropdowns in the
sidebar ("Type A" and "Type B"). Three panels stacked vertically:

- **Magnitude response** — reference in grey (context), A in blue
  solid, B in red dashed.
- **Phase response** — same styling.
- **Impulse response (samples 0–255)** — this is where the two
  dtypes actually diverge visibly, since `filt.process` runs through
  the quantized state/sample types.

A four-metric strip below the plot reports:

| Metric | Meaning |
|--------|---------|
| `SQNR A vs B` | How close A and B are to each other |
| `max |A − B|` | Worst per-sample disagreement |
| `SQNR A vs ref` | How close A is to the reference (double) |
| `SQNR B vs ref` | How close B is to the reference |

Useful for deciding which of two candidate dtypes to deploy — the
A-vs-B SQNR tells you "are these interchangeable?" and the two
A-vs-ref / B-vs-ref numbers tell you "which one is closer to truth?"

### Tab 6: Summary

A single-row SQNR heatmap across all 7 arithmetic configurations
plus the **precision-cost frontier** (SQNR vs bits/sample scatter
with dtype labels). Both are computed live from the current design
— not from a pre-collected CSV — so the visualizations move as you
slide family / order / cutoff in the sidebar. The frontier is the
go-to visualization for deciding "at this bit budget, which dtype
gives the highest SQNR?". Pareto-optimal dtypes sit on the upper
envelope of the scatter.

---

## Interpreting the mixed-precision comparison

Knowing what the numbers mean saves hours. Rules of thumb accumulated
from the included notebooks:

| SQNR | What it usually means |
|------|-----------------------|
| `inf` | Bit-exact with the reference. Your chosen dtype is effectively double for this design. |
| > 120 dB | Numerically indistinguishable from reference for most applications. `gpu_baseline` sits here. |
| 60 – 120 dB | High-quality audio / instrumentation regime. `cf24`, `posit_full`, `ml_hw` typical. |
| 40 – 60 dB | Acceptable for telephony, control, many sensor paths. `half` typical. |
| 20 – 40 dB | Noise-floor regime — only OK for already-noisy signals. |
| < 20 dB | The filter has **hit the precision floor**. `tiny_posit` lands here on high-Q filters; the output is dominated by quantization error, not the filter's designed shape. |
| negative (and growing negative over time) | The quantized filter has gone **unstable**. Compare to the reference's stability margin — most likely the margin was already tight. |

Two common pitfalls surface on this tab:

1. **High-order, low-cutoff filters collapse first.** When poles cluster
   near the unit circle (high Q), small coefficient displacement knocks
   them out or around. Watch the pole-displacement bars for the
   low-precision dtypes before deciding "it works."
2. **Different test signals produce different SQNR.** A chirp excites
   the full band; a 440 Hz sine in a lowpass that has a 1 kHz cutoff
   basically measures the passband, which is the easy region. The
   SQNR from `chirp` is closer to worst-case; `sine` is closer to
   best-case. Be explicit about which you're reporting.

---

## Exports

Every tab that produces a figure has a **Download PNG** button. The
pole/zero tab additionally exports the biquad coefficients as CSV,
and the mixed-precision tab exports its comparison DataFrame as CSV.

Filenames encode the design so multiple experiments can be diff'd
later:

```
butterworth_lowpass_n4_freq.png
butterworth_lowpass_n4_polezero.png
butterworth_lowpass_n4_coefficients.csv
butterworth_lowpass_n4_time.png
butterworth_lowpass_n4_comparison.csv
butterworth_lowpass_n4_displacement.png

rbj_lowshelf_freq.png          # no order in filename for RBJ
```

The pattern is `{family}_{topology}_n{order}_{view}.{ext}` for the
classical families and `rbj_{topology}_{view}.{ext}` for RBJ biquads
(which have no `order` concept).

---

## Extending the dashboard

The dashboard is a single file, ~470 lines. Two extension points are
worth knowing.

### Adding a new filter family

Add an entry to `ORDER_FAMILIES` in `scripts/plot_dashboard.py` if
it's a classical order-based design, or to `RBJ_VARIANTS` if it's a
new biquad variant. For an order-based family the schema is:

```python
"MyFamily": FamilySpec(
    "MyFamily",
    extra_params=("my_extra_param",),   # names passed to the makers
    makers={
        "lowpass":  mpdsp.myfamily_lowpass,
        "highpass": mpdsp.myfamily_highpass,
        "bandpass": mpdsp.myfamily_bandpass,
        "bandstop": mpdsp.myfamily_bandstop,
    },
),
```

And add the extra parameter's `(min, max, default, step)` tuple to the
`default_range` dict inside `main()` so the sidebar slider knows how
to render it.

### Adding a new view (tab)

Write a pure function `plot_my_view(filt, ...) -> matplotlib.Figure`
and add a new `st.tabs([...])` entry alongside the existing four.
Keep the helper pure-function (no Streamlit calls inside) — that lets
the same function run headlessly for testing, which is how the
existing four helpers are validated in CI.

### Dispatching on a new arithmetic config

New configs come from upstream `mp-dsp-python` via `src/types.hpp`
and surface through `mpdsp.available_dtypes()`. The dashboard picks
them up automatically — no dashboard changes needed. Just remember
to rebuild the wheel.
