"""Tests for the precision-profile bindings and reductions (issue #133).

The expected values are the numbers published in Universal's
decimals-of-accuracy tutorial (docs/tutorials/decimals-of-accuracy.md),
which docs/decimals_of_accuracy.md reproduces. Pinning them here means a
drift in upstream's precision_profile.hpp, or in a number system's
encoding, fails loudly instead of silently changing the page.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)

from mpdsp import precision as pr  # noqa: E402


@pytest.fixture(scope="module")
def profile():
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = mpdsp.precision_profile(name)
        return cache[name]
    return get


# ---- binding surface ----------------------------------------------------


TUTORIAL_TYPES = [
    "int16", "Q15", "fp16", "lns<16,10>",
    "bposit<16,6,5>", "bposit<16,6,3>", "bposit<16,6,2>",
    "bposit<16,5,2>", "bposit<16,4,2>", "bposit<16,3,1>",
    "int32", "Q31", "float", "lns<32,23>",
    "bposit<32,6,5>", "bposit<32,5,2>", "bposit<32,4,2>",
]


def test_available_types_cover_the_tutorial():
    available = mpdsp.available_profile_types()
    for name in TUTORIAL_TYPES:
        assert name in available


def test_available_types_cover_the_dtype_scalars():
    available = mpdsp.available_profile_types()
    for name in ["double", "cf24", "half", "posit<16,1>", "posit<32,2>",
                 "fixpnt<32,24>", "fixpnt<16,12>", "integer<8>",
                 "integer<6>"]:
        assert name in available


@pytest.mark.parametrize("name", mpdsp.available_profile_types())
def test_every_type_profiles(name):
    p = mpdsp.precision_profile(name)
    assert len(p) > 0
    assert 0.0 < p.minpos < p.maxpos
    # ascending magnitudes, consistent columns
    assert np.all(np.diff(p.magnitude) > 0)
    assert p.magnitude.shape == p.decimals.shape == p.fraction_bits.shape
    np.testing.assert_allclose(p.log2_magnitude, np.log2(p.magnitude))


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="available_profile_types"):
        mpdsp.precision_profile("posit<64,3>")


def test_zero_samples_per_binade_raises():
    with pytest.raises(ValueError, match="samples_per_binade"):
        mpdsp.precision_profile("float", samples_per_binade=0)


def test_half_is_an_alias_for_fp16(profile):
    half = mpdsp.precision_profile("half")
    assert half.label == "fp16"
    np.testing.assert_array_equal(half.decimals, profile("fp16").decimals)


def test_exhaustive_and_sampled(profile):
    # 16-bit types decode every encoding: ~32k positive values each
    for name in ["int16", "Q15", "lns<16,10>", "bposit<16,6,5>"]:
        p = profile(name)
        assert p.exhaustive
        assert len(p) == 32767
    assert len(profile("fp16")) == 31743     # less the inf/NaN encodings
    for name in ["float", "Q31", "bposit<32,4,2>"]:
        assert not profile(name).exhaustive


def test_samples_per_binade_densifies_sampled_profiles():
    coarse = mpdsp.precision_profile("float", samples_per_binade=2)
    fine = mpdsp.precision_profile("float", samples_per_binade=8)
    assert len(fine) > len(coarse)


def test_decimals_definition(profile):
    p = profile("fp16")
    # decimals = log10(2) * (fraction_bits + 1) at every point
    np.testing.assert_allclose(p.decimals,
                               math.log10(2.0) * (p.fraction_bits + 1.0))


def test_decimals_at_is_zero_outside_the_range(profile):
    q15 = profile("Q15")
    assert q15.decimals_at(1.0) == 0.0            # no headroom above 1.0
    assert q15.decimals_at(math.ldexp(1.0, -16)) == 0.0
    assert profile("int16").decimals_at(0.5) == 0.0


def test_write_csv_matches_universal_layout(profile, tmp_path):
    path = tmp_path / "fp16.csv"
    profile("fp16").write_csv(str(path))
    lines = path.read_text().splitlines()
    assert lines[0] == "# label: fp16"
    assert lines[4] == "# sampling: exhaustive"
    assert lines[5] == "magnitude,log2_magnitude,decimals,fraction_bits"
    assert len(lines) == 6 + 31743


def test_repr(profile):
    assert repr(profile("fp16")) == \
        "PrecisionProfile('fp16', 31743 points, exhaustive)"


# ---- the tutorial's published numbers -----------------------------------


@pytest.mark.parametrize("name, band", [
    ("Q15", 4.52), ("fp16", 3.31), ("lns<16,10>", 3.47),
    ("bposit<16,6,5>", 2.71), ("bposit<16,4,2>", 3.61),
    ("bposit<16,3,1>", 3.91),
    ("float", 7.22), ("bposit<32,6,5>", 7.53), ("bposit<32,4,2>", 8.43),
])
def test_signal_band_decimals(profile, name, band):
    assert profile(name).decimals_at(0.5) == pytest.approx(band, abs=0.005)


def test_lns_is_flat(profile):
    d = profile("lns<16,10>").decimals
    assert d.max() - d.min() < 0.01
    assert d.mean() == pytest.approx(3.47, abs=0.01)


def test_int16_peaks_at_4_8_decimals(profile):
    assert profile("int16").decimals.max() == pytest.approx(4.8, abs=0.05)


@pytest.mark.parametrize("name, share", [
    ("bposit<16,6,5>", 21.1), ("bposit<16,6,3>", 67.2),
    ("bposit<16,6,2>", 89.8), ("bposit<16,5,2>", 89.8),
    ("bposit<16,4,2>", 92.2), ("bposit<16,3,1>", 100.0),
    ("fp16", 85.5), ("lns<16,10>", 84.4),
])
def test_share_in_roi(profile, name, share):
    assert 100.0 * pr.share_in_roi(profile(name)) == \
        pytest.approx(share, abs=0.05)


def test_share_in_roi_is_none_for_sampled_profiles(profile):
    assert pr.share_in_roi(profile("float")) is None


RANGE_FIT_16 = [
    # label, range, band, worst, floor
    ("bposit<16,6,5>", (-192, 192), 2.71, 2.71, 4),
    ("bposit<16,6,3>", (-48, 48), 3.31, 3.01, 6),
    ("bposit<16,6,2>", (-24, 24), 3.61, 2.71, 7),
    ("bposit<16,5,2>", (-20, 20), 3.61, 2.71, 8),
    ("bposit<16,4,2>", (-16, 16), 3.61, 3.01, 9),
    ("bposit<16,3,1>", (-6, 6), 3.91, 0.00, 11),
]

RANGE_FIT_32 = [
    ("float", (-149, 128), 7.22, 7.22, 0),
    ("bposit<32,6,5>", (-192, 192), 7.53, 7.53, 20),
    ("bposit<32,5,2>", (-20, 20), 8.43, 7.53, 24),
    ("bposit<32,4,2>", (-16, 16), 8.43, 7.83, 25),
]


@pytest.mark.parametrize("expected", RANGE_FIT_16 + RANGE_FIT_32,
                         ids=lambda e: e[0])
def test_range_fit_table(profile, expected):
    label, rng, band, worst, floor = expected
    (row,) = mpdsp.range_fit_table([profile(label)])
    assert row["label"] == label
    assert row["range"] == rng
    assert row["band"] == pytest.approx(band, abs=0.005)
    assert row["worst"] == pytest.approx(worst, abs=0.005)
    assert row["floor_bits"] == pytest.approx(floor, abs=1e-9)


@pytest.mark.parametrize("name", [
    "bposit<16,6,5>", "bposit<16,6,3>", "bposit<16,6,2>",
    "bposit<16,5,2>", "bposit<16,4,2>", "fp16", "lns<16,10>",
])
def test_binade_starts_find_the_exhaustive_minimum(profile, name):
    """min_decimals_over() samples only 2^k; the page claims that suffices.

    Within a binade precision is lowest at its start, so the minimum over
    the binade starts equals the minimum over every encoding in the region.
    """
    p = profile(name)
    m = p.magnitude
    inside = (m >= 2.0 ** -15) & (m <= 2.0 ** 12)
    assert p.decimals[inside].min() == pytest.approx(pr.min_decimals_over(p))


# Where in the region each configuration reaches its worst case, as the
# "Reading the table" section of docs/decimals_of_accuracy.md states.
WORST_BINADES = [
    ("bposit<16,6,5>", 8, list(range(-15, 13))),
    ("bposit<16,6,3>", 9, list(range(-15, -8)) + list(range(8, 13))),
    ("bposit<16,6,2>", 8, [-15, -14, -13, 12]),
    ("bposit<16,5,2>", 8, [-15, -14, -13, 12]),
    ("bposit<16,4,2>", 9, list(range(-15, -8)) + list(range(8, 13))),
]


@pytest.mark.parametrize("name, bits, binades", WORST_BINADES,
                         ids=lambda e: e if isinstance(e, str) else "")
def test_where_the_worst_case_lands(profile, name, bits, binades):
    p = profile(name)
    worst = pr.min_decimals_over(p)
    # decimals at a binade start = log10(2) * (fraction bits + 1)
    assert worst == pytest.approx(math.log10(2.0) * (bits + 1))
    at = [k for k in range(-15, 13)
          if p.decimals_at(2.0 ** k) == pytest.approx(worst)]
    assert at == binades


def test_narrow_bposit_is_zero_outside_its_range(profile):
    p = profile("bposit<16,3,1>")
    at = [k for k in range(-15, 13) if p.decimals_at(2.0 ** k) == 0.0]
    assert at == list(range(-15, -5)) + list(range(6, 13))


def test_tutorial_assertions(profile):
    """The checks Universal's dsp_precision_profiles application asserts."""
    q15, q31, fp16 = profile("Q15"), profile("Q31"), profile("fp16")
    bp16std, bp16fit = profile("bposit<16,6,5>"), profile("bposit<16,4,2>")
    narrow = profile("bposit<16,3,1>")
    # Q15 has no headroom; at 2^-15 it is down to its last bit
    assert q15.decimals_at(0.5) > fp16.decimals_at(0.5)
    assert q31.decimals_at(1.0) == 0.0
    assert q15.decimals_at(2.0 ** -15) < 0.5
    assert fp16.decimals_at(2.0 ** -15) > 2.9
    # the standard b-posit trails fp16 in the signal band
    assert bp16std.decimals_at(0.5) < fp16.decimals_at(0.5)
    # bposit<16,3,1> does not cover the region
    assert narrow.decimals_at(2.0 ** -15) == 0.0
    assert narrow.decimals_at(1024.0) == 0.0
    # the tightest fit keeps 3 decimals across the region, 9-bit floor
    assert pr.min_decimals_over(bp16fit) >= 3.0
    assert pr.min_fraction_bits(bp16fit) >= 9.0 - 1e-9


# ---- summary and formatting ---------------------------------------------


def test_summary_table(profile):
    (fp16,) = mpdsp.summary_table([profile("fp16")])
    assert fp16["range"] == (-24, 16)
    assert fp16["band"] == pytest.approx(3.31, abs=0.005)
    # 10 fraction bits at the start of [0.5, 1) -> 60.2 dB
    assert fp16["sqnr_db"] == pytest.approx(60.2, abs=0.05)
    assert fp16["at_floor"] == pytest.approx(3.01, abs=0.005)
    assert fp16["at_fft_1024"] == pytest.approx(3.31, abs=0.005)


def test_sqnr_db_of_unrepresentable_is_zero():
    assert pr.sqnr_db(0.0) == 0.0


def test_format_markdown_table(profile):
    table = pr.format_markdown_table(
        mpdsp.range_fit_table([profile("bposit<16,6,5>"), profile("fp16")]),
        [("type", "label", pr.fmt_label),
         ("range (log2)", "range", pr.fmt_range),
         ("share", "share", pr.fmt_share)])
    assert table.splitlines() == [
        "| type | range (log2) | share |",
        "|---|---|---|",
        "| `bposit<16,6,5>` | -192 .. 192 | 21.1 % |",
        "| fp16 | -24 .. 16 | 85.5 % |",
    ]


# ---- plotting -----------------------------------------------------------


def test_plot_precision_profiles(profile):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = [profile("Q15"), profile("fp16"), profile("bposit<32,4,2>")]
    ax = mpdsp.plot_precision_profiles(profiles, dsp=True, xmin=-32,
                                       xmax=24, title="t")
    try:
        lines = [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")]
        assert [ln.get_label() for ln in lines] == ["Q15", "fp16",
                                                    "bposit<32,4,2>"]
        # sampled profiles are drawn as steps, exhaustive ones are not
        assert lines[1].get_drawstyle() == "default"
        assert lines[2].get_drawstyle() == "steps-post"
        # curves are closed to 0 at both ends of the range
        y = lines[0].get_ydata()
        assert y[0] == 0.0 and y[-1] == 0.0
        assert ax.get_xlim() == (-32, 24)
        assert ax.get_title() == "t"
    finally:
        plt.close(ax.figure)
