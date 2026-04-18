"""Tests for image processing bindings (scaffold: generators + convolve2d)."""

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


class TestCheckerboard:
    def test_shape_and_dtype(self):
        img = mpdsp.checkerboard(8, 12, block_size=2)
        assert img.shape == (8, 12)
        assert img.dtype == np.float64

    def test_two_values(self):
        img = mpdsp.checkerboard(4, 4, block_size=1, low=-1.0, high=2.0)
        unique = np.unique(img)
        assert set(unique.tolist()) == {-1.0, 2.0}

    def test_block_size_controls_square(self):
        """With block_size=3, each 3x3 region should be uniform."""
        img = mpdsp.checkerboard(6, 6, block_size=3)
        # Top-left 3x3 block is all one value
        assert len(np.unique(img[:3, :3])) == 1
        # Top-right 3x3 block is the other value
        assert len(np.unique(img[:3, 3:])) == 1
        assert img[0, 0] != img[0, 3]

    def test_zero_block_size_raises(self):
        with pytest.raises(ValueError):
            mpdsp.checkerboard(4, 4, block_size=0)

    @pytest.mark.parametrize("rows,cols", [(0, 4), (4, 0), (0, 0)])
    def test_zero_dims_rejected(self, rows, cols):
        with pytest.raises(ValueError):
            mpdsp.checkerboard(rows, cols, block_size=2)


class TestGaussianBlob:
    def test_shape(self):
        img = mpdsp.gaussian_blob(16, 20, sigma=3.0)
        assert img.shape == (16, 20)
        assert img.dtype == np.float64

    def test_peak_is_center(self):
        """The Gaussian blob peaks at the image center."""
        img = mpdsp.gaussian_blob(17, 17, sigma=3.0)
        # argmax in flattened space; unravel to (row, col)
        peak_idx = np.unravel_index(np.argmax(img), img.shape)
        assert peak_idx == (8, 8)

    def test_amplitude_scales_peak(self):
        img_low = mpdsp.gaussian_blob(9, 9, sigma=2.0, amplitude=0.5)
        img_high = mpdsp.gaussian_blob(9, 9, sigma=2.0, amplitude=2.0)
        # Peak amplitude ratio matches the parameter ratio exactly.
        assert img_high.max() == pytest.approx(img_low.max() * 4.0)

    def test_non_positive_sigma_raises(self):
        with pytest.raises(ValueError):
            mpdsp.gaussian_blob(8, 8, sigma=0.0)
        with pytest.raises(ValueError):
            mpdsp.gaussian_blob(8, 8, sigma=-1.0)

    @pytest.mark.parametrize("rows,cols", [(0, 4), (4, 0)])
    def test_zero_dims_rejected(self, rows, cols):
        with pytest.raises(ValueError):
            mpdsp.gaussian_blob(rows, cols, sigma=1.0)


class TestGradientHorizontal:
    def test_shape(self):
        img = mpdsp.gradient_horizontal(6, 10)
        assert img.shape == (6, 10)

    def test_linear_left_to_right(self):
        img = mpdsp.gradient_horizontal(4, 5, start=0.0, end=1.0)
        expected_row = np.linspace(0.0, 1.0, 5)
        # Every row equals the same gradient.
        for r in range(img.shape[0]):
            np.testing.assert_allclose(img[r], expected_row)

    def test_start_end_parameters(self):
        img = mpdsp.gradient_horizontal(2, 3, start=10.0, end=20.0)
        assert img[0, 0] == 10.0
        assert img[0, -1] == 20.0

    @pytest.mark.parametrize("rows,cols", [(0, 5), (5, 0)])
    def test_zero_dims_rejected(self, rows, cols):
        with pytest.raises(ValueError):
            mpdsp.gradient_horizontal(rows, cols)


class TestGradientVertical:
    def test_linear_top_to_bottom(self):
        img = mpdsp.gradient_vertical(5, 4, start=0.0, end=1.0)
        expected_col = np.linspace(0.0, 1.0, 5)
        for c in range(img.shape[1]):
            np.testing.assert_allclose(img[:, c], expected_col)


class TestGradientRadial:
    def test_peak_at_center(self):
        img = mpdsp.gradient_radial(9, 9, center_val=1.0, edge_val=0.0)
        peak = np.unravel_index(np.argmax(img), img.shape)
        assert peak == (4, 4)

    def test_edge_is_lower(self):
        img = mpdsp.gradient_radial(9, 9, center_val=1.0, edge_val=0.0)
        assert img[0, 0] < img[4, 4]


class TestStripes:
    @pytest.mark.parametrize("fn", ["stripes_horizontal", "stripes_vertical"])
    def test_two_values(self, fn):
        img = getattr(mpdsp, fn)(8, 8, stripe_width=2, low=-1.0, high=2.0)
        assert set(np.unique(img).tolist()) == {-1.0, 2.0}

    def test_horizontal_stripes_vary_in_rows(self):
        img = mpdsp.stripes_horizontal(6, 4, stripe_width=1)
        # Each row is uniform; adjacent rows alternate.
        for r in range(img.shape[0]):
            assert len(np.unique(img[r, :])) == 1
        assert img[0, 0] != img[1, 0]

    def test_vertical_stripes_vary_in_cols(self):
        img = mpdsp.stripes_vertical(4, 6, stripe_width=1)
        for c in range(img.shape[1]):
            assert len(np.unique(img[:, c])) == 1
        assert img[0, 0] != img[0, 1]

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            mpdsp.stripes_horizontal(4, 4, stripe_width=0)
        with pytest.raises(ValueError):
            mpdsp.stripes_vertical(4, 4, stripe_width=0)


class TestGrid:
    def test_shape(self):
        img = mpdsp.grid(12, 12, spacing=4)
        assert img.shape == (12, 12)

    def test_line_at_multiples_of_spacing(self):
        img = mpdsp.grid(10, 10, spacing=3, background=0.0, line=1.0)
        # Rows and columns at multiples of the spacing are lines (==1).
        assert img[0, 0] == 1.0
        assert img[3, 3] == 1.0
        assert img[6, 6] == 1.0

    def test_zero_spacing_raises(self):
        with pytest.raises(ValueError):
            mpdsp.grid(8, 8, spacing=0)


class TestCircle:
    def test_center_is_foreground(self):
        img = mpdsp.circle(11, 11, radius=3, foreground=1.0, background=0.0)
        assert img[5, 5] == 1.0

    def test_corner_is_background(self):
        img = mpdsp.circle(11, 11, radius=3, foreground=1.0, background=0.0)
        assert img[0, 0] == 0.0

    def test_area_grows_with_radius(self):
        small = mpdsp.circle(21, 21, radius=2).sum()
        large = mpdsp.circle(21, 21, radius=6).sum()
        assert large > small


class TestRectangle:
    def test_filled_region(self):
        img = mpdsp.rectangle(10, 10, y=2, x=3, h=4, w=5,
                              foreground=1.0, background=0.0)
        # Pixels inside [y:y+h, x:x+w] are foreground
        assert np.all(img[2:6, 3:8] == 1.0)

    def test_outside_is_background(self):
        img = mpdsp.rectangle(10, 10, y=2, x=3, h=4, w=5,
                              foreground=1.0, background=0.5)
        assert img[0, 0] == 0.5
        assert img[9, 9] == 0.5


class TestZonePlate:
    def test_shape_and_range(self):
        """Upstream normalizes the cosine output to [0, 1]."""
        img = mpdsp.zone_plate(32, 32)
        assert img.shape == (32, 32)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_nontrivial_output(self):
        """Zone plate should have many unique values (it's a chirp)."""
        img = mpdsp.zone_plate(32, 32)
        assert len(np.unique(img.round(3))) > 10


class TestNoiseGenerators:
    def test_uniform_noise_in_range(self):
        img = mpdsp.uniform_noise_image(64, 64, low=0.0, high=1.0, seed=7)
        assert img.min() >= 0.0
        assert img.max() <= 1.0
        # 64x64 uniform samples should cover most of the range.
        assert img.min() < 0.1
        assert img.max() > 0.9

    def test_uniform_noise_seed_reproducible(self):
        a = mpdsp.uniform_noise_image(16, 16, seed=123)
        b = mpdsp.uniform_noise_image(16, 16, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_uniform_noise_different_seeds_differ(self):
        a = mpdsp.uniform_noise_image(16, 16, seed=0)
        b = mpdsp.uniform_noise_image(16, 16, seed=1)
        assert not np.array_equal(a, b)

    def test_gaussian_noise_stats(self):
        img = mpdsp.gaussian_noise_image(256, 256, mean=0.0, stddev=1.0, seed=5)
        assert abs(img.mean() - 0.0) < 0.05
        assert abs(img.std() - 1.0) < 0.05

    def test_gaussian_noise_negative_stddev_raises(self):
        with pytest.raises(ValueError):
            mpdsp.gaussian_noise_image(8, 8, stddev=-0.1)

    def test_salt_and_pepper_density_zero_is_uniform(self):
        img = mpdsp.salt_and_pepper(32, 32, density=0.0, low=0.0, high=1.0)
        # density=0 means no pixels flipped; all at the midpoint (0.5).
        np.testing.assert_array_equal(img, 0.5)

    def test_salt_and_pepper_density_out_of_range_raises(self):
        with pytest.raises(ValueError):
            mpdsp.salt_and_pepper(8, 8, density=-0.1)
        with pytest.raises(ValueError):
            mpdsp.salt_and_pepper(8, 8, density=1.5)


class TestAddNoise:
    def test_zero_stddev_unchanged(self):
        base = mpdsp.checkerboard(8, 8, 2)
        out = mpdsp.add_noise(base, stddev=0.0)
        np.testing.assert_array_equal(out, base)

    def test_stddev_scales_deviation(self):
        base = np.zeros((64, 64))
        low_noise  = mpdsp.add_noise(base, stddev=0.1, seed=0)
        high_noise = mpdsp.add_noise(base, stddev=1.0, seed=0)
        assert high_noise.std() > low_noise.std() * 5

    def test_seed_reproducible(self):
        base = mpdsp.gradient_horizontal(16, 16)
        a = mpdsp.add_noise(base, stddev=0.2, seed=99)
        b = mpdsp.add_noise(base, stddev=0.2, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_negative_stddev_raises(self):
        base = mpdsp.checkerboard(8, 8, 2)
        with pytest.raises(ValueError):
            mpdsp.add_noise(base, stddev=-0.1)

    def test_empty_image_raises(self):
        with pytest.raises(ValueError):
            mpdsp.add_noise(np.zeros((0, 5)), stddev=0.1)


class TestThreshold:
    def test_binary_output(self):
        img = mpdsp.gradient_horizontal(4, 8)  # values 0 .. 1
        out = mpdsp.threshold(img, thresh=0.5)
        assert set(np.unique(out).tolist()) == {0.0, 1.0}

    def test_custom_low_high(self):
        img = mpdsp.gradient_horizontal(4, 8)
        out = mpdsp.threshold(img, thresh=0.5, low=-1.0, high=2.0)
        assert set(np.unique(out).tolist()) == {-1.0, 2.0}

    def test_threshold_splits_correctly(self):
        img = np.array([[0.1, 0.4], [0.6, 0.9]])
        out = mpdsp.threshold(img, thresh=0.5)
        np.testing.assert_array_equal(out, [[0.0, 0.0], [1.0, 1.0]])

    def test_boundary_pixel_goes_high(self):
        """Upstream contract: pixels >= thresh map to `high`; pixels
        strictly below map to `low`. Pin the boundary behavior so a
        future upstream shift from >= to > would fail this test."""
        img = np.array([[0.5]])
        out = mpdsp.threshold(img, thresh=0.5, low=-7.0, high=7.0)
        assert out[0, 0] == 7.0

    def test_just_below_threshold_goes_low(self):
        """Counterpart to test_boundary_pixel_goes_high."""
        img = np.array([[0.5 - 1e-12]])
        out = mpdsp.threshold(img, thresh=0.5, low=-7.0, high=7.0)
        assert out[0, 0] == -7.0

    def test_empty_image_raises(self):
        with pytest.raises(ValueError):
            mpdsp.threshold(np.zeros((0, 5)), thresh=0.5)


# ---------------------------------------------------------------------------
# convolve2d
# ---------------------------------------------------------------------------


class TestConvolve2d:
    def test_identity_kernel_round_trips(self):
        """A 3x3 kernel with 1.0 at the center is the identity operation —
        output must equal input (at reference dtype, exactly)."""
        img = mpdsp.checkerboard(8, 8, block_size=2)
        kernel = np.array([[0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0]])
        out = mpdsp.convolve2d(img, kernel)
        np.testing.assert_array_equal(out, img)

    def test_shape_preserved(self):
        img = mpdsp.gaussian_blob(12, 16, sigma=2.0)
        kernel = np.ones((5, 5)) / 25.0
        out = mpdsp.convolve2d(img, kernel)
        assert out.shape == img.shape
        assert out.dtype == np.float64

    def test_box_blur_reduces_high_frequency(self):
        """A box-blur kernel should strictly reduce the peak-to-peak range of
        a high-frequency signal (like a checkerboard)."""
        img = mpdsp.checkerboard(16, 16, block_size=1)  # alternating 0/1
        kernel = np.ones((3, 3)) / 9.0
        out = mpdsp.convolve2d(img, kernel)
        # Interior pixels mix so the range tightens around 0.5.
        interior = out[2:-2, 2:-2]
        assert interior.max() < 1.0
        assert interior.min() > 0.0

    def test_unknown_border_raises(self):
        img = mpdsp.checkerboard(8, 8, block_size=2)
        kernel = np.ones((3, 3)) / 9.0
        with pytest.raises(ValueError):
            mpdsp.convolve2d(img, kernel, border="not_a_border")

    def test_constant_border_with_pad(self):
        """With border='constant' and pad=0, output border pixels mix
        in zeros from the virtual padding — so the output is smaller."""
        img = np.ones((5, 5))
        kernel = np.ones((3, 3)) / 9.0
        out_reflect = mpdsp.convolve2d(img, kernel, border="reflect_101")
        out_constant = mpdsp.convolve2d(img, kernel, border="constant", pad=0.0)
        # Reflect border replicates the ones, so output stays uniform at 1.
        np.testing.assert_allclose(out_reflect, 1.0)
        # Constant-zero border puts zeros at the edges, so corner values drop.
        assert out_constant[0, 0] < out_constant[2, 2]

    def test_empty_image_raises(self):
        kernel = np.ones((3, 3))
        with pytest.raises(ValueError):
            mpdsp.convolve2d(np.zeros((0, 5)), kernel)

    def test_empty_kernel_raises(self):
        with pytest.raises(ValueError):
            mpdsp.convolve2d(np.ones((5, 5)), np.zeros((0, 3)))

    def test_unknown_dtype_raises(self):
        img = np.ones((4, 4))
        kernel = np.ones((3, 3)) / 9.0
        with pytest.raises(ValueError):
            mpdsp.convolve2d(img, kernel, dtype="not_a_dtype")


class TestConvolve2dDtypeDispatch:
    @pytest.mark.parametrize("dtype", [
        "reference", "gpu_baseline", "ml_hw", "cf24", "half",
        "posit_full", "tiny_posit",
    ])
    def test_runs_under_each_dtype(self, dtype):
        img = mpdsp.gradient_horizontal(8, 8)
        kernel = np.ones((3, 3)) / 9.0
        out = mpdsp.convolve2d(img, kernel, dtype=dtype)
        assert out.shape == img.shape
        assert out.dtype == np.float64
        assert np.all(np.isfinite(out))

    def test_reduced_precision_close_to_reference(self):
        """For a smooth input and small kernel, `half` output tracks the
        reference within a few percent."""
        img = mpdsp.gaussian_blob(16, 16, sigma=3.0)
        kernel = np.ones((3, 3)) / 9.0
        ref = mpdsp.convolve2d(img, kernel, dtype="reference")
        alt = mpdsp.convolve2d(img, kernel, dtype="half")
        err = np.max(np.abs(ref - alt)) / (np.max(np.abs(ref)) + 1e-12)
        assert err < 0.02
