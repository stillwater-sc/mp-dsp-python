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


# ---------------------------------------------------------------------------
# Separable filter
# ---------------------------------------------------------------------------


class TestSeparableFilter:
    def test_identity_kernels_round_trip(self):
        img = mpdsp.checkerboard(8, 8, block_size=2)
        one = np.array([1.0])
        out = mpdsp.separable_filter(img, one, one)
        np.testing.assert_array_equal(out, img)

    def test_matches_convolve2d_for_outer_product(self):
        """Separable with row/col kernels should match convolve2d with the
        outer-product kernel to within floating-point tolerance."""
        img = mpdsp.gaussian_blob(16, 16, sigma=3.0)
        r = np.array([0.25, 0.5, 0.25])
        c = np.array([1.0, 2.0, 1.0])
        sep = mpdsp.separable_filter(img, r, c)
        ref = mpdsp.convolve2d(img, np.outer(c, r))
        np.testing.assert_allclose(sep, ref, atol=1e-12)

    def test_empty_image_raises(self):
        with pytest.raises(ValueError):
            mpdsp.separable_filter(np.zeros((0, 5)),
                                    np.array([1.0]), np.array([1.0]))

    def test_empty_kernel_raises(self):
        with pytest.raises(ValueError):
            mpdsp.separable_filter(np.ones((5, 5)),
                                    np.array([]), np.array([1.0]))


# ---------------------------------------------------------------------------
# Gaussian + box blur
# ---------------------------------------------------------------------------


class TestGaussianBlur:
    def test_shape_preserved(self):
        img = mpdsp.gaussian_blob(16, 16, sigma=3.0)
        out = mpdsp.gaussian_blur(img, sigma=1.5)
        assert out.shape == img.shape

    def test_blurring_reduces_high_frequency(self):
        """A checkerboard has max high-frequency content; blurring should
        reduce the peak-to-peak range of interior pixels."""
        img = mpdsp.checkerboard(16, 16, block_size=1)
        blurred = mpdsp.gaussian_blur(img, sigma=1.0)
        interior = blurred[2:-2, 2:-2]
        assert interior.max() < 1.0
        assert interior.min() > 0.0

    def test_explicit_radius(self):
        img = mpdsp.gaussian_blob(24, 24, sigma=3.0)
        out = mpdsp.gaussian_blur(img, sigma=1.0, radius=5)
        assert out.shape == img.shape

    def test_non_positive_sigma_raises(self):
        with pytest.raises(ValueError):
            mpdsp.gaussian_blur(np.ones((5, 5)), sigma=0.0)

    def test_empty_image_raises(self):
        with pytest.raises(ValueError):
            mpdsp.gaussian_blur(np.zeros((0, 5)), sigma=1.0)


class TestBoxBlur:
    def test_uniform_input_unchanged(self):
        """Box blur of a uniform image with reflect_101 border is the same
        uniform image (average of equal values is the same value)."""
        img = np.full((10, 10), 0.42)
        out = mpdsp.box_blur(img, size=3)
        np.testing.assert_allclose(out, 0.42)

    def test_reduces_variance_on_noise(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((32, 32))
        out = mpdsp.box_blur(img, size=5)
        assert out.std() < img.std()

    def test_zero_size_raises(self):
        with pytest.raises(ValueError):
            mpdsp.box_blur(np.ones((5, 5)), size=0)


# ---------------------------------------------------------------------------
# Edge detection — Sobel / Prewitt / gradient_magnitude / Canny
# ---------------------------------------------------------------------------


class TestSobel:
    def _vertical_edge(self):
        """Half-white / half-black image with a vertical edge at x=16."""
        return mpdsp.rectangle(32, 32, y=0, x=16, h=32, w=16)

    def test_sobel_x_detects_vertical_edge(self):
        img = self._vertical_edge()
        sx = mpdsp.sobel_x(img)
        # sobel_x responds to horizontal derivative; a vertical edge
        # at column 16 produces non-zero values along that column.
        assert np.max(np.abs(sx[:, 15:18])) > 0.5
        # Interior of the uniform regions should be ~0.
        assert np.max(np.abs(sx[10:20, 2:8])) < 1e-9
        assert np.max(np.abs(sx[10:20, 24:30])) < 1e-9

    def test_sobel_y_zero_on_vertical_edge(self):
        """A purely vertical edge has no vertical derivative — sobel_y → 0."""
        img = self._vertical_edge()
        sy = mpdsp.sobel_y(img)
        assert np.max(np.abs(sy)) < 1e-9

    def test_sobel_y_detects_horizontal_edge(self):
        img = mpdsp.rectangle(32, 32, y=16, x=0, h=16, w=32)
        sy = mpdsp.sobel_y(img)
        assert np.max(np.abs(sy[15:18, :])) > 0.5

    def test_empty_image_raises(self):
        with pytest.raises(ValueError):
            mpdsp.sobel_x(np.zeros((0, 5)))


class TestPrewitt:
    def test_prewitt_x_detects_vertical_edge(self):
        img = mpdsp.rectangle(32, 32, y=0, x=16, h=32, w=16)
        px = mpdsp.prewitt_x(img)
        assert np.max(np.abs(px[:, 15:18])) > 0.5

    def test_prewitt_y_zero_on_vertical_edge(self):
        img = mpdsp.rectangle(32, 32, y=0, x=16, h=32, w=16)
        py = mpdsp.prewitt_y(img)
        assert np.max(np.abs(py)) < 1e-9


class TestGradientMagnitude:
    def test_zeros_input_zero_output(self):
        z = np.zeros((8, 8))
        gm = mpdsp.gradient_magnitude(z, z)
        np.testing.assert_array_equal(gm, 0.0)

    def test_sqrt_sum_of_squares(self):
        gx = np.array([[3.0, 0.0], [0.0, 4.0]])
        gy = np.array([[4.0, 0.0], [0.0, 3.0]])
        gm = mpdsp.gradient_magnitude(gx, gy)
        # sqrt(3^2 + 4^2) = 5 on both diagonal elements.
        assert gm[0, 0] == pytest.approx(5.0)
        assert gm[1, 1] == pytest.approx(5.0)
        assert gm[0, 1] == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mpdsp.gradient_magnitude(np.zeros((4, 4)), np.zeros((4, 5)))


class TestCanny:
    def test_binary_output(self):
        img = mpdsp.rectangle(32, 32, y=0, x=16, h=32, w=16)
        edges = mpdsp.canny(img, low_threshold=0.1, high_threshold=0.3)
        assert set(np.unique(edges).tolist()) == {0.0, 1.0}

    def test_edges_appear_near_true_edge(self):
        """For a half-image, Canny should place edge pixels near x=16."""
        img = mpdsp.rectangle(32, 32, y=0, x=16, h=32, w=16)
        edges = mpdsp.canny(img, low_threshold=0.1, high_threshold=0.3)
        edge_cols = np.where(edges > 0.5)[1]
        assert len(edge_cols) > 0
        # Edges cluster near column 16.
        assert abs(np.median(edge_cols) - 16) <= 2

    def test_no_edges_on_uniform_image(self):
        img = np.full((16, 16), 0.5)
        edges = mpdsp.canny(img, low_threshold=0.1, high_threshold=0.3)
        assert edges.sum() == 0.0

    def test_invalid_thresholds_raise(self):
        img = np.ones((8, 8))
        # low > high
        with pytest.raises(ValueError):
            mpdsp.canny(img, low_threshold=0.5, high_threshold=0.1)
        # negative low
        with pytest.raises(ValueError):
            mpdsp.canny(img, low_threshold=-0.1, high_threshold=0.3)

    def test_non_positive_sigma_raises(self):
        img = np.ones((8, 8))
        with pytest.raises(ValueError):
            mpdsp.canny(img, low_threshold=0.1, high_threshold=0.3, sigma=0.0)


# ---------------------------------------------------------------------------
# Processor dtype dispatch — one parametrized sanity test covers all 9
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [
    "reference", "gpu_baseline", "ml_hw", "cf24", "half",
    "posit_full", "tiny_posit",
])
class TestProcessorDtypeDispatch:
    def test_gaussian_blur_runs(self, dtype):
        img = mpdsp.gaussian_blob(16, 16, sigma=3.0)
        out = mpdsp.gaussian_blur(img, sigma=1.0, dtype=dtype)
        assert out.shape == img.shape
        assert np.all(np.isfinite(out))

    def test_sobel_x_runs(self, dtype):
        img = mpdsp.rectangle(16, 16, y=0, x=8, h=16, w=8)
        out = mpdsp.sobel_x(img, dtype=dtype)
        assert out.shape == img.shape

    def test_gradient_magnitude_runs(self, dtype):
        gx = mpdsp.gaussian_blob(8, 8, sigma=2.0)
        gy = mpdsp.gaussian_blob(8, 8, sigma=2.0)
        out = mpdsp.gradient_magnitude(gx, gy, dtype=dtype)
        assert out.shape == gx.shape


def test_canny_posit_differs_from_reference():
    """Acceptance criterion from issue #7: canny(..., dtype='tiny_posit')
    produces measurably different edges than reference."""
    rng = np.random.default_rng(3)
    img = mpdsp.gaussian_blob(32, 32, sigma=4.0) + 0.02 * rng.standard_normal((32, 32))
    ref   = mpdsp.canny(img, low_threshold=0.05, high_threshold=0.15, dtype="reference")
    posit = mpdsp.canny(img, low_threshold=0.05, high_threshold=0.15, dtype="tiny_posit")
    # Each is a binary edge map; differences are pixels where one has an
    # edge and the other doesn't.
    different_pixels = int((ref != posit).sum())
    assert different_pixels > 0


# ---------------------------------------------------------------------------
# Morphology — structuring element constructors
# ---------------------------------------------------------------------------


class TestElementConstructors:
    def test_rect_element_shape_and_dtype(self):
        elem = mpdsp.make_rect_element(3, 5)
        assert elem.shape == (3, 5)
        assert elem.dtype == np.bool_

    def test_rect_element_all_true(self):
        elem = mpdsp.make_rect_element(4, 4)
        assert elem.all()

    def test_cross_element_shape(self):
        elem = mpdsp.make_cross_element(5)
        assert elem.shape == (5, 5)
        assert elem.dtype == np.bool_

    def test_cross_element_center_row_col_true(self):
        elem = mpdsp.make_cross_element(5)
        # Center row (row 2) and center column (col 2) are all True.
        assert elem[2, :].all()
        assert elem[:, 2].all()
        # Off-cross corners are False.
        assert not elem[0, 0]
        assert not elem[0, 4]
        assert not elem[4, 0]
        assert not elem[4, 4]

    def test_ellipse_element_shape(self):
        elem = mpdsp.make_ellipse_element(5)
        assert elem.shape == (5, 5)
        assert elem.dtype == np.bool_

    def test_ellipse_element_center_true(self):
        elem = mpdsp.make_ellipse_element(7)
        # Center pixel must be inside the ellipse.
        assert elem[3, 3]

    def test_ellipse_element_corners_false(self):
        elem = mpdsp.make_ellipse_element(7)
        assert not elem[0, 0]
        assert not elem[0, 6]
        assert not elem[6, 0]
        assert not elem[6, 6]

    @pytest.mark.parametrize("ctor,args", [
        ("make_rect_element", (0, 3)),
        ("make_rect_element", (3, 0)),
        ("make_cross_element", (0,)),
        ("make_ellipse_element", (0,)),
    ])
    def test_zero_dims_rejected(self, ctor, args):
        with pytest.raises(ValueError):
            getattr(mpdsp, ctor)(*args)


# ---------------------------------------------------------------------------
# Morphology — dilate / erode / open / close / gradient / tophat / blackhat
# ---------------------------------------------------------------------------


class TestDilateErode:
    def test_dilate_grows_foreground(self):
        """Dilation expands bright regions under a rect element."""
        img = mpdsp.circle(21, 21, radius=3)
        elem = mpdsp.make_rect_element(3, 3)
        out = mpdsp.dilate(img, elem)
        assert (out > 0.5).sum() > (img > 0.5).sum()

    def test_erode_shrinks_foreground(self):
        """Erosion shrinks bright regions."""
        img = mpdsp.circle(21, 21, radius=5)
        elem = mpdsp.make_rect_element(3, 3)
        out = mpdsp.erode(img, elem)
        assert (out > 0.5).sum() < (img > 0.5).sum()

    def test_dilate_identity_element_unchanged(self):
        """A 1x1 element (single true pixel) is the identity for dilation."""
        img = mpdsp.gaussian_blob(9, 9, sigma=2.0)
        elem = mpdsp.make_rect_element(1, 1)
        out = mpdsp.dilate(img, elem)
        np.testing.assert_array_equal(out, img)

    def test_erode_identity_element_unchanged(self):
        img = mpdsp.gaussian_blob(9, 9, sigma=2.0)
        elem = mpdsp.make_rect_element(1, 1)
        out = mpdsp.erode(img, elem)
        np.testing.assert_array_equal(out, img)

    def test_empty_image_raises(self):
        elem = mpdsp.make_rect_element(3, 3)
        with pytest.raises(ValueError):
            mpdsp.dilate(np.zeros((0, 5)), elem)

    def test_empty_element_raises(self):
        img = mpdsp.circle(11, 11, radius=3)
        with pytest.raises(ValueError):
            mpdsp.dilate(img, np.zeros((0, 3), dtype=bool))


class TestMorphologicalOpenClose:
    def test_open_removes_small_peaks(self):
        """Opening is erode-then-dilate; it removes isolated salt peaks."""
        img = np.zeros((15, 15))
        img[7, 7] = 1.0  # isolated peak
        elem = mpdsp.make_rect_element(3, 3)
        out = mpdsp.morphological_open(img, elem)
        assert out[7, 7] == 0.0

    def test_close_fills_small_holes(self):
        """Closing is dilate-then-erode; it fills isolated pepper holes."""
        img = np.ones((15, 15))
        img[7, 7] = 0.0  # isolated hole
        elem = mpdsp.make_rect_element(3, 3)
        out = mpdsp.morphological_close(img, elem)
        assert out[7, 7] == 1.0

    def test_open_then_close_roughly_idempotent(self):
        """Opening followed by closing should largely preserve a smooth
        shape like a circle (aside from the edge-pixel effects)."""
        img = mpdsp.circle(31, 31, radius=8)
        elem = mpdsp.make_rect_element(3, 3)
        oc = mpdsp.morphological_close(mpdsp.morphological_open(img, elem), elem)
        # Majority of pixels unchanged.
        diff = np.abs(oc - img)
        assert (diff < 1e-9).mean() > 0.9


class TestMorphologicalGradientTophatBlackhat:
    def test_gradient_highlights_edges(self):
        """Morphological gradient = dilate - erode; it's positive only at
        edges of foreground regions."""
        img = mpdsp.circle(21, 21, radius=5)
        elem = mpdsp.make_rect_element(3, 3)
        out = mpdsp.morphological_gradient(img, elem)
        # Interior of the circle (far from edge): should be ~0.
        assert out[10, 10] < 1e-9
        # On the circle boundary (radius=5 from center): should be > 0.
        assert out[10, 5] > 0.5 or out[5, 10] > 0.5

    def test_tophat_extracts_small_bright(self):
        """White tophat = image - opening; extracts small bright features
        that opening removes."""
        base = np.zeros((15, 15))
        base[7, 7] = 1.0
        elem = mpdsp.make_rect_element(3, 3)
        out = mpdsp.tophat(base, elem)
        # The isolated peak survives tophat (opening removes it, so
        # image - opening preserves it).
        assert out[7, 7] == pytest.approx(1.0)

    def test_blackhat_extracts_small_dark(self):
        """Black tophat = closing - image; extracts small dark features
        that closing fills."""
        base = np.ones((15, 15))
        base[7, 7] = 0.0
        elem = mpdsp.make_rect_element(3, 3)
        out = mpdsp.blackhat(base, elem)
        # The isolated hole becomes 1 in blackhat (closing fills it).
        assert out[7, 7] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Morphology — dtype dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [
    "reference", "gpu_baseline", "ml_hw", "cf24", "half",
    "posit_full", "tiny_posit",
])
def test_dilate_runs_under_each_dtype(dtype):
    img = mpdsp.circle(11, 11, radius=3)
    elem = mpdsp.make_rect_element(3, 3)
    out = mpdsp.dilate(img, elem, dtype=dtype)
    assert out.shape == img.shape
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Multi-channel — rgb_to_gray + apply_per_channel
# ---------------------------------------------------------------------------


class TestRgbToGray:
    def test_pure_red_gives_luminance_weight(self):
        """Y = 0.299*R + 0.587*G + 0.114*B. Pure R=1 yields 0.299."""
        r = np.ones((4, 4))
        g = np.zeros((4, 4))
        b = np.zeros((4, 4))
        gray = mpdsp.rgb_to_gray(r, g, b)
        assert gray.shape == r.shape
        np.testing.assert_allclose(gray, 0.299)

    def test_pure_green_gives_luminance_weight(self):
        r = np.zeros((4, 4))
        g = np.ones((4, 4))
        b = np.zeros((4, 4))
        gray = mpdsp.rgb_to_gray(r, g, b)
        np.testing.assert_allclose(gray, 0.587)

    def test_pure_blue_gives_luminance_weight(self):
        r = np.zeros((4, 4))
        g = np.zeros((4, 4))
        b = np.ones((4, 4))
        gray = mpdsp.rgb_to_gray(r, g, b)
        np.testing.assert_allclose(gray, 0.114)

    def test_white_input_white_output(self):
        """R=G=B=1 -> Y = 0.299 + 0.587 + 0.114 = 1.0 exactly."""
        r = np.ones((4, 4))
        gray = mpdsp.rgb_to_gray(r, r, r)
        np.testing.assert_allclose(gray, 1.0)

    def test_shape_mismatch_raises(self):
        r = np.ones((4, 4))
        g = np.ones((5, 4))  # wrong shape
        b = np.ones((4, 4))
        with pytest.raises(ValueError):
            mpdsp.rgb_to_gray(r, g, b)


class TestApplyPerChannel:
    def test_applies_to_each_plane_independently(self):
        r = np.ones((3, 3))
        g = np.ones((3, 3)) * 2.0
        b = np.ones((3, 3)) * 3.0
        out_r, out_g, out_b = mpdsp.apply_per_channel(r, g, b,
                                                       lambda p: p * 10.0)
        np.testing.assert_allclose(out_r, 10.0)
        np.testing.assert_allclose(out_g, 20.0)
        np.testing.assert_allclose(out_b, 30.0)

    def test_composes_with_mpdsp_processors(self):
        """Common usage: per-channel Gaussian blur."""
        r = mpdsp.gaussian_blob(16, 16, sigma=3.0)
        g = mpdsp.gaussian_blob(16, 16, sigma=4.0)
        b = mpdsp.gaussian_blob(16, 16, sigma=5.0)
        out_r, out_g, out_b = mpdsp.apply_per_channel(
            r, g, b, lambda p: mpdsp.gaussian_blur(p, sigma=1.0))
        assert out_r.shape == r.shape
        assert out_g.shape == g.shape
        assert out_b.shape == b.shape

    def test_shape_mismatch_raises(self):
        r = np.ones((4, 4))
        g = np.ones((5, 4))
        b = np.ones((4, 4))
        with pytest.raises(ValueError):
            mpdsp.apply_per_channel(r, g, b, lambda p: p)


# ---------------------------------------------------------------------------
# File I/O — PGM, PPM, BMP round-trips
# ---------------------------------------------------------------------------


import os
import tempfile


class TestPgmIO:
    def test_round_trip_preserves_shape(self):
        img = mpdsp.gaussian_blob(16, 16, sigma=3.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.pgm")
            mpdsp.write_pgm(path, img)
            loaded = mpdsp.read_pgm(path)
        assert loaded.shape == img.shape
        assert loaded.dtype == np.float64

    def test_round_trip_within_quantization(self):
        """8-bit PGM loses ~1/255 ≈ 0.004 per pixel in quantization."""
        img = mpdsp.gradient_horizontal(32, 32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gradient.pgm")
            mpdsp.write_pgm(path, img, max_val=255)
            loaded = mpdsp.read_pgm(path)
        assert np.max(np.abs(img - loaded)) < 0.005

    def test_higher_max_val_reduces_error(self):
        img = mpdsp.gradient_horizontal(32, 32)
        with tempfile.TemporaryDirectory() as d:
            path8 = os.path.join(d, "8.pgm")
            path16 = os.path.join(d, "16.pgm")
            mpdsp.write_pgm(path8, img, max_val=255)
            mpdsp.write_pgm(path16, img, max_val=65535)
            loaded8 = mpdsp.read_pgm(path8)
            loaded16 = mpdsp.read_pgm(path16)
        err8 = np.max(np.abs(img - loaded8))
        err16 = np.max(np.abs(img - loaded16))
        assert err16 < err8

    def test_write_rejects_empty_image(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "empty.pgm")
            with pytest.raises(ValueError):
                mpdsp.write_pgm(path, np.zeros((0, 5)))

    def test_read_missing_file_raises(self):
        with pytest.raises(RuntimeError):
            mpdsp.read_pgm("/tmp/this_file_does_not_exist_xyz.pgm")


class TestPpmIO:
    def test_round_trip_three_channels(self):
        r = mpdsp.gradient_horizontal(16, 16)
        g = mpdsp.gradient_vertical(16, 16)
        b = mpdsp.gradient_radial(16, 16)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rgb.ppm")
            mpdsp.write_ppm(path, r, g, b)
            r2, g2, b2 = mpdsp.read_ppm(path)
        assert r2.shape == r.shape
        assert g2.shape == g.shape
        assert b2.shape == b.shape
        assert np.max(np.abs(r - r2)) < 0.005
        assert np.max(np.abs(g - g2)) < 0.005
        assert np.max(np.abs(b - b2)) < 0.005

    def test_shape_mismatch_raises(self):
        r = np.ones((4, 4))
        g = np.ones((5, 4))
        b = np.ones((4, 4))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.ppm")
            with pytest.raises(ValueError):
                mpdsp.write_ppm(path, r, g, b)


class TestBmpIO:
    def test_grayscale_round_trip(self):
        img = mpdsp.gaussian_blob(16, 16, sigma=3.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gray.bmp")
            mpdsp.write_bmp(path, img)
            r, g, b, is_grayscale = mpdsp.read_bmp(path)
        assert r.shape == img.shape
        # write_bmp(grayscale) stores R=G=B.
        np.testing.assert_allclose(r, g)
        np.testing.assert_allclose(r, b)
        assert np.max(np.abs(r - img)) < 0.005

    def test_rgb_round_trip(self):
        r = mpdsp.gradient_horizontal(12, 12)
        g = mpdsp.gradient_vertical(12, 12)
        b = mpdsp.gradient_radial(12, 12)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rgb.bmp")
            mpdsp.write_bmp_rgb(path, r, g, b)
            r2, g2, b2, is_grayscale = mpdsp.read_bmp(path)
        assert not is_grayscale
        assert np.max(np.abs(r - r2)) < 0.005
        assert np.max(np.abs(g - g2)) < 0.005
        assert np.max(np.abs(b - b2)) < 0.005

    def test_shape_mismatch_raises(self):
        r = np.ones((4, 4))
        g = np.ones((5, 4))
        b = np.ones((4, 4))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.bmp")
            with pytest.raises(ValueError):
                mpdsp.write_bmp_rgb(path, r, g, b)


# ---------------------------------------------------------------------------
# Image plotting helpers
# ---------------------------------------------------------------------------


class TestImagePlottingHelpers:
    def test_plot_image_returns_axes(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpdsp.image import plot_image

        img = mpdsp.gaussian_blob(16, 16, sigma=3.0)
        ax = plot_image(img, title="blob")
        assert ax is not None
        plt.close("all")

    def test_plot_image_grid_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpdsp.image import plot_image_grid

        imgs = [mpdsp.gaussian_blob(8, 8, sigma=s) for s in (1.0, 2.0, 3.0)]
        fig = plot_image_grid(imgs, titles=["s=1", "s=2", "s=3"], ncols=3)
        assert fig is not None
        assert len(fig.axes) >= 3
        plt.close(fig)

    def test_plot_image_grid_rejects_empty(self):
        from mpdsp.image import plot_image_grid
        with pytest.raises(ValueError):
            plot_image_grid([])

    def test_plot_pipeline_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpdsp.image import plot_pipeline

        clean = mpdsp.gaussian_blob(32, 32, sigma=4.0)
        noisy = mpdsp.add_noise(clean, stddev=0.1)
        blurred = mpdsp.gaussian_blur(noisy, sigma=1.0)
        fig = plot_pipeline([clean, noisy, blurred],
                             titles=["clean", "noisy", "blurred"])
        assert fig is not None
        plt.close(fig)

    def test_plot_pipeline_rejects_empty(self):
        from mpdsp.image import plot_pipeline
        with pytest.raises(ValueError):
            plot_pipeline([])
