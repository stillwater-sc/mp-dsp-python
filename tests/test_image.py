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
