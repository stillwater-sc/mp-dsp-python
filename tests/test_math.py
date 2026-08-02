"""Tests for math/ bindings (gap-analysis Phase 5 / #113):
polynomial ops, quadratic solver, elliptic integral, RootFinder.
"""

import math

import numpy as np
import pytest

mpdsp = pytest.importorskip("mpdsp", reason="mpdsp C++ module not built")
if not mpdsp.HAS_CORE:
    pytest.skip("mpdsp._core not available", allow_module_level=True)


class TestEvaluatePolynomial:
    def test_constant_polynomial(self):
        # p(x) = 3
        assert mpdsp.evaluate_polynomial(np.array([3.0]), 5.0) == 3.0

    def test_linear_polynomial(self):
        # p(x) = 2 + 3x
        coeffs = np.array([2.0, 3.0])
        assert mpdsp.evaluate_polynomial(coeffs, 0.0) == 2.0
        assert mpdsp.evaluate_polynomial(coeffs, 1.0) == 5.0
        assert mpdsp.evaluate_polynomial(coeffs, 10.0) == 32.0

    def test_quadratic_polynomial(self):
        # p(x) = 1 + 2x + x^2 = (1+x)^2
        coeffs = np.array([1.0, 2.0, 1.0])
        for x in [-2.0, -1.0, 0.0, 1.0, 3.0]:
            expected = (1 + x) ** 2
            assert mpdsp.evaluate_polynomial(coeffs, x) == pytest.approx(expected)

    def test_matches_numpy_polyval(self):
        # numpy.polyval expects DESCENDING order; our convention is ascending.
        rng = np.random.default_rng(0)
        coeffs = rng.standard_normal(6)
        x = 1.5
        expected = np.polyval(coeffs[::-1], x)
        assert mpdsp.evaluate_polynomial(coeffs, x) == pytest.approx(expected)

    def test_empty_coeffs_returns_zero(self):
        assert mpdsp.evaluate_polynomial(np.array([]), 5.0) == 0.0

    def test_complex_x_returns_complex(self):
        # p(x) = 1 + x^2 at x=i gives 1 + i^2 = 0
        coeffs = np.array([1.0, 0.0, 1.0])
        result = mpdsp.evaluate_polynomial(coeffs, 1j)
        assert isinstance(result, complex)
        assert result == pytest.approx(0.0)

    def test_complex_x_with_real_coeffs(self):
        # p(x) = 1 + x at x=(1+j) gives (2+j)
        coeffs = np.array([1.0, 1.0])
        result = mpdsp.evaluate_polynomial(coeffs, 1.0 + 1.0j)
        assert result == pytest.approx(2.0 + 1.0j)


class TestMultiplyPolynomials:
    def test_matches_numpy_convolve(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal(5)
        b = rng.standard_normal(4)
        expected = np.convolve(a, b, mode="full")
        result = mpdsp.multiply_polynomials(a, b)
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected)

    def test_degree_addition(self):
        # (1 + x) * (1 + x) = 1 + 2x + x^2 (degree 2)
        a = np.array([1.0, 1.0])
        b = np.array([1.0, 1.0])
        result = mpdsp.multiply_polynomials(a, b)
        np.testing.assert_allclose(result, [1.0, 2.0, 1.0])

    def test_empty_input_gives_empty(self):
        assert mpdsp.multiply_polynomials(np.array([]), np.array([1.0])).shape == (0,)
        assert mpdsp.multiply_polynomials(np.array([1.0]), np.array([])).shape == (0,)

    def test_zero_polynomial(self):
        # p * 0 = 0 (in the polynomial sense, all coefficients zero)
        result = mpdsp.multiply_polynomials(np.array([1.0, 2.0]), np.array([0.0]))
        np.testing.assert_allclose(result, [0.0, 0.0])


class TestSolveQuadratic:
    def test_real_distinct_roots(self):
        # x^2 - 3x + 2 = 0 -> x = 1, 2
        r1, r2 = mpdsp.solve_quadratic(1.0, -3.0, 2.0)
        # Roots come back in any order; sort by real part for comparison.
        roots = sorted([r1, r2], key=lambda z: z.real)
        assert roots[0] == pytest.approx(1.0 + 0j)
        assert roots[1] == pytest.approx(2.0 + 0j)

    def test_complex_roots(self):
        # x^2 + 1 = 0 -> x = +/- i
        r1, r2 = mpdsp.solve_quadratic(1.0, 0.0, 1.0)
        imags = sorted([r1.imag, r2.imag])
        assert imags[0] == pytest.approx(-1.0)
        assert imags[1] == pytest.approx(+1.0)
        assert abs(r1.real) < 1e-12
        assert abs(r2.real) < 1e-12

    def test_repeated_root(self):
        # (x - 5)^2 = x^2 - 10x + 25
        r1, r2 = mpdsp.solve_quadratic(1.0, -10.0, 25.0)
        assert r1 == pytest.approx(5.0 + 0j)
        assert r2 == pytest.approx(5.0 + 0j)

    def test_solve_quadratic_1_vs_2(self):
        # r1 = (-b + sqrt(D)) / 2a; r2 = (-b - sqrt(D)) / 2a
        a, b, c = 2.0, -1.0, -3.0
        r1 = mpdsp.solve_quadratic_1(a, b, c)
        r2 = mpdsp.solve_quadratic_2(a, b, c)
        # Sum of roots = -b/a; product = c/a
        assert (r1 + r2).real == pytest.approx(-b / a)
        assert (r1 * r2).real == pytest.approx(c / a)

    def test_rejects_a_zero(self):
        with pytest.raises(ValueError):
            mpdsp.solve_quadratic(0.0, 1.0, 2.0)
        with pytest.raises(ValueError):
            mpdsp.solve_quadratic_1(0.0, 1.0, 2.0)
        with pytest.raises(ValueError):
            mpdsp.solve_quadratic_2(0.0, 1.0, 2.0)


class TestEllipticK:
    def test_K_at_zero(self):
        # K(0) = pi/2
        assert mpdsp.elliptic_K(0.0) == pytest.approx(math.pi / 2)

    def test_K_at_half(self):
        # K(1/sqrt(2)) ~ 1.8540746773 (well-known value)
        assert mpdsp.elliptic_K(1.0 / math.sqrt(2)) == pytest.approx(
            1.8540746773013719, rel=1e-9)

    def test_K_at_small_positive(self):
        # K(0.5) ~ 1.6857503548
        assert mpdsp.elliptic_K(0.5) == pytest.approx(1.6857503548125963, rel=1e-9)

    def test_K_monotonic_increasing(self):
        # K is monotone increasing on [0, 1).
        prev = mpdsp.elliptic_K(0.0)
        for k in np.linspace(0.05, 0.95, 10):
            curr = mpdsp.elliptic_K(k)
            assert curr > prev
            prev = curr

    def test_rejects_out_of_domain(self):
        # K(1) is unbounded; k > 1 is out of the real-valued domain.
        with pytest.raises(ValueError):
            mpdsp.elliptic_K(1.0)
        with pytest.raises(ValueError):
            mpdsp.elliptic_K(1.5)
        with pytest.raises(ValueError):
            mpdsp.elliptic_K(-0.1)


class TestRootFinder:
    def test_construction_defaults(self):
        rf = mpdsp.RootFinder()
        assert rf.degree == 0
        assert rf.max_degree == 32

    def test_solve_without_coefficients_raises(self):
        rf = mpdsp.RootFinder()
        with pytest.raises(RuntimeError):
            rf.solve()

    def test_roots_without_solve_raises(self):
        rf = mpdsp.RootFinder()
        with pytest.raises(RuntimeError):
            rf.roots()

    def test_x_squared_minus_one(self):
        # x^2 - 1 = 0 -> roots +/- 1
        rf = mpdsp.RootFinder()
        rf.set_coefficients(np.array([-1.0, 0.0, 1.0], dtype=np.complex128))
        rf.solve()
        assert rf.degree == 2
        roots = rf.roots()
        assert roots.shape == (2,)
        # Sort by real part; roots must be near ±1 with zero imaginary.
        real_parts = sorted(r.real for r in roots)
        assert real_parts[0] == pytest.approx(-1.0)
        assert real_parts[1] == pytest.approx(+1.0)
        for r in roots:
            assert abs(r.imag) < 1e-9

    def test_x_cubed_minus_one(self):
        # x^3 - 1 = 0: roots are 1, cis(120deg), cis(240deg).
        rf = mpdsp.RootFinder()
        rf.set_coefficients(np.array([-1.0, 0.0, 0.0, 1.0],
                                       dtype=np.complex128))
        rf.solve()
        roots = rf.roots()
        assert roots.shape == (3,)
        # All roots have unit magnitude.
        for r in roots:
            assert abs(r) == pytest.approx(1.0, abs=1e-9)

    def test_complex_coefficients(self):
        # (x - (1+j)) * (x - (1-j)) = x^2 - 2x + 2
        rf = mpdsp.RootFinder()
        rf.set_coefficients(np.array([2+0j, -2+0j, 1+0j], dtype=np.complex128))
        rf.solve()
        roots = rf.roots()
        # Roots should be {1+j, 1-j}
        imags = sorted(r.imag for r in roots)
        assert imags[0] == pytest.approx(-1.0)
        assert imags[1] == pytest.approx(+1.0)
        for r in roots:
            assert r.real == pytest.approx(1.0)

    def test_empty_coefficients_raises(self):
        rf = mpdsp.RootFinder()
        with pytest.raises(ValueError):
            rf.set_coefficients(np.array([], dtype=np.complex128))

    def test_degree_exceeds_max_raises(self):
        rf = mpdsp.RootFinder()
        oversized = np.zeros(35, dtype=np.complex128)   # degree 34, > 32
        with pytest.raises(ValueError):
            rf.set_coefficients(oversized)

    def test_polish_and_sort_flags_dont_crash(self):
        rf = mpdsp.RootFinder()
        rf.set_coefficients(np.array([-1.0, 0.0, 1.0], dtype=np.complex128))
        # Try all four combinations of polish/sort.
        for polish in [True, False]:
            for sort in [True, False]:
                rf.solve(polish=polish, sort=sort)
                roots = rf.roots()
                real_parts = sorted(r.real for r in roots)
                assert real_parts[0] == pytest.approx(-1.0, abs=1e-9)
                assert real_parts[1] == pytest.approx(+1.0, abs=1e-9)

    def test_degree_updated_by_set_coefficients(self):
        rf = mpdsp.RootFinder()
        rf.set_coefficients(np.array([1, 2, 3, 4], dtype=np.complex128))
        assert rf.degree == 3
        rf.set_coefficients(np.array([1, 2], dtype=np.complex128))
        assert rf.degree == 1
