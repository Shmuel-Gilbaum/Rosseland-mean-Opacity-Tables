"""The compiled routines against the published ones they were transcribed from.

These two checks are the reason the compiled routines can be trusted. A rewrite
that is merely plausible is worth nothing; only agreement with the published
routine is.
"""
import numpy as np
import pytest

from rm_tables.sources import _compiled as K
from rm_tables.sources import semenov

import _semenov_reference as F


def test_the_compiled_semenov_is_bit_identical_to_the_port():
    """Not 'close'. The same numbers. Compiling forced two changes and neither
    touches the arithmetic: the polynomial is evaluated term by term rather
    than by a NumPy call, and its coefficients are reversed once beforehand."""
    gas_grid = semenov._gas_grid()
    exact = 0
    n = 0
    for rho in (1e-18, 1e-16, 1e-14, 1e-11, 1e-8):
        for T in np.linspace(5.0, 9990.0, 400):
            a = F.cop(F.eR, gas_grid, rho, T)
            b = K.semenov_kappa(K.DUST_COEFFS_HI_FIRST, gas_grid, rho, T, 1.0)
            n += 1
            if a == b:
                exact += 1
    assert exact == n, f"only {exact} of {n} identical"


def test_the_metallicity_multiply_reaches_dust_and_not_gas():
    gas_grid = semenov._gas_grid()
    rho = 1e-14
    for T, expected in ((500.0, 2.0), (3000.0, 1.0)):
        one = K.semenov_kappa(K.DUST_COEFFS_HI_FIRST, gas_grid, rho, T, 1.0)
        two = K.semenov_kappa(K.DUST_COEFFS_HI_FIRST, gas_grid, rho, T, 2.0)
        assert one > 0.0
        assert two / one == pytest.approx(expected, rel=1e-12)


def test_bilinear_reproduces_the_grid_at_its_own_nodes():
    x = np.array([0.0, 1.0, 2.5, 4.0])
    y = np.array([-2.0, 0.0, 3.0])
    block = np.arange(12.0).reshape(4, 3)
    for i, xv in enumerate(x):
        for j, yv in enumerate(y):
            assert K.bilinear(x, y, block, xv, yv) == pytest.approx(block[i, j])


def test_bilinear_holds_the_edge_outside_the_grid():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0])
    block = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert K.bilinear(x, y, block, -99.0, -99.0) == pytest.approx(1.0)
    assert K.bilinear(x, y, block, 99.0, 99.0) == pytest.approx(6.0)


def test_bilinear_is_exact_on_a_linear_function():
    x = np.linspace(0.0, 3.0, 4)
    y = np.linspace(-1.0, 2.0, 4)
    block = 2.0 * x[:, None] - 3.0 * y[None, :] + 0.5
    for xv, yv in ((0.7, 0.3), (2.2, -0.4), (1.0, 1.0)):
        assert K.bilinear(x, y, block, xv, yv) == pytest.approx(
            2.0 * xv - 3.0 * yv + 0.5)


def test_caching_degrades_instead_of_refusing_to_import():
    """numba saves its compiled code beside the source file so a later session
    need not compile again. A package installed somewhere read-only offers
    nowhere to save it, and numba refuses at the moment the package is
    imported, so the whole package became unusable rather than merely slower.
    """
    from rm_tables.sources._compiled import cached_njit

    def add(a, b):
        return a + b

    def boom(fn):
        raise RuntimeError("cannot cache function: no locator available")

    import numba
    real = numba.njit
    try:
        numba.njit = lambda *a, **k: (boom if k.get("cache") else real(*a, **k))
        from rm_tables.sources import _compiled
        _compiled.njit = lambda *a, **k: (boom if k.get("cache") else real(*a, **k))
        f = cached_njit(add)
        assert f(2.0, 3.0) == 5.0
    finally:
        numba.njit = real
        _compiled.njit = real
