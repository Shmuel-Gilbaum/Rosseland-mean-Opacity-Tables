"""The compiled kernels against the references they were transcribed from.

These two checks are the reason the compiled path can be trusted. A rewrite
that is merely plausible is worth nothing; only agreement with the reference is.
"""
import numpy as np
import pytest

from rm_tables.sources import _compiled as K
from rm_tables.sources import _semenov_fit as F
from rm_tables.sources import semenov


def test_the_compiled_semenov_is_bit_identical_to_the_port():
    """Not 'close'. Identical. The only changes numba forced were replacing
    np.polyval with a Horner loop and reversing the coefficients once."""
    eG = semenov._gas_grid()
    exact = 0
    n = 0
    for rho in (1e-18, 1e-16, 1e-14, 1e-11, 1e-8):
        for T in np.linspace(5.0, 9990.0, 400):
            a = F.cop(F.eR, eG, rho, T)
            b = K.semenov_kappa(K.ED_HI_FIRST, eG, rho, T, 1.0)
            n += 1
            if a == b:
                exact += 1
    assert exact == n, f"only {exact} of {n} identical"


def test_the_metallicity_multiply_reaches_dust_and_not_gas():
    eG = semenov._gas_grid()
    rho = 1e-14
    for T, expected in ((500.0, 2.0), (3000.0, 1.0)):
        one = K.semenov_kappa(K.ED_HI_FIRST, eG, rho, T, 1.0)
        two = K.semenov_kappa(K.ED_HI_FIRST, eG, rho, T, 2.0)
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
    """numba writes its cache beside the source file. A package installed
    read-only offers nowhere, and numba raises when the decorator RUNS, which
    is at import: the whole package became unimportable rather than slower.
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
