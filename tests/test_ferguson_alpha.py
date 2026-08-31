"""The alpha-enhanced Ferguson sets, and what selecting one is allowed to move.

Alpha enhancement raises oxygen, magnesium and silicon against iron at a fixed
total metal fraction. No pair of `X` and `Z` reaches it, which is why the sets
ship rather than being scaled from one another.
"""
import warnings

import numpy as np
import pytest

import rm_tables
from rm_tables.coverage import CoverageError
from rm_tables.sources import ferguson


def test_six_enhancements_are_tabulated_and_evenly_spaced():
    a = ferguson.alphas()
    assert a.shape == (6,)
    assert np.allclose(a, [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    assert np.allclose(np.diff(a), 0.2)


def test_every_set_is_on_one_composition_grid():
    """The six are separate calculations, so nothing guarantees a shared grid
    until it is checked. Interpolating in hydrogen and metals assumes it."""
    X0, Z0 = ferguson.compositions()
    T0, R0 = ferguson.axes()
    for a in list(ferguson.alphas()) + [None]:
        X, Z = ferguson.compositions(a)
        T, R = ferguson.axes(a)
        assert np.array_equal(np.sort(X), np.sort(X0))
        assert np.array_equal(np.sort(Z), np.sort(Z0))
        assert np.array_equal(T, T0) and np.array_equal(R, R0)


def test_the_dust_opacity_falls_as_alpha_rises():
    """Enhancement holds the metal fraction fixed, so more oxygen, magnesium
    and silicon means less iron, and iron is a major grain opacity carrier."""
    log_T, _ = ferguson.axes()
    dust = (log_T >= 2.70) & (log_T < 3.00)
    medians = [np.median(ferguson.table_at(0.7, 0.02, alpha=a)[dust])
               for a in ferguson.alphas()]
    assert np.all(np.diff(medians) < 0), medians
    assert medians[0] - medians[-1] == pytest.approx(0.315, abs=0.01)


def test_an_untabulated_enhancement_raises_rather_than_interpolating():
    """The six are separate calculations at each enhancement, not points on an
    axis this package interpolates."""
    with pytest.raises(CoverageError) as e:
        rm_tables.opacity(cold="ferguson", alpha=0.3)
    assert "0.2" in str(e.value) and "0.4" in str(e.value)


def test_an_enhancement_outside_the_tabulated_range_raises():
    for bad in (-0.4, 1.0):
        with pytest.raises(CoverageError) as e:
            rm_tables.opacity(cold="ferguson", alpha=bad)
        assert "-0.2" in str(e.value) and "0.8" in str(e.value)


def test_an_enhancement_given_to_semenov_warns_and_is_ignored():
    """Semenov's coefficients carry no alpha enhancement. Accepting the
    argument silently would report an enhancement that never reached a table."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        k = rm_tables.opacity(cold="semenov", alpha=0.6)
    mine = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(mine) == 1 and "ignored" in str(mine[0].message)
    assert k.provenance["cold_set"] == "semenov"
    plain = rm_tables.opacity(cold="semenov")
    assert k(800.0, 1e-12) == plain(800.0, 1e-12)


def test_no_enhancement_selects_the_set_that_shipped_before_the_others():
    """`alpha=None` is the Grevesse & Noels 1993 set, at different metal ratios
    from the six. It must still read the file it always read."""
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(ferguson.__file__)))
    d = np.load(os.path.join(here, "data", "ferguson_f05_g93.npz"))
    X, Z, log_T, log_R, K = (d[k].astype(float)
                             for k in ("X", "Z", "log_T", "log_R", "kappa"))
    i = int(np.argmin(np.abs(X - 0.7) + np.abs(Z - 0.02)))
    block = K[i][np.ix_(np.argsort(log_T), np.argsort(log_R))]
    assert np.allclose(ferguson.table_at(0.7, 0.02, alpha=None), block)
    assert ferguson.set_name(None) == "f05.g93"


def test_the_provenance_names_the_published_archive():
    for a, name in [(None, "f05.g93"), (-0.2, "f05.gs98-.2"),
                    (0.0, "f05.gs98"), (0.6, "f05.gs98+.6")]:
        p = rm_tables.opacity(cold="ferguson", alpha=a).provenance
        assert p["cold_set"] == name


def test_alpha_does_not_reach_above_the_handover():
    """Above 10,000 K the hot source answers alone, so the cold set must make
    no difference there. `opal_set` is what selects the mixture that high."""
    lo = rm_tables.opacity(cold="ferguson", alpha=-0.2)
    hi = rm_tables.opacity(cold="ferguson", alpha=0.8)
    for T, rho in ((1e5, 1e-10), (1e6, 1e-8)):
        assert lo(T, rho) == hi(T, rho)
    assert lo(800.0, 1e-12) != hi(800.0, 1e-12)
