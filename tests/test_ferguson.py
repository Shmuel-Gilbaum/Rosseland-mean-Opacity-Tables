"""The Ferguson source: its axes, its composition interpolation, its reach."""
import numpy as np
import pytest

import rm_tables
from rm_tables.coverage import CoverageError
from rm_tables.sources import ferguson, opal


def test_the_axes_are_the_published_ones():
    log_T, log_R = ferguson.axes()
    assert (log_T.min(), log_T.max()) == pytest.approx((2.70, 4.50))
    assert (log_R.min(), log_R.max()) == pytest.approx((-8.0, 1.0))
    assert log_T.size == 85 and log_R.size == 19


def test_there_are_155_compositions_and_helium_is_never_an_axis():
    X, Z = ferguson.compositions()
    assert X.size == Z.size == 155
    pairs = {(round(a, 6), round(b, 6)) for a, b in zip(X, Z)}
    assert len(pairs) == 155, "two tables would differ only in helium"


def test_a_tabulated_composition_is_returned_unchanged():
    """Interpolating onto a point that is in the table must not move it."""
    X, Z = ferguson.compositions()
    i = int(np.argmin(np.abs(X - 0.7) + np.abs(Z - 0.02)))
    got = ferguson.table_at(float(X[i]), float(Z[i]))
    from rm_tables.sources.ferguson import _load
    _, _, log_T, log_R, K = _load()
    want = K[i][np.ix_(np.argsort(log_T), np.argsort(log_R))]
    assert np.abs(got - want).max() == 0.0


def test_there_are_no_blank_entries_anywhere():
    """Ferguson tabulates its whole rectangle. OPAL leaves 42 of its entries
    blank at solar composition, so a caller cannot assume one behaves as the
    other."""
    _, _, _, _, K = ferguson._load()
    assert K.shape == (155, 85, 19)
    assert np.isfinite(K).all()
    assert np.isnan(opal.table_at(0.7, 0.02)).sum() == 42


def test_the_grid_returns_nan_outside_rather_than_extrapolating():
    out = ferguson.grid(np.array([2.0, 3.0]), np.array([-8.0, 5.0]), 0.7, 0.02)
    assert np.isnan(out[0, 0]), "2.0 is below Ferguson's temperature floor"
    assert np.isnan(out[1, 1]), "+5.0 is above Ferguson's density ceiling"
    assert out[1, 0] == pytest.approx(-1.606, abs=1e-3)


def test_it_reaches_the_top_of_the_density_axis_at_every_temperature():
    """Ferguson holds a value at log10 R = +1.0 over its whole temperature
    range. That is the reach Semenov's density-bounded gas grid lacks."""
    log_T, _ = ferguson.axes()
    top = ferguson.grid(log_T, np.array([1.0]), 0.7381, 0.0134)
    assert top.shape == (85, 1)
    assert np.isfinite(top).all()


def test_ferguson_reaches_the_full_density_ceiling_in_a_single_grid():
    """Ferguson is a rectangle to log10 R = +1.0 at every temperature it holds,
    so no split is needed. Semenov's gas grid runs out at 11 - 3 log10 T, so the
    same request either splits or raises. Choosing the cold source is what
    decides whether one dense grid is possible."""
    kw = dict(log_T_range=(2.71, 7.1), log_R_range=(-8.0, 1.0), n_T=60, n_R=40)
    t = rm_tables.build(cold="ferguson", split=False, **kw)
    assert not t.is_split
    assert t.cold.shape == (60, 40)
    assert np.isfinite(t.cold).all()
    assert t.cold_log_R[-1] == pytest.approx(1.0)

    with pytest.raises(CoverageError) as e:
        rm_tables.build(cold="semenov", split=False, **kw)
    assert "ferguson" in str(e.value)


def test_the_two_cold_sources_disagree_in_the_dust_regime():
    """In the dust regime the answer comes from the cold source alone, and the
    two destroy grains at different temperatures, so they must differ.

    501.2 K, not 500 K: Ferguson's lowest tabulated row sits at
    ``log10 T = 2.70``, which is 501.19 K, and anything below it is refused.
    """
    f = rm_tables.opacity(cold="ferguson")
    s = rm_tables.opacity(cold="semenov")
    assert f(501.2, 1e-14) == pytest.approx(0.45509, rel=1e-3)
    assert s(501.2, 1e-14) == pytest.approx(1.75555, rel=1e-3)


def test_ferguson_refuses_below_its_own_floor_rather_than_holding_the_edge():
    """The compiled lookup clamps into the grid, which would return the 501 K
    value for any colder temperature. That is a silent wrong answer, so the
    floor is checked before the lookup runs."""
    f = rm_tables.opacity(cold="ferguson")
    with pytest.raises(rm_tables.coverage.CoverageError) as e:
        f(100.0, 1e-14)
    assert "501" in str(e.value) and "semenov" in str(e.value)
    with pytest.raises(rm_tables.coverage.CoverageError):
        f(np.array([600.0, 100.0]), 1e-14)
    assert rm_tables.opacity(cold="semenov")(100.0, 1e-14) > 0.0


def test_the_two_cold_sources_agree_exactly_where_opal_answers():
    """Above the ramp neither cold source contributes, so the choice of cold
    source must make no difference at all."""
    f = rm_tables.opacity(cold="ferguson")
    s = rm_tables.opacity(cold="semenov")
    for T, rho in ((1e5, 1e-10), (1e5, 1e-8), (1e6, 1e-8)):
        assert f(T, rho) == s(T, rho)
    assert f(1e5, 1e-10) == pytest.approx(0.468214, rel=1e-5)


def test_the_provenance_names_ferguson_when_ferguson_is_chosen():
    p = rm_tables.opacity(cold="ferguson").provenance
    assert p["cold"] == "ferguson"
    assert "Ferguson" in p["cold_reference"]
    assert "Iglesias" in p["hot_reference"]
    assert p["reference_Z"] == ferguson.REFERENCE_Z == 0.02
