"""The OPAL source: its axes, its composition interpolation, its edges."""
import numpy as np
import pytest

from rm_tables.sources import opal


def test_the_axes_are_the_published_ones():
    log_T, log_R = opal.axes()
    assert (log_T.min(), log_T.max()) == pytest.approx((3.75, 8.70))
    assert (log_R.min(), log_R.max()) == pytest.approx((-8.0, 1.0))
    assert log_T.size == 70 and log_R.size == 19


def test_there_are_126_compositions_and_helium_is_never_an_axis():
    X, Z = opal.compositions()
    assert X.size == Z.size == 126
    pairs = {(round(a, 6), round(b, 6)) for a, b in zip(X, Z)}
    assert len(pairs) == 126, "two tables would differ only in helium"


def test_a_tabulated_composition_is_returned_unchanged():
    """Interpolating onto a point that is in the table must not move it."""
    X, Z = opal.compositions()
    i = int(np.argmin(np.abs(X - 0.7) + np.abs(Z - 0.02)))
    got = opal.table_at(float(X[i]), float(Z[i]))
    from rm_tables.sources.opal import _load
    _, _, _, log_R, K = _load()
    want = K[i][:, np.argsort(log_R)]
    both = np.isfinite(got) & np.isfinite(want)
    assert np.abs(got[both] - want[both]).max() < 1e-6


def test_the_hot_dilute_corner_approaches_electron_scattering():
    """A Rosseland mean cannot fall below electron scattering in a fully
    ionised gas. This is the check that catches a units error."""
    X = 0.7
    block = opal.table_at(X, 0.02)
    log_T, log_R = opal.axes()
    i = int(np.argmin(np.abs(log_T - 8.0)))
    j = int(np.argmin(np.abs(log_R + 8.0)))
    kappa_es = 0.2 * (1.0 + X)
    assert 0.5 < 10.0 ** block[i, j] / kappa_es < 2.0


def test_the_grid_returns_nan_outside_rather_than_extrapolating():
    out = opal.grid(np.array([3.0, 5.0]), np.array([-8.0, 5.0]), 0.7, 0.02)
    assert np.isnan(out[0, 0]), "3.0 is below OPAL's temperature floor"
    assert np.isnan(out[1, 1]), "+5.0 is above OPAL's density ceiling"
    assert np.isfinite(out[1, 0])


def test_metallicity_moves_the_opacity_in_the_right_direction():
    """More metals, more bound-free absorption, at the iron bump."""
    log_T, log_R = opal.axes()
    i = int(np.argmin(np.abs(log_T - 6.4)))
    j = int(np.argmin(np.abs(log_R + 1.5)))
    poor = opal.table_at(0.7, 0.0001)[i, j]
    rich = opal.table_at(0.7, 0.04)[i, j]
    assert rich > poor


def test_all_seventy_seven_published_sets_are_shipped():
    names = opal.sets()
    assert len(names) == 77
    assert opal.DEFAULT_SET in names
    for one in ("GS98hz", "AGS04hz", "W95hz", "GN93hz.CtoN", "Gz020.x70"):
        assert one in names


def test_the_shipped_sets_hold_6766_tables():
    assert sum(opal.compositions(s)[0].size for s in opal.sets()) == 6766


def test_the_float32_store_keeps_opal_s_own_print_precision():
    """OPAL prints three decimals. The 32-bit store must not move a value
    further from its printed multiple of 0.001 than the cast itself costs."""
    _, _, _, _, K = opal._load()
    k = K[np.isfinite(K)]
    assert np.abs(k - np.round(k, 3)).max() < 1e-6


def test_an_unknown_set_name_is_refused():
    with pytest.raises(KeyError):
        opal.table_at(0.7, 0.02, dataset="no_such_set")


def test_grevesse_sauval_1998_differs_from_grevesse_noels_1993():
    """Two solar abundance patterns at one (X, Z) are not the same opacity."""
    log_T, log_R = opal.axes()
    i = int(np.argmin(np.abs(log_T - 6.4)))
    j = int(np.argmin(np.abs(log_R + 1.5)))
    gn93 = opal.table_at(0.7, 0.02)[i, j]
    gs98 = opal.table_at(0.7, 0.02, dataset="GS98hz")[i, j]
    assert abs(gs98 - gn93) > 1e-3
    assert abs(gs98 - gn93) < 0.1, "a different metal mix, not a different table"


def test_a_carbon_rich_set_matches_the_default_where_it_carries_no_excess():
    """Gz020.x70 is X=0.7, Z=0.02 with a grid of excess carbon and oxygen.
    Its dXc = dXo = 0 corner is the same composition as GN93hz there."""
    X, Z = opal.compositions("Gz020.x70")
    assert np.unique(Z).size == 1
    plain = opal.table_at(0.7, 0.02)
    corner = opal.table_at(0.7, 0.02, dataset="Gz020.x70")
    both = np.isfinite(plain) & np.isfinite(corner)
    assert np.abs(plain[both] - corner[both]).max() < 1e-6


def test_excess_carbon_raises_the_opacity_at_the_iron_bump():
    log_T, log_R = opal.axes()
    i = int(np.argmin(np.abs(log_T - 6.4)))
    j = int(np.argmin(np.abs(log_R + 1.5)))
    none = opal.table_at(0.7, 0.02, dataset="Gz020.x70")[i, j]
    rich = opal.table_at(0.7, 0.02, dataset="Gz020.x70", dXc=0.1)[i, j]
    assert rich > none


def test_an_untabulated_excess_is_refused_rather_than_interpolated():
    dXc, dXo = opal.excess("Gz020.x70")
    assert dXc.size == dXo.size == 32
    assert 0.123 not in set(np.round(dXc, 6))
    with pytest.raises(ValueError):
        opal.table_at(0.7, 0.02, dataset="Gz020.x70", dXc=0.123)


def test_the_default_set_carries_no_excess_carbon_or_oxygen():
    dXc, dXo = opal.excess()
    assert dXc.max() == dXo.max() == 0.0
