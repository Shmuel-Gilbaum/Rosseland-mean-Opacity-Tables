"""Assembly: the two tables, their join, and what happens on a bad request."""
import numpy as np
import pytest

from rm_tables import tables as B
from rm_tables.coverage import CoverageError


@pytest.fixture(scope="module")
def tables():
    return B.build(n_T=80, n_R=60)


def test_it_builds_at_the_defaults_and_nothing_is_missing(tables):
    assert np.isfinite(tables.cold).all()
    assert np.isfinite(tables.hot).all()


def test_the_hot_table_reaches_a_density_the_cold_one_cannot(tables):
    assert tables.hot_log_R[-1] > tables.cold_log_R[-1]
    assert tables.hot_log_R[-1] == pytest.approx(1.0)
    assert tables.cold_log_R[-1] == pytest.approx(-0.25)


def test_the_selection_rule_sends_hot_and_dense_points_to_the_hot_table(tables):
    assert tables.which(5.0, -4.0)                # hot by temperature
    assert tables.which(3.9, 0.5)                 # hot by density
    assert not tables.which(2.0, -4.0)            # cold


def test_the_two_tables_agree_where_they_overlap(tables):
    """Above the ramp both are pure OPAL, so they must agree closely."""
    lt, lr = 5.0, -4.0
    c = _at(tables.cold_log_T, tables.cold_log_R, tables.cold, lt, lr)
    h = _at(tables.hot_log_T, tables.hot_log_R, tables.hot, lt, lr)
    assert abs(c - h) < 0.01, (c, h)


def _at(log_T, log_R, block, lt, lr):
    i = int(np.argmin(np.abs(log_T - lt)))
    return float(np.interp(lr, log_R, block[i]))


def test_the_join_has_no_kink(tables):
    """The slope must fall through the ramp, not spike at either edge."""
    lr = -6.0
    col = np.array([_at(tables.cold_log_T, tables.cold_log_R, tables.cold, t, lr)
                    for t in tables.cold_log_T])
    slope = np.gradient(col, tables.cold_log_T)
    window = (tables.cold_log_T >= 3.6) & (tables.cold_log_T <= 4.2)
    assert np.abs(slope[window]).max() < np.abs(slope).max()


def test_a_density_ceiling_semenov_cannot_reach_needs_a_split():
    """Semenov's gas grid stops at 11 - 3 log10 T, so one grid cannot hold both
    the 5 K floor and a ceiling of +1.0. A pair can: the cold grid keeps
    Semenov's own ceiling and the hot grid, OPAL alone, reaches the full
    height."""
    t = B.build(log_R_range=(-8.0, 1.0), n_T=80, n_R=60)
    assert t.is_split
    assert t.cold_log_R[-1] == pytest.approx(-0.25)
    assert t.hot_log_R[-1] == pytest.approx(1.0)


def test_forbidding_the_split_makes_that_same_request_raise():
    """The flag permits splitting. Forbidding it must not silently clip."""
    with pytest.raises(CoverageError) as e:
        B.build(log_R_range=(-8.0, 1.0), n_T=80, n_R=60, allow_split=False)
    assert "ferguson" in str(e.value)


def test_a_range_inside_one_region_is_one_grid_even_when_splitting_is_allowed():
    """A second grid would be empty, so it is not made."""
    for kw in (dict(log_T_range=(0.699, 3.0)), dict(log_T_range=(4.5, 7.0))):
        t = B.build(n_T=40, n_R=25, allow_split=True, **kw)
        assert not t.is_split
        assert t.hot is None


def test_a_single_grid_reports_that_every_point_is_its_own():
    t = B.build(n_T=40, n_R=25, allow_split=False)
    assert not t.is_split
    assert bool(t.which(5.0, 0.5)) is False
    assert t.which(np.array([2.0, 6.0]), -4.0).tolist() == [False, False]


def test_asking_below_semenovs_floor_raises():
    with pytest.raises(CoverageError):
        B.build(log_T_range=(0.0, 7.1), n_T=80, n_R=60)


def test_an_unknown_cold_source_raises():
    with pytest.raises(KeyError):
        B.build(cold="not_a_source", n_T=20, n_R=20)


def test_metallicity_reaches_the_dust_and_not_the_hot_table():
    poor = B.build(Z=0.001, n_T=80, n_R=60)
    rich = B.build(Z=0.04, n_T=80, n_R=60)
    cold_i = int(np.argmin(np.abs(poor.cold_log_T - 2.7)))    # 500 K, dust
    assert rich.cold[cold_i].mean() > poor.cold[cold_i].mean()
    # The hot table is OPAL alone, so metallicity moves it through OPAL only,
    # never through the dust multiply.
    assert not np.array_equal(rich.hot, poor.hot)


def test_the_provenance_records_the_resolution_and_the_sources(tables):
    p = tables.provenance
    assert p["n_T"] == 80 and p["n_R"] == 60
    assert p["units"] == "cm^2/g"
    assert "Semenov" in p["cold_reference"]
    assert "Iglesias" in p["hot_reference"]
    assert p["reference_Z"] == 0.0189
    assert p["X"] + p["Y"] + p["Z"] == pytest.approx(1.0)


def test_the_check_covers_the_whole_grid_that_gets_assembled():
    """The cold grid is assembled over the WHOLE temperature range, not only
    below the split, so checking only its cold half let a range past OPAL's
    ceiling reach assembly and fail with the package's own internal message."""
    with pytest.raises(CoverageError, match="not covered by opal"):
        B.build(log_T_range=(0.699, 9.0), n_T=40, n_R=25)


def test_a_range_inside_opals_ceiling_still_builds():
    """The guard above must not refuse what the data does hold."""
    t = B.build(log_T_range=(0.699, 7.1), n_T=40, n_R=25)
    assert t.cold.shape == (40, 25)


def test_a_single_grid_prints_without_crashing():
    """Printing the object was enough to raise an error, because the summary
    line reads the hot grid's shape and a single-grid build has no hot grid."""
    one = B.build(log_T_range=(2.9, 3.9), n_T=30, n_R=20)
    assert not one.is_split
    assert "(30, 20)" in repr(one)
    assert "+" not in repr(one)
    two = B.build(n_T=30, n_R=20)
    assert "+" in repr(two)
