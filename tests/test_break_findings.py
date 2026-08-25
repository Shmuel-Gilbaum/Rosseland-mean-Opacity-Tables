"""Every defect an adversarial pass found, pinned so none returns.

Each test names the wrong answer that used to come back. A test here failing
means the package has resumed lying rather than merely regressing.
"""
import os

import numpy as np
import pytest

import rm_tables
from rm_tables.coverage import CoverageError
from rm_tables.sources import ferguson, opal


# --- the ragged composition grid ------------------------------------------

@pytest.mark.parametrize("cold", ["semenov", "ferguson"])
def test_hydrogen_above_nine_tenths_is_not_nan(cold):
    """Ferguson returned NaN for every hydrogen fraction above 0.9: its
    bracketing divided by zero on a column holding one metallicity."""
    for X in (0.9, 0.92, 0.94, 0.95):
        v = rm_tables.opacity(X=X, Z=0.0134, cold=cold)(700.0, 1e-14)
        assert np.isfinite(v) and v > 0, f"{cold} at X={X} gave {v}"


def test_the_opacity_is_monotone_across_hydrogen_not_a_sawtooth():
    """At hydrogen 0.92 OPAL holds one helium-free table. Using it for any
    metallicity asked returned 1.6827 where its interpolable neighbours
    bracket 0.6919 and 0.6839, a factor of 2.4."""
    v = [rm_tables.opacity(X=X, Z=0.001)(1e5, 1e-8) for X in (0.9, 0.92, 0.95)]
    assert v[0] > v[1] > v[2], v
    assert abs(v[1] - 0.5 * (v[0] + v[2])) < 0.02 * v[1]


def test_a_hydrogen_fraction_no_column_can_interpolate_is_refused():
    """Above 0.95 every column holds one helium-free pair, so no metallicity
    but that pair's own can be reached."""
    with pytest.raises(ValueError, match="cannot reach"):
        rm_tables.opacity(X=0.96, Z=0.001)


def test_an_exactly_tabulated_pair_is_taken_as_it_stands():
    """0.92 with metals at 0.08 sums to 1 and is tabulated. It must be read,
    not refused for lying outside the interpolable region."""
    v = rm_tables.opacity(X=0.92, Z=0.08)(1e5, 1e-8)
    assert np.isfinite(v) and v > 0


def test_a_tabulated_fraction_survives_its_own_32_bit_storage():
    """0.95 is stored as 0.949999988079071. Comparing the request against that
    put a tabulated value outside its own grid."""
    for cold in ("semenov", "ferguson"):
        assert np.isfinite(rm_tables.opacity(X=0.95, Z=0.0134, cold=cold)(700.0, 1e-14))


# --- the composition guard -------------------------------------------------

@pytest.mark.parametrize("X,Z", [(0.7, float("nan")), (float("nan"), 0.02),
                                 (0.7, float("inf")), (float("-inf"), 0.02)])
def test_a_non_finite_mass_fraction_is_refused(X, Z):
    """Every comparison against NaN is false, so all three guards passed it and
    the callable came back finite in the gas branch and NaN elsewhere."""
    with pytest.raises(CoverageError, match="finite"):
        rm_tables.opacity(X=X, Z=Z)


# --- reading a built pair --------------------------------------------------

def test_a_cool_dense_point_does_not_get_an_ionised_gas_opacity():
    """Any point above the cold density ceiling went to the hot grid, whose
    temperature floor is 5623 K, so 2000 K read OPAL's 5623 K row: 28 times
    too large. 200 K was 20 times too small."""
    t = rm_tables.build(n_T=60, n_R=40)
    k = rm_tables.opacity()
    for T, log_R in ((200.0, -0.2), (500.0, -0.2), (1000.0, -0.2), (2000.0, 0.5)):
        rho = 10.0 ** log_R * (T / 1e6) ** 3
        ratio = t.kappa(T, rho) / k(T, rho)
        assert 0.5 < ratio < 2.0, f"{T} K, log R {log_R}: {ratio:.3f}x"


def test_the_hot_grid_still_answers_where_it_should():
    """The fix must not send hot points to the cold grid."""
    t = rm_tables.build(n_T=60, n_R=40)
    assert bool(t.which(5.0, -13.0))
    assert bool(t.which(np.log10(2e4), 0.5))


def test_a_single_grid_prints_and_reads():
    one = rm_tables.build(log_T_range=(2.9, 3.9), n_T=30, n_R=20)
    assert not one.is_split and "+" not in repr(one)
    assert np.isfinite(one.kappa(3000.0, 1e-14))


# --- axes that cannot be interpolated --------------------------------------

@pytest.mark.parametrize("kw", [
    dict(log_T_range=(6.0, 3.0)), dict(log_R_range=(-2.0, -8.0)),
    dict(log_T_range=(4.0, 4.0)), dict(log_R_range=(-4.0, -4.0)),
])
def test_a_reversed_or_zero_width_axis_is_refused(kw):
    """A descending axis built a correct block whose every lookup was wrong, by
    up to 1.8e10, with no error and no warning."""
    with pytest.raises(CoverageError, match="must ascend"):
        rm_tables.build(n_T=20, n_R=15, **kw)


@pytest.mark.parametrize("kw", [dict(n_T=1), dict(n_R=1), dict(n_T=0)])
def test_one_grid_point_in_an_axis_is_refused(kw):
    """These built, and every lookup in them returned NaN."""
    with pytest.raises(CoverageError, match="at least 2"):
        rm_tables.build(**({"n_T": 20, "n_R": 15} | kw))


def test_a_zero_width_hot_density_axis_is_refused():
    """A ceiling equal to the floor was accepted and produced a hot grid whose
    every lookup was NaN while its block was fully finite."""
    with pytest.raises(CoverageError, match="must exceed the density floor"):
        rm_tables.build(n_T=20, n_R=15, hot_log_R_max=-8.0)


# --- densities -------------------------------------------------------------

@pytest.mark.parametrize("rho", [0.0, -1e-14, -1e30, float("inf"), float("nan")])
def test_a_density_that_is_not_positive_is_refused(rho):
    """The cold branch clamped into its gas grid and answered -1e30 at the
    dilute edge; the hot branch took a logarithm and returned NaN. One nonsense
    input, two silent answers."""
    with pytest.raises(CoverageError, match="finite and positive"):
        rm_tables.opacity()(3000.0, rho)


def test_an_empty_array_comes_back_empty_from_both_readers():
    """The callable raised inside NumPy's minimum while the table reader
    returned an empty array."""
    e = np.array([])
    t = rm_tables.build(n_T=20, n_R=15)
    assert rm_tables.opacity()(e, e).shape == (0,)
    assert t.kappa(e, e).shape == (0,)


# --- saving and loading ----------------------------------------------------

def test_save_writes_the_filename_it_was_given(tmp_path):
    """NumPy appends '.npz' to any path not already ending in it, so 'T.NPZ'
    landed as 'T.NPZ.npz' and load could not find it."""
    p = tmp_path / "T.NPZ"
    rm_tables.build(n_T=20, n_R=15).save(str(p))
    assert p.exists() and not (tmp_path / "T.NPZ.npz").exists()
    assert rm_tables.load(str(p)).cold.shape == (20, 15)


def test_an_unknown_format_is_refused_rather_than_written_as_hdf5(tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        rm_tables.build(n_T=20, n_R=15).save(str(tmp_path / "x.npz"), fmt="banana")


@pytest.mark.parametrize("ext", [".npz", ".txt", ".h5"])
def test_every_format_round_trips_the_provenance_unchanged(tmp_path, ext):
    """Text invented a key named 'R' from a prose line holding an equals sign,
    lost the split temperature, and turned the default temperature range into a
    string. HDF5 dropped the split temperature. The three disagreed."""
    if ext == ".h5":
        pytest.importorskip("h5py")
    t = rm_tables.build(n_T=20, n_R=15)
    p = tmp_path / ("t" + ext)
    t.save(str(p))
    back = rm_tables.load(str(p))
    assert set(back.provenance) == set(t.provenance)
    for k, v in t.provenance.items():
        w = back.provenance[k]
        if isinstance(v, (tuple, list)):
            assert tuple(np.ravel(v)) == tuple(np.ravel(w)), k
        else:
            assert v == w, f"{k}: {v!r} -> {w!r}"


# --- what the provenance claims --------------------------------------------

def test_the_provenance_density_range_is_the_one_the_grid_has():
    """It recorded the range asked for, though the cold grid is clipped to its
    source's ceiling on a split build."""
    t = rm_tables.build(log_R_range=(-8.0, 1.0), n_T=30, n_R=20)
    assert t.provenance["log_R_range"] == (float(t.cold_log_R[0]),
                                           float(t.cold_log_R[-1]))
    assert t.provenance["log_R_range_requested"] == (-8.0, 1.0)


def test_a_cold_source_that_contributed_nothing_is_not_cited():
    """Two builds naming different papers were bit-identical, because both were
    entirely above the handover and OPAL alone produced them."""
    a = rm_tables.build(cold="semenov", log_T_range=(4.1, 7.1), n_T=30, n_R=20)
    b = rm_tables.build(cold="ferguson", log_T_range=(4.6, 7.1), n_T=30, n_R=20)
    for t in (a, b):
        assert t.provenance["cold_contributed"] is False
        assert t.provenance["cold"] is None
        assert t.provenance["cold_reference"] is None
    assert rm_tables.build(n_T=30, n_R=20).provenance["cold_contributed"] is True


# --- coverage --------------------------------------------------------------

@pytest.mark.parametrize("n_T", [8, 50, 100, 400])
def test_coverage_does_not_depend_on_the_resolution(n_T):
    """The check sampled only the grid nodes, so a band thinner than one cell
    was invisible and the same request was legal at 50 points and illegal at
    100. The resolution is documented as not physics."""
    with pytest.raises(CoverageError):
        rm_tables.build(split=False, log_R_range=(-8.0, -0.20), n_R=6, n_T=n_T)


def test_the_coverage_error_does_not_offer_a_remedy_already_in_force():
    """It said 'no source covers it' and then offered a two-table build, which
    was already the default and gave the identical error."""
    with pytest.raises(CoverageError) as e:
        rm_tables.build(log_T_range=(3.9, 8.7), n_T=5, n_R=5)
    text = str(e.value)
    assert not ("No source covers it" in text
                and "a two-table build covers the range" in text)


def test_holding_the_cold_density_ceiling_is_accurate_for_most_of_the_disc():
    """The ceiling is held rather than refused, so pin what that costs. It was
    measured at a median 0.001 dex a twentieth of a decade past, with the
    failures confined to the dust-destruction cliff."""
    t = rm_tables.build()
    k = rm_tables.opacity()
    lt = np.linspace(np.log10(100.0), np.log10(5623.0), 200)
    log_R = float(t.cold_log_R[-1]) + 0.05
    T = 10.0 ** lt
    rho = 10.0 ** log_R * (T / 1e6) ** 3
    d = np.abs(np.log10(t.kappa(T, rho) / k(T, rho)))
    assert np.median(d) < 0.01, np.median(d)
    assert int((d > 0.5).sum()) <= 3, int((d > 0.5).sum())


@pytest.mark.parametrize("T,rho", [(-3000.0, 1e-14), (3000.0, -1e-14),
                                   (3000.0, 0.0), (float("nan"), 1e-14)])
def test_both_readers_refuse_the_same_impossible_point(T, rho):
    """The callable refused these while the table reader took a logarithm and
    returned NaN for a negative temperature, and answered a density of zero
    with its dilute-edge value."""
    t = rm_tables.build(n_T=20, n_R=15)
    with pytest.raises(CoverageError):
        rm_tables.opacity()(T, rho)
    with pytest.raises(CoverageError):
        t.kappa(T, rho)
