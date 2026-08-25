"""Coverage is the contract that replaced every stitch. It gets tested first."""
import numpy as np
import pytest

from rm_tables.coverage import (OPAL, FERGUSON, SEMENOV, CoverageError,
                                     covers)


def test_opal_and_ferguson_share_the_density_axis():
    assert OPAL.log_R_at(4.0) == FERGUSON.log_R_at(4.0) == (-8.0, 1.0)


def test_semenov_is_not_a_rectangle():
    """Below 1000 K the dust polynomial answers anywhere; above it the gas
    grid bounds the answer, and the bound narrows with temperature."""
    wide = SEMENOV.log_R_at(2.5)
    narrow = SEMENOV.log_R_at(4.0)
    assert wide[1] - wide[0] > narrow[1] - narrow[0]
    assert narrow[1] < wide[1]


def test_semenov_reaches_five_kelvin_and_ferguson_does_not():
    assert SEMENOV.log_T[0] == pytest.approx(np.log10(5.0))
    assert FERGUSON.log_T[0] == pytest.approx(2.70)


def test_a_covered_range_does_not_raise():
    """Semenov plus OPAL over the box a disc actually visits."""
    log_T = np.linspace(1.0, 8.0, 40)[:, None]
    log_R = np.linspace(-8.0, -1.0, 15)[None, :]
    covers(("semenov", "opal"), log_T, log_R)


def test_semenov_and_opal_leave_a_wedge_at_high_density():
    """Between about 1900 and 5600 K, Semenov's gas grid runs out below the top
    of the density axis and OPAL has not started. Nothing covers that wedge, and
    the builder says so rather than filling it."""
    with pytest.raises(CoverageError) as e:
        covers(("semenov", "opal"), 3.5, 1.0)
    assert "ferguson" in str(e.value)


def test_ferguson_closes_that_wedge():
    covers(("semenov", "ferguson", "opal"), 3.5, 1.0)


def test_asking_ferguson_for_five_kelvin_raises_and_names_semenov():
    with pytest.raises(CoverageError) as e:
        covers(("ferguson", "opal"), np.log10(5.0), -8.0)
    assert "semenov" in str(e.value)
    assert "5 K" in str(e.value) or "5.000" in str(e.value)


def test_the_error_names_the_offending_corner():
    with pytest.raises(CoverageError) as e:
        covers(("opal",), 2.0, -8.0)
    assert "log10 T = 2.000" in str(e.value)


def test_no_source_covers_below_the_density_floor_at_high_temperature():
    """OPAL and Ferguson both stop at log10 R = -8, and Semenov has no dust
    at a million kelvin. That region is held at the edge, not tabulated."""
    with pytest.raises(CoverageError) as e:
        covers(("semenov", "ferguson", "opal"), 6.0, -9.0)
    assert "No source covers it" in str(e.value)


def test_every_source_states_its_reference():
    for c in (OPAL, FERGUSON, SEMENOV):
        assert c.reference.count(",") >= 2, c.reference


def test_a_tabulated_declaration_matches_its_axis_exactly():
    """A bound declared at a nominal value claims a sliver the 32-bit axis does
    not have. A request landing there passed the check and then found no value.
    Ferguson over-claimed at its floor and OPAL at its ceiling: both directions
    of the same mistake."""
    from rm_tables.coverage import _BY_NAME
    from rm_tables.sources import ferguson, opal
    for name, mod in (("ferguson", ferguson), ("opal", opal)):
        axis_T, axis_R = mod.axes()
        declared = _BY_NAME[name]
        assert declared.log_T == (float(axis_T[0]), float(axis_T[-1])), name
        assert declared.log_R_at(4.0) == (float(axis_R[0]), float(axis_R[-1])), name


@pytest.mark.parametrize("cold,lo,hi", [("ferguson", 2.70, 4.5)])
def test_a_range_starting_on_a_tabulated_floor_raises_coverage_not_a_bug(cold, lo, hi):
    """It used to reach assembly and fail with the package's own internal
    message, which tells a user nothing they can act on."""
    import rm_tables
    with pytest.raises(CoverageError, match="not covered by"):
        rm_tables.build(cold=cold, log_T_range=(lo, hi), n_T=40, n_R=25)
