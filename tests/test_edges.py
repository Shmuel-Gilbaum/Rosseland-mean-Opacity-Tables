"""No source may hold its edge silently.

Both compiled lookups clamp into their grids, so a request outside a source's
range would come back as that source's nearest tabulated value instead of an
error. Those tests are parametrised over every registered source, so a new one
cannot be added without declaring the range it holds.
"""
import numpy as np
import pytest

import rm_tables
from rm_tables.coverage import _BY_NAME, CoverageError
from rm_tables.tables import _COLD_SOURCES

COLD = sorted(_COLD_SOURCES)


@pytest.mark.parametrize("cold", COLD)
def test_every_cold_source_refuses_below_its_own_floor(cold):
    k = rm_tables.opacity(cold=cold)
    floor = 10.0 ** _BY_NAME[cold].log_T[0]
    inside = k(floor * 1.001, 1e-14)
    assert inside > 0.0
    with pytest.raises(CoverageError) as e:
        k(floor * 0.5, 1e-14)
    assert f"{floor:,.0f}" in str(e.value)


@pytest.mark.parametrize("cold", COLD)
def test_the_refusal_survives_being_hidden_in_an_array(cold):
    """One bad entry among good ones must not pass unnoticed."""
    k = rm_tables.opacity(cold=cold)
    floor = 10.0 ** _BY_NAME[cold].log_T[0]
    good = np.full(50, floor * 10.0)
    good[37] = floor * 0.5
    with pytest.raises(CoverageError):
        k(good, 1e-14)


@pytest.mark.parametrize("cold", COLD)
def test_every_source_refuses_above_opals_ceiling(cold):
    k = rm_tables.opacity(cold=cold)
    ceiling = 10.0 ** _BY_NAME["opal"].log_T[1]
    with pytest.raises(CoverageError):
        k(ceiling * 2.0, 1e-14)


@pytest.mark.parametrize("cold", COLD)
def test_a_refusal_names_a_source_that_would_answer_when_one_exists(cold):
    """An error that only says no is worth less than one that says what to do."""
    k = rm_tables.opacity(cold=cold)
    floor = 10.0 ** _BY_NAME[cold].log_T[0]
    rescuers = [n for n in COLD
                if n != cold and 10.0 ** _BY_NAME[n].log_T[0] < floor]
    if not rescuers:
        pytest.skip(f"{cold} already reaches the lowest floor")
    with pytest.raises(CoverageError) as e:
        k(floor * 0.5, 1e-14)
    assert any(r in str(e.value) for r in rescuers)


def test_the_floor_comes_from_the_coverage_declaration_not_a_literal():
    """Reading the bound from the source's declaration is what makes a new
    source impossible to add without one."""
    for cold in COLD:
        k = rm_tables.opacity(cold=cold)
        assert k._floor_K == pytest.approx(10.0 ** _BY_NAME[cold].log_T[0])


def test_holding_the_edge_would_have_returned_a_plausible_wrong_number():
    """The reason this is checked rather than trusted: the clamped answer looks
    entirely reasonable, so nothing downstream would notice."""
    from rm_tables.sources import _compiled as C
    from rm_tables.sources import ferguson
    f_T, f_R = ferguson.axes()
    block = ferguson.table_at(0.7381, 0.0134)
    at_floor = C.bilinear(f_T, f_R, block, f_T[0], -6.0)
    far_below = C.bilinear(f_T, f_R, block, np.log10(100.0), -6.0)
    assert far_below == at_floor
    assert 0.0 < 10.0 ** far_below < 10.0
