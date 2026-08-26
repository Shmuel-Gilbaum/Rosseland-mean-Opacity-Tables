"""A Rosseland mean is never zero, and the callable must never return one.

Semenov's routine returns 0.0 to mean "no value here", which stands for a
missing entry rather than for an opacity. Passing it on sends an optical depth to zero and a
radiative-transfer factor divides by it, so the wrong answer arrives far from
where it was made.

These cover the whole box a solver's search can reach, not the box a converged
solution sits in.
"""
import numpy as np
import pytest

import rm_tables
from rm_tables.sources import _compiled as C
from rm_tables.sources import semenov

COLD = ["semenov", "ferguson"]


@pytest.mark.parametrize("cold", COLD)
def test_no_zero_anywhere_a_solver_can_reach(cold):
    """The search box, not the solution box. A branch scan runs many decades
    past the tabulated density edge."""
    k = rm_tables.opacity(cold=cold)
    floor = 10.0 ** rm_tables.coverage._BY_NAME[cold].log_T[0]
    lt = np.linspace(np.log10(floor * 1.001), 7.0, 160)
    lr = np.linspace(-25.0, 1.0, 160)
    T = np.broadcast_to(10.0 ** lt[:, None], (160, 160)).copy()
    rho = 10.0 ** lr[None, :] * (T / 1e6) ** 3
    vals = k(T, rho)
    assert np.isfinite(vals).all()
    assert (vals > 0.0).all(), f"{int((vals == 0).sum())} zeros"


def test_the_dust_destruction_band_is_where_it_used_to_fail():
    """The zeros sat between 1005 and 5314 K, where Semenov's dust has gone and
    its gas grid is bounded in density. That band gets its own check."""
    k = rm_tables.opacity(cold="semenov")
    T = np.linspace(1000.0, 5400.0, 200)
    for log_R in (-25.0, -12.0, -8.0, 0.0, 1.0):
        rho = 10.0 ** log_R * (T / 1e6) ** 3
        vals = k(T, rho)
        assert (vals > 0.0).all(), f"zeros at log10 R = {log_R}"


def test_holding_the_density_edge_leaves_the_value_flat_beyond_it():
    """Holding is only defensible because the tabulated opacity is flat at the
    dilute edge. Past the edge the answer must stop moving."""
    eG = semenov._gas_grid()
    T = 3000.0
    at_edge = C.semenov_kappa_held(C.ED_HI_FIRST, eG, C.GAS_RHO_MIN, T, 1.0)
    for rho in (C.GAS_RHO_MIN * 0.1, 1e-25, 1e-40):
        assert C.semenov_kappa_held(C.ED_HI_FIRST, eG, rho, T, 1.0) == at_edge


def test_holding_changes_nothing_inside_the_valid_range():
    """The held variant must be the unmodified routine wherever the routine has
    an answer, or it would be a second implementation."""
    eG = semenov._gas_grid()
    worst = 0
    for T in np.linspace(6.0, 9900.0, 300):
        for rho in (1e-18, 1e-14, 1e-10, 1e-8):
            plain = C.semenov_kappa(C.ED_HI_FIRST, eG, rho, T, 1.0)
            held = C.semenov_kappa_held(C.ED_HI_FIRST, eG, rho, T, 1.0)
            if plain > 0.0:
                assert held == plain
                worst += 1
    assert worst > 500, "too few points had a value to make this meaningful"


def test_the_bounds_are_the_ones_the_routine_checks():
    assert C.GAS_RHO_MIN == 1.0e-19
    assert C.GAS_RHO_MAX == 1.0e-7
