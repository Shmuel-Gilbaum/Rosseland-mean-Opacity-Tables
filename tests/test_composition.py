"""A composition the physics or the tables cannot hold must be refused.

OPAL interpolates linearly in hydrogen and in the logarithm of the metallicity,
and continues that straight line past its own edge. Before this guard existed a
hydrogen fraction of 1.5 returned 5.5e-136 cm^2/g and a metal fraction of 2,
twice the total mass, returned 5.02. Both are numbers with nothing behind them.
"""
import numpy as np
import pytest

import rm_tables
from rm_tables.coverage import CoverageError, check_composition
from rm_tables.sources import opal

ENTRY = [
    ("opacity", lambda **kw: rm_tables.opacity(**kw)),
    ("build", lambda **kw: rm_tables.build(n_T=20, n_R=15, **kw)),
]
IMPOSSIBLE = [
    (-0.1, 0.02, "negative"),
    (0.7, -0.01, "negative"),
    (0.9, 0.3, "helium"),
    (1.5, 0.0, "helium"),
    (0.7, 0.5, "helium"),
]


@pytest.mark.parametrize("name,fn", ENTRY, ids=[e[0] for e in ENTRY])
@pytest.mark.parametrize("X,Z,why", IMPOSSIBLE)
def test_both_entry_points_refuse_an_impossible_composition(name, fn, X, Z, why):
    """Both, not one. A guard on the callable alone leaves the builder open."""
    with pytest.raises(CoverageError, match=why):
        fn(X=X, Z=Z)


def test_the_solar_default_is_accepted():
    """A guard that refuses the default would be caught by everything else, but
    state it anyway so the boundary is pinned from both sides."""
    X, _, Z = rm_tables.defaults.SOLAR_MASS_FRACTIONS
    check_composition(X, Z, rm_tables.defaults.OPAL_SET)


def test_the_tabulated_corners_are_inside_not_outside():
    """Pure hydrogen and the metal ceiling are tabulated values. A guard that
    refuses its own table's corners is worse than no guard."""
    xs, zs = opal.compositions()
    x_hi, z_hi = float(np.max(xs)), float(np.max(zs))
    check_composition(x_hi, 0.0, "GN93hz")
    check_composition(0.0, z_hi, "GN93hz")


def test_a_nominal_value_is_not_refused_by_its_own_32_bit_axis():
    """The set holds one hydrogen fraction, stored as 0.699999988079071. A guard
    comparing 0.7 against that float refuses a value the set contains. Three
    bugs in this package have come from that mistake."""
    check_composition(0.70, 0.02, "Gz020.x70")
    assert rm_tables.opacity(X=0.70, Z=0.02, dataset="Gz020.x70")(1e5, 1e-8) > 0


def test_a_set_holding_one_hydrogen_fraction_refuses_any_other():
    """Before the guard, hydrogen was silently ignored for these sets: every
    value returned the one tabulated table."""
    with pytest.raises(CoverageError, match="tabulates hydrogen"):
        rm_tables.opacity(X=0.35, Z=0.02, dataset="Gz020.x70")


def test_metals_past_the_set_ceiling_are_refused():
    """Hydrogen is dropped to zero so the helium check does not fire first: a
    metal fraction of 0.5 with solar hydrogen is caught as negative helium
    before it ever reaches the ceiling."""
    with pytest.raises(CoverageError, match="tabulates metals up to"):
        rm_tables.opacity(X=0.0, Z=0.5)


def test_the_message_names_the_value_and_the_bound():
    """An error a user cannot act on is barely better than a wrong number."""
    with pytest.raises(CoverageError) as e:
        rm_tables.opacity(X=0.0, Z=0.5)
    assert "0.1" in str(e.value) and "0.5" in str(e.value)
