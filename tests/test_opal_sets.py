"""The metal mixture is an argument, and it reaches both entry points.

OPAL published 77 Type-1 files at different metal mixtures. All 77 ship here,
so a caller who cares which one answered must be able to say, and the answer
must record what it used.
"""
import numpy as np
import pytest

import rm_tables
from rm_tables.sources import opal

ALTERNATIVES = ["GS98hz", "AGS04hz", "W95hz"]


def test_every_shipped_set_is_readable():
    """A set listed but not loadable is worse than one that never shipped."""
    names = opal.sets()
    assert len(names) == 77
    assert "GN93hz" in names
    for name in ALTERNATIVES:
        block = opal.table_at(0.7381, 0.0134, opal_set=name)
        assert block.shape == (70, 19)
        assert np.isfinite(block).any()


@pytest.mark.parametrize("name", ALTERNATIVES)
def test_the_callable_takes_a_set_and_the_answer_moves(name):
    """A forwarded argument that changes nothing is not forwarded."""
    base = rm_tables.opacity()
    other = rm_tables.opacity(opal_set=name)
    assert other.provenance["opal_set"] == name
    assert base(1e5, 1e-8) != other(1e5, 1e-8)


@pytest.mark.parametrize("name", ALTERNATIVES)
def test_the_builder_takes_a_set_and_the_hot_grid_moves(name):
    base = rm_tables.build(n_T=30, n_R=20)
    other = rm_tables.build(n_T=30, n_R=20, opal_set=name)
    assert other.provenance["opal_set"] == name
    assert np.nanmax(np.abs(base.hot - other.hot)) > 0.0


def test_the_default_set_is_what_was_built_before_the_argument_existed():
    """Naming the default explicitly must change nothing, or every table built
    before this argument existed can no longer be reproduced."""
    a = rm_tables.build(n_T=30, n_R=20)
    b = rm_tables.build(n_T=30, n_R=20, opal_set="GN93hz")
    assert np.array_equal(a.cold, b.cold)
    assert np.array_equal(a.hot, b.hot)


def test_the_default_set_has_no_excess_carbon_or_oxygen():
    """The sets that span hydrogen and metals tabulate only (0, 0). Asking for
    an enhanced mixture from one of them must fail rather than silently give
    the unenhanced table."""
    cs, os_ = opal.excess()
    assert sorted({(float(a), float(b)) for a, b in zip(cs, os_)}) == [(0.0, 0.0)]


def test_excess_carbon_reaches_the_callable_through_a_set_that_has_it():
    """The enhanced grids live in the fixed-composition files, whose hydrogen
    and metal fractions are set by the file rather than interpolated."""
    base = rm_tables.opacity(X=0.70, Z=0.02, opal_set="Gz020.x70")
    rich = rm_tables.opacity(X=0.70, Z=0.02, opal_set="Gz020.x70", dXc=0.1)
    assert rich.provenance["dXc"] == 0.1
    assert base(1e5, 1e-8) != rich(1e5, 1e-8)


def test_an_untabulated_excess_is_refused_by_name():
    with pytest.raises(ValueError, match="no table at"):
        rm_tables.opacity(dXc=0.5)


def test_the_set_survives_a_save_and_a_load(tmp_path):
    t = rm_tables.build(n_T=30, n_R=20, opal_set="GS98hz")
    for ext in (".npz", ".txt"):
        p = tmp_path / ("t" + ext)
        t.save(str(p))
        assert rm_tables.load(str(p)).provenance["opal_set"] == "GS98hz"
