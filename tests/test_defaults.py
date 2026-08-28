"""The defaults must be inside what the default sources actually cover.

This is the test that fails if someone widens a default without checking, which
is exactly how a table ends up carrying invented values. Every value is read off
`rm_tables.build` itself, so there is no second copy to drift.
"""
import inspect

import numpy as np

import rm_tables
from rm_tables.coverage import covers

D = {p.name: p.default
     for p in inspect.signature(rm_tables.build).parameters.values()}


def test_the_default_box_is_fully_covered_by_the_default_sources():
    log_T = np.linspace(*D["log_T_range"], D["n_T"])[:, None]
    log_R = np.linspace(*D["log_R_range"], D["n_R"])[None, :]
    covers((D["cold"], "opal"), log_T, log_R)


def test_the_default_ceiling_is_the_highest_that_stays_covered():
    """Any higher must fail, or the default is leaving covered range unused.

    The limit is arithmetic: Semenov's density ceiling is 11 - 3 log10 T, and
    OPAL takes over at log10 T = 3.75, so 11 - 3(3.75) = -0.25.
    """
    import pytest
    from rm_tables.coverage import CoverageError
    assert D["log_R_range"][1] == 11.0 - 3.0 * 3.75
    log_T = np.linspace(*D["log_T_range"], D["n_T"])[:, None]
    higher = np.linspace(D["log_R_range"][0], 0.0, D["n_R"])[None, :]
    with pytest.raises(CoverageError):
        covers((D["cold"], "opal"), log_T, higher)


def test_the_default_ceiling_holds_at_any_grid_size():
    """-0.25 is the grid-INDEPENDENT limit. A finer grid puts a point closer to
    OPAL's floor, where Semenov's ceiling is lower, so a ceiling tuned to one
    grid would fail on another."""
    for n_T in (100, 200, 400, 800, 1600):
        log_T = np.linspace(*D["log_T_range"], n_T)[:, None]
        log_R = np.linspace(*D["log_R_range"], 50)[None, :]
        covers((D["cold"], "opal"), log_T, log_R)


def test_ferguson_reaches_the_full_height_that_semenov_cannot():
    """Span Ferguson's declared range rather than its nominal one. The declared
    floor is the 32-bit value its axis actually holds, a hair above 2.70, and
    2.70 itself is outside the data."""
    from rm_tables.coverage import FERGUSON
    log_T = np.linspace(*FERGUSON.log_T, 60)[:, None]
    log_R = np.linspace(*FERGUSON.log_R_at(4.0), 19)[None, :]
    covers(("ferguson", "opal"), log_T, log_R)


def test_the_solar_helium_remainder_is_positive():
    """Helium is whatever hydrogen and metals leave, so the pair must not
    reach one."""
    assert 0.0 < 1.0 - D["X"] - D["Z"] < 1.0


def test_the_two_entry_points_agree_on_what_they_share():
    """A composition or a source named in one signature and not the other is a
    silent difference between building a table and calling the sources."""
    o = {p.name: p.default
         for p in inspect.signature(rm_tables.opacity).parameters.values()}
    for name in ("X", "Z", "cold", "dataset", "dXc", "dXo"):
        assert o[name] == D[name], name


def test_every_default_is_documented():
    """A default with no explanation is a number someone picked."""
    for fn in (rm_tables.build, rm_tables.opacity):
        doc = fn.__doc__
        for p in inspect.signature(fn).parameters.values():
            assert p.name in doc, (fn.__name__, p.name)
