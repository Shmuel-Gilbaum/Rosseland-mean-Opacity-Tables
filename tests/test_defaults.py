"""The defaults must be inside what the default sources actually cover.

This is the test that fails if someone widens a default without checking, which
is exactly how a table ends up carrying invented values.
"""
import numpy as np

from rm_tables import defaults
from rm_tables.coverage import covers


def test_the_default_box_is_fully_covered_by_the_default_sources():
    log_T = np.linspace(*defaults.LOG_T_RANGE, defaults.N_T)[:, None]
    log_R = np.linspace(*defaults.LOG_R_RANGE, defaults.N_R)[None, :]
    covers((defaults.COLD, "opal"), log_T, log_R)


def test_the_default_ceiling_is_the_highest_that_stays_covered():
    """Any higher must fail, or the default is leaving covered range unused.

    The limit is arithmetic: Semenov's density ceiling is 11 - 3 log10 T, and
    OPAL takes over at log10 T = 3.75, so 11 - 3(3.75) = -0.25.
    """
    import pytest
    from rm_tables.coverage import CoverageError
    assert defaults.LOG_R_RANGE[1] == 11.0 - 3.0 * 3.75
    log_T = np.linspace(*defaults.LOG_T_RANGE, defaults.N_T)[:, None]
    higher = np.linspace(defaults.LOG_R_RANGE[0], 0.0, defaults.N_R)[None, :]
    with pytest.raises(CoverageError):
        covers((defaults.COLD, "opal"), log_T, higher)


def test_the_default_ceiling_holds_at_any_grid_size():
    """-0.25 is the grid-INDEPENDENT limit. A finer grid puts a point closer to
    OPAL's floor, where Semenov's ceiling is lower, so a ceiling tuned to one
    grid would fail on another."""
    for n_T in (100, 200, 400, 800, 1600):
        log_T = np.linspace(*defaults.LOG_T_RANGE, n_T)[:, None]
        log_R = np.linspace(*defaults.LOG_R_RANGE, 50)[None, :]
        covers((defaults.COLD, "opal"), log_T, log_R)


def test_ferguson_reaches_the_full_height_that_semenov_cannot():
    """Span Ferguson's declared range rather than its nominal one. The declared
    floor is the 32-bit value its axis actually holds, a hair above 2.70, and
    2.70 itself is outside the data."""
    from rm_tables.coverage import FERGUSON
    log_T = np.linspace(*FERGUSON.log_T, 60)[:, None]
    log_R = np.linspace(*FERGUSON.log_R_at(4.0), 19)[None, :]
    covers(("ferguson", "opal"), log_T, log_R)


def test_the_reference_metallicity_is_anders_and_grevesse():
    assert defaults.REFERENCE_Z == 0.0189


def test_the_mass_fractions_sum_to_one():
    assert sum(defaults.SOLAR_MASS_FRACTIONS) == 1.0


def test_every_default_documents_why():
    """A default with no docstring is a number someone picked."""
    import rm_tables.defaults as d
    for name in d.__all__:
        assert name in d.__dict__
    src = open(d.__file__).read()
    for name in d.__all__:
        assert f"{name} " in src or f"{name}," in src
    # Each public name is followed by a docstring, not a bare comment.
    assert src.count('"""') >= 2 * len(d.__all__) - 2
