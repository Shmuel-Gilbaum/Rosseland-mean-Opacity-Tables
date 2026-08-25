"""The Semenov source, and the one modification made to it."""
import numpy as np
import pytest

from rm_tables.sources import semenov


def test_the_reference_metallicity_reproduces_the_unmodified_routine():
    """The metallicity multiply must be invisible at the reference."""
    from rm_tables.sources import _semenov_fit as f
    eG = semenov._gas_grid()
    worst = 0.0
    for lt in np.linspace(0.7, 4.0, 60):
        T = 10.0 ** lt
        for lr in np.linspace(-8.0, 1.0, 19):
            rho = 10.0 ** lr * (T / 1e6) ** 3
            a = f.cop(f.eR, eG, rho, T)
            b = semenov.kappa(rho, T)
            worst = max(worst, abs(a - b))
    assert worst == 0.0, worst


def test_dust_scales_with_metallicity_and_gas_does_not():
    """Below the dust destruction the opacity is proportional to Z. Above it,
    on the gas grid, metallicity must not reach the answer at all."""
    rho = 1e-14
    for T, expected in ((500.0, 2.0), (3000.0, 1.0)):
        one = semenov.kappa(rho, T, semenov.REFERENCE_Z)
        two = semenov.kappa(rho, T, 2.0 * semenov.REFERENCE_Z)
        assert one > 0.0
        assert two / one == pytest.approx(expected, rel=1e-9)


def test_a_zero_means_no_answer_not_a_small_opacity():
    """Outside the gas grid's density bounds the routine returns 0."""
    assert semenov.kappa(1e-30, 5000.0) == 0.0


def test_the_grid_marks_missing_points_with_nan_rather_than_filling():
    log_T = np.array([3.9])                       # 7943 K, on the gas grid
    log_R = np.array([-8.0, 1.0])                 # the second is out of bounds
    out = semenov.grid(log_T, log_R)
    assert np.isfinite(out[0, 0])
    assert np.isnan(out[0, 1])


def test_the_condensed_fractions_come_from_the_paper():
    assert semenov.CONDENSED_FRACTION[
        "silicates, iron and troilite, 440 to 675 K"] == 4.304e-03
    assert semenov.REFERENCE_Z == 0.0189
