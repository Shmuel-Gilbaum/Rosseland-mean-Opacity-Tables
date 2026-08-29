"""The callable opacity: shapes, agreement with the sources, and the join."""
import numpy as np
import pytest

import rm_tables
from rm_tables.sources import _compiled as C
from rm_tables.sources import opal, semenov


@pytest.fixture(scope="module")
def kappa():
    return rm_tables.opacity()


def test_a_scalar_in_gives_a_float_out(kappa):
    v = kappa(500.0, 1.25e-19)
    assert isinstance(v, float)
    assert v > 0.0


def test_an_array_in_keeps_its_shape(kappa):
    T = np.full((2, 3), 1e4)
    rho = np.full((2, 3), 1e-12)
    assert kappa(T, rho).shape == (2, 3)


def test_scalars_and_arrays_agree(kappa):
    T = np.array([300.0, 900.0, 5e3, 2e5])
    rho = np.array([1e-18, 1e-16, 1e-13, 1e-10])
    many = kappa(T, rho)
    one = np.array([kappa(float(a), float(b)) for a, b in zip(T, rho)])
    assert np.abs(many - one).max() == 0.0


def test_it_broadcasts(kappa):
    T = np.array([500.0, 5000.0])
    assert kappa(T, 1e-14).shape == (2,)
    assert kappa(1e4, np.ones(3) * 1e-12).shape == (3,)


def test_below_the_ramp_it_is_the_cold_source_exactly(kappa):
    """No table, no resampling: the answer must be the source's own value."""
    gas_grid = semenov._gas_grid()
    for T in (50.0, 300.0, 1500.0, 4000.0):
        rho = 1e-14
        want = C.semenov_kappa(C.DUST_COEFFS_HI_FIRST, gas_grid, rho, T, kappa._z_scale)
        assert kappa(T, rho) == want


def test_above_the_ramp_it_is_opal_exactly(kappa):
    """Same again on the hot side, against a bilinear read of OPAL's grid."""
    own_T, own_R = opal.axes()
    for T, rho in ((5e4, 1e-10), (1e6, 1e-8)):
        lt = np.log10(T)
        lr = np.log10(rho / (T / 1e6) ** 3)
        want = 10.0 ** C.bilinear(kappa._opal_T, kappa._opal_R,
                                  kappa._opal_k, lt, lr)
        assert kappa(T, rho) == want


def test_the_join_is_continuous(kappa):
    """Stepping across the ramp must not jump. The two sources agree there to
    within a factor of about 1.2, so a step would mean the ramp is broken."""
    rho = 1e-14
    Ts = np.logspace(np.log10(4000.0), np.log10(2e4), 400)
    k = kappa(Ts, rho)
    jump = np.abs(np.diff(np.log10(k)))
    assert jump.max() < 0.05, f"largest step {jump.max():.4f} dex"


def test_metallicity_reaches_the_dust_and_not_the_molecular_gas():
    poor = rm_tables.opacity(Z=0.001)
    rich = rm_tables.opacity(Z=0.04)
    assert rich(500.0, 1e-18) > 10.0 * poor(500.0, 1e-18)
    assert rich(3000.0, 1e-16) == poor(3000.0, 1e-16)


def test_an_unknown_cold_source_says_what_is_available():
    """A misspelt source must name both cold sources, since which one is
    chosen changes the answer below 4000 K and neither is a default fallback."""
    with pytest.raises(KeyError) as e:
        rm_tables.opacity(cold="draine")
    assert "draine" in str(e.value)
    assert "semenov" in str(e.value) and "ferguson" in str(e.value)


def test_the_provenance_names_both_sources(kappa):
    p = kappa.provenance
    assert "Semenov" in p["cold_reference"]
    assert "Iglesias" in p["hot_reference"]
    assert p["units"] == "cm^2/g"
