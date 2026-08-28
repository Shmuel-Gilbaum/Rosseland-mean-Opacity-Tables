"""Semenov et al. 2003 dust and gas opacity, with metallicity applied to dust.

A&A 410, 611. Covers 5 K to 10,000 K. Above the dust-to-gas transition the gas
grid restricts it to densities of 1e-19 to 1e-7 g/cm^3; below it the dust
polynomial has no density dependence.

Metallicity multiplies the dust and leaves the gas alone. The Rosseland mean is
a harmonic average, so scaling every monochromatic opacity scales the mean by
the same factor, and dust opacity per gram of gas is proportional to dust mass
per gram of gas. That holds at fixed grain sizes and mineralogy, which the
dust model nrm/h/s fixes and which no argument here changes. At the reference
metallicity the multiply is the identity and this module reproduces the
published routine.
"""
import os

import numpy as np

from . import _semenov_fit as _f

__all__ = ["kappa", "grid", "REFERENCE_Z", "CONDENSED_FRACTION", "REFERENCE"]

REFERENCE = "Semenov et al. 2003, A&A 410, 611"

REFERENCE_Z = 0.0189
"""Metal mass fraction of Anders & Grevesse 1989, which this source adopts."""

CONDENSED_FRACTION = {
    "all present, below 155 K": 1.399e-02,
    "no ice, 165 to 270 K": 8.436e-03,
    "no volatile organics, 280 to 410 K": 7.834e-03,
    "silicates, iron and troilite, 440 to 675 K": 4.304e-03,
    "silicates and iron, 685 to 1500 K": 3.536e-03,
}
"""Condensed mass per gram of gas, by temperature region, for normal silicates.

Summed from Semenov et al. 2003, Table 1. At 500 K the condensed mass is 22.8%
of the metals in Anders & Grevesse 1989.
"""

_GAS = None


def _gas_grid():
    """The Helling gas table, read once."""
    global _GAS
    if _GAS is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _GAS = _f.read_gas(os.path.join(here, "data", "kR_h2001.dat"))
    return _GAS


def kappa(T, rho, Z=REFERENCE_Z):
    """Rosseland mean opacity, cm^2/g, or 0 where Semenov has no value.

    Parameters
    ----------
    T : float
        Temperature, K.
    rho : float
        Gas density, g/cm^3.
    Z : float, optional
        Metal mass fraction. The dust is multiplied by ``Z / REFERENCE_Z``; the
        gas is not. The default reproduces the unmodified routine.

    Returns
    -------
    float
        Opacity in cm^2/g, or 0.0 outside the model's range. A zero is not a
        small opacity; it means this source has no answer there.
    """
    eD, eG = _f.eR, _gas_grid()
    zfac = max(Z, 1e-12) / REFERENCE_Z
    if T < 1.0:
        return 0.0
    T_ev = [_f._bint(rho, _f.RO_EV, _f.TT[i]) for i in range(4)]
    edges = [T_ev[0], 275.0, 425.0, 680.0,
             min(max(T_ev[1], T_ev[2]), max(T_ev[2], T_ev[3]))]
    widths = [5.0, 5.0, 15.0, 5.0, 100.0]

    region = 5                                   # 5 means the gas grid
    if T <= edges[0] + widths[0]:
        region = 0
    for k in range(1, 5):
        if edges[k - 1] + widths[k - 1] < T <= edges[k] + widths[k]:
            region = k

    if region == 5:
        return _f.gop(eG, rho, T)                # pure gas, never scaled

    blended = any(abs(T - edges[i]) <= widths[i] for i in range(5))
    if not blended:
        return zfac * np.polyval(eD[region][::-1], T)     # pure dust, scaled

    lo = edges[region] - widths[region]
    hi = edges[region] + widths[region]
    left = zfac * np.polyval(eD[region][::-1], lo)        # dust side, scaled
    if region == 4:
        right = _f.gop(eG, rho, hi)                       # gas side, not scaled
    else:
        right = zfac * np.polyval(eD[region + 1][::-1], hi)
    amp, mid = 0.5 * (left - right), 0.5 * (left + right)
    return mid - amp * np.sin(np.pi / 2.0 / widths[region] * (T - edges[region]))


def grid(log_T, log_R, Z=REFERENCE_Z):
    """``log10`` opacity on a grid, NaN where Semenov has no value.

    Parameters
    ----------
    log_T : ndarray of shape (nT,)
        ``log10`` temperature, K.
    log_R : ndarray of shape (nR,)
        ``log10 R``, where ``R = rho / (T / 1e6)**3`` in g/cm^3.
    Z : float, optional
        Metal mass fraction.

    Returns
    -------
    ndarray of shape (nT, nR)
        ``log10`` of the opacity in cm^2/g. NaN marks a point this source does
        not answer at; it is never filled here.
    """
    out = np.full((np.size(log_T), np.size(log_R)), np.nan)
    for i, lt in enumerate(np.ravel(log_T)):
        T = 10.0 ** lt
        scale = (T / 1e6) ** 3
        for j, lr in enumerate(np.ravel(log_R)):
            v = kappa(T, 10.0 ** lr * scale, Z)
            if v > 0.0:
                out[i, j] = np.log10(v)
    return out
