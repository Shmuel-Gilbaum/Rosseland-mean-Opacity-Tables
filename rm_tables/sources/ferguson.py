"""Ferguson et al. 2005 low-temperature opacity, interpolated in composition.

ApJ 623, 585. Covers 501 K to 31,623 K over the same density axis as OPAL,
`log10 R` from -8.0 to 1.0, with no blank entries.

The shipped set is `f05.g93`: 155 tables on a grid of hydrogen and metal mass
fraction, at Grevesse & Noels 1993 metal ratios, which is the abundance set
OPAL's shipped tables also use.

Grain abundances come from a chemical equilibrium solve, so composition reaches
the dust through the calculation rather than through a multiply. Grains form at
the condensation temperature here; Semenov destroys them at the evaporation
temperature, which is lower. Section 4 of the paper states the difference.

Stored as 32-bit floats, which costs 4.730e-07 dex.
"""
import os

import numpy as np

from . import _composition

__all__ = ["table_at", "grid", "axes", "compositions", "kappa",
           "REFERENCE", "REFERENCE_Z"]

REFERENCE = "Ferguson et al. 2005, ApJ 623, 585"

REFERENCE_Z = 0.02
"""Metallicity the shipped tables are keyed against.

Unused by `table_at`, which interpolates among the tabulated compositions
instead of scaling one.
"""

_DATA = None


def _load():
    global _DATA
    if _DATA is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d = np.load(os.path.join(here, "data", "ferguson_f05_g93.npz"))
        _DATA = tuple(d[k].astype(float)
                      for k in ("X", "Z", "log_T", "log_R", "kappa"))
    return _DATA


def axes():
    """The tabulated ``(log_T, log_R)`` axes, ascending in both.

    Returns
    -------
    log_T : ndarray of shape (85,)
        ``log10`` of temperature in K, 2.7000000477 (501 K) to 4.5 (31,623 K).
    log_R : ndarray of shape (19,)
        ``log10`` of ``rho / (T / 1e6)**3`` in g/cm^3, -8.0 to 1.0.
    """
    _, _, log_T, log_R, _ = _load()
    return np.sort(log_T), np.sort(log_R)


def compositions():
    """The hydrogen and metal fractions the 155 tables are given at."""
    X, Z, _, _, _ = _load()
    return X, Z


def table_at(X, Z, z_floor=1e-5):
    """``log10`` opacity at one composition, on Ferguson's own axes.

    Parameters
    ----------
    X : float
        Hydrogen mass fraction.
    Z : float
        Metal mass fraction.
    z_floor : float, optional
        Added to both metallicities before taking the logarithm, so that a
        metallicity of zero remains interpolable. Smaller than OPAL's, because
        Ferguson tabulates down to 1e-5.

    Returns
    -------
    ndarray of shape (85, 19)
        ``log10`` of the opacity in cm^2/g, ascending in temperature and in
        density parameter.
    """
    Xs, Zs, log_T, log_R, K = _load()
    t_order = np.argsort(log_T)
    r_order = np.argsort(log_R)
    block = _composition.interpolate(Xs, Zs, K, X, Z, z_floor, "ferguson")
    return block[np.ix_(t_order, r_order)]


def grid(log_T, log_R, X, Z):
    """``log10`` opacity on requested axes, NaN outside Ferguson's own.

    Parameters
    ----------
    log_T : array_like
        ``log10`` of temperature in K.
    log_R : array_like
        ``log10`` of ``rho / (T / 1e6)**3`` in g/cm^3.
    X : float
        Hydrogen mass fraction.
    Z : float
        Metal mass fraction.

    Returns
    -------
    ndarray of shape (log_T.size, log_R.size)
        ``log10`` of the opacity in cm^2/g.
    """
    own_T, own_R = axes()
    block = table_at(X, Z)
    log_T = np.ravel(log_T)
    log_R = np.ravel(log_R)

    inside_T = (log_T >= own_T[0]) & (log_T <= own_T[-1])
    on_T = np.empty((log_T.size, own_R.size))
    for j in range(own_R.size):
        on_T[:, j] = np.where(inside_T,
                              np.interp(log_T, own_T, block[:, j]), np.nan)

    inside_R = (log_R >= own_R[0]) & (log_R <= own_R[-1])
    out = np.empty((log_T.size, log_R.size))
    for i in range(log_T.size):
        out[i] = np.where(inside_R, np.interp(log_R, own_R, on_T[i]), np.nan)
    return out


def kappa(rho, T, X, Z):
    """Rosseland mean opacity in cm^2/g at one point, or 0 outside the tables.

    Parameters
    ----------
    rho : float
        Density, g/cm^3.
    T : float
        Temperature, K.
    X, Z : float
        Hydrogen and metal mass fractions.

    Returns
    -------
    float
        Opacity in cm^2/g. A zero means this source has no value there.
    """
    log_T = np.log10(T)
    log_R = np.log10(rho / (T / 1e6) ** 3)
    own_T, own_R = axes()
    if not (own_T[0] <= log_T <= own_T[-1] and own_R[0] <= log_R <= own_R[-1]):
        return 0.0
    return float(10.0 ** grid(np.array([log_T]), np.array([log_R]), X, Z)[0, 0])
