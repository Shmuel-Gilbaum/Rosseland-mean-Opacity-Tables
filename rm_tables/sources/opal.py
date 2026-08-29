"""OPAL Rosseland mean opacity, interpolated in hydrogen and metals.

Iglesias & Rogers 1996, ApJ 464, 943.

The 77 shipped OPAL Type-1 files hold 6766 tables between them. Each file is a
*set*, and ``GN93hz`` is the default: 126 compositions on a grid of hydrogen
and metal mass fraction, at Grevesse & Noels 1993 metal ratios. The other sets
carry other solar abundance patterns, CNO-processed metal ratios, and excess
carbon and oxygen. :func:`sets` lists them.

Every set shares the same axes, 3.75 to 8.6999998 in ``log10 T`` with T in K,
and -8.0 to 1.0 in ``log10 R`` where ``R = rho / (T / 1e6)**3`` in g/cm^3.
Opacity is in cm^2/g and is returned as ``log10``.

A table also carries excess carbon ``dXc`` and excess oxygen ``dXo``, the mass
fractions of carbon and of oxygen beyond what the metal fraction Z already
supplies. The 40 ``Gz???.x??`` sets vary those two at one fixed ``(X, Z)``;
every other set holds ``dXc = dXo = 0`` throughout. :func:`table_at` selects a
tabulated pair through its ``dXc`` and ``dXo`` arguments and :func:`excess`
lists the pairs a set provides.

The published tables are blank in the cool dilute corner and the hot dense one.
:func:`table_at` returns NaN there. :func:`grid` interpolates along temperature
through a blank and holds the last tabulated value past one.

Stored as 32-bit floats, which costs 2.365e-07 dex.
"""
import os

import numpy as np

from . import _composition

__all__ = ["table_at", "grid", "axes", "compositions", "excess", "sets",
           "DEFAULT_OPAL_SET", "REFERENCE"]

REFERENCE = "Iglesias & Rogers 1996, ApJ 464, 943"

DEFAULT_OPAL_SET = "GN93hz"

_ARCHIVE = None
_CACHE = {}


def _archive():
    global _ARCHIVE
    if _ARCHIVE is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _ARCHIVE = np.load(os.path.join(here, "data", "opal_type1.npz"))
    return _ARCHIVE


def _load(opal_set=DEFAULT_OPAL_SET):
    if opal_set not in _CACHE:
        z = _archive()
        if opal_set + "::kappa" not in z.files:
            raise KeyError("no OPAL set %r; see rm_tables.sources.opal.sets()"
                           % (opal_set,))
        _CACHE[opal_set] = tuple(
            z[k].astype(float) for k in (opal_set + "::X", opal_set + "::Z",
                                         "log_T", "log_R", opal_set + "::kappa"))
    return _CACHE[opal_set]


def _co(opal_set):
    z = _archive()
    return z[opal_set + "::dXc"].astype(float), z[opal_set + "::dXo"].astype(float)


def sets():
    """The names of the shipped OPAL Type-1 sets.

    Returns
    -------
    tuple of str
        77 names, sorted. Each names one published OPAL Type-1 file.
    """
    return tuple(sorted({k.split("::")[0] for k in _archive().files
                         if "::" in k}))


def axes():
    """The tabulated ``(log_T, log_R)`` axes, ascending in both.

    Returns
    -------
    log_T : ndarray of shape (70,)
        ``log10`` of temperature in K, 3.75 to 8.6999998.
    log_R : ndarray of shape (19,)
        ``log10`` of ``rho / (T / 1e6)**3`` in g/cm^3, -8.0 to 1.0.
    """
    _, _, log_T, log_R, _ = _load()
    return log_T, np.sort(log_R)


def compositions(opal_set=DEFAULT_OPAL_SET):
    """The hydrogen and metal mass fractions one set is tabulated at.

    Parameters
    ----------
    opal_set : str, optional
        A name from :func:`sets`.

    Returns
    -------
    X : ndarray
        Hydrogen mass fraction, one entry per table.
    Z : ndarray
        Metal mass fraction, one entry per table.
    """
    X, Z, _, _, _ = _load(opal_set)
    return X, Z


def excess(opal_set=DEFAULT_OPAL_SET):
    """The excess carbon and oxygen mass fractions one set is tabulated at.

    Parameters
    ----------
    opal_set : str, optional
        A name from :func:`sets`.

    Returns
    -------
    dXc : ndarray
        Carbon mass fraction beyond that in Z, one entry per table.
    dXo : ndarray
        Oxygen mass fraction beyond that in Z, one entry per table.
    """
    _load(opal_set)
    return _co(opal_set)


def _bracket(values, target):
    """Index of the lower bracketing node and the weight of the upper one."""
    if values.size < 2:
        return 0, 0.0
    i = int(np.clip(np.searchsorted(values, target) - 1, 0, values.size - 2))
    span = values[i + 1] - values[i]
    return i, 0.0 if span == 0 else (target - values[i]) / span


def table_at(X, Z, z_floor=1e-4, opal_set=DEFAULT_OPAL_SET, dXc=0.0, dXo=0.0):
    """``log10`` opacity at one composition, on OPAL's own axes.

    Parameters
    ----------
    X : float
        Hydrogen mass fraction.
    Z : float
        Metal mass fraction.
    z_floor : float, optional
        Added to both metallicities before taking the logarithm, so that a
        metallicity of zero remains interpolable.
    opal_set : str, optional
        A name from :func:`sets`.
    dXc : float, optional
        Carbon mass fraction beyond that in Z. Must be one the set tabulates.
    dXo : float, optional
        Oxygen mass fraction beyond that in Z. Must be one the set tabulates.

    Returns
    -------
    ndarray of shape (70, 19)
        ``log10`` of the opacity in cm^2/g, ascending in temperature and in
        density parameter. NaN where OPAL's table is blank.

    Raises
    ------
    ValueError
        If the set holds no table at the requested ``(dXc, dXo)``.

    Notes
    -----
    Linear in hydrogen and linear in ``log10(Z + z_floor)``, at the selected
    ``(dXc, dXo)``. This does not reproduce OPAL's own interpolator exactly.
    """
    Xs, Zs, log_T, log_R, K = _load(opal_set)
    order = np.argsort(log_R)

    cs, os_ = _co(opal_set)
    keep = np.where((np.abs(cs - dXc) < 1e-6) & (np.abs(os_ - dXo) < 1e-6))[0]
    if keep.size == 0:
        raise ValueError(
            "set %r has no table at dXc=%g, dXo=%g; see "
            "rm_tables.sources.opal.excess(%r)" % (opal_set, dXc, dXo, opal_set))
    Xs, Zs, K = Xs[keep], Zs[keep], K[keep]
    block = _composition.interpolate(Xs, Zs, K, X, Z, z_floor, opal_set)
    return block[:, order]


def grid(log_T, log_R, X, Z, opal_set=DEFAULT_OPAL_SET, dXc=0.0, dXo=0.0):
    """``log10`` opacity on requested axes, NaN outside OPAL's own.

    Interpolates in temperature and in density parameter separately, and does
    not extrapolate: a point outside OPAL's rectangle comes back NaN. A blank
    inside the rectangle does not come back NaN, because the interpolation in
    temperature runs through the blanks and holds the last tabulated value past
    them.

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
    opal_set : str, optional
        A name from :func:`sets`.
    dXc : float, optional
        Carbon mass fraction beyond that in Z.
    dXo : float, optional
        Oxygen mass fraction beyond that in Z.

    Returns
    -------
    ndarray of shape (log_T.size, log_R.size)
        ``log10`` of the opacity in cm^2/g.
    """
    own_T, own_R = axes()
    block = table_at(X, Z, opal_set=opal_set, dXc=dXc, dXo=dXo)
    log_T = np.ravel(log_T)
    log_R = np.ravel(log_R)

    on_T = np.empty((log_T.size, own_R.size))
    inside_T = (log_T >= own_T[0]) & (log_T <= own_T[-1])
    for j in range(own_R.size):
        col = block[:, j]
        g = np.isfinite(col)
        on_T[:, j] = np.where(inside_T,
                              np.interp(log_T, own_T[g], col[g]), np.nan)

    out = np.empty((log_T.size, log_R.size))
    inside_R = (log_R >= own_R[0]) & (log_R <= own_R[-1])
    for i in range(log_T.size):
        out[i] = np.where(inside_R, np.interp(log_R, own_R, on_T[i]), np.nan)
    return out
