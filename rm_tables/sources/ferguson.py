"""Ferguson et al. 2005 low-temperature opacity, interpolated in composition.

ApJ 623, 585. Covers 501 K to 31,623 K over the same density axis as OPAL,
`log10 R` from -8.0 to 1.0, with no blank entries.

Seven sets ship, each 155 tables on a grid of hydrogen and metal mass fraction.
Six are at Grevesse & Sauval 1998 metal ratios and differ in alpha enhancement,
the abundance of oxygen, magnesium and silicon relative to iron at fixed total
metal fraction. `alpha` selects among them. Enhancement holds the metal
fraction fixed, so raising the alpha elements lowers iron, and iron is a major
grain opacity carrier: the dust opacity falls by 0.054 to 0.079 dex per step.

The seventh is `f05.g93`, at Grevesse & Noels 1993 ratios, selected by
`compilation`. It was this module's only set before the alpha sets were added,
and it is the cold partner for OPAL's carbon- and oxygen-rich tables, which are
published at those ratios and at no others. No alpha-enhanced version of it
exists, so asking for one raises. Composition interpolation never crosses
between the two compilations, since they are different metal mixtures rather
than different points on one axis.

Grain abundances come from a chemical equilibrium solve, so composition reaches
the dust through the calculation rather than through a multiply. Grains form at
the condensation temperature here; Semenov destroys them at the evaporation
temperature, which is lower. Section 4 of the paper states the difference.

Stored as 32-bit floats, which costs 4.730e-07 dex.
"""
import os

import numpy as np

from . import _composition

__all__ = ["table_at", "grid", "axes", "compositions", "kappa", "alphas",
           "set_name", "REFERENCE", "REFERENCE_Z", "DEFAULT_ALPHA",
           "ALPHA_RANGE", "COMPILATIONS", "DEFAULT_COMPILATION"]

REFERENCE = "Ferguson et al. 2005, ApJ 623, 585"

DEFAULT_ALPHA = 0.0
"""Alpha enhancement the module reads when none is asked for."""

ALPHA_RANGE = (-0.2, 0.8)
"""Lowest and highest tabulated alpha enhancement."""

_ALPHA_KEYS = ("-0.2", "+0.0", "+0.2", "+0.4", "+0.6", "+0.8")
"""The six tabulated enhancements, as they key the shipped archive."""

COMPILATIONS = ("gs98", "g93")
"""The abundance compilations that ship: Grevesse & Sauval 1998 and Noels 1993."""

DEFAULT_COMPILATION = "gs98"
"""The compilation read when none is asked for. Only it carries enhancements."""

REFERENCE_Z = 0.02
"""Metallicity the shipped tables are keyed against.

Unused by `table_at`, which interpolates among the tabulated compositions
instead of scaling one.
"""

_CACHE = {}
_ALPHA_ARCHIVE = None


def _data_dir():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "data")


def alphas():
    """The tabulated alpha enhancements, ascending.

    Returns
    -------
    ndarray of shape (6,)
        -0.2 to 0.8 in steps of 0.2.
    """
    return np.array([float(k) for k in _ALPHA_KEYS])


def _key(alpha, compilation=DEFAULT_COMPILATION):
    """The archive key for one enhancement, or None for the 1993 set.

    Raises
    ------
    CoverageError
        If `alpha` is outside the tabulated range, if it falls between two
        tabulated values, or if it is non-zero on a compilation that has no
        enhanced tables. Nothing here interpolates in alpha.
    """
    from ..coverage import CoverageError
    if compilation not in COMPILATIONS:
        raise KeyError("unknown compilation %r. Available: %s."
                       % (compilation, ", ".join(repr(c) for c in COMPILATIONS)))
    if compilation == "g93":
        if alpha:
            raise CoverageError(
                "alpha=%g was asked for on the Grevesse & Noels 1993 tables, "
                "which Ferguson published at no enhancement only. The "
                "enhanced tables are on Grevesse & Sauval 1998." % alpha)
        return None
    if alpha is None:
        alpha = DEFAULT_ALPHA
    alpha = float(alpha)
    lo, hi = ALPHA_RANGE
    if not lo - 1e-9 <= alpha <= hi + 1e-9:
        raise CoverageError(
            "Ferguson tabulates alpha enhancement from %.1f to %.1f, and "
            "alpha=%g was requested." % (lo, hi, alpha))
    i = int(np.argmin(np.abs(alphas() - alpha)))
    if abs(alphas()[i] - alpha) > 1e-9:
        raise CoverageError(
            "Ferguson tabulates alpha enhancement at %s, and alpha=%g was "
            "requested. The tables are separate calculations at each value, "
            "not points on an axis this package interpolates."
            % (", ".join("%.1f" % a for a in alphas()), alpha))
    return _ALPHA_KEYS[i]


def set_name(alpha=DEFAULT_ALPHA, compilation=DEFAULT_COMPILATION):
    """The published archive this enhancement is read from.

    Returns
    -------
    str
        The name Ferguson distributes the set under, for the provenance record.

    Examples
    --------
    >>> from rm_tables.sources import ferguson
    >>> ferguson.set_name(0.6)
    'f05.gs98+.6'
    >>> ferguson.set_name(compilation="g93")
    'f05.g93'
    """
    key = _key(alpha, compilation)
    if key is None:
        return "f05.g93"
    if key == "+0.0":
        return "f05.gs98"
    return "f05.gs98" + key[0] + key[2:]      # '+0.6' is distributed as '+.6'



def _load(alpha=DEFAULT_ALPHA, compilation=DEFAULT_COMPILATION):
    global _ALPHA_ARCHIVE
    key = _key(alpha, compilation)
    if key not in _CACHE:
        if key is None:
            d = np.load(os.path.join(_data_dir(), "ferguson_f05_g93.npz"))
            block = d["kappa"]
        else:
            if _ALPHA_ARCHIVE is None:
                _ALPHA_ARCHIVE = np.load(
                    os.path.join(_data_dir(), "ferguson_alpha_gs98.npz"))
            d = _ALPHA_ARCHIVE
            block = d[key + "::kappa"]
        _CACHE[key] = tuple(
            np.asarray(a, float)
            for a in (d["X"], d["Z"], d["log_T"], d["log_R"], block))
    return _CACHE[key]


def axes(alpha=DEFAULT_ALPHA, compilation=DEFAULT_COMPILATION):
    """The tabulated ``(log_T, log_R)`` axes, ascending in both.

    Parameters
    ----------
    alpha : float or None, optional
        Alpha enhancement, one of the values `alphas` lists.
    compilation : {'gs98', 'g93'}, optional
        Abundance compilation. Only ``'gs98'`` carries enhanced tables. Every set shares these axes.

    Returns
    -------
    log_T : ndarray of shape (85,)
        ``log10`` of temperature in K, 2.7000000477 (501 K) to 4.5 (31,623 K).
    log_R : ndarray of shape (19,)
        ``log10`` of ``rho / (T / 1e6)**3`` in g/cm^3, -8.0 to 1.0.
    """
    _, _, log_T, log_R, _ = _load(alpha, compilation)
    return np.sort(log_T), np.sort(log_R)


def compositions(alpha=DEFAULT_ALPHA, compilation=DEFAULT_COMPILATION):
    """The hydrogen and metal fractions the 155 tables are given at."""
    X, Z, _, _, _ = _load(alpha, compilation)
    return X, Z


def table_at(X, Z, z_floor=1e-5, alpha=DEFAULT_ALPHA,
             compilation=DEFAULT_COMPILATION):
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
    alpha : float or None, optional
        Alpha enhancement, one of the values `alphas` lists.
    compilation : {'gs98', 'g93'}, optional
        Abundance compilation. Only ``'gs98'`` carries enhanced tables.

    Returns
    -------
    ndarray of shape (85, 19)
        ``log10`` of the opacity in cm^2/g, ascending in temperature and in
        density parameter.
    """
    Xs, Zs, log_T, log_R, K = _load(alpha, compilation)
    t_order = np.argsort(log_T)
    r_order = np.argsort(log_R)
    block = _composition.interpolate(Xs, Zs, K, X, Z, z_floor, "ferguson")
    return block[np.ix_(t_order, r_order)]


def grid(log_T, log_R, X, Z, alpha=DEFAULT_ALPHA,
         compilation=DEFAULT_COMPILATION):
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
    alpha : float or None, optional
        Alpha enhancement, one of the values `alphas` lists.
    compilation : {'gs98', 'g93'}, optional
        Abundance compilation. Only ``'gs98'`` carries enhanced tables.

    Returns
    -------
    ndarray of shape (log_T.size, log_R.size)
        ``log10`` of the opacity in cm^2/g.
    """
    own_T, own_R = axes(alpha, compilation)
    block = table_at(X, Z, alpha=alpha, compilation=compilation)
    return _composition.on_axes(own_T, own_R, block, log_T, log_R)


def kappa(T, rho, X, Z, alpha=DEFAULT_ALPHA,
          compilation=DEFAULT_COMPILATION):
    """Rosseland mean opacity in cm^2/g at one point, or 0 outside the tables.

    Parameters
    ----------
    T : float
        Temperature, K.
    rho : float
        Density, g/cm^3.
    X, Z : float
        Hydrogen and metal mass fractions.
    alpha : float or None, optional
        Alpha enhancement, one of the values `alphas` lists.
    compilation : {'gs98', 'g93'}, optional
        Abundance compilation. Only ``'gs98'`` carries enhanced tables.

    Returns
    -------
    float
        Opacity in cm^2/g. A zero means this source has no value there.
    """
    log_T = np.log10(T)
    log_R = np.log10(rho / (T / 1e6) ** 3)
    own_T, own_R = axes(alpha, compilation)
    if not (own_T[0] <= log_T <= own_T[-1] and own_R[0] <= log_R <= own_R[-1]):
        return 0.0
    return float(10.0 ** grid(np.array([log_T]), np.array([log_R]), X, Z,
                              alpha, compilation)[0, 0])
