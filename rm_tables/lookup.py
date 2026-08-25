"""Opacity from temperature and density, with no table in between.

`opacity` returns a callable. Below the handover it evaluates Semenov's routine
at the point requested, or interpolates Ferguson's published grid; above it, it
interpolates OPAL's. Nothing is resampled and no spline is fitted, so the answer
carries no grid of this package's choosing.

Outside a source's range in density the nearest value it holds is returned,
matching what a tabulated grid does past its own span. A temperature no source
reaches raises instead. Semenov reports "no value" as a zero, and a zero is
never returned as an opacity. OPAL's tables are blank in two corners, and the
value there is carried in from the nearest tabulated density.

Per call over an array of 200,000 points, compiled: 168 ns through Semenov, 102
ns through OPAL.
"""
import numpy as np

from . import defaults
from .coverage import _BY_NAME, CoverageError, check_composition
from .sources import _compiled as _c
from .sources import ferguson, opal, semenov

__all__ = ["opacity", "Opacity"]

RAMP = (3.75, 4.00)
"""Temperature window over which the cold source hands to OPAL, ``log10 T``."""


@_c.cached_njit
def _one(eD, eG, opal_T, opal_R, opal_k, rho, T, zfac, lo, hi):
    """One opacity, cm^2/g. Ramps between the two sources across ``lo`` to ``hi``."""
    lt = np.log10(T)
    if lt >= hi:
        return 10.0 ** _c.bilinear(opal_T, opal_R, opal_k, lt,
                                   np.log10(rho / (T / 1e6) ** 3))
    cold = _c.semenov_kappa_held(eD, eG, rho, T, zfac)
    if lt <= lo:
        return cold
    w = (lt - lo) / (hi - lo)
    hot = 10.0 ** _c.bilinear(opal_T, opal_R, opal_k, lt,
                              np.log10(rho / (T / 1e6) ** 3))
    if cold <= 0.0:
        return hot
    return 10.0 ** ((1.0 - w) * np.log10(cold) + w * np.log10(hot))


@_c.cached_njit
def _many(eD, eG, opal_T, opal_R, opal_k, rho, T, zfac, lo, hi, out):
    for i in range(out.size):
        out[i] = _one(eD, eG, opal_T, opal_R, opal_k, rho[i], T[i], zfac, lo, hi)
    return out


@_c.cached_njit
def _one_grids(cold_T, cold_R, cold_k, opal_T, opal_R, opal_k,
               rho, T, lo, hi):
    """One opacity when the cold source is tabulated rather than computed."""
    lt = np.log10(T)
    lr = np.log10(rho / (T / 1e6) ** 3)
    hot = _c.bilinear(opal_T, opal_R, opal_k, lt, lr)
    if lt >= hi:
        return 10.0 ** hot
    cold = _c.bilinear(cold_T, cold_R, cold_k, lt, lr)
    if lt <= lo:
        return 10.0 ** cold
    w = (lt - lo) / (hi - lo)
    return 10.0 ** ((1.0 - w) * cold + w * hot)


@_c.cached_njit
def _many_grids(cold_T, cold_R, cold_k, opal_T, opal_R, opal_k,
                rho, T, lo, hi, out):
    for i in range(out.size):
        out[i] = _one_grids(cold_T, cold_R, cold_k, opal_T, opal_R, opal_k,
                            rho[i], T[i], lo, hi)
    return out


class Opacity:
    """A callable opacity at one composition.

    Call it with scalars or arrays.

    Attributes
    ----------
    provenance : dict
        The composition, the sources and their references, the OPAL set, and
        the excess carbon and oxygen.
    """

    def __init__(self, X, Z, cold, provenance, dataset, dXc, dXo):
        self._cold = cold
        self._eD = _c.ED_HI_FIRST
        self._eG = semenov._gas_grid()
        self._zfac = max(Z, 1e-12) / semenov.REFERENCE_Z
        # The compiled lookups clamp into their grids, so a request below a
        # source's floor would come back as that source's coldest value rather
        # than as an error. The floor is read from the coverage declaration.
        self._floor_K = 10.0 ** _BY_NAME[cold].log_T[0]
        self._ceiling_K = 10.0 ** _BY_NAME["opal"].log_T[1]
        if cold == "ferguson":
            f_T, f_R = ferguson.axes()
            self._cold_T = np.ascontiguousarray(f_T)
            self._cold_R = np.ascontiguousarray(f_R)
            self._cold_k = np.ascontiguousarray(ferguson.table_at(X, Z))
        own_T, own_R = opal.axes()
        block = opal.table_at(X, Z, dataset=dataset, dXc=dXc, dXo=dXo)
        # OPAL's published tables are blank in two corners. Carry the nearest
        # tabulated value in along density rather than returning nothing.
        for i in range(block.shape[0]):
            g = np.where(np.isfinite(block[i]))[0]
            if g.size and g.size < block.shape[1]:
                block[i] = np.interp(np.arange(block.shape[1]), g, block[i][g])
        self._opal_T = np.ascontiguousarray(own_T)
        self._opal_R = np.ascontiguousarray(own_R)
        self._opal_k = np.ascontiguousarray(block)
        self.provenance = provenance

    def _check_temperature(self, T):
        """Refuse a temperature no source holds, rather than holding an edge.

        Raises
        ------
        CoverageError
            Naming the bound that was crossed and, where one exists, the source
            that would answer.
        """
        if T.size == 0:
            return
        # Every comparison against NaN is false, so a NaN temperature passed
        # both bounds and came back as a NaN opacity.
        if not np.all(np.isfinite(T) & (T > 0.0)):
            bad = float(T[~(np.isfinite(T) & (T > 0.0))].ravel()[0])
            raise CoverageError(
                f"temperature must be finite and positive, and {bad:g} K was "
                f"requested.")
        lo, hi = float(np.min(T)), float(np.max(T))
        if lo < self._floor_K:
            other = [n for n, c in _BY_NAME.items()
                     if n != "opal" and n != self._cold
                     and 10.0 ** c.log_T[0] <= lo]
            hint = (f" Select cold={other[0]!r}, which reaches "
                    f"{10.0 ** _BY_NAME[other[0]].log_T[0]:,.0f} K."
                    if other else " No source reaches that temperature.")
            raise CoverageError(
                f"{self._cold} has no value below {self._floor_K:,.0f} K, "
                f"and {lo:,.4g} K was requested.{hint}")
        if hi > self._ceiling_K:
            raise CoverageError(
                f"opal stops at {self._ceiling_K:,.4g} K, and {hi:,.4g} K was "
                f"requested. No source reaches that temperature.")

    def __call__(self, T, rho):
        """Rosseland mean opacity in cm^2/g.

        Parameters
        ----------
        T : float or array_like
            Temperature, K.
        rho : float or array_like
            Density, g/cm^3.

        Returns
        -------
        float or ndarray
            Opacity in cm^2/g, of the shape the inputs broadcast to.

        Raises
        ------
        CoverageError
            If a temperature falls outside what the chosen sources hold.
        """
        Ta, ra = np.broadcast_arrays(np.asarray(T, float), np.asarray(rho, float))
        self._check_temperature(Ta)
        # A density at or below zero has no opacity. The cold branch clamps into
        # its gas grid, so -1e30 came back as the dilute-edge value, while the
        # hot branch took its logarithm and returned NaN: the same nonsense
        # input answered two different ways.
        if Ta.size and not np.all(np.isfinite(ra) & (ra > 0.0)):
            bad = float(ra[~(np.isfinite(ra) & (ra > 0.0))].ravel()[0])
            raise CoverageError(
                f"density must be finite and positive, and {bad:g} g/cm^3 was "
                f"requested.")
        tabulated = self._cold == "ferguson"
        if Ta.ndim == 0:
            if tabulated:
                v = _one_grids(self._cold_T, self._cold_R, self._cold_k,
                               self._opal_T, self._opal_R, self._opal_k,
                               float(ra), float(Ta), RAMP[0], RAMP[1])
            else:
                v = _one(self._eD, self._eG, self._opal_T, self._opal_R,
                         self._opal_k, float(ra), float(Ta), self._zfac,
                         RAMP[0], RAMP[1])
            return float(v)
        if Ta.size == 0:
            return np.empty(Ta.shape)
        flat_T = np.ascontiguousarray(Ta.ravel())
        flat_r = np.ascontiguousarray(ra.ravel())
        out = np.empty(flat_T.size)
        if tabulated:
            _many_grids(self._cold_T, self._cold_R, self._cold_k,
                        self._opal_T, self._opal_R, self._opal_k,
                        flat_r, flat_T, RAMP[0], RAMP[1], out)
        else:
            _many(self._eD, self._eG, self._opal_T, self._opal_R, self._opal_k,
                  flat_r, flat_T, self._zfac, RAMP[0], RAMP[1], out)
        return out.reshape(Ta.shape)

    def __repr__(self):
        p = self.provenance
        return f"Opacity(X={p['X']:.4f}, Z={p['Z']:.4f}, cold={p['cold']!r})"


def opacity(X=None, Z=None, cold=None, dataset=None, dXc=0.0, dXo=0.0):
    """Build a callable opacity at one composition.

    Parameters
    ----------
    X, Z : float, optional
        Hydrogen and metal mass fractions. Helium is the remainder. Default:
        `defaults.SOLAR_MASS_FRACTIONS`.
    cold : {'semenov', 'ferguson'}, optional
        Cold source. Default: `defaults.COLD`.
    dataset : str, optional
        Which OPAL file supplies the hot opacity, from
        `rm_tables.sources.opal.sets()`. Selects the metal mixture. Default:
        `defaults.OPAL_SET`.
    dXc, dXo : float, optional
        Carbon and oxygen mass fractions beyond those in `Z`. Must be a pair
        the chosen set tabulates; `rm_tables.sources.opal.excess()` lists them.

    Returns
    -------
    Opacity
        Call it with a temperature in K and a density in g/cm^3.

    Examples
    --------
    >>> import numpy as np
    >>> import rm_tables
    >>> kappa = rm_tables.opacity(X=0.7381, Z=0.0134)
    >>> float(round(kappa(500.0, 1.25e-19), 4))
    1.7535

    It takes arrays as readily as scalars.

    >>> T = np.array([500.0, 3000.0, 1e5])
    >>> rho = np.array([1.25e-19, 2.7e-17, 1e-16])
    >>> np.round(kappa(T, rho), 5)
    array([1.75354e+00, 6.00000e-05, 4.46020e-01])
    """
    X0, _, Z0 = defaults.SOLAR_MASS_FRACTIONS
    X = X0 if X is None else float(X)
    Z = Z0 if Z is None else float(Z)
    cold = defaults.COLD if cold is None else cold
    dataset = defaults.OPAL_SET if dataset is None else dataset
    check_composition(X, Z, dataset)
    if cold not in ("semenov", "ferguson"):
        raise KeyError(f"unknown cold source {cold!r}. "
                       f"Available: 'semenov', 'ferguson'.")
    provenance = {
        "X": X, "Y": 1.0 - X - Z, "Z": Z, "cold": cold,
        "cold_reference": (ferguson.REFERENCE if cold == "ferguson"
                           else semenov.REFERENCE),
        "hot_reference": opal.REFERENCE,
        "opal_set": dataset, "dXc": float(dXc), "dXo": float(dXo),
        "reference_Z": (ferguson.REFERENCE_Z if cold == "ferguson"
                        else semenov.REFERENCE_Z),
        "units": "cm^2/g",
    }
    return Opacity(X, Z, cold, provenance, dataset, float(dXc), float(dXo))
