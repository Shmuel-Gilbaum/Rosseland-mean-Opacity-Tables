"""What each source covers, and the error raised when a request leaves it.

Coverage is not always a rectangle. OPAL and Ferguson are tabulated on a grid of
``log10 T`` and ``log10 R``, so theirs are. Semenov is a function of temperature
and density, and its gas branch is bounded in density, so its coverage narrows
with temperature above 1000 K.

A request outside coverage raises and names a source that would answer it.
Nothing here fills, holds or extrapolates.
"""
import numpy as np

__all__ = ["Coverage", "CoverageError", "OPAL", "FERGUSON", "SEMENOV",
           "covers", "check_composition"]


class CoverageError(ValueError):
    """A requested temperature or density lies outside every chosen source."""


class Coverage:
    """The region of ``(log10 T, log10 R)`` one source can answer in.

    Parameters
    ----------
    name : str
        The source, as it appears in an error message.
    log_T : tuple of float
        Lowest and highest ``log10`` of temperature in K.
    log_R : tuple of float or callable
        Lowest and highest ``log10 R``, where ``R = rho / (T / 1e6)**3`` in
        g/cm^3. A callable takes ``log10 T`` and returns the pair, for a source
        whose density range depends on temperature.
    reference : str
        Author, year and the journal reference, for the provenance record.
    """

    def __init__(self, name, log_T, log_R, reference):
        self.name = name
        self.log_T = tuple(log_T)
        self._log_R = log_R
        self.reference = reference

    def log_R_at(self, log_T):
        """Lowest and highest ``log10 R`` this source answers at one temperature.

        Both are NaN at a temperature the source answers at no density.
        """
        if callable(self._log_R):
            return self._log_R(log_T)
        return self._log_R

    def covers(self, log_T, log_R):
        """True where this source has a value. Broadcasts over arrays."""
        log_T = np.asarray(log_T, float)
        log_R = np.asarray(log_R, float)
        inside = (log_T >= self.log_T[0]) & (log_T <= self.log_T[1])
        lo, hi = np.vectorize(self.log_R_at)(log_T)
        return inside & (log_R >= lo) & (log_R <= hi)

    def __repr__(self):
        return (f"Coverage({self.name!r}, log_T={self.log_T}, "
                f"log_R={'variable' if callable(self._log_R) else self._log_R})")


def _f32(x):
    """The bound as the stored axis holds it.

    Both tabulated sources ship their axes in 32-bit floats, so an axis written
    from 2.70 reads back as 2.700000047683716. A bound declared at the nominal
    value claims a sliver the data does not have, and a request landing in that
    sliver passes the coverage check and then finds no value.
    """
    return float(np.float32(x))


# Iglesias & Rogers 1996, ApJ 464, 943. A rectangle, as tabulated.
OPAL = Coverage("opal", (_f32(3.75), _f32(8.70)), (_f32(-8.0), _f32(1.0)),
                "Iglesias & Rogers 1996, ApJ 464, 943")

# Ferguson et al. 2005, ApJ 623, 585. Also a rectangle, on the same density axis.
FERGUSON = Coverage("ferguson", (_f32(2.70), _f32(4.50)),
                    (_f32(-8.0), _f32(1.0)),
                    "Ferguson et al. 2005, ApJ 623, 585")


def _semenov_log_R(log_T):
    """Semenov's density range at one temperature, in the table's coordinates.

    Semenov is bounded in density, not in the density parameter. Its gas grid
    runs 1e-19 to 1e-7 g/cm^3. Since ``R = rho / (T / 1e6)**3``, a constant
    density bound is a straight line of slope -3 in ``log10 R``:

        log10 T     3.30     3.40     3.70     4.00
        lower     -10.90   -11.20   -12.10   -13.00
        upper       1.10     0.80    -0.10    -1.00

    Below the dust-to-gas transition the polynomial has no density dependence
    and answers at any density. The transition sits 100 K above the interpolated
    evaporation temperature of iron, olivine and orthopyroxene, so it moves with
    density: 1001.21 K in the dilute limit, 1001.29 K at 1e-19 g/cm^3 and
    1514.73 K at 1e-7 g/cm^3. The two bounds therefore begin binding at
    different temperatures, ``log10 T`` of 3.0005 and 3.1803.

    Past the dense bound the routine returns to the dust polynomial once the
    density is high enough to carry the transition temperature above ``T``, at
    ``log10 R`` of 2.08 at ``log10 T = 3.20`` and 4.30 at 3.30. That second
    region is not claimed, so the pair returned is a subset of where the source
    has a value rather than all of it.
    """
    if log_T > 4.0:
        return (np.nan, np.nan)                # above 10,000 K Semenov stops
    lo = -1.0 - 3.0 * log_T if log_T >= 3.0005 else -24.0
    hi = 11.0 - 3.0 * log_T if log_T >= 3.1803 else 4.0
    return (lo, hi)


# Semenov et al. 2003, A&A 410, 611. NOT a rectangle.
SEMENOV = Coverage("semenov", (np.log10(5.0), 4.0), _semenov_log_R,
                   "Semenov et al. 2003, A&A 410, 611")

_BY_NAME = {c.name: c for c in (OPAL, FERGUSON, SEMENOV)}


def covers(sources, log_T, log_R, what="the requested range"):
    """Raise unless every point is answered by at least one source.

    Parameters
    ----------
    sources : sequence of str
        Source names, from ``'opal'``, ``'ferguson'`` and ``'semenov'``.
    log_T, log_R : ndarray
        The grid to check, broadcast against each other.
    what : str, optional
        How the range is described in the error message.

    Raises
    ------
    CoverageError
        Naming the worst uncovered corner, and any source that would cover it.
    """
    chosen = [_BY_NAME[s] for s in sources]
    log_T, log_R = np.broadcast_arrays(np.asarray(log_T, float),
                                       np.asarray(log_R, float))
    ok = np.zeros(log_T.shape, bool)
    for c in chosen:
        ok |= c.covers(log_T, log_R)
    if ok.all():
        return
    i = int(np.argmin(ok.ravel()))
    bad_T, bad_R = log_T.ravel()[i], log_R.ravel()[i]
    rescue = [c.name for c in _BY_NAME.values()
              if c.name not in sources and bool(c.covers(bad_T, bad_R))]
    hint = (f" Add {' or '.join(rescue)} to cover it."
            if rescue else " No source covers it.")
    raise CoverageError(
        f"{what} is not covered by {', '.join(sources)}: "
        f"{int(np.size(ok) - ok.sum())} of {np.size(ok)} points fall outside, "
        f"the first at log10 T = {bad_T:.3f} ({10 ** bad_T:.4g} K), "
        f"log10 R = {bad_R:.3f}.{hint}")


def check_composition(X, Z, dataset):
    """Refuse a composition the physics or the tables cannot hold.

    Parameters
    ----------
    X : float
        Hydrogen mass fraction.
    Z : float
        Metal mass fraction.
    dataset : str
        The OPAL set the hot opacity comes from, from
        `rm_tables.sources.opal.sets`.

    Raises
    ------
    CoverageError
        If helium would be negative, if either fraction is negative, or if the
        pair lies outside what the chosen set tabulates. OPAL interpolates
        linearly in hydrogen and in the logarithm of the metallicity, and
        continues that straight line past its own edge, so a request outside
        returns a number with nothing behind it. Hydrogen of 1.5 returns
        5.5e-136 cm^2/g at 1e5 K.
    """
    from .sources import opal
    X, Z = float(X), float(Z)
    if X < 0.0 or Z < 0.0:
        raise CoverageError(
            f"mass fractions cannot be negative, and X={X:g}, Z={Z:g} "
            f"was requested.")
    if X + Z > 1.0:
        raise CoverageError(
            f"X={X:g} and Z={Z:g} leave helium at {1.0 - X - Z:g}. "
            f"Helium is the remainder, so X + Z cannot exceed 1.")
    xs, zs = opal.compositions(dataset)
    x_hi, z_hi = float(np.max(xs)), float(np.max(zs))
    x_lo = float(np.min(xs))
    # The tabulated fractions are stored in 32-bit floats, so 0.7 reads back as
    # 0.699999988079071. Compare at that precision or an exactly tabulated
    # value falls outside its own set.
    Xc, Zc = float(np.float32(X)), float(np.float32(Z))
    if not x_lo <= Xc <= x_hi:
        raise CoverageError(
            f"set {dataset!r} tabulates hydrogen from {x_lo:g} to {x_hi:g}, "
            f"and X={X:g} was requested."
            + (f" That set holds one hydrogen fraction only."
               if x_lo == x_hi else ""))
    if Zc > z_hi:
        raise CoverageError(
            f"set {dataset!r} tabulates metals up to {z_hi:g}, and Z={Z:g} "
            f"was requested. rm_tables.sources.opal.sets() lists the others.")
