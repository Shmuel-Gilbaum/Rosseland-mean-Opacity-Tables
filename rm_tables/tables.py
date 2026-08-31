"""Assemble a pair of opacity tables at a requested composition.

    cold   the cold source below OPAL's floor, ramped into OPAL across their
           overlap, up to the cold source's ceiling in density
    hot    OPAL alone, from the ramp floor to the requested temperature
           ceiling, and up to `hot_log_R_max` in density

A requested range outside what the chosen sources hold raises. Nothing is
filled, held or extrapolated at build time.
"""
import math

import numpy as np

from .coverage import _BY_NAME, CoverageError, covers, check_composition
from .sources import ferguson, opal, semenov

__all__ = ["Tables", "build"]

RAMP_LOG_T = (3.75, 4.00)
"""Temperature window over which the cold source hands to OPAL, ``log10 T``.

Both sources hold values across it. At the default composition and grid,
Semenov divided by OPAL runs 0.62 to 1.65 over the window, median 0.81.
"""

_COLD_SOURCES = {"semenov": semenov, "ferguson": ferguson}


def _check_or_advise(sources, log_T, log_R, split_log_T, hot_log_R_max,
                     allow_split=True):
    """Raise unless the requested box is covered, saying what would fix it.

    A single rectangle is what a caller usually wants. Where one cannot hold the
    range, this reports whether splitting the table at `split_log_T` would,
    and if not,
    whether another source would.

    Parameters
    ----------
    sources : sequence of str
        The chosen cold source and ``'opal'``.
    log_T, log_R : ndarray
        The requested axes.
    split_log_T : float
        Temperature at which a two-table build would hand over, ``log10 T``.
    hot_log_R_max : float
        Density ceiling the hot table would reach.
    allow_split : bool, optional
        Whether splitting is permitted. Where it already is, splitting is not
        offered as a remedy, since it is in force and did not help.

    Raises
    ------
    CoverageError
        Carrying the remedy where one exists.
    """
    # The nodes first, so the message counts the points the caller asked for.
    try:
        covers(sources, log_T[:, None], log_R[None, :],
               what="the requested range")
    except CoverageError as exc:
        first = str(exc)          # Python clears the name at the clause's end
        axis_T, axis_R = log_T, log_R
    else:
        # Then between them. A band narrower than one cell is invisible to a
        # node-only check, which made the same physical request legal at 50
        # temperature points and illegal at 100.
        fine_T = np.linspace(log_T[0], log_T[-1], max(log_T.size, 512))
        fine_R = np.linspace(log_R[0], log_R[-1], max(log_R.size, 512))
        try:
            covers(sources, fine_T[:, None], fine_R[None, :])
            return
        except CoverageError:
            gap_T, gap_R = np.broadcast_arrays(fine_T[:, None], fine_R[None, :])
            held = np.zeros(gap_T.shape, bool)
            for s in sources:
                held |= _BY_NAME[s].covers(gap_T, gap_R)
            i = int(np.argmax(~held.ravel()))
            first = (
                f"the requested range is not covered by {', '.join(sources)}. "
                f"Every grid point is covered, but a band between them is not, "
                f"the first at log10 T = {gap_T.ravel()[i]:.3f} "
                f"({10 ** gap_T.ravel()[i]:,.0f} K), "
                f"log10 R = {gap_R.ravel()[i]:.3f}. Raising the resolution "
                f"would expose it rather than fix it.")
            axis_T, axis_R = fine_T, fine_R

    # Would splitting help? A split lets the hot half reach `hot_log_R_max`
    # while the cold half keeps its own ceiling, so it helps only when
    # everything uncovered lies above the split.
    # Analyse the grid that actually failed: on a between-node gap the caller's
    # own nodes are all covered, so nothing there would be uncovered to reason
    # about.
    grid_T, grid_R = np.broadcast_arrays(axis_T[:, None], axis_R[None, :])
    ok = np.zeros(grid_T.shape, bool)
    for s in sources:
        ok |= _BY_NAME[s].covers(grid_T, grid_R)
    uncovered_T = grid_T[~ok]
    uncovered_R = grid_R[~ok]
    # Splitting helps only where the hot table, OPAL alone up to
    # `hot_log_R_max`,
    # actually holds every uncovered point. Testing the temperature alone
    # offered a two-table build for points OPAL cannot reach at any density,
    # and offered it when splitting was already in force.
    hot_would_hold = (_BY_NAME["opal"]
                      .covers(uncovered_T,
                              np.minimum(uncovered_R, hot_log_R_max)).all()
                      and uncovered_R.max() <= hot_log_R_max)
    if not allow_split and uncovered_T.min() >= split_log_T and hot_would_hold:
        raise CoverageError(
            f"{first}\nEvery uncovered point is above "
            f"{10 ** split_log_T:,.0f} K and inside what OPAL holds, so a "
            f"two-table "
            f"build covers the range: the cold table keeps its ceiling and the "
            f"hot table, OPAL alone, reaches log10 R = {hot_log_R_max}. "
            f"Splitting is permitted by default; this build passed "
            f"allow_split=False.")

    # Would another cold source? Report honestly whether it covers the whole
    # box or only the offending corner, since those are different remedies.
    for other in (s for s in _COLD_SOURCES if s not in sources):
        cand = _BY_NAME[other]
        if not cand.covers(uncovered_T, uncovered_R).all():
            continue
        try:
            covers((other, "opal"), log_T[:, None], log_R[None, :])
        except CoverageError:
            raise CoverageError(
                f"{first}\nSplitting the table does not help. "
                f"cold={other!r} covers that corner but not the whole range: "
                f"it starts at {10 ** cand.log_T[0]:,.0f} K, and "
                f"{10 ** log_T[0]:,.1f} K was requested. Raise the temperature "
                f"floor to use it.") from None
        raise CoverageError(
            f"{first}\nSplitting the table does not help. "
            f"cold={other!r} covers the whole range.") from None

    raise CoverageError(
        f"{first}\nSplitting the table does not help, and no other source "
        f"covers that corner. The requested range is outside the data.")


class Tables:
    """A cold table, a hot table, and the record of how they were made.

    `Tables.kappa` reads them, as ``kappa(T, rho)``: the temperature in K
    first and the density in g/cm^3 second, returning cm^2/g.

    Attributes
    ----------
    cold_log_T, cold_log_R, cold : ndarray
        The cold table's axes and its ``log10`` opacity in cm^2/g.
    hot_log_T, hot_log_R, hot : ndarray or None
        The same for the hot table, and ``None`` where one grid covers the
        range.
    split_log_T : float
        Take the hot table at or above this ``log10`` temperature, or where the
        density parameter exceeds ``cold_log_R[-1]`` and the temperature is at
        or above ``hot_log_T[0]``.
    provenance : dict
        Everything needed to reproduce this pair, including the resolution,
        which changes the answer and is not physics.
    """

    def __init__(self, cold_log_T, cold_log_R, cold,
                 hot_log_T, hot_log_R, hot, split_log_T, provenance):
        self.cold_log_T, self.cold_log_R, self.cold = cold_log_T, cold_log_R, cold
        self.hot_log_T, self.hot_log_R, self.hot = hot_log_T, hot_log_R, hot
        self.split_log_T = split_log_T
        self.provenance = provenance

    @property
    def is_split(self):
        """``True`` when this holds two grids, ``False`` when one covers it."""
        return self.hot is not None

    def is_hot(self, log_T, log_R):
        """``True`` where the hot table answers, ``False`` where the cold does.

        Always ``False`` on a single table, since one grid covers everything.

        A point denser than the cold table reaches goes to the hot table only
        where the hot table holds its temperature. Below that, the cold table
        answers and holds its density edge. Crossing a density edge costs 4
        percent under 2000 K, where the tabulated opacity is nearly flat in
        density; crossing the hot table's temperature floor instead returns an
        ionised-gas opacity for cold material, measured 67 times too large at
        2000 K.

        Parameters
        ----------
        log_T : float or array_like
            ``log10`` of temperature in K.
        log_R : float or array_like
            ``log10 R``, where ``R = rho / (T / 1e6)**3`` in g/cm^3.

        Returns
        -------
        bool or ndarray of bool
            ``True`` where the hot table answers, of the shape the inputs
            broadcast to.
        """
        log_T = np.asarray(log_T)
        log_R = np.asarray(log_R)
        shape = np.broadcast(log_T, log_R).shape
        if not self.is_split:
            return np.zeros(shape, bool) if shape else False
        dense = (log_R > self.cold_log_R[-1]) & (log_T >= self.hot_log_T[0])
        return (log_T >= self.split_log_T) | dense

    def kappa(self, T, rho):
        """Rosseland mean opacity in cm^2/g, from temperature and density.

        Bilinear in ``log10 T`` and ``log10 R``, held at the edge outside the
        grid.

        Holding the cold table's density ceiling is exact for most of the disc
        and poor at a few temperatures. Over 200 temperatures from 100 K to
        5623 K, a point a twentieth of a decade past the ceiling reads to a
        median 0.001 dex, and one temperature of the 200 is out by 0.709 dex.
        A point 1.25 decades past reads to a median 0.004 dex with 18 of the
        200 out by more than 0.5. The bad ones sit where dust is being
        destroyed and the opacity falls steeply with density.
        `rm_tables.opacity` reads the source rather than a grid and answers
        there: Semenov reaches ``log10 R`` of 1.47 at 1500 K where a default
        grid stops at -0.25. Anything needing a smoother interpolant should
        fit its own spline to `cold` and `hot` and use `is_hot` to choose
        between them.

        Parameters
        ----------
        T : float or ndarray
            Temperature, K.
        rho : float or ndarray
            Density, g/cm^3.

        Returns
        -------
        float or ndarray
            Opacity in cm^2/g, of the same shape as the broadcast inputs.

        Raises
        ------
        CoverageError
            If a temperature or a density is not finite and positive.

        Examples
        --------
        >>> import rm_tables
        >>> t = rm_tables.build()
        >>> float(round(t.kappa(500.0, 1.25e-19), 4))
        1.7541
        """
        T, rho = np.broadcast_arrays(np.asarray(T, float),
                                     np.asarray(rho, float))
        # The callable refuses these; this reader took their logarithm and
        # returned NaN for a negative temperature while answering a density of
        # zero with the dilute-edge value. Two readers, two silent outcomes.
        if T.size:
            bad = ~(np.isfinite(T) & (T > 0.0) & np.isfinite(rho) & (rho > 0.0))
            if np.any(bad):
                raise CoverageError(
                    f"temperature and density must be finite and positive, "
                    f"and T={float(T[bad].ravel()[0]):g} K with "
                    f"rho={float(rho[bad].ravel()[0]):g} g/cm^3 was requested.")
        log_T = np.log10(T)
        log_R = np.log10(rho / (T / 1e6) ** 3)
        out = np.empty(log_T.shape)
        hot = np.asarray(self.is_hot(log_T, log_R))
        pieces = [(~hot, self.cold_log_T, self.cold_log_R, self.cold)]
        if self.is_split:
            pieces.append((hot, self.hot_log_T, self.hot_log_R, self.hot))
        for mask, lt, lr, block in pieces:
            if not np.any(mask):
                continue
            x = np.clip(log_T[mask], lt[0], lt[-1])
            y = np.clip(log_R[mask], lr[0], lr[-1])
            i = np.clip(np.searchsorted(lt, x) - 1, 0, lt.size - 2)
            j = np.clip(np.searchsorted(lr, y) - 1, 0, lr.size - 2)
            u = (x - lt[i]) / (lt[i + 1] - lt[i])
            v = (y - lr[j]) / (lr[j + 1] - lr[j])
            out[mask] = ((1 - u) * (1 - v) * block[i, j]
                         + u * (1 - v) * block[i + 1, j]
                         + (1 - u) * v * block[i, j + 1]
                         + u * v * block[i + 1, j + 1])
        out = 10.0 ** out
        return out if out.shape else float(out)

    def save(self, path, fmt=None):
        """Write this pair to disk.

        Parameters
        ----------
        path : str
            Destination. Its extension picks the format: ``.npz`` for a
            compressed NumPy archive, ``.txt`` or ``.dat`` for plain text that
            any language can read, ``.h5`` or ``.hdf5`` for HDF5.
        fmt : {'numpy', 'text', 'hdf5'}, optional
            Overrides the extension.

        Raises
        ------
        ValueError
            If neither the extension nor `fmt` names a format.
        ImportError
            If HDF5 was asked for and `h5py` is not installed.

        Notes
        -----
        Every format loses precision. ``.npz`` and ``.h5`` store the opacity as
        32-bit floats and ``.txt`` writes six decimals, which on a round trip
        of a 40 by 25 build cost at most 2.4e-07, 2.4e-07 and 5.0e-07 in
        ``log10`` opacity. Provenance tuples come back as lists from ``.npz``
        and ``.h5``, so a loaded pair does not compare equal to the built one
        on provenance.
        """
        from .io import save as _save
        _save(self, path, fmt)

    def __repr__(self):
        p = self.provenance
        shape = (f"{self.cold.shape} + {self.hot.shape}" if self.is_split
                 else f"{self.cold.shape}")
        return (f"Tables(X={p['X']:.4f}, Z={p['Z']:.4f}, cold={p['cold']!r}, "
                f"{shape})")



def build(X=0.7381, Z=0.0134, cold="semenov",
          log_T_range=(math.log10(5.0), 7.1), log_R_range=(-8.0, -0.25),
          hot_log_R_max=1.0, n_T=200, n_R=500, split_log_T=4.0,
          allow_split=True,
          opal_set="GN93hz", dXc=0.0, dXo=0.0):
    """Build a cold and a hot opacity table at one composition.

    Parameters
    ----------
    X, Z : float, optional
        Hydrogen and metal mass fractions. Helium is the remainder.
    cold : {'semenov', 'ferguson'}, optional
        Cold source. The two are never combined. Semenov gives evaporation
        temperatures and Ferguson gives condensation temperatures, so Semenov
        suits material that is heating and Ferguson material that is cooling.
        Ferguson reaches a higher density but stops at 501 K.
    log_T_range : tuple of float, optional
        Lowest and highest ``log10`` temperature in K. The hot table shares the
        ceiling and starts at the ramp, ``log10 T = 3.75``. The floor 5 K is
        Semenov's lowest tabulated temperature. The default ceiling 7.1 is
        12.6 million K. The highest that can be asked for is
        ``log10 T = 8.6999998``, the last value OPAL's axis holds; that axis is
        stored in 32-bit floats, so 8.70 falls outside it.
    log_R_range : tuple of float, optional
        Lowest and highest ``log10 R`` of the cold table, where
        ``R = rho / (T / 1e6)**3`` in g/cm^3. The hot table shares the floor.
        The ceiling -0.25 is where Semenov's gas grid runs out: that grid is
        bounded in density, so its reach falls as ``11 - 3 log10 T``, and
        ``11 - 3(3.75) = -0.25`` is the highest value it holds at every
        temperature below OPAL's floor. Passing ``cold="ferguson"`` reaches
        +1.0 over 501 to 31,623 K. Below -8.0 only Semenov has data, and
        Semenov stops at 10,000 K, so a range that far dilute must stay below
        that. Its own floor is -24 below 1000 K and ``-1 - 3 log10 T`` above.
    hot_log_R_max : float, optional
        The hot table's density ceiling as ``log10 R``, which OPAL supports to
        1.0.
    n_T, n_R : int, optional
        Grid points in temperature and in density parameter, in each table.
        The resolution changes the table: Semenov's dust destruction spans
        192 K, which at 200 temperature points is two and a half grid cells, so
        the tabulated cliff is gentler than the model's own. Every table records
        the resolution it was built at, and two built at different resolutions
        are not comparable. Raising it costs 9% per lookup at 400 temperature
        points and 27% at 800.
    split_log_T : float, optional
        Temperature at which a lookup switches to the hot table, if there are
        two, as ``log10 T``. A lookup also takes the hot table wherever the
        density parameter exceeds the cold table's ceiling.
    allow_split : bool, optional
        Whether splitting into two grids is permitted. It is never forced: a
        range one grid covers comes back as one grid either way. False
        requires a single grid and raises instead.
    opal_set : str, optional
        Which OPAL file supplies the hot opacity, from
        `rm_tables.sources.opal.sets()`. Selects the metal mixture.
    dXc, dXo : float, optional
        Carbon and oxygen mass fractions beyond those in `Z`. Must be a pair
        the chosen set tabulates; `rm_tables.sources.opal.excess()` lists them.

    Returns
    -------
    Tables
        One grid where one covers the range, two where the extra density
        height is needed. `Tables.is_split` says which.

    Raises
    ------
    CoverageError
        If the requested range leaves what the chosen sources hold, or if it
        needs splitting and `allow_split` is False. The message names the
        offending corner and any source that would cover it. Also if `n_T` or
        `n_R` is below 2, if either range does not ascend, or if
        `hot_log_R_max` does not exceed the density floor.
    KeyError
        If `cold` is not a known source.

    Examples
    --------
    >>> import numpy as np
    >>> import rm_tables
    >>> t = rm_tables.build(X=0.7381, Z=0.0134)
    >>> t.cold.shape, t.hot.shape
    ((200, 500), (200, 500))

    The hot table reaches a density the cold one does not.

    >>> float(t.cold_log_R[-1]), float(t.hot_log_R[-1])
    (-0.25, 1.0)

    `Tables.is_hot` picks the table for a point, by temperature or by density.

    >>> bool(t.is_hot(5.0, 0.5))
    True

    Reading the opacity at 500 K and ``log10 R = -6``, in the dust regime.

    >>> i = int(np.argmin(np.abs(t.cold_log_T - np.log10(500.0))))
    >>> kappa = 10.0 ** np.interp(-6.0, t.cold_log_R, t.cold[i])
    >>> float(round(kappa, 4))
    1.7427
    """
    X, Z = float(X), float(Z)
    hot_log_R_max = float(hot_log_R_max)
    n_T, n_R = int(n_T), int(n_R)
    split_log_T = float(split_log_T)
    allow_split = bool(allow_split)
    source = _COLD_SOURCES[cold]
    check_composition(X, Z, opal_set)

    # A descending or zero-width axis silently breaks every lookup: the reader
    # clips and searches assuming ascending order. Refuse both here rather than
    # returning a table whose values are right and whose lookups are not.
    for name, rng in (("log_T_range", log_T_range), ("log_R_range", log_R_range)):
        lo, hi = float(rng[0]), float(rng[1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise CoverageError(f"{name}={rng} must be finite.")
        if hi <= lo:
            raise CoverageError(
                f"{name}={rng} must ascend. A descending or zero-width axis "
                f"gives a table whose lookups are wrong.")
    if hot_log_R_max <= log_R_range[0]:
        raise CoverageError(
            f"hot_log_R_max={hot_log_R_max} must exceed the density floor "
            f"{log_R_range[0]}. A hot grid of zero width holds values but "
            f"every lookup in it returns NaN.")
    for name, n in (("n_T", n_T), ("n_R", n_R)):
        if n < 2:
            raise CoverageError(
                f"{name}={n} must be at least 2. One grid point in an axis "
                f"leaves nothing to interpolate between, and every lookup "
                f"returns NaN.")

    log_T = np.linspace(log_T_range[0], log_T_range[1], n_T)
    log_R = np.linspace(log_R_range[0], log_R_range[1], n_R)

    # Two grids only when the range actually spans both regions AND splitting
    # is permitted. A range living on one side of the split is one grid either
    # way, because a second would be empty.
    spans_both = log_T[0] < split_log_T < log_T[-1]
    single = not (allow_split and spans_both)
    if single:
        _check_or_advise((cold, "opal"), log_T, log_R, split_log_T, hot_log_R_max,
                         allow_split)
    else:
        # The cold grid stops at the cold source's own ceiling; the hot grid,
        # OPAL alone, reaches further in density. Clip the cold grid rather than
        # refusing, since the hot grid covers what it gives up.
        ceiling = min(log_R[-1], -0.25)
        if ceiling < log_R[-1]:
            log_R = np.linspace(log_R[0], ceiling, n_R)
        # The cold grid is assembled over the whole temperature range, not
        # only below the split, so it is checked over the whole of it.
        _check_or_advise((cold, "opal"), log_T[log_T < split_log_T], log_R,
                         split_log_T, hot_log_R_max)
        _check_or_advise(("opal",), log_T[log_T >= split_log_T], log_R,
                         split_log_T, hot_log_R_max)

    block = _assemble(source, cold, log_T, log_R, X, Z, opal_set, dXc, dXo)
    built_R = (float(log_R[0]), float(log_R[-1]))
    provenance = _provenance(X, Z, cold, source, n_T, n_R, log_T_range,
                             built_R, hot_log_R_max, split_log_T, single,
                             opal_set, dXc, dXo)
    provenance["log_R_range_requested"] = (float(log_R_range[0]),
                                           float(log_R_range[1]))
    if not _cold_contributed(log_T):
        provenance["cold"] = None
        provenance["cold_reference"] = None
        provenance["reference_Z"] = None
        provenance["cold_contributed"] = False
    else:
        provenance["cold_contributed"] = True
    if single:
        return Tables(log_T, log_R, block, None, None, None, split_log_T,
                      provenance)

    hot_T = np.linspace(RAMP_LOG_T[0], log_T_range[1], n_T)
    hot_R = np.linspace(log_R_range[0], hot_log_R_max, n_R)
    _check_or_advise(("opal",), hot_T, hot_R, split_log_T, hot_log_R_max)
    return Tables(log_T, log_R, block, hot_T, hot_R,
                  opal.grid(hot_T, hot_R, X, Z, opal_set=opal_set,
                            dXc=dXc, dXo=dXo), split_log_T, provenance)


def _cold_contributed(log_T):
    """Whether any row lies below the top of the handover.

    A range starting above it is OPAL alone, and naming a cold source in the
    provenance would cite a paper the numbers never came from.
    """
    return bool(np.any(np.asarray(log_T) < RAMP_LOG_T[1]))


def _assemble(source, cold, log_T, log_R, X, Z, opal_set, dXc, dXo):
    """The cold source below OPAL's floor, ramped into OPAL across the overlap."""
    if cold == "ferguson":
        below = source.grid(log_T, log_R, X, Z)
    else:
        below = source.grid(log_T, log_R, Z)
    above = opal.grid(log_T, log_R, X, Z, opal_set=opal_set, dXc=dXc, dXo=dXo)
    w = np.clip((log_T - RAMP_LOG_T[0]) / (RAMP_LOG_T[1] - RAMP_LOG_T[0]), 0.0, 1.0)[:, None]
    w = np.broadcast_to(w, below.shape).copy()

    # The cold source can run out inside the ramp while OPAL continues. Use
    # OPAL alone there. It leaves a step along the cold source's ceiling, 0.1415
    # dex at the defaults.
    w[~np.isfinite(below) & np.isfinite(above)] = 1.0
    block = np.where(w >= 1.0, above,
                     np.where(w <= 0.0, below, (1 - w) * below + w * above))

    if not np.isfinite(block).all():
        bad = np.argwhere(~np.isfinite(block))
        i, j = bad[0]
        raise ValueError(
            f"the table has {len(bad)} points no source answered at, the "
            f"first at log10 T = {log_T[i]:.3f}, log10 R = {log_R[j]:.3f}. "
            f"The coverage check should have caught this; it is a bug.")
    return block


def _provenance(X, Z, cold, source, n_T, n_R, log_T_range, log_R_range,
                hot_log_R_max, split_log_T, single, opal_set, dXc, dXo):
    """Everything needed to reproduce a build."""
    return {
        "X": X, "Y": 1.0 - X - Z, "Z": Z,
        "cold": cold,
        "cold_reference": source.REFERENCE,
        "hot_reference": opal.REFERENCE,
        "opal_set": opal_set, "dXc": float(dXc), "dXo": float(dXo),
        "reference_Z": source.REFERENCE_Z,
        "n_T": n_T, "n_R": n_R,
        "log_T_range": tuple(float(v) for v in log_T_range),
        "log_R_range": tuple(float(v) for v in log_R_range),
        "hot_log_R_max": float(hot_log_R_max),
        "split_log_T": split_log_T,
        "is_split": not single,
        "ramp": RAMP_LOG_T,
        "units": "cm^2/g",
    }
