"""Assemble a pair of opacity tables at a requested composition.

    cold   the cold source below OPAL's floor, ramped into OPAL across their
           overlap, up to the cold source's ceiling in density
    hot    OPAL alone, from the ramp floor to the requested temperature
           ceiling, and up to `hot_log_R_max` in density

A requested range outside what the chosen sources hold raises. Nothing is
filled, held or extrapolated at build time.
"""
import numpy as np

from . import defaults
from .coverage import _BY_NAME, CoverageError, covers, check_composition
from .sources import ferguson, opal, semenov

__all__ = ["Tables", "build", "load"]

RAMP = (3.75, 4.00)
"""Temperature window over which the cold source hands to OPAL, ``log10 T``.

Both sources hold values across it. At the default composition and grid,
Semenov divided by OPAL runs 0.62 to 1.65 over the window, median 0.81.
"""

_COLD_SOURCES = {"semenov": semenov, "ferguson": ferguson}


def _check_or_advise(sources, log_T, log_R, split, hot_max):
    """Raise unless the requested box is covered, saying what would fix it.

    A single rectangle is what a caller usually wants. Where one cannot hold the
    range, this reports whether splitting the table at `split` would, and if not,
    whether another source would.

    Parameters
    ----------
    sources : sequence of str
        The chosen cold source and ``'opal'``.
    log_T, log_R : ndarray
        The requested axes.
    split : float
        Temperature at which a two-table build would hand over, ``log10 T``.
    hot_max : float
        Density ceiling the hot table would reach.

    Raises
    ------
    CoverageError
        Carrying the remedy where one exists.
    """
    try:
        covers(sources, log_T[:, None], log_R[None, :],
               what="the requested range")
        return
    except CoverageError as exc:
        first = str(exc)          # Python clears the name at the clause's end

    # Would splitting help? A split lets the hot half reach `hot_max` while the
    # cold half keeps its own ceiling, so it helps only when everything
    # uncovered lies above the split.
    grid_T, grid_R = np.broadcast_arrays(log_T[:, None], log_R[None, :])
    ok = np.zeros(grid_T.shape, bool)
    for s in sources:
        ok |= _BY_NAME[s].covers(grid_T, grid_R)
    uncovered_T = grid_T[~ok]
    if uncovered_T.min() >= split:
        raise CoverageError(
            f"{first}\nEvery uncovered point is above "
            f"{10 ** split:,.0f} K, so a two-table build covers the range: "
            f"the cold table keeps its ceiling and the hot table, OPAL alone, "
            f"reaches log10 R = {hot_max}.")

    # Would another cold source? Report honestly whether it covers the whole
    # box or only the offending corner, since those are different remedies.
    for other in (s for s in _COLD_SOURCES if s not in sources):
        cand = _BY_NAME[other]
        if not cand.covers(uncovered_T, grid_R[~ok]).all():
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

    Attributes
    ----------
    cold_log_T, cold_log_R, cold : ndarray
        The cold table's axes and its ``log10`` opacity in cm^2/g.
    hot_log_T, hot_log_R, hot : ndarray or None
        The same for the hot table, and ``None`` where one grid covers the
        range.
    split_log_T : float
        Take the hot table at or above this ``log10`` temperature, or wherever
        the density parameter exceeds ``cold_log_R[-1]``.
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

    def which(self, log_T, log_R):
        """``True`` where the hot table answers, ``False`` where the cold does.

        Always ``False`` on a single table, since one grid covers everything.
        """
        shape = np.broadcast(np.asarray(log_T), np.asarray(log_R)).shape
        if not self.is_split:
            return np.zeros(shape, bool) if shape else False
        return ((np.asarray(log_T) >= self.split_log_T)
                | (np.asarray(log_R) > self.cold_log_R[-1]))

    def kappa(self, T, rho):
        """Rosseland mean opacity in cm^2/g, from temperature and density.

        Bilinear in ``log10 T`` and ``log10 R``, held at the edge outside the
        grid. Anything needing a smoother interpolant should fit its own spline
        to `cold` and `hot` and use `which` to choose between them.

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

        Examples
        --------
        >>> import rm_tables
        >>> t = rm_tables.build()
        >>> float(round(t.kappa(500.0, 1.25e-19), 4))
        1.7541
        """
        T, rho = np.broadcast_arrays(np.asarray(T, float),
                                     np.asarray(rho, float))
        log_T = np.log10(T)
        log_R = np.log10(rho / (T / 1e6) ** 3)
        out = np.empty(log_T.shape)
        hot = np.asarray(self.which(log_T, log_R))
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
        """
        from .io import save as _save
        _save(self, path, fmt)

    def __repr__(self):
        p = self.provenance
        shape = (f"{self.cold.shape} + {self.hot.shape}" if self.is_split
                 else f"{self.cold.shape}")
        return (f"Tables(X={p['X']:.4f}, Z={p['Z']:.4f}, cold={p['cold']!r}, "
                f"{shape})")


def load(path, fmt=None):
    """Read a table pair written by `Tables.save`.

    Parameters
    ----------
    path : str
        The file. Its extension picks the format.
    fmt : {'numpy', 'text', 'hdf5'}, optional
        Overrides the extension.

    Returns
    -------
    Tables
    """
    from .io import load as _load
    return _load(path, fmt)


def build(X=None, Z=None, cold=None, log_T_range=None, log_R_range=None,
          hot_log_R_max=None, n_T=None, n_R=None, split_log_T=None,
          split=None, dataset=None, dXc=0.0, dXo=0.0):
    """Build a cold and a hot opacity table at one composition.

    Parameters
    ----------
    X, Z : float, optional
        Hydrogen and metal mass fractions. Helium is the remainder. Default:
        `defaults.SOLAR_MASS_FRACTIONS`.
    cold : {'semenov', 'ferguson'}, optional
        Cold source. The two cold sources are never combined; see
        `defaults.COLD`.
    log_T_range : tuple of float, optional
        Lowest and highest ``log10`` temperature in K. The hot table shares the
        ceiling and starts at the ramp, ``log10 T = 3.75``. Default:
        `defaults.LOG_T_RANGE`.
    log_R_range : tuple of float, optional
        Lowest and highest ``log10 R`` of the cold table, where
        ``R = rho / (T / 1e6)**3`` in g/cm^3. The hot table shares the floor.
        Default: `defaults.LOG_R_RANGE`.
    hot_log_R_max : float, optional
        The hot table's density ceiling as ``log10 R``, which OPAL supports to
        1.0. Default: `defaults.HOT_LOG_R_MAX`.
    n_T, n_R : int, optional
        Grid points in temperature and in density parameter, in each table. The
        resolution changes the table; see `defaults.N_T`.
    split_log_T : float, optional
        Temperature at which a lookup switches to the hot table, if there are
        two, as ``log10 T``. Default: `defaults.SPLIT_LOG_T`.
    split : bool, optional
        Whether splitting into two grids is permitted. It is never forced: a
        range one grid covers comes back as one grid either way. Default: True,
        so a range needing the extra density height gets it. Pass False to
        require a single grid and raise instead.
    dataset : str, optional
        Which OPAL file supplies the hot opacity, from
        `rm_tables.sources.opal.sets()`. Selects the metal mixture. Default:
        `defaults.OPAL_SET`.
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
        needs splitting and `split` is False. The message names the offending
        corner and any source that would cover it.
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

    `Tables.which` picks the table for a point, by temperature or by density.

    >>> bool(t.which(5.0, 0.5))
    True

    Reading the opacity at 500 K and ``log10 R = -6``, in the dust regime.

    >>> i = int(np.argmin(np.abs(t.cold_log_T - np.log10(500.0))))
    >>> kappa = 10.0 ** np.interp(-6.0, t.cold_log_R, t.cold[i])
    >>> float(round(kappa, 4))
    1.7427
    """
    X0, _, Z0 = defaults.SOLAR_MASS_FRACTIONS
    X = X0 if X is None else float(X)
    Z = Z0 if Z is None else float(Z)
    cold = defaults.COLD if cold is None else cold
    log_T_range = defaults.LOG_T_RANGE if log_T_range is None else log_T_range
    log_R_range = defaults.LOG_R_RANGE if log_R_range is None else log_R_range
    hot_max = defaults.HOT_LOG_R_MAX if hot_log_R_max is None else hot_log_R_max
    n_T = defaults.N_T if n_T is None else int(n_T)
    n_R = defaults.N_R if n_R is None else int(n_R)
    split_T = defaults.SPLIT_LOG_T if split_log_T is None else float(split_log_T)
    may_split = True if split is None else bool(split)
    source = _COLD_SOURCES[cold]
    dataset = defaults.OPAL_SET if dataset is None else dataset
    check_composition(X, Z, dataset)

    log_T = np.linspace(log_T_range[0], log_T_range[1], n_T)
    log_R = np.linspace(log_R_range[0], log_R_range[1], n_R)

    # Two grids only when the range actually spans both regions AND splitting
    # is permitted. A range living on one side of the split is one grid either
    # way, because a second would be empty.
    spans_both = log_T[0] < split_T < log_T[-1]
    single = not (may_split and spans_both)
    if single:
        _check_or_advise((cold, "opal"), log_T, log_R, split_T, hot_max)
    else:
        # The cold grid stops at the cold source's own ceiling; the hot grid,
        # OPAL alone, reaches further in density. Clip the cold grid rather than
        # refusing, since the hot grid covers what it gives up.
        ceiling = min(log_R[-1], defaults.LOG_R_RANGE[1])
        if ceiling < log_R[-1]:
            log_R = np.linspace(log_R[0], ceiling, n_R)
        # The cold grid is assembled over the whole temperature range, not
        # only below the split, so it is checked over the whole of it.
        _check_or_advise((cold, "opal"), log_T[log_T < split_T], log_R,
                         split_T, hot_max)
        _check_or_advise(("opal",), log_T[log_T >= split_T], log_R,
                         split_T, hot_max)

    block = _assemble(source, cold, log_T, log_R, X, Z, dataset, dXc, dXo)
    provenance = _provenance(X, Z, cold, source, n_T, n_R, log_T_range,
                             log_R_range, hot_max, split_T, single,
                             dataset, dXc, dXo)
    if single:
        return Tables(log_T, log_R, block, None, None, None, split_T,
                      provenance)

    hot_T = np.linspace(RAMP[0], log_T_range[1], n_T)
    hot_R = np.linspace(log_R_range[0], hot_max, n_R)
    _check_or_advise(("opal",), hot_T, hot_R, split_T, hot_max)
    return Tables(log_T, log_R, block, hot_T, hot_R,
                  opal.grid(hot_T, hot_R, X, Z, dataset=dataset,
                            dXc=dXc, dXo=dXo), split_T, provenance)


def _assemble(source, cold, log_T, log_R, X, Z, dataset, dXc, dXo):
    """The cold source below OPAL's floor, ramped into OPAL across the overlap."""
    if cold == "ferguson":
        below = source.grid(log_T, log_R, X, Z)
    else:
        below = source.grid(log_T, log_R, Z)
    above = opal.grid(log_T, log_R, X, Z, dataset=dataset, dXc=dXc, dXo=dXo)
    w = np.clip((log_T - RAMP[0]) / (RAMP[1] - RAMP[0]), 0.0, 1.0)[:, None]
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
                hot_max, split_T, single, dataset, dXc, dXo):
    """Everything needed to reproduce a build."""
    return {
        "X": X, "Y": 1.0 - X - Z, "Z": Z,
        "cold": cold,
        "cold_reference": source.REFERENCE,
        "hot_reference": opal.REFERENCE,
        "opal_set": dataset, "dXc": float(dXc), "dXo": float(dXo),
        "reference_Z": source.REFERENCE_Z,
        "n_T": n_T, "n_R": n_R,
        "log_T_range": tuple(log_T_range),
        "log_R_range": tuple(log_R_range),
        "hot_log_R_max": float(hot_max),
        "split_log_T": split_T,
        "is_split": not single,
        "ramp": RAMP,
        "units": "cm^2/g",
    }
