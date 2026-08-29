"""Interpolation shared by OPAL and Ferguson: composition, then axes.

Both sources tabulate a SET of composition pairs rather than a rectangle. Above
a hydrogen mass fraction of 0.9 most columns hold a single table, and it is the
helium-free one: 0.92 with metals at 0.08, 0.94 with 0.06, each summing to
exactly 1. A column like that cannot be interpolated in metallicity, and using
its one table for any metallicity asked returns a number belonging to a
different composition. At hydrogen 0.92 that was 2.4 times the value its two
interpolable neighbours bracket.

So a column joins the interpolation only where it holds the requested
metallicity between two of its own nodes. An exactly tabulated pair is taken as
it stands. Anything else raises.

`on_axes` is the other half: once a table is chosen, both sources put it on
requested temperature and density axes the same way, holding no value outside
their own rectangle.
"""
import numpy as np

__all__ = ["interpolate", "on_axes", "CompositionError"]

_TOL = 1e-6
"""Match tolerance for an exactly tabulated pair, absolute in mass fraction.

The tabulated values are stored in 32-bit floats, so 0.7 reads back as
0.699999988079071 and an exact request must still match.
"""


class CompositionError(ValueError):
    """A composition pair neither tabulated nor bracketed by tabulated ones."""


def _usable(Xs, Zs, target, z_floor):
    """Hydrogen nodes holding the requested metallicity between two of their own."""
    out = []
    for x in np.unique(Xs):
        lz = np.sort(np.log10(np.unique(Zs[Xs == x]) + z_floor))
        if lz.size >= 2 and lz[0] <= target <= lz[-1]:
            out.append(float(x))
    return np.array(out)


def _column(Xs, Zs, K, x, target, z_floor):
    """One hydrogen column interpolated to the requested metallicity."""
    sel = np.where(Xs == x)[0]
    order = np.argsort(Zs[sel])
    sel = sel[order]
    lz = np.log10(Zs[sel] + z_floor)
    j = int(np.clip(np.searchsorted(lz, target) - 1, 0, lz.size - 2))
    span = lz[j + 1] - lz[j]
    w = 0.0 if span == 0 else (target - lz[j]) / span
    return (1 - w) * K[sel[j]] + w * K[sel[j + 1]]


def interpolate(Xs, Zs, K, X, Z, z_floor, source):
    """Opacity blocks interpolated to one composition on a ragged grid.

    Parameters
    ----------
    Xs, Zs : ndarray of shape (n,)
        Hydrogen and metal mass fraction of each tabulated block.
    K : ndarray of shape (n, ...)
        The blocks, in the same order.
    X, Z : float
        The requested hydrogen and metal mass fractions.
    z_floor : float
        Added to both metallicities before taking the logarithm, so that a
        metallicity of zero remains interpolable.
    source : str
        Named in the error message.

    Returns
    -------
    ndarray
        One block, linear in hydrogen and in ``log10(Z + z_floor)``.

    Raises
    ------
    CompositionError
        If the pair is neither tabulated nor bracketed by hydrogen columns that
        hold the requested metallicity.
    """
    # The tabulated fractions are stored in 32-bit floats, so the node nearest
    # 0.95 reads back as 0.949999988079071. Comparing a request of 0.95 against
    # that puts a tabulated value outside its own grid.
    X, Z = float(np.float32(X)), float(np.float32(Z))
    exact = np.where((np.abs(Xs - X) < _TOL) & (np.abs(Zs - Z) < _TOL))[0]
    if exact.size:
        return K[exact[0]]

    target = np.log10(Z + z_floor)
    xs = _usable(Xs, Zs, target, z_floor)
    if xs.size == 0:
        raise CompositionError(
            f"{source} tabulates no composition at Z={Z:g}. Its metallicities "
            f"run {float(np.min(Zs)):g} to {float(np.max(Zs)):g}.")
    if not xs[0] <= X <= xs[-1]:
        near = xs[0] if X < xs[0] else xs[-1]
        pairs = np.sort(Zs[np.abs(Xs - _nearest(np.unique(Xs), X)) < _TOL])
        raise CompositionError(
            f"{source} cannot reach X={X:g} at Z={Z:g}. It interpolates in "
            f"hydrogen from {xs[0]:g} to {xs[-1]:g} at this metallicity, "
            f"because its higher-hydrogen columns hold one helium-free table "
            f"each. The nearest tabulated pair is X={_nearest(np.unique(Xs), X):g} "
            f"with Z={float(pairs[-1]):g}. Use X={near:g}, or that pair exactly.")

    i = int(np.clip(np.searchsorted(xs, X) - 1, 0, xs.size - 2))
    x0, x1 = xs[i], xs[i + 1]
    wx = 0.0 if x1 == x0 else (X - x0) / (x1 - x0)
    lo = _column(Xs, Zs, K, x0, target, z_floor)
    hi = _column(Xs, Zs, K, x1, target, z_floor)
    return (1 - wx) * lo + wx * hi


def _nearest(values, target):
    return float(values[int(np.argmin(np.abs(values - target)))])


def on_axes(own_T, own_R, block, log_T, log_R):
    """`block`, tabulated on `own_T` by `own_R`, on the requested axes.

    Interpolates in temperature and in density parameter separately and never
    extrapolates: a point outside the tabulated rectangle comes back NaN. A
    blank INSIDE it does not, because the temperature interpolation runs
    through the blanks and holds the last tabulated value past them.
    """
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
