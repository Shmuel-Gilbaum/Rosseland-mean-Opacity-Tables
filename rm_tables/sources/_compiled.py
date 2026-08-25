"""Compiled routines: Semenov's routine, and a bilinear lookup for OPAL.

Semenov's routine is a transcription of the published Fortran, changed only
where numba refuses to compile the original, and the test suite checks it
against that transcription bit for bit. The bilinear lookup is checked at a
grid's own nodes and on a linear function.

Compiling all of them takes about 2 s, and only happens if a compiled path is
used. numba caches the result beside the source file, so a later session
reloads it in about 0.2 s instead. Where the cache cannot be written, every
session compiles.
"""
import numpy as np
from numba import njit

from . import _semenov_fit as _f

__all__ = ["semenov_kappa", "bilinear", "ED_HI_FIRST", "cached_njit"]


def cached_njit(fn):
    """``njit(cache=True)``, degrading to no cache where numba cannot place one.

    numba writes its cache beside the source file, or under ``NUMBA_CACHE_DIR``
    if that is set. Where neither is writable it raises as the decorator runs,
    which is on import, and the fallback keeps the import working at the cost
    of compiling once per session.
    """
    try:
        return njit(cache=True)(fn)
    except RuntimeError:
        return njit(cache=False)(fn)

# numba has no ``np.polyval`` and reversed-stride slicing is awkward inside it.
# Reversing the coefficients once here means the compiled code needs neither.
ED_HI_FIRST = np.ascontiguousarray(_f.eR[:, ::-1])
_RO_EV = np.ascontiguousarray(_f.RO_EV)
_TT = np.ascontiguousarray(_f.TT)
_RHO_G = np.ascontiguousarray(_f.RHO_G)
_T_G = np.ascontiguousarray(_f.T_G)


@cached_njit
def _polyval(c, x):
    """``np.polyval``, coefficients highest order first."""
    acc = 0.0
    for k in range(c.size):
        acc = acc * x + c[k]
    return acc


@cached_njit
def _bint(xa, x, ri):
    """Linear in the evaporation table, clamped. `opacity.f` lines 765 to 822."""
    n = x.size
    if xa <= x[0]:
        i2 = 1
    elif xa >= x[n - 1]:
        i2 = n - 1
    else:
        low = 0
        hi = n - 1
        while hi - low > 1:
            mid = (low + hi) // 2
            if x[mid] < xa:
                low = mid
            else:
                hi = mid
                if xa >= x[mid - 1]:
                    break
        i2 = hi
    i1 = i2 - 1
    return (ri[i2] - ri[i1]) / (x[i2] - x[i1]) * (xa - x[i1]) + ri[i1]


@cached_njit
def _eint(X, Y, Dm, XP, YP):
    """Linear in Y then X on the gas grid. `opacity.f` lines 824 to 902.

    X descends. Past its end the routine extrapolates quadratically.
    """
    N = X.size
    M = Y.size
    ip = 0
    for i in range(M):
        if YP <= Y[i]:
            ip = i
            break
    j = ip - 1
    xt = np.empty(N)
    for k in range(N):
        xt[k] = (Dm[k, j] + (Dm[k, j + 1] - Dm[k, j])
                 / (Y[j + 1] - Y[j]) * (YP - Y[j]))
    ip = 0
    for i in range(N):
        if XP >= X[i]:
            ip = i
            break
    if ip != 0:
        k = ip - 1
        return xt[k] + (xt[k + 1] - xt[k]) / (X[k + 1] - X[k]) * (XP - X[k])
    a = (((xt[N-1]-xt[N-2])/(X[N-1]-X[N-2])
          - (xt[N-2]-xt[N-3])/(X[N-2]-X[N-3])) / (X[N-1] - X[N-3]))
    b = (xt[N-1]-xt[N-2])/(X[N-1]-X[N-2]) - a*(X[N-1]+X[N-2])
    c = xt[N-1] - a*X[N-1]**2 - b*X[N-1]
    return a*XP*XP + b*XP + c


@cached_njit
def _gop(eG, rho, T):
    """Gas opacity, 0 outside the grid's own bounds."""
    if rho > 1.0e-7 or rho < 1.0e-19 or T < 500.0 or T > 10000.0:
        return 0.0
    return 10.0 ** _eint(_RHO_G, _T_G, eG, rho, T)


GAS_RHO_MIN = 1.0e-19
GAS_RHO_MAX = 1.0e-7
"""Density bounds of Semenov's gas grid, g/cm^3, checked in `opacity.f`.

The dust polynomial has no density dependence, so only the gas branch is bounded.
Outside these the routine returns zero, meaning it has no value rather than that
the opacity vanishes.
"""


@cached_njit
def semenov_kappa_held(eD, eG, rho, T_in, zfac):
    """`semenov_kappa`, holding the nearest valid density instead of returning 0.

    Outside the gas grid's density bounds the routine reports no value as a
    zero. This evaluates at the nearest bound instead, which is what a
    tabulated grid does outside its own span.
    """
    v = semenov_kappa(eD, eG, rho, T_in, zfac)
    if v > 0.0:
        return v
    held = min(max(rho, GAS_RHO_MIN), GAS_RHO_MAX)
    if held != rho:
        v = semenov_kappa(eD, eG, held, T_in, zfac)
    return v


@cached_njit
def semenov_kappa(eD, eG, rho, T_in, zfac):
    """Rosseland mean, cm^2/g, or 0 where Semenov has no value.

    `opacity.f` lines 510 to 666. ``zfac`` multiplies the dust polynomials and
    never the gas grid; at ``zfac = 1`` this is the unmodified routine.
    """
    if T_in < 1.0:
        return 0.0
    T_ev = np.empty(4)
    for i in range(4):
        T_ev[i] = _bint(rho, _RO_EV, _TT[i])
    T = np.empty(5)
    T[0] = T_ev[0]
    T[1] = 275.0
    T[2] = 425.0
    T[3] = 680.0
    T[4] = min(max(T_ev[1], T_ev[2]), max(T_ev[2], T_ev[3]))
    dT = np.empty(5)
    dT[0] = 5.0
    dT[1] = 5.0
    dT[2] = 15.0
    dT[3] = 5.0
    dT[4] = 100.0

    kk = 5
    if T_in <= T[0] + dT[0]:
        kk = 0
    for it in range(1, 5):
        if T[it - 1] + dT[it - 1] < T_in <= T[it] + dT[it]:
            kk = it
    if kk == 5:
        return _gop(eG, rho, T_in)

    smooth = False
    for i in range(5):
        if abs(T_in - T[i]) <= dT[i]:
            smooth = True
    if not smooth:
        return zfac * _polyval(eD[kk], T_in)

    T1 = T[kk] - dT[kk]
    T2 = T[kk] + dT[kk]
    left = zfac * _polyval(eD[kk], T1)
    if kk == 4:
        right = _gop(eG, rho, T2)
    else:
        right = zfac * _polyval(eD[kk + 1], T2)
    amp = 0.5 * (left - right)
    mid = 0.5 * (left + right)
    return mid - amp * np.sin(np.pi / 2.0 / dT[kk] * (T_in - T[kk]))


@cached_njit
def bilinear(x_ax, y_ax, block, x, y):
    """Bilinear on an unevenly spaced grid, held at the edges."""
    nx = x_ax.size
    ny = y_ax.size
    if x <= x_ax[0]:
        x = x_ax[0]
    elif x >= x_ax[nx - 1]:
        x = x_ax[nx - 1]
    if y <= y_ax[0]:
        y = y_ax[0]
    elif y >= y_ax[ny - 1]:
        y = y_ax[ny - 1]
    i = 0
    while i < nx - 2 and x_ax[i + 1] < x:
        i += 1
    j = 0
    while j < ny - 2 and y_ax[j + 1] < y:
        j += 1
    u = (x - x_ax[i]) / (x_ax[i + 1] - x_ax[i])
    v = (y - y_ax[j]) / (y_ax[j + 1] - y_ax[j])
    return ((1 - u) * (1 - v) * block[i, j] + u * (1 - v) * block[i + 1, j]
            + (1 - u) * v * block[i, j + 1] + u * v * block[i + 1, j + 1])
