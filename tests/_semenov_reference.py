"""Semenov's published routine, transcribed and left alone.

The package computes the opacity through a compiled rewrite and through
`rm_tables.sources.semenov.kappa`, which adds the metallicity scaling. Both are
checked against this transcription, so it stays here rather than in the package:
its only job is to be the thing they are compared to.

`opacity.f` lines 510 to 666, Semenov et al. 2003, A&A 410, 611.
"""
import numpy as np

from rm_tables.sources._semenov_fit import RO_EV, TT, _bint, eR, gop

__all__ = ["cop", "eR"]


def cop(dust_coeffs, gas_grid, rho, T_in):
    """Rosseland or Planck mean extinction, cm^2/g."""
    if T_in < 1.0:
        return 0.0
    T_ev = [_bint(rho, RO_EV, TT[i]) for i in range(4)]
    T = [T_ev[0], 275.0, 425.0, 680.0,
         min(max(T_ev[1], T_ev[2]), max(T_ev[2], T_ev[3]))]
    dT = [5.0, 5.0, 15.0, 5.0, 100.0]
    kk = 5                                   # 0-based; 5 means gas
    if T_in <= T[0] + dT[0]:
        kk = 0
    for it in range(1, 5):
        if T[it-1] + dT[it-1] < T_in <= T[it] + dT[it]:
            kk = it
    if kk == 5:
        return gop(gas_grid, rho, T_in)
    smooth = any(abs(T_in - T[i]) <= dT[i] for i in range(5))
    if not smooth:
        return np.polyval(dust_coeffs[kk][::-1], T_in)
    T1, T2, TD = T[kk] - dT[kk], T[kk] + dT[kk], T_in - T[kk]
    aKrL = np.polyval(dust_coeffs[kk][::-1], T1)
    aKrR = gop(gas_grid, rho, T2) if kk == 4 else np.polyval(dust_coeffs[kk+1][::-1], T2)
    AA, BB = 0.5*(aKrL - aKrR), 0.5*(aKrL + aKrR)
    return BB - AA*np.sin(np.pi/2.0/dT[kk]*TD)
