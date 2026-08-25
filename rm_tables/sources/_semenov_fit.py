"""Semenov et al. 2003 dust and gas opacity, ported from `opacity.f`.

A&A 410, 611. Dust model nrm/h/s: normal iron content, homogeneous, spherical,
which is the one Ferguson et al. 2005 chose for its own comparison.

The dust opacity is a fifth-degree polynomial per temperature region with its
coefficients carried below, so no Fortran compiler is needed. Only the gas
opacity above roughly 1500 K reads a file. Against the compiled Fortran over 24
runs of 201 temperatures, the maximum relative difference is 1.776e-05, which is
the Fortran's own printed precision rather than a porting error.

Licence, from `opacity.f` lines 8 and 9, verbatim: "one can freely use, modify,
or redistribute all parts of the code." Copyright Dmitry Semenov, AIU Jena,
2001-2002.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --- dust fit coefficients, nrm_h_s, opacity.f:910-978. Order a0..a5 per region
eR = np.array([
 [ 0.244153190819473e-03,-0.191588103585833e-03, 0.229132294521843e-03,
   0.454088516995579e-06,-0.389280273285906e-08, 0.319006401785413e-11],
 [ 0.292517586013911e+01,-0.801305197566973e-01, 0.923001367150898e-03,
  -0.349287851393541e-05, 0.587151269508960e-08,-0.371333580736933e-11],
 [-0.125893536782728e+02, 0.160197849829283e+00,-0.582601311634824e-03,
   0.110792133181219e-05,-0.105506373600323e-08, 0.404997080931974e-12],
 [-0.192550086994197e+01, 0.354301116460647e-01,-0.127355043519226e-03,
   0.233773506650911e-06,-0.207587683535083e-09, 0.728005983960845e-13],
 [ 0.165244978116957e+01,-0.260479348963077e-02, 0.590717655258634e-05,
  -0.300202906476749e-08, 0.803836688553263e-12,-0.916709776405312e-16]])
eP = np.array([
 [ 0.423062838413742e-03,-0.191556936842122e-02, 0.130732588474620e-02,
  -0.108371821805323e-04, 0.427373694560880e-07,-0.664969066656727e-10],
 [-0.173587657890234e+01, 0.302186734201724e-01, 0.311604311624702e-03,
  -0.210704968520099e-05, 0.472321713029246e-08,-0.371134628305626e-11],
 [-0.638046050114383e+01, 0.120954274022502e+00,-0.436299138970822e-03,
   0.784339065020565e-06,-0.681138400996940e-09, 0.233228177925061e-12],
 [-0.345042341508906e+01, 0.664915248499724e-01,-0.240971082612604e-03,
   0.417313950448598e-06,-0.349467552126090e-09, 0.115933380913977e-12],
 [ 0.667823366719512e+01,-0.166789974601131e-01, 0.238329845360675e-04,
  -0.140583908595144e-07, 0.417753047583439e-11,-0.503398974713655e-15]])

# --- evaporation-temperature table, opacity.f:521-541
RO_EV = np.array([1e-18,1e-16,1e-14,1e-12,1e-10,1e-08,1e-06,1e-04])
TT = np.array([
 [1.090e2,1.180e2,1.290e2,1.430e2,1.590e2,1.800e2,2.070e2,2.440e2],
 [8.350e2,9.080e2,9.940e2,1.100e3,1.230e3,1.395e3,1.612e3,1.908e3],
 [9.020e2,9.800e2,1.049e3,1.129e3,1.222e3,1.331e3,1.462e3,1.621e3],
 [9.290e2,9.970e2,1.076e3,1.168e3,1.277e3,1.408e3,1.570e3,1.774e3]])

# --- gas grid axes, opacity.f:669-724
T_G = np.array([
 500.00,521.86,544.68,568.50,593.35,619.30,646.38,674.64,704.14,734.93,
 767.06,800.60,835.61,872.15,910.28,950.08,991.63,1034.99,1080.24,1127.47,
 1176.77,1228.23,1281.93,1337.99,1396.49,1457.55,1521.28,1587.80,1657.23,
 1729.69,1805.32,1884.26,1966.65,2052.64,2142.39,2236.07,2333.84,2435.89,
 2542.40,2653.56,2769.59,2890.69,3017.09,3149.01,3286.70,3430.41,3580.41,
 3736.96,3900.36,4070.91,4248.91,4434.69,4628.60,4830.98,5042.22,5262.69,
 5492.80,5732.98,5983.65,6245.29,6518.36,6803.38,7100.86,7411.34,7735.41,
 8073.64,8426.66,8795.12,9179.68,9581.07,10000.00])
RHO_G = np.array([
 0.2364e-06,0.1646e-06,0.1147e-06,0.7985e-07,0.5560e-07,0.3872e-07,
 0.2697e-07,0.1878e-07,0.1308e-07,0.9107e-08,0.6342e-08,0.4417e-08,
 0.3076e-08,0.2142e-08,0.1492e-08,0.1039e-08,0.7234e-09,0.5038e-09,
 0.3508e-09,0.2443e-09,0.1701e-09,0.1185e-09,0.8252e-10,0.5746e-10,
 0.4002e-10,0.2787e-10,0.1941e-10,0.1352e-10,0.9412e-11,0.6554e-11,
 0.4565e-11,0.3179e-11,0.2214e-11,0.1542e-11,0.1074e-11,0.7476e-12,
 0.5206e-12,0.3626e-12,0.2525e-12,0.1758e-12,0.1225e-12,0.8528e-13,
 0.5939e-13,0.4136e-13,0.2880e-13,0.2006e-13,0.1397e-13,0.9727e-14,
 0.6774e-14,0.4717e-14,0.3285e-14,0.2288e-14,0.1593e-14,0.1109e-14,
 0.7726e-15,0.5381e-15,0.3747e-15,0.2609e-15,0.1817e-15,0.1265e-15,
 0.8813e-16,0.6137e-16,0.4274e-16,0.2976e-16,0.2073e-16,0.1443e-16,
 0.1005e-16,0.7000e-17,0.4875e-17,0.3395e-17,0.2364e-17])


def read_gas(path):
    """eG[i, j], i over RHO_G, j over T_G. opacity.f:474-483."""
    v = np.array(open(path).read().split()[1:], dtype=float)
    return v.reshape(71, 71)


def _bint(xa, x, ri):
    """opacity.f:765-822. Linear, clamped index, never extrapolates below."""
    if xa < x[0]:
        i2 = 1
    elif xa > x[-1]:
        i2 = len(x) - 1
    else:
        low, hi = 0, len(x)
        while True:
            mid = (hi + low + 1) // 2
            if xa > x[mid]:
                low = mid
            else:
                hi = mid
                if xa >= x[mid - 1]:
                    break
        i2 = hi
    i1 = i2 - 1
    return (ri[i2] - ri[i1]) / (x[i2] - x[i1]) * (xa - x[i1]) + ri[i1]


def _eint(X, Y, Dm, XP, YP):
    """opacity.f:824-902. Bilinear in Y then X, quadratic extrapolation in X."""
    N, M = len(X), len(Y)
    ip = 0
    for i in range(M):
        if YP <= Y[i]:
            ip = i
            break
    if ip == 0:
        raise ValueError("EINT: YP outside the range of Y")
    j = ip - 1                       # 0-based left knot
    xt = Dm[:, j] + (Dm[:, j + 1] - Dm[:, j]) / (Y[j + 1] - Y[j]) * (YP - Y[j])
    ip = 0
    for i in range(N):
        if XP >= X[i]:
            ip = i
            break
    if ip != 0:
        k = ip - 1
        return xt[k] + (xt[k + 1] - xt[k]) / (X[k + 1] - X[k]) * (XP - X[k])
    a = ((xt[N-1]-xt[N-2])/(X[N-1]-X[N-2]) - (xt[N-2]-xt[N-3])/(X[N-2]-X[N-3])) \
        / (X[N-1] - X[N-3])
    b = (xt[N-1]-xt[N-2])/(X[N-1]-X[N-2]) - a*(X[N-1]+X[N-2])
    c = xt[N-1] - a*X[N-1]**2 - b*X[N-1]
    return a*XP*XP + b*XP + c


def gop(eG, rho, T):
    """opacity.f:668-763. Zero outside the gas grid's own bounds."""
    if rho > 1.0e-7 or rho < 1.0e-19 or T < 500.0 or T > 10000.0:
        return 0.0
    return 10.0 ** _eint(RHO_G, T_G, eG, rho, T)


def cop(eD, eG, rho, T_in):
    """opacity.f:510-666. Rosseland or Planck mean extinction, cm^2/g."""
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
        return gop(eG, rho, T_in)
    smooth = any(abs(T_in - T[i]) <= dT[i] for i in range(5))
    if not smooth:
        return np.polyval(eD[kk][::-1], T_in)
    T1, T2, TD = T[kk] - dT[kk], T[kk] + dT[kk], T_in - T[kk]
    aKrL = np.polyval(eD[kk][::-1], T1)
    aKrR = gop(eG, rho, T2) if kk == 4 else np.polyval(eD[kk+1][::-1], T2)
    AA, BB = 0.5*(aKrL - aKrR), 0.5*(aKrL + aKrR)
    return BB - AA*np.sin(np.pi/2.0/dT[kk]*TD)
