# Using rm-tables

Rosseland mean opacity at a requested composition, either as a callable or as a
tabulated grid. Every number below is what the example printed.

`opacity` answers at a point, reading the sources directly. `build` returns
grids for anyone fitting their own interpolant or writing a file. `load` reads
such a file back.

## An opacity at a point, with `opacity`

`opacity` returns a callable. Building it reads the tables once, so it is worth
keeping and calling many times.

```python
import numpy as np
import rm_tables

kappa = rm_tables.opacity()
print(kappa(3000.0, 1e-14))    # -> 6.635618312601376e-05
```

The call is `kappa(T, rho)`: the temperature in K first, the density in
g/cm^3 second, and the result in cm^2/g. Both arguments are plain numbers, so
a swap cannot be detected and gives an opacity that is silently wrong.

Arrays broadcast against each other and the result takes the broadcast shape.

```python
T = np.array([500.0, 3000.0, 1e5])
rho = np.array([1.25e-19, 1e-14, 1e-16])
print(kappa(T, rho))    # -> [1.7535...e+00 6.6356...e-05 4.5123...e-01]
```

A column of temperatures against a row of densities gives the rectangle.

```python
column = np.array([1e3, 1e4])[:, None]
row = np.array([1e-14, 1e-13, 1e-12])[None, :]
print(kappa(column, row).shape)    # -> (2, 3)
```

## Composition

Hydrogen and metals are arguments and helium is the remainder. Opacity tables
are indexed by hydrogen and metals only.

```python
solar = rm_tables.opacity(X=0.7381, Z=0.0134)
poor = rm_tables.opacity(X=0.7381, Z=0.00134)
print(solar(500.0, 1e-14))    # -> 1.7535359703708098
print(poor(500.0, 1e-14))     # -> 0.17535359703708098
```

A tenth of the metals gives a tenth of the opacity at 500 K. In the dust regime
the Rosseland mean is a harmonic average over grain absorption, so scaling every
monochromatic opacity scales the mean by the same factor. That holds at fixed
grain sizes and mineralogy, which no argument here changes. It applies to the
dust alone; the gas keeps its tabulated values.

The default set tabulates metals from 0 to 0.1 at eight hydrogen fractions,
0 to 0.9. Hydrogen of 0.95 reaches metals of 0.04. Otherwise above 0.9 it holds
one table per hydrogen fraction, the helium-free one where `X + Z = 1`, so 0.97
exists only with metals at 0.03. A request that no pair brackets raises, as
does a pair leaving helium negative. Both raise a `ValueError`.

```python
try:
    rm_tables.opacity(X=0.9, Z=0.3)
except ValueError as e:
    print(e)    # -> X=0.9 and Z=0.3 leave helium at -0.2. Helium is the ...
```

OPAL's fixed-composition sets hold one hydrogen fraction each, named in the
file. Asking one of those for a different hydrogen fraction raises.

```python
try:
    rm_tables.opacity(X=0.35, Z=0.02, opal_set="Gz020.x70")
except ValueError as e:
    print(str(e).split(", and")[0])    # -> set 'Gz020.x70' tabulates hydrogen from 0.7 to 0.7
```

## Choosing the cold source

Semenov et al. 2003 tabulates the temperature at which existing grains
evaporate. Ferguson et al. 2005 tabulates the temperature at which grains
condense out of cooling gas. Grains are destroyed at a lower temperature than
they form at, which Ferguson et al. 2005 states in its section 4.

Heating material takes Semenov, as in an accretion flow. Cooling material takes
Ferguson, as in an atmosphere.

```python
print(rm_tables.opacity(cold="semenov")(700.0, 1e-14))     # -> 1.3268899275683468
print(rm_tables.opacity(cold="ferguson")(700.0, 1e-14))    # -> 0.5447430109726219
```

Two Rosseland means cannot be averaged, so the package offers no mixture.

Ferguson stops at 501 K and Semenov reaches 5 K. Ferguson reaches a higher
density than Semenov above 2154 K, and carries a composition through its own
grain chemistry rather than by the scaling above.

## Choosing the metal mixture

OPAL supplies the hot end. Its 77 published files carry different metal
mixtures, and all of them ship.

```python
from rm_tables.sources import opal

print(len(opal.sets()))       # -> 77
print(opal.sets()[28:33])     # -> ('GN93hz.CNOtoNe', 'GN93hz.COtoN', ...
```

`opal_set` selects one. The mixture moves the answer at 1e5 K by about 9 percent
across the four main sets.

```python
for name in ("GN93hz", "GS98hz", "AGS04hz", "W95hz"):
    print("%-9s %.6f" % (name, rm_tables.opacity(opal_set=name)(1e5, 1e-8)))

# -> GN93hz    0.927698
# -> GS98hz    0.936628
# -> AGS04hz   0.967055
# -> W95hz     0.885030
```

The default is `GS98hz`, at Grevesse & Sauval 1998 metal ratios. `GN93hz` is
the same mixture as revised in 1993 and differs by a median 0.002 dex.

## Alpha enhancement

Alpha enhancement is the abundance of oxygen, magnesium and silicon against
iron, at fixed total metal fraction. No choice of `X` and `Z` reaches it, since
those set how much metal there is and not which metals. Ferguson computed a
separate set of 155 tables at each of six enhancements, and all six ship.

```python
from rm_tables.sources import ferguson

print(ferguson.alphas())     # -> [-0.2  0.   0.2  0.4  0.6  0.8]
```

`alpha` selects one. Raising it lowers the dust opacity, because the metal
fraction is held fixed, so more oxygen, magnesium and silicon means less iron,
and iron is a major grain opacity carrier.

```python
for a in ferguson.alphas():
    print("%+.1f  %.6f" % (a, rm_tables.opacity(cold="ferguson", alpha=a)(800.0, 1e-12)))

# -> -0.2  0.635919
# -> +0.0  0.569248
# -> +0.2  0.486439
# -> +0.4  0.436135
# -> +0.6  0.354785
# -> +0.8  0.304262
```

The default is 0.0. Those six sets are at Grevesse & Sauval 1998 metal ratios.
`alpha=None` selects `f05.g93`, at Grevesse & Noels 1993 ratios, which was this
package's only cold Ferguson table before the others were added.

```python
print(rm_tables.opacity(cold="ferguson", alpha=None)(800.0, 1e-12))  # -> 0.5331353949488168
```

The six are separate calculations rather than points on an axis this package
interpolates, so an untabulated enhancement raises.

```python
try:
    rm_tables.opacity(cold="ferguson", alpha=0.3)
except rm_tables.coverage.CoverageError as e:
    print(e)    # -> Ferguson tabulates alpha enhancement at -0.2, 0.0, 0.2, ...
```

Alpha reaches the dust and molecular opacity only. Above 10,000 K the hot
source answers and `opal_set` selects its mixture. OPAL's own enhanced and
unenhanced tables differ by 0.015 dex there, against 0.189 dex for the same
enhancement across the cold table, so the hot end is left to `opal_set` rather
than moved on the caller's behalf. Semenov carries no alpha enhancement, and
passing one with `cold="semenov"` warns and is ignored.

The set actually read is in the provenance.

```python
print(rm_tables.opacity(cold="ferguson", alpha=0.6).provenance["cold_set"])
# -> f05.gs98+.6
```

Carbon and oxygen beyond what `Z` carries are `dXc` and `dXo`. Those grids live
in OPAL's fixed-composition files, whose hydrogen and metal fractions come from
the file rather than being interpolated, so a set carrying them is chosen by
name.

```python
at = dict(X=0.70, Z=0.02, opal_set="Gz020.x70")
print(rm_tables.opacity(**at)(1e5, 1e-8))              # -> 0.9931160483613025
print(rm_tables.opacity(dXc=0.1, **at)(1e5, 1e-8))     # -> 1.0209394827969092
```

`rm_tables.sources.opal.excess` lists the pairs one set tabulates. A pair it
does not hold raises and names the set.

## Which entry point to use

`opacity` reads the sources rather than a grid of this package's choosing, and
suits work driven from Python: one-off values, a plot, a sweep over arrays. The
OPAL interpolation and Ferguson's grid are read once, when the callable is
built; only Semenov's polynomial is evaluated per point. On arrays it costs
246 ns a point. One point at a time from Python costs 6,841 ns, almost all of
it Python overhead.

A compiled solver cannot call the object: numba refuses a Python object holding
Python arrays. `Opacity.as_compiled` returns the same opacity as a numba
function, which can.

```python
import numpy as np
import rm_tables
from numba import njit

kappa = rm_tables.opacity(X=0.7381, Z=0.0134).as_compiled()

@njit
def optical_depth(T, rho, height):
    return kappa(T, rho) * rho * height

print(optical_depth(3000.0, 1e-14, 1e13))    # -> 6.6356...e-06
```

It costs 157 ns a point inside a compiled loop. Compiling costs about 0.4 s
once, so build it once and keep it: building it again compiles it again.

A compiled function cannot raise, so the refusal the object makes an exception
is a NaN here. `kappa(1e9, 1e-8)` and `kappa(3000.0, -1.0)` are NaN, at the
same bounds the object raises on. The range check is 1.006 times the unchecked
call, median of nine interleaved runs of 200,000 points.

### Fitting an interpolant instead

A spline over a built table is slower than either, at 513 ns a point, and it
resamples the sources. It earns its place where the solver needs smooth
derivatives, since the compiled lookup above is bilinear in OPAL and has a
kinked derivative there.

`scijit` supplies scipy-compatible routines that compile under numba:

```python
import numpy as np
import rm_tables
from numba import njit
from scijit.interpolate import RectBivariateSpline, bispeu

t = rm_tables.build(X=0.7381, Z=0.0134)

# A spline cannot carry NaN, and OPAL leaves two corners blank.
cold = RectBivariateSpline(t.cold_log_T, t.cold_log_R,
                           np.nan_to_num(t.cold, nan=-4.0))
hot = RectBivariateSpline(t.hot_log_T, t.hot_log_R,
                          np.nan_to_num(t.hot, nan=-4.0))

# Read the knots and coefficients out once, so the lookup below holds numbers
# rather than a Python object, which numba cannot compile.
ctx, cty, cc, ckx, cky = cold.tx, cold.ty, cold.c, cold.kx, cold.ky
htx, hty, hc, hkx, hky = hot.tx, hot.ty, hot.c, hot.kx, hot.ky
split = float(t.split_log_T)
cold_r_max = float(t.cold_log_R[-1])
hot_t_min = float(t.hot_log_T[0])

@njit
def kappa(T, rho):
    """Rosseland mean opacity in cm^2/g, from K and g/cm^3."""
    x = np.empty(1)
    y = np.empty(1)
    x[0] = np.log10(T)
    y[0] = np.log10(rho / (T * 1e-6) ** 3)
    # `Tables.is_hot`, inlined: numba cannot call the method.
    if x[0] >= split or (y[0] > cold_r_max and x[0] >= hot_t_min):
        return 10.0 ** bispeu(x, y, htx, hty, hc, hkx, hky)[0]
    return 10.0 ** bispeu(x, y, ctx, cty, cc, ckx, cky)[0]

print(kappa(1e6, 1e-8))    # -> 0.3500...
```

The fit is not free. Against the sources, over a grid spanning the cold table,
it reads to a median 0.0004 dex, with 8.6 percent of points past 0.1 dex and a
worst case of 1.87 dex at 1381 K in the dilute corner, where a bicubic spline
overshoots the dust-destruction cliff. Raising `n_T` narrows the cliff rather
than the overshoot.

`agndisks` carries a worked version of this, with the spline cached per
composition and the table optionally read from a file.

## A tabulated grid, with `build`

`build` returns `log10` opacity in cm^2/g on a regular grid of `log10 T` and
`log10 R`, where `R = rho / (T / 1e6)**3` in g/cm^3. Two grids come back.
`cold` carries the cold source ramped into OPAL. `hot` carries OPAL alone.
Building the default takes about 1.4 s.

```python
t = rm_tables.build(X=0.7381, Z=0.0134)
print(t.cold.shape, t.hot.shape)    # -> (200, 500) (200, 500)
```

The range and the resolution are arguments. A disc that never leaves 100 K to
1e6 K needs no table below that.

```python
small = rm_tables.build(log_T_range=(2.0, 6.0), log_R_range=(-8.0, -2.0),
                        n_T=300, n_R=400)
print(small.cold.shape)         # -> (300, 400)
print(small.cold_log_T[0])      # -> 2.0
print(small.cold_log_R[-1])     # -> -2.0
```

Both ranges describe the cold grid. The hot grid runs from OPAL's floor of
5623 K to the same ceiling, over its own density range, whose top is the
`hot_log_R_max` argument, 1.0 by default.

The resolution changes the table. Semenov's dust destruction spans 192 K, which
is two and a half cells at the default grid, so the tabulated cliff is gentler
than the model's own. Two tables built at different resolutions are not
comparable, so every table records the resolution it was built at.

```python
print(t.provenance["n_T"], t.provenance["n_R"])    # -> 200 500
```

`help(rm_tables.build)` documents all thirteen arguments, the value each falls
back to, and the measurement that set it.

## The pair of grids

The two grids work together. `cold` carries the dust and molecular opacity,
`hot` carries OPAL, and a lookup takes whichever holds the point.

    cold   5 K to 1.26e7 K       log R -8 to -0.25    Semenov, ramped into OPAL
    hot    5623 K to 1.26e7 K    log R -8 to +1.0     OPAL alone

The pair exists because of the cold ceiling of -0.25. Semenov's gas grid is
bounded in density, so its reach in `log R` falls as `11 - 3 log10 T`, dropping
below OPAL's ceiling at 2154 K and reaching -0.25 by the time OPAL starts at
5623 K. One rectangle covering both would have to stop at -0.25 everywhere,
throwing away the density height OPAL supports. A second grid, OPAL alone,
carries the full height.

`Tables.is_split` says which came back, and `hot` is `None` on a single grid.

```python
one = rm_tables.build(log_T_range=(0.699, 3.0), n_T=40, n_R=25)
print(one.is_split)    # -> False

pair = rm_tables.build(n_T=40, n_R=25)
print(pair.is_split)             # -> True
print(pair.cold_log_R[-1])       # -> -0.25
print(pair.hot_log_R[-1])        # -> 1.0
```

`Tables.kappa` reads the right grid without being told which.

```python
print(pair.kappa(np.array([3e3, 1e5]), np.array([1e-14, 1e-16])))
# -> [5.7967...e-05 4.6252...e-01]
```

Fitting a smoother interpolant means fitting one per grid and choosing between
them with `Tables.is_hot`, which answers `True` where the hot grid holds the
point. A point denser than the cold grid reaches goes to the hot grid only
where the hot grid holds its temperature, which starts at 5623 K. Below that
the cold grid answers and holds its density edge. Over 200 temperatures from
100 K to 5623 K that reads to a median 0.001 dex, with the failures confined to
the dust-destruction cliff, where one temperature of the 200 is out by 0.709
dex.

```python
cool, warm = np.log10(3000.0), np.log10(2e4)
print(pair.is_hot(cool, -6.0), pair.is_hot(cool, 0.5))    # -> False False
print(pair.is_hot(warm, 0.5), pair.is_hot(5.0, -13.0))    # -> True True
```

`Tables.split_log_T` is the temperature it switches at, 4.0 in `log10` by
default. `help` on the returned object documents every attribute.

Splitting is permitted rather than forced. A range that sits on one side of the
split temperature comes back as one grid whatever `allow_split` says, because
the second would be empty. `allow_split=False` requires a single grid and
raises rather
than clipping.

```python
try:
    rm_tables.build(log_R_range=(-8.0, 1.0), n_T=40, n_R=25, allow_split=False)
except ValueError as e:
    print(str(e).splitlines()[-1])    # -> Splitting the table does not help. ...
```

## Saving and loading

The extension picks the format.

    .npz          compressed NumPy archive. Compact, Python only.
    .txt, .dat    plain text with a readable header. Any language, any editor.
    .h5, .hdf5    HDF5. Any scientific language, needs h5py installed.

Each carries the composition, the sources, their references, the resolution and
the units alongside the numbers, so a file found later can be traced to what
produced it.

```python
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "solar.txt")
t.save(path)
print(rm_tables.load(path))
# -> Tables(X=0.7381, Z=0.0134, cold='semenov', (200, 500) + (200, 500))
```

## A request outside the sources

A range leaving what the chosen sources hold raises. The message names the
offending corner and any source that would cover it.

```python
try:
    rm_tables.build(log_T_range=(0.0, 4.0), n_T=40, n_R=25)
except ValueError as e:
    print(str(e).splitlines()[0])
# -> the requested range is not covered by semenov, opal: 175 of 1000 points ...
```

`opacity` holds the nearest value past a density edge, matching what a
tabulated grid does past its own span. A temperature no source reaches raises
instead, because holding across a temperature edge would return a dust opacity
for an ionised gas. `Opacity.as_compiled` returns NaN at those same bounds.

```python
try:
    rm_tables.opacity()(2.0, 1e-14)
except ValueError as e:
    print(e)    # -> semenov has no value below 5 K, and 2 K was requested. ...
```

The error is `rm_tables.coverage.CoverageError`, which subclasses `ValueError`,
so catching `ValueError` is enough.

A zero is never returned. Semenov reports "no value here" as a zero, which is a
sentinel and not an opacity: a Rosseland mean cannot be zero, and passing the
sentinel on sends an optical depth to zero. The value at the nearest density
Semenov holds is returned in its place.

OPAL's own tables stop short in the cool dilute corner and the hot dense one.
Those blanks sit inside its declared range, so the coverage check passes and the
nearest tabulated value is held across them.

## Compiled routines

The lookups are compiled with numba, which turns a Python function into machine
code the first time it runs. Compiling all of them takes about 1.3 s, once per
session, and the result is cached on disk beside the installed package.

Where that location is not writable, `NUMBA_CACHE_DIR` points them somewhere
that is. Without it the routines recompile each session and nothing else
changes.

## Citation

The hot opacity always comes from OPAL, and the cold opacity from whichever
source `cold` names. A published result cites both, and this package.

| what | reference | bibcode |
| --- | --- | --- |
| hot opacity, always | Iglesias & Rogers 1996, ApJ 464, 943 | [1996ApJ...464..943I](https://ui.adsabs.harvard.edu/abs/1996ApJ...464..943I/abstract) |
| cold opacity, `cold="semenov"` | Semenov et al. 2003, A&A 410, 611 | [2003A&A...410..611S](https://ui.adsabs.harvard.edu/abs/2003A%26A...410..611S/abstract) |
| cold opacity, `cold="ferguson"` | Ferguson et al. 2005, ApJ 623, 585 | [2005ApJ...623..585F](https://ui.adsabs.harvard.edu/abs/2005ApJ...623..585F/abstract) |

A result that depends on the metal mixture rather than on the metal mass
fraction alone carries one more. The default OPAL set `GS98hz` is at Grevesse
& Sauval 1998 ratios, as are the six alpha-enhanced Ferguson sets,
[1993oee..conf...15G](https://ui.adsabs.harvard.edu/abs/1993oee..conf...15G/abstract).
`rm_tables.sources.opal.sets()` names the other sets, each carrying its own
mixture. Semenov's coefficients are computed at Anders &
Grevesse 1989,
[1989GeCoA..53..197A](https://ui.adsabs.harvard.edu/abs/1989GeCoA..53..197A/abstract),
and a requested metallicity is scaled from that composition.

`CITATION.cff` in the repository root carries the same list in machine-readable
form, and ADS exports BibTeX from any of these bibcodes.
