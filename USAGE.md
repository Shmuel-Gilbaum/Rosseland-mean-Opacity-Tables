# Using rm-tables

Rosseland mean opacity at a requested composition, either as a callable or as a
tabulated grid. Every number below is what the example printed.

`opacity` answers at a point and builds nothing. `build` returns grids for
anyone fitting their own interpolant or writing a file. `load` reads such a
file back.

## An opacity at a point, with `opacity`

`opacity` returns a callable. One build at a composition serves any number of
calls.

```python
>>> import numpy as np
>>> import rm_tables
>>> kappa = rm_tables.opacity()
>>> float(round(kappa(3000.0, 1e-14), 9))
6.6356e-05

```

Temperature is in K, density in g/cm^3, and the result in cm^2/g.

Arrays broadcast against each other and the result takes the broadcast shape.

```python
>>> T = np.array([500.0, 3000.0, 1e5])
>>> rho = np.array([1.25e-19, 1e-14, 1e-16])
>>> np.array2string(kappa(T, rho), precision=6)
'[1.753536e+00 6.635618e-05 4.460250e-01]'

```

A column of temperatures against a row of densities gives the rectangle.

```python
>>> kappa(np.array([1e3, 1e4])[:, None], np.array([1e-14, 1e-13, 1e-12])[None, :]).shape
(2, 3)

```

## Composition

Hydrogen and metals are arguments and helium is the remainder. Opacity tables
are indexed by hydrogen and metals only.

```python
>>> solar = rm_tables.opacity(X=0.7381, Z=0.0134)
>>> poor = rm_tables.opacity(X=0.7381, Z=0.00134)
>>> float(round(solar(500.0, 1e-14), 6)), float(round(poor(500.0, 1e-14), 6))
(1.753536, 0.175354)

```

A tenth of the metals gives a tenth of the opacity at 500 K, exactly. In the
dust regime the Rosseland mean is a harmonic average over grain absorption, so
scaling every monochromatic opacity scales the mean by the same factor. The
scaling is applied to the dust and never to the gas.

The default set tabulates hydrogen from 0 to 1 and metals from 0 to 0.1. A
request outside that raises, and so does a pair leaving helium negative. OPAL
interpolates linearly in hydrogen and in the logarithm of the metallicity and
continues that line past its own edge, so an unguarded request returns a number
with nothing behind it: hydrogen of 1.5 gave 5.5e-136 cm^2/g at 1e5 K.

```python
>>> try:
...     rm_tables.opacity(X=0.9, Z=0.3)
... except ValueError as e:
...     print(e)
X=0.9 and Z=0.3 leave helium at -0.2. Helium is the remainder, so X + Z cannot exceed 1.

```

OPAL's fixed-composition sets hold one hydrogen fraction each, named in the
file. Asking one of those for a different hydrogen fraction raises rather than
returning the single table it holds.

```python
>>> try:
...     rm_tables.opacity(X=0.35, Z=0.02, dataset="Gz020.x70")
... except ValueError as e:
...     print(str(e).split(", and")[0])
set 'Gz020.x70' tabulates hydrogen from 0.7 to 0.7

```

## Choosing the cold source

Semenov et al. 2003 tabulates the temperature at which existing grains
evaporate. Ferguson et al. 2005 tabulates the temperature at which grains
condense out of cooling gas. Grains are destroyed at a lower temperature than
they form at, which Ferguson et al. 2005 states in its section 4.

Heating material takes Semenov, as in an accretion flow. Cooling material takes
Ferguson, as in an atmosphere.

```python
>>> float(round(rm_tables.opacity(cold="semenov")(700.0, 1e-14), 6))
1.32689
>>> float(round(rm_tables.opacity(cold="ferguson")(700.0, 1e-14), 6))
0.511706

```

The two are alternatives and are never combined. Two Rosseland means cannot be
averaged, so no mixture of them is offered.

Ferguson stops at 501 K and Semenov reaches 5 K. Ferguson reaches a higher
density than Semenov above 2154 K, and carries a composition through its own
grain chemistry rather than by the scaling above.

## Choosing the metal mixture

OPAL supplies the hot end. Its 77 published files carry different metal
mixtures, and all of them ship.

```python
>>> from rm_tables.sources import opal
>>> len(opal.sets())
77
>>> opal.sets()[28:33]
('GN93hz.CNOtoNe', 'GN93hz.COtoN', 'GN93hz.CtoN', 'GS98hz', 'GS98hz.CNOtoNe')

```

`dataset` selects one. The mixture moves the answer at 1e5 K by about 9 percent
across the four main sets.

```python
>>> for name in ("GN93hz", "GS98hz", "AGS04hz", "W95hz"):
...     print("%-9s %.6f" % (name, rm_tables.opacity(dataset=name)(1e5, 1e-8)))
GN93hz    0.927698
GS98hz    0.936628
AGS04hz   0.967055
W95hz     0.885030

```

The default is `GN93hz`, at Grevesse & Noels 1993 metal ratios.

Carbon and oxygen beyond what `Z` carries are `dXc` and `dXo`. Those grids live
in OPAL's fixed-composition files, whose hydrogen and metal fractions come from
the file rather than being interpolated, so a set carrying them is chosen by
name.

```python
>>> at = dict(X=0.70, Z=0.02, dataset="Gz020.x70")
>>> float(round(rm_tables.opacity(**at)(1e5, 1e-8), 6))
0.993116
>>> float(round(rm_tables.opacity(dXc=0.1, **at)(1e5, 1e-8), 6))
1.020939

```

`rm_tables.sources.opal.excess` lists the pairs one set tabulates. A pair it
does not hold raises and names the set.

## A tabulated grid, with `build`

`build` returns `log10` opacity in cm^2/g on a regular grid of `log10 T` and
`log10 R`, where `R = rho / (T / 1e6)**3` in g/cm^3. It returns two of them:
`cold` carries the cold source ramped into OPAL, `hot` carries OPAL alone.
Building the default takes about 1.4 s.

```python
>>> t = rm_tables.build(X=0.7381, Z=0.0134)
>>> t.cold.shape, t.hot.shape
((200, 500), (200, 500))

```

The range and the resolution are arguments. A disc that never leaves 100 K to
1e6 K needs no table below that.

```python
>>> small = rm_tables.build(log_T_range=(2.0, 6.0), log_R_range=(-8.0, -2.0),
...                         n_T=300, n_R=400)
>>> small.cold.shape, float(small.cold_log_T[0]), float(small.cold_log_R[-1])
((300, 400), 2.0, -2.0)

```

Both ranges describe the cold grid. The hot grid runs from OPAL's floor of
5623 K to the same ceiling, over its own density range, whose top is
`hot_log_R_max`.

The resolution changes the table. Semenov's dust destruction spans 192 K, which
is two and a half cells at the default grid, so the tabulated cliff is gentler
than the model's own. Two tables built at different resolutions are not
comparable, which is why every table records the resolution it was built at.

```python
>>> t.provenance["n_T"], t.provenance["n_R"]
(200, 500)

```

`help(rm_tables.build)` documents all thirteen arguments and
`rm_tables.defaults` holds the value each falls back to, with the measurement
that set it.

## The pair of grids

The two grids are a pair, not two independent tables. `cold` carries the dust
and molecular opacity and `hot` carries OPAL, and a lookup takes whichever holds
the point.

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
>>> rm_tables.build(log_T_range=(0.699, 3.0), n_T=40, n_R=25).is_split
False
>>> pair = rm_tables.build(n_T=40, n_R=25)
>>> pair.is_split, float(pair.cold_log_R[-1]), float(pair.hot_log_R[-1])
(True, -0.25, 1.0)

```

`Tables.kappa` reads the right grid without being told which.

```python
>>> np.round(pair.kappa(np.array([3e3, 1e5]), np.array([1e-14, 1e-16])), 6)
array([5.80000e-05, 4.57623e-01])

```

Fitting a smoother interpolant means fitting one per grid and choosing between
them with `Tables.which`, which answers `True` where the hot grid holds the
point. It answers on temperature or on density, so a cool point above the cold
ceiling still goes to the hot grid.

```python
>>> cool = np.log10(3000.0)
>>> bool(pair.which(cool, -6.0)), bool(pair.which(cool, 0.5))
(False, True)

```

`Tables.split_log_T` is the temperature it switches at, 4.0 in `log10` by
default. `help` on the returned object documents every attribute.

Splitting is permitted, never forced. A range that sits on one side of the
split temperature comes back as one grid whatever `split` says, because the
second would be empty. `split=False` requires a single grid and raises rather than clipping.

```python
>>> try:
...     rm_tables.build(log_R_range=(-8.0, 1.0), n_T=40, n_R=25, split=False)
... except ValueError as e:
...     print(str(e).splitlines()[-1])
Splitting the table does not help. cold='ferguson' covers that corner but not the whole range: it starts at 501 K, and 5.0 K was requested. Raise the temperature floor to use it.

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
>>> import os, tempfile
>>> path = os.path.join(tempfile.mkdtemp(), "solar.txt")
>>> t.save(path)
>>> rm_tables.load(path)
Tables(X=0.7381, Z=0.0134, cold='semenov', (200, 500) + (200, 500))

```

## A request outside the sources

A range leaving what the chosen sources hold raises rather than being filled or
extrapolated, and the message names the offending corner and any source that
would cover it.

```python
>>> try:
...     rm_tables.build(log_T_range=(0.0, 4.0), n_T=40, n_R=25)
... except ValueError as e:
...     print(str(e).splitlines()[0])
the requested range is not covered by semenov, opal: 175 of 1000 points fall outside, the first at log10 T = 0.000 (1 K), log10 R = -8.000. No source covers it.

```

`opacity` holds the nearest value past a density edge, matching what a
tabulated grid does past its own span. A temperature no source reaches raises
instead, because holding across a temperature edge would return a dust opacity
for an ionised gas.

```python
>>> try:
...     rm_tables.opacity()(2.0, 1e-14)
... except ValueError as e:
...     print(e)
semenov has no value below 5 K, and 2 K was requested. No source reaches that temperature.

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
