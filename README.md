# Rosseland mean Opacity Tables

Rosseland mean opacity from published data, at a requested composition, as a
callable or as a built table. The default covers temperatures from 5 K to 12.6
million K and densities from 1e-24 to 2e4 g/cm^3. At the cold end the opacity
comes from dust grains, at the hot end from ionised gas.

Semenov et al. 2003 and Ferguson et al. 2005 tabulate dust and molecular gas at
the cold end. OPAL tabulates the ionised gas from 5623 K up. The three sit on
different grids and at different compositions, and the two cold ones disagree
about the temperature at which grains survive.

This package reads them, interpolates in hydrogen and metal mass fraction, and
joins a cold source to OPAL across the temperature range where both hold
values. Nothing is refitted. A request outside their coverage raises an error
naming the corner and the source that would answer it.

The data ships inside the package, so nothing is downloaded at run time.

## The data

| source | reference | range | supplies |
| --- | --- | --- | --- |
| OPAL | Iglesias & Rogers 1996, ApJ 464, 943 | 5623 K to 5e8 K | ionised gas |
| Semenov | Semenov et al. 2003, A&A 410, 611 | 5 K to 10,000 K | dust and molecular gas |
| Ferguson | Ferguson et al. 2005, ApJ 623, 585 | 501 K to 31,600 K | dust and molecular gas |

The caller chooses one of Semenov or Ferguson for the cold end.

**OPAL** ships as all 77 published Type-1 files, 6766 tables between them, on a
grid of hydrogen and metal mass fraction. The default is `GS98hz`, at Grevesse
& Sauval 1998 metal ratios, matching the cold default. `GN93hz` is the same at
Grevesse & Noels 1993 ratios and differs by a median 0.002 dex. Obtained from
Arnold Boothroyd's public mirror at the Canadian Institute for Theoretical
Astrophysics.

**Ferguson** ships as seven sets of 155 tables. Six are at Grevesse & Sauval
1998 metal ratios, at alpha enhancements of -0.2, 0.0, 0.2, 0.4, 0.6 and 0.8,
selected with `alpha`; the default is 0.0. Alpha enhancement raises oxygen,
magnesium and silicon against iron at fixed total metal fraction, which no
choice of `X` and `Z` can reach. It moves the dust opacity by 0.054 to 0.079
dex per step and the whole table by 0.189 dex at 0.8, against 0.025 dex for
swapping the abundance compilation. The seventh set is `f05.g93` at Grevesse &
Noels 1993 ratios, reached with `alpha=None`, and was this package's only cold
Ferguson table before the alpha sets were added.

Alpha reaches the dust and molecular opacity only. Above 10,000 K the hot
source answers and `opal_set` selects its mixture; OPAL's own enhanced and
unenhanced tables differ by 0.015 dex, so nothing there is chosen for the
caller.

**Semenov** ships as a Python translation of the published `opacity.f`, dust
model `nrm/h/s`: normal iron content, homogeneous, spherical. Its dust opacity
is a fifth-degree polynomial per temperature region, so no Fortran compiler is
involved. The translation matches the compiled original to 1.776e-05 relative,
which is that program's own printed precision. Its gas table ships unchanged.

A published result cites the papers above. Bibcodes and the rule for which
ones apply are under [Citation](#citation).

## Install

    pip install rm-tables

Requires NumPy and Numba. `h5py` is needed only to read and write HDF5 files.

## Usage

Every opacity in this package is called as `kappa(T, rho)`: the temperature in
K first, the density in g/cm^3 second, returning cm^2/g. Both arguments are
plain numbers, so a swap cannot be detected. Every argument below is optional
and is written at its default, so `rm_tables.opacity()` is the same call.

```python
import numpy as np
import rm_tables

kappa = rm_tables.opacity(X=0.7381, Z=0.0134, cold="semenov",
                          opal_set="GS98hz", alpha=0.0, dXc=0.0, dXo=0.0)
print(kappa(3000.0, 1e-14))    # -> 6.635618312601376e-05
```

`X` and `Z` are the hydrogen and metal mass fractions and helium is the
remainder, so `X + Z` cannot exceed 1. `cold` picks the cold source, `opal_set`
picks which of OPAL's 77 published files supplies the hot end, `alpha` picks the
alpha enhancement of the cold Ferguson tables, and `dXc` and `dXo` add carbon
and oxygen beyond what `Z` carries.

A composition outside what the chosen set tabulates raises rather than
extrapolating.

```python
try:
    rm_tables.opacity(X=0.0, Z=0.5)
except ValueError as e:
    print(e)    # -> set 'GS98hz' tabulates metals up to 0.1, and Z=0.5 was ...
```

Arrays broadcast against each other and the result takes the broadcast shape.

```python
T = np.array([500.0, 3000.0, 1e5])
rho = np.array([1.25e-19, 1e-14, 1e-16])
print(kappa(T, rho))    # -> [1.7535...e+00 6.6356...e-05 4.5123...e-01]
```

A compiled solver cannot call the object, because numba refuses a Python
object holding Python arrays. `as_compiled` returns the same opacity as a numba
function, at 157 ns a point. A compiled function cannot raise, so it returns
NaN where the object raises.

```python
from numba import njit

fast = rm_tables.opacity().as_compiled()

@njit
def optical_depth(T, rho, height):
    return fast(T, rho) * rho * height

print(optical_depth(3000.0, 1e-14, 1e13))    # -> 6.6356...e-06
```

`build` returns tabulated grids instead, for fitting an interpolant or writing
a file. `cold` carries the cold source ramped into OPAL. `hot` carries OPAL
alone and reaches a higher density.

```python
t = rm_tables.build(X=0.7381, Z=0.0134)
print(t.cold.shape, t.hot.shape)    # -> (200, 500) (200, 500)
```

`build` takes the same six arguments and seven more, all optional, for the
range, the resolution and how the pair is split. The ranges are in `log10`, and
`R` is `rho / (T / 1e6)**3` in g/cm^3, the density parameter rather than the
density.

```python
t2 = rm_tables.build(log_T_range=(2.0, 6.0), log_R_range=(-8.0, -2.0),
                     n_T=300, n_R=400)
print(t2.cold.shape, t2.cold_log_T[0], t2.cold_log_R[-1])    # -> (300, 400) 2.0 -2.0
```

Every argument's default is in the signature, and `help(rm_tables.build)`
carries the measurement that set it.

## The rest of the interface

- **Composition.** Hydrogen and metals are arguments and helium is the
  remainder. Metallicity scales the dust, at fixed grain sizes and mineralogy,
  and leaves the gas alone.
- **The cold source.** Semenov tabulates where grains evaporate, Ferguson
  where they condense. Heating material suits Semenov, cooling material suits
  Ferguson.
- **The metal mixture.** All 77 OPAL sets are selectable, and so are the
  carbon and oxygen enhanced grids inside them.
- **Range and resolution.** Both are arguments, and every table records what
  it was built at.
- **Saving.** Compressed NumPy, plain text or HDF5, chosen by the file
  extension, each carrying the provenance beside the numbers.
- **Compiled code.** `opacity(...).as_compiled()` is callable from inside numba,
  and a fitted spline over a built table is the route where the solver needs
  smooth derivatives.

Full guide with runnable examples for all of it:
[USAGE.md](https://github.com/Shmuel-Gilbaum/Rosseland-mean-Opacity-Tables/blob/main/USAGE.md).

## Limits

- **Resolution.** Semenov's dust destruction spans 192 K, which is two and a
  half cells at the default grid, so the tabulated cliff is gentler than the
  model's own. Two tables built at different resolutions are not comparable.
- **Semenov's density floor.** The evaporation temperatures that set where
  dust survives are tabulated from 1e-18 g/cm^3 up. The cold table's lowest
  density column falls below that under 464 K, where the evaporation
  temperature is a linear extrapolation of the first tabulated interval.
- **OPAL's blank corners.** The published tables stop short in the cool dilute
  corner and the hot dense one, and the nearest tabulated value is held across
  them. At solar composition the blanks begin at 15.8 million K, above the
  default ceiling of 12.6 million K.
- **A temperature no source reaches.** The call raises rather than holding.
  Holding across a temperature edge would return a dust opacity for an ionised
  gas. The compiled form returns NaN at the same bounds, and at a density that
  is not finite and positive.
- **The compiled routines.** They are cached next to the installed package.
  Where that location is not writable they recompile each session, about 1.3 s,
  and `NUMBA_CACHE_DIR` points them somewhere writable.

## Citation

A published result carries the hot source, the cold source in use, and this
package.

| what | reference | bibcode |
| --- | --- | --- |
| hot opacity, always | Iglesias & Rogers 1996, ApJ 464, 943 | [1996ApJ...464..943I](https://ui.adsabs.harvard.edu/abs/1996ApJ...464..943I/abstract) |
| cold opacity, `cold="semenov"` | Semenov et al. 2003, A&A 410, 611 | [2003A&A...410..611S](https://ui.adsabs.harvard.edu/abs/2003A%26A...410..611S/abstract) |
| cold opacity, `cold="ferguson"` | Ferguson et al. 2005, ApJ 623, 585 | [2005ApJ...623..585F](https://ui.adsabs.harvard.edu/abs/2005ApJ...623..585F/abstract) |

OPAL is read at every temperature above 5623 K, so it is cited whatever the
cold source is. From 5623 to 10,000 K the answer is a blend in `log10` of the
cold source and OPAL; above 10,000 K OPAL is alone. One cold source is read per
call, and the `cold` argument names it.

Two composition references apply where a result depends on the metal mixture
rather than on the metal mass fraction alone. The default OPAL set `GN93hz` is
at Grevesse & Noels 1993 ratios,
[1993oee..conf...15G](https://ui.adsabs.harvard.edu/abs/1993oee..conf...15G/abstract);
the other sets carry their own mixture, and
`rm_tables.sources.opal.sets()` names them. Semenov's coefficients are computed at Anders &
Grevesse 1989,
[1989GeCoA..53..197A](https://ui.adsabs.harvard.edu/abs/1989GeCoA..53..197A/abstract),
which is the composition a requested metallicity is scaled from.

ADS exports BibTeX from any of these bibcodes.

## Licence

This package is MIT licensed. See `LICENSE`.

The data it redistributes carries its own terms.

Semenov's source grants free use, modification and redistribution, quoted at the
top of `rm_tables/sources/_semenov_fit.py` with its copyright line.

No licence statement was found for the OPAL or Ferguson data. Both are
distributed publicly and without registration, and this package redistributes
them. Anyone republishing that data should confirm the terms independently.
