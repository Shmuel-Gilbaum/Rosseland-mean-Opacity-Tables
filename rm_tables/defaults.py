"""Default ranges, grid and composition.

Every limit here comes from what the sources hold.
"""
import numpy as np

__all__ = ["LOG_T_RANGE", "LOG_R_RANGE", "HOT_LOG_R_MAX", "SPLIT_LOG_T",
           "N_T", "N_R", "COLD", "OPAL_SET", "REFERENCE_Z",
           "SOLAR_MASS_FRACTIONS"]

LOG_T_RANGE = (np.log10(5.0), 7.1)
"""Temperature limits of both tables, as ``log10 T`` in K: 5 K to 1.26e7 K.

The floor is Semenov's lowest tabulated temperature. Raising the ceiling is
free up to ``log10 T = 8.6999998``, the last temperature OPAL's axis holds.
That axis is stored in 32-bit floats, so a request at 8.70 falls outside it.
"""

SPLIT_LOG_T = 4.0
"""Temperature at which a lookup moves to the hot table, as ``log10 T``.

A pair of tables covers more than one can. The cold table carries the cold
source ramped into OPAL; the hot table is OPAL alone and reaches a higher
density. Take the hot table at or above this temperature, or wherever the
density parameter exceeds the cold table's ceiling.
"""

LOG_R_RANGE = (-8.0, -0.25)
"""Density limits of the cold table, as ``log10 R``.

``R = rho / (T / 1e6)**3`` in g/cm^3.

The ceiling is where Semenov's gas grid runs out. That grid is bounded in
density, so its reach in ``log10 R`` falls as ``11 - 3 log10 T``, and -0.25 is
the highest value it holds at every temperature below OPAL's floor.

For a higher ceiling, pass ``cold="ferguson"``, which reaches +1.0 over 501 to
31,623 K. Requesting more than the chosen sources hold raises rather than
filling.

Below -8.0 no source has data. The opacity is held at the edge there.
"""

HOT_LOG_R_MAX = 1.0
"""Density ceiling of the hot table, as ``log10 R``. OPAL reaches +1.0."""

N_T, N_R = 200, 500
"""Grid points in temperature and in density parameter.

The resolution changes the table. Semenov's dust destruction spans 192 K, which
at 200 temperature points is two and a half grid cells, so the tabulated cliff
is gentler than the model's own. Every table records the resolution it was built
at; two built at different resolutions are not comparable.

Raising it costs 9% per lookup at 400 temperature points and 27% at 800.
"""

COLD = "semenov"
"""Cold source, ``'semenov'`` or ``'ferguson'``. The two are never combined.

Semenov gives evaporation temperatures, Ferguson gives condensation
temperatures, and grains are destroyed at the lower of the two. Choose Semenov
where material is heating, as in an accretion flow, and Ferguson where it is
cooling, as in an atmosphere. Neither is more accurate.

Ferguson also carries a composition through its own grain chemistry and reaches
a higher density, but stops at 501 K.

Ferguson et al. 2005, ApJ 623, 585, section 4.
"""

REFERENCE_Z = 0.0189
"""Metal mass fraction Semenov's coefficients are computed at.

Anders & Grevesse 1989. The metallicity scaling multiplies the dust by
``Z / REFERENCE_Z``, so a table built at this value is the unmodified source.
"""

SOLAR_MASS_FRACTIONS = (0.7381, 0.2485, 0.0134)
"""Hydrogen, helium and metal mass fractions.

Helium is the remainder and is not an independent argument. Opacity tables are
indexed by hydrogen and metals only.
"""

OPAL_SET = "GN93hz"
"""Which of OPAL's 77 published Type-1 files supplies the hot opacity.

Grevesse & Noels 1993 metal ratios, tabulated to a metal mass fraction of 0.1.
`rm_tables.sources.opal.sets` lists the others, which carry different metal
mixtures: Grevesse & Sauval 1998, Asplund et al. 2004, and neon-enhanced or
alpha-enhanced variants of each.
"""
