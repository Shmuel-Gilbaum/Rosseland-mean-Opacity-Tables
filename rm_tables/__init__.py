"""Rosseland mean opacity for gas and dust, at any composition.

Two ways in. `opacity` returns a callable and builds no table: it evaluates the
sources at the point requested. `build` returns a tabulated grid for anyone who
wants to fit their own interpolant or write the result to a file.

Sources are used as their authors published them. OPAL supplies the ionised gas
above 5623 K; either Semenov 2003 or Ferguson 2005 supplies dust and molecules
below. The two cold sources are alternatives and are never combined; the README
says which to choose. Both entry points take a `dataset` naming the OPAL file,
which sets the metal mixture, and the excess carbon and oxygen that file
tabulates.

`build` raises `rm_tables.coverage.CoverageError` for a range the chosen sources
do not hold. A callable from `opacity` raises it for a temperature no source
reaches, and holds the nearest value it has in density. `CoverageError`
subclasses `ValueError`.
"""

from . import coverage, defaults
from .lookup import opacity
from .tables import build, load

__all__ = ["opacity", "build", "load"]
__version__ = "0.1.0"
