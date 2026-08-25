"""The published sources, each read as its authors supply it.

No source is modified except by the metallicity multiply in `semenov`, which is
documented there and reduces to the unmodified routine at the reference
composition.
"""
from . import ferguson, opal, semenov

__all__ = ["ferguson", "opal", "semenov"]
