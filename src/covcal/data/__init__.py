"""Dataset loaders and splits manifest.

Public entry points:

* :func:`covcal.data.load_math500` — filtered MATH-500 short-answer subset.
* :func:`covcal.data.load_amc_aime` — robustness subset (AMC/AIME-style).
* :func:`covcal.data.make_splits` — deterministic dev/cal/test split + JSON manifest I/O.
"""

from .amc_aime import load_amc_aime
from .filters import FilterReport, FilterResult, filter_problems, log_exclusions
from .math500 import load_math500
from .splits import SplitsManifest, load_splits, make_splits, write_splits

__all__ = [
    "FilterReport",
    "FilterResult",
    "SplitsManifest",
    "filter_problems",
    "load_amc_aime",
    "load_math500",
    "load_splits",
    "log_exclusions",
    "make_splits",
    "write_splits",
]
