"""Coverage diagnostics (Sec. 4 of the paper).

Given the formal observation O(x), compute:

* `C_typ(x)  = sum_c Q_c(x) * T_c(x)`         typed coverage
* `C_prf(x)  = sum_c Q_c(x) * P_c(x)`         proved coverage
* `c+(x)     = argmax_{P_c=1} Q_c(x)`         highest-weight proved class
* `R_unres(x) = max_{c != c+, P_c=0} Q_c(x)`  unresolved rival mass
* `M(x)      = Q_{c+}(x) - R_unres(x)`        formal margin (-inf if nothing proved)
* `conflict(x)`: True iff at least two *inequivalent* classes are both proved.

Everything is computed from a `FormalObservation` only; no labels are consulted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import ClassRecord, FormalObservation


@dataclass(frozen=True, slots=True)
class Diagnostics:
    typed_coverage: float
    proved_coverage: float
    proved_winner: str | None  # label of c+, or None if nothing proved
    proved_winner_weight: float  # 0.0 if nothing proved
    unresolved_rival_mass: float  # 0.0 if no unresolved rival or nothing proved
    margin: float  # -inf if nothing proved
    conflict: bool


def typed_coverage(obs: FormalObservation) -> float:
    return sum(c.weight for c in obs.classes if c.typed)


def proved_coverage(obs: FormalObservation) -> float:
    return sum(c.weight for c in obs.classes if c.proved)


def proved_winner(obs: FormalObservation) -> ClassRecord | None:
    """The highest-weight proved class (c+ in the paper).

    Ties are broken by descending weight, then label, to match `aggregate_classes` ordering.
    Returns None if no class is proved.
    """
    proved = [c for c in obs.classes if c.proved]
    if not proved:
        return None
    return min(proved, key=lambda c: (-c.weight, c.label))


def unresolved_rival_mass(obs: FormalObservation, winner: ClassRecord | None) -> float:
    """`R_unres`: largest weight among unresolved (not-proved) classes other than the winner.

    "Unresolved" means `P_c = 0`, which is exactly "not in proved_classes". By the paper:
    a class that is also proved (so P_c=1) is not a rival here; it would instead trigger
    a `conflict` (see :func:`has_conflict`).
    """
    if winner is None:
        return 0.0
    rivals = [c.weight for c in obs.classes if c.label != winner.label and not c.proved]
    return max(rivals) if rivals else 0.0


def formal_margin(winner: ClassRecord | None, unres: float) -> float:
    if winner is None:
        return -math.inf
    return winner.weight - unres


def has_conflict(obs: FormalObservation) -> bool:
    """Two or more inequivalent proved classes for the same problem.

    This is not read as a Lean inconsistency. It signals an
    upstream issue (normalization, semantic mismatch, autoformalization disagreement),
    and the formal selector rejects by default.
    """
    proved = [c for c in obs.classes if c.proved]
    return len({c.label for c in proved}) >= 2


def compute_diagnostics(obs: FormalObservation) -> Diagnostics:
    typ = typed_coverage(obs)
    prf = proved_coverage(obs)
    winner = proved_winner(obs)
    unres = unresolved_rival_mass(obs, winner)
    margin = formal_margin(winner, unres)
    return Diagnostics(
        typed_coverage=typ,
        proved_coverage=prf,
        proved_winner=winner.label if winner else None,
        proved_winner_weight=winner.weight if winner else 0.0,
        unresolved_rival_mass=unres,
        margin=margin,
        conflict=has_conflict(obs),
    )
