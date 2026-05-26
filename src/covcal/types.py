"""Core types for the CovCal pipeline.

The data model follows Sec. 3 of the paper:

* a problem `x` has K candidate answers `a_1, ..., a_K` with weights `q_j`;
* candidates are normalized into answer classes `c \\in C(x)` with class weights `Q_c`;
* a verifier assigns each candidate one of five statuses; statuses lift to per-class
  indicators `P_c` (proved) and `T_c` (typed).

Everything in this module is plain dataclasses + enums. No I/O, no math beyond aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Sentinel used by the formal selector and CovCal to indicate abstention. The paper writes \bot;
# we use the string "ABSTAIN" throughout so it survives JSON serialization unchanged.
ABSTAIN: str = "ABSTAIN"


class Status(str, Enum):
    """Per-artifact formal status. Mirrors `S` in Sec. 3 of the paper.

    The ordering of statuses determines the *best status* aggregation when one class has
    several artifacts: PROVED dominates TYPECHECKED, which dominates TIMEOUT, which dominates
    ILLTYPED, which dominates UNFORMALIZED. `Status.best_of(...)` encodes this.
    """

    PROVED = "proved"
    TYPECHECKED = "typechecked"
    TIMEOUT = "timeout"
    ILLTYPED = "illtyped"
    UNFORMALIZED = "unformalized"

    @classmethod
    def best_of(cls, statuses: list[Status]) -> Status:
        order = {
            cls.PROVED: 0,
            cls.TYPECHECKED: 1,
            cls.TIMEOUT: 2,
            cls.ILLTYPED: 3,
            cls.UNFORMALIZED: 4,
        }
        if not statuses:
            return cls.UNFORMALIZED
        return min(statuses, key=lambda s: order[s])


# Per the paper, S_typ = {proved, typechecked, timeout}: any of these means the statement
# elaborated and reached proof search (timeout counts as "well-formed but undecided in budget").
TYPED_STATUSES: frozenset[Status] = frozenset(
    {Status.PROVED, Status.TYPECHECKED, Status.TIMEOUT}
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One raw candidate answer for a problem, with its sampling weight."""

    answer_text: str  # raw text the generator produced (post `\boxed{...}` extraction)
    weight: float  # nonneg; self-consistency frequency by default. Sums to 1 over j.
    sample_id: int = -1  # generator-side index, for debugging only


@dataclass(slots=True)
class ArtifactOutcome:
    """One Lean attempt for a (class, route) pair."""

    status: Status
    tactic_used: str | None = None
    elapsed_seconds: float = 0.0
    log: str = ""
    source: str = "template"  # "template" | "autoformalizer" | "autoformalizer_repair"


@dataclass(slots=True)
class ClassRecord:
    """All evidence the verifier produced for a single answer class."""

    label: str  # canonical class label (e.g., "1/2", "42", "(1,2,3)")
    weight: float  # Q_c, the aggregated class weight
    candidate_indices: list[int] = field(default_factory=list)
    artifacts: list[ArtifactOutcome] = field(default_factory=list)

    @property
    def proved(self) -> bool:
        """`P_c` from the paper: any artifact for this class kernel-checked."""
        return any(a.status is Status.PROVED for a in self.artifacts)

    @property
    def typed(self) -> bool:
        """`T_c` from the paper: any artifact reached at least proof search."""
        return any(a.status in TYPED_STATUSES for a in self.artifacts)

    @property
    def best_status(self) -> Status:
        return Status.best_of([a.status for a in self.artifacts])


@dataclass(slots=True)
class FormalObservation:
    """`O(x)` from Sec. 3: everything the selector is allowed to see for problem x.

    The reference class label is **not** stored here; it lives in the labeled split
    used for calibration / evaluation, never inside the observation passed to a selector.
    """

    problem_id: str
    classes: list[ClassRecord]
    candidates: list[Candidate] = field(default_factory=list)
    prover_budget_seconds: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def class_by_label(self, label: str) -> ClassRecord | None:
        for c in self.classes:
            if c.label == label:
                return c
        return None

    def proved_classes(self) -> list[ClassRecord]:
        return [c for c in self.classes if c.proved]


@dataclass(frozen=True, slots=True)
class Thresholds:
    """A single point in the threshold grid `T` from Eq. (7)."""

    typ: float
    prf: float
    margin: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.typ, self.prf, self.margin)


@dataclass(slots=True)
class SelectorOutput:
    """Result of any selector. `selected is ABSTAIN` means abstention (paper's bot)."""

    selected: str  # class label, or the ABSTAIN sentinel
    reason: str = ""  # short tag for diagnostics ("proved_winner", "no_proof", ...)

    @property
    def abstained(self) -> bool:
        return self.selected == ABSTAIN
