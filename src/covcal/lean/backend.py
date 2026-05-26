"""Abstract Lean backend interface.

A `LeanTask` is the Python-side mirror of the Lean `Task` schema in
`lean/CovCal/Runner.lean`. It is serialized to JSON by :meth:`LeanTask.to_dict`
and the result is parsed back into a :class:`LeanOutcome` by the backend.

Both subprocess and mock backends accept a batch of tasks for two reasons:
(a) the real Runner amortises a single Mathlib boot across many tasks; (b) the mock
backend can reorder, deduplicate or sort tasks before lookup without changing the
caller's contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from ..types import Status


@dataclass(frozen=True, slots=True)
class LeanTask:
    """A single attempt request for the Lean backend."""

    name: str
    statement: str
    tactics: tuple[str, ...]
    max_heartbeats_per_tactic: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "statement": self.statement,
            "tactics": list(self.tactics),
        }
        if self.max_heartbeats_per_tactic is not None:
            d["maxHeartbeatsPerTactic"] = self.max_heartbeats_per_tactic
        return d


@dataclass(slots=True)
class LeanOutcome:
    """One outcome from the Lean runner, normalised into the Python `Status` enum."""

    name: str
    status: Status
    tactic_used: str | None
    elapsed_seconds: float
    log: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = self.status.value
        return out

    @classmethod
    def from_runner_dict(cls, d: dict[str, Any]) -> LeanOutcome:
        status_str = d.get("status", "unformalized")
        try:
            status = Status(status_str)
        except ValueError:
            status = Status.UNFORMALIZED
        return cls(
            name=d.get("name", "<missing>"),
            status=status,
            tactic_used=d.get("tactic_used") or d.get("tacticUsed"),
            elapsed_seconds=float(d.get("elapsed_seconds", d.get("elapsedSeconds", 0.0))),
            log=str(d.get("log", "")),
        )


class LeanBackend(ABC):
    """Process a batch of `LeanTask`s and return one `LeanOutcome` per task.

    Implementations must preserve task order (outcome[i] corresponds to task[i]).
    """

    @abstractmethod
    def check(self, tasks: list[LeanTask]) -> list[LeanOutcome]: ...

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def __enter__(self) -> LeanBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
