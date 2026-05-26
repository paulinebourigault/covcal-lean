"""Deterministic mock Lean backend.

Looks up canned outcomes from a fixture file (or in-memory dict) keyed by either:

* the task `name`, or
* `sha256(statement || "\\n" || "|".join(tactics))`.

The second form is used in unit tests and fixture replays so that adding a new test does
not require coordinating a unique name space across files.

If neither key is found, the mock returns a configurable default (UNFORMALIZED by default,
which causes the diagnostics to treat the class as having no formal evidence).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import Status
from .backend import LeanBackend, LeanOutcome, LeanTask


def task_content_hash(task: LeanTask) -> str:
    payload = task.statement + "\n" + "|".join(task.tactics)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _Entry:
    status: Status
    tactic_used: str | None = None
    elapsed_seconds: float = 0.0
    log: str = ""


class MockLeanBackend(LeanBackend):
    """Deterministic replay of canned Lean outcomes.

    Outcomes can be registered by name (`add_by_name`) or by statement hash
    (`add_by_statement`). A `default_status` (UNFORMALIZED by default) is returned for any
    task whose key is not in the table.
    """

    def __init__(self, *, default_status: Status = Status.UNFORMALIZED) -> None:
        self._by_name: dict[str, _Entry] = {}
        self._by_hash: dict[str, _Entry] = {}
        self._default = default_status
        self.history: list[LeanTask] = []  # for assertion in tests

    # --- registration ---

    def add_by_name(
        self,
        name: str,
        status: Status,
        *,
        tactic_used: str | None = None,
        elapsed_seconds: float = 0.0,
        log: str = "",
    ) -> None:
        self._by_name[name] = _Entry(status, tactic_used, elapsed_seconds, log)

    def add_by_statement(
        self,
        statement: str,
        tactics: tuple[str, ...],
        status: Status,
        *,
        tactic_used: str | None = None,
        elapsed_seconds: float = 0.0,
        log: str = "",
    ) -> None:
        h = task_content_hash(LeanTask(name="_", statement=statement, tactics=tactics))
        self._by_hash[h] = _Entry(status, tactic_used, elapsed_seconds, log)

    def load_fixture(self, path: str | Path) -> None:
        """Load a JSON fixture file with entries:

        ``{"name": "...", "status": "proved", "tactic_used": "norm_num", "log": "..."}``
        or
        ``{"statement_hash": "<hex>", "status": "...", ...}``
        """
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"fixture {p} must be a JSON list")
        for item in data:
            status = Status(item["status"])
            entry = _Entry(
                status=status,
                tactic_used=item.get("tactic_used"),
                elapsed_seconds=float(item.get("elapsed_seconds", 0.0)),
                log=item.get("log", ""),
            )
            if "name" in item:
                self._by_name[item["name"]] = entry
            elif "statement_hash" in item:
                self._by_hash[item["statement_hash"]] = entry
            else:
                raise ValueError(f"fixture entry needs 'name' or 'statement_hash': {item}")

    # --- backend interface ---

    def check(self, tasks: list[LeanTask]) -> list[LeanOutcome]:
        out: list[LeanOutcome] = []
        for t in tasks:
            self.history.append(t)
            entry = self._by_name.get(t.name) or self._by_hash.get(task_content_hash(t))
            if entry is None:
                out.append(
                    LeanOutcome(
                        name=t.name,
                        status=self._default,
                        tactic_used=None,
                        elapsed_seconds=0.0,
                        log="mock: no fixture entry",
                    )
                )
            else:
                out.append(
                    LeanOutcome(
                        name=t.name,
                        status=entry.status,
                        tactic_used=entry.tactic_used,
                        elapsed_seconds=entry.elapsed_seconds,
                        log=entry.log,
                    )
                )
        return out

    def close(self) -> None:
        pass

    # --- diagnostics for tests ---

    def __len__(self) -> int:
        return len(self._by_name) + len(self._by_hash)

    def to_summary(self) -> dict[str, Any]:
        return {
            "by_name": len(self._by_name),
            "by_hash": len(self._by_hash),
            "default": self._default.value,
            "history_size": len(self.history),
        }
