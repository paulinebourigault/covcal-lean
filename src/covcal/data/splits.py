"""Deterministic dev/cal/test splits + JSON manifest I/O.

Splits are produced by a *seeded* shuffle of problem ids and then sliced by the configured
fractions. The output is a JSON manifest of the form:

```
{
  "name": "minimal",
  "seed": 0,
  "fractions": {"dev": 0.20, "cal": 0.40, "test": 0.40},
  "n_total": 200,
  "dev":  ["math500_0001", ...],
  "cal":  [...],
  "test": [...]
}
```

The same `SplitsManifest` is consumed by `covcal calibrate`/`evaluate` to determine which
problem ids belong to which split. Splits are *fixed before* calibration labels are inspected,
which is the precondition for the paper's Theorem 1.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SplitName = str  # "dev" | "cal" | "test"


@dataclass(slots=True)
class SplitsManifest:
    name: str
    seed: int
    fractions: dict[SplitName, float]
    splits: dict[SplitName, list[str]] = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return sum(len(v) for v in self.splits.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed,
            "fractions": self.fractions,
            "n_total": self.n_total,
            **self.splits,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> SplitsManifest:
        fractions = dict(d["fractions"])  # type: ignore[arg-type]
        splits: dict[str, list[str]] = {}
        for k in fractions:
            ids = d.get(k)
            if not isinstance(ids, list):
                raise ValueError(f"splits manifest missing key {k!r}")
            splits[k] = [str(x) for x in ids]
        return cls(
            name=str(d.get("name", "<unnamed>")),
            seed=int(d.get("seed", 0)),
            fractions=fractions,
            splits=splits,
        )


def make_splits(
    problem_ids: Iterable[str],
    *,
    name: str,
    seed: int,
    fractions: dict[SplitName, float],
) -> SplitsManifest:
    """Deterministically partition `problem_ids` into the named splits.

    `fractions` must contain *only* the split names you want (typically dev/cal/test)
    and sum to a value in (0, 1].  Any leftover items (when fractions don't sum to 1) are
    discarded -> make this explicit by setting `fractions["dev"]=0.0` to drop the dev split.
    """
    pids = sorted({str(p) for p in problem_ids})  # de-dupe + canonical order
    if not pids:
        raise ValueError("no problem ids provided")
    rng = random.Random(seed)
    rng.shuffle(pids)

    total_frac = sum(fractions.values())
    if total_frac <= 0 or total_frac > 1.0 + 1e-9:
        raise ValueError(f"fractions must sum to (0, 1]; got {total_frac}")

    n = len(pids)
    splits: dict[str, list[str]] = {}
    cursor = 0
    # Process keys in a stable order to make the split deterministic.
    for split_name in sorted(fractions.keys()):
        frac = fractions[split_name]
        if frac <= 0:
            splits[split_name] = []
            continue
        take = round(n * frac)
        splits[split_name] = pids[cursor : cursor + take]
        cursor += take
    # Catch rounding drift: stuff the leftovers into the largest split.
    leftover = pids[cursor:]
    if leftover:
        biggest = max(splits, key=lambda k: len(splits[k]) or -1)
        splits[biggest].extend(leftover)
        logger.debug("appended %d leftover ids to split %r", len(leftover), biggest)

    return SplitsManifest(name=name, seed=seed, fractions=fractions, splits=splits)


def write_splits(manifest: SplitsManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    logger.info("wrote splits manifest %s (%d ids)", path, manifest.n_total)


def load_splits(path: Path) -> SplitsManifest:
    return SplitsManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
