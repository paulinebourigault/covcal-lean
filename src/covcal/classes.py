"""Aggregate candidates into answer classes and class weights.

Implements Sec. 3 of the paper:

    Q_c(x) = sum_{j: e_x(a_j) = c} q_j(x),  sum_c Q_c(x) = 1.

The artifact attachment step (lifting per-candidate statuses to class-level T_c, P_c) lives
here because it is also an aggregation: it groups artifacts by their target class.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

from .normalization import normalize_answer
from .types import ArtifactOutcome, Candidate, ClassRecord


def aggregate_classes(candidates: Iterable[Candidate]) -> list[ClassRecord]:
    """Group candidates into normalized answer classes and aggregate weights.

    Output classes are sorted by descending weight, then by label for determinism.
    The summed class weights must equal sum(q_j); a downstream caller can require
    sum(q_j) == 1 via :func:`assert_normalized_weights`.
    """
    by_label: dict[str, ClassRecord] = {}
    for j, cand in enumerate(candidates):
        label = normalize_answer(cand.answer_text)
        rec = by_label.get(label)
        if rec is None:
            rec = ClassRecord(label=label, weight=0.0, candidate_indices=[], artifacts=[])
            by_label[label] = rec
        rec.weight += float(cand.weight)
        # `sample_id` is the canonical index; fall back to the enumeration position when -1.
        rec.candidate_indices.append(cand.sample_id if cand.sample_id >= 0 else j)
    classes = list(by_label.values())
    classes.sort(key=lambda c: (-c.weight, c.label))
    return classes


def assert_normalized_weights(classes: Iterable[ClassRecord], *, tol: float = 1e-6) -> None:
    total = sum(c.weight for c in classes)
    if not math.isclose(total, 1.0, abs_tol=tol):
        raise ValueError(f"class weights must sum to 1 (got {total!r}); did you normalize q_j?")


def self_consistency_candidates(answers: Iterable[str]) -> list[Candidate]:
    """Convenience: build K equally-weighted candidates from raw sampled answer strings.

    This is the paper's default weighting scheme (Sec. 4.3): Q_c is the self-consistency
    frequency among K samples.
    """
    answers = list(answers)
    if not answers:
        return []
    w = 1.0 / len(answers)
    return [Candidate(answer_text=a, weight=w, sample_id=i) for i, a in enumerate(answers)]


def attach_artifacts(
    classes: list[ClassRecord],
    artifacts_by_label: Mapping[str, list[ArtifactOutcome]],
) -> None:
    """In-place: attach formal artifacts to the matching class records by label.

    Labels in `artifacts_by_label` that don't match any class are ignored (they correspond
    to lower-weight classes that the pipeline chose not to formalize). 
    Unresolved classes simply have no artifacts and `T_c = P_c = 0`.
    """
    for cls in classes:
        artefacts = artifacts_by_label.get(cls.label)
        if artefacts:
            cls.artifacts.extend(artefacts)


def top_k_by_weight(classes: list[ClassRecord], k: int) -> list[ClassRecord]:
    """Return the top-k classes by descending weight (tie-break by label)."""
    return sorted(classes, key=lambda c: (-c.weight, c.label))[:k]


def total_weight(classes: Iterable[ClassRecord]) -> float:
    return sum(c.weight for c in classes)


def class_label_to_record(classes: Iterable[ClassRecord]) -> dict[str, ClassRecord]:
    """Build a defensive label->record map, raising on duplicate labels."""
    out: dict[str, ClassRecord] = defaultdict(lambda: None)  # type: ignore[arg-type]
    for c in classes:
        if c.label in out and out[c.label] is not None:
            raise ValueError(f"duplicate class label: {c.label!r}")
        out[c.label] = c
    return dict(out)
