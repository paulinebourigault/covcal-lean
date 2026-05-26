"""Dataset filters, shared by all loaders.

Our protocol excludes three kinds of MATH-500 / AMC items:

  1) diagram-dependent geometry (image, asy block, or geometry references the model can't see);
  2) proof-only questions (no extractable final answer);
  3) examples whose official answer cannot be normalised by the rules in
     :mod:`covcal.normalization`.

Exclusion counts are reported per-reason.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..normalization import is_unnormalized, normalize_answer
from ..pipeline import Problem

logger = logging.getLogger(__name__)


# -- regexes that flag exclusion reasons --------------------------------------------------

_ASY_BLOCK_RE = re.compile(r"\[asy\]", re.IGNORECASE)
_IMG_REF_RE = re.compile(r"(?:figure|diagram|picture)\s+(?:above|below|on the right|on the left)",
                         re.IGNORECASE)
_PROOF_ONLY_RE = re.compile(r"\b(?:prove|show that|demonstrate)\b", re.IGNORECASE)


@dataclass(slots=True)
class FilterResult:
    keep: bool
    reason: str = "kept"


@dataclass(slots=True)
class FilterReport:
    """Aggregate counts. Sums to len(raw inputs)."""

    kept: list[Problem] = field(default_factory=list)
    excluded_by_reason: Counter[str] = field(default_factory=Counter)
    total_seen: int = 0

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_excluded(self) -> int:
        return self.total_seen - self.n_kept

    def as_dict(self) -> dict[str, object]:
        return {
            "total_seen": self.total_seen,
            "n_kept": self.n_kept,
            "n_excluded": self.n_excluded,
            "excluded_by_reason": dict(self.excluded_by_reason),
        }


def classify(problem_text: str, reference_answer: str) -> FilterResult:
    """Classify a single (problem, answer) pair as keep/exclude. Reason is the *first* hit."""
    if _ASY_BLOCK_RE.search(problem_text):
        return FilterResult(keep=False, reason="diagram_asy_block")
    if _IMG_REF_RE.search(problem_text):
        return FilterResult(keep=False, reason="diagram_image_reference")
    # "Prove" / "show that" usually means no extractable final answer (proof-only).
    if _PROOF_ONLY_RE.search(problem_text) and not reference_answer.strip():
        return FilterResult(keep=False, reason="proof_only")
    if not reference_answer or not reference_answer.strip():
        return FilterResult(keep=False, reason="missing_reference_answer")
    normalized = normalize_answer(reference_answer)
    if is_unnormalized(normalized):
        return FilterResult(keep=False, reason="non_normalizable_reference")
    return FilterResult(keep=True)


def filter_problems(
    items: Iterable[Problem],
    *,
    max_examples: int | None = None,
) -> FilterReport:
    """Apply :func:`classify` to every input, keeping up to `max_examples` kept items."""
    report = FilterReport()
    for problem in items:
        report.total_seen += 1
        verdict = classify(problem.problem_text, problem.reference_answer)
        if verdict.keep:
            if max_examples is not None and report.n_kept >= max_examples:
                report.excluded_by_reason["over_max_examples"] += 1
                continue
            report.kept.append(problem)
        else:
            report.excluded_by_reason[verdict.reason] += 1
    return report


def log_exclusions(report: FilterReport) -> None:
    logger.info(
        "filter: kept %d / %d (excluded %d)",
        report.n_kept,
        report.total_seen,
        report.n_excluded,
    )
    for reason, n in sorted(report.excluded_by_reason.items(), key=lambda x: -x[1]):
        logger.info("  excluded[%s] = %d", reason, n)
