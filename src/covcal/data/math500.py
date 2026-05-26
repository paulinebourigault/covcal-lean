"""MATH-500 loader.

MATH-500 is the 500-problem evaluation subset of the MATH benchmark used by many recent
papers (PRM, Self-Refine, etc.). Several mirrors exist on HuggingFace; we default to the
`HuggingFaceH4/MATH-500` mirror, which has the canonical answer format with `\\boxed{...}`
already extracted into the `answer` field. Each item also carries a `level` and `subject`
which we map to the paper's `domain` field for Table 3.

The reference answer is the contents of the dataset's `answer` column (a string). We pass
it through the same normalisation as candidate answers (Sec. 4.3), so `c*(x)` lives in the
same answer-class universe as the predictions.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

from ..normalization import normalize_answer
from ..pipeline import Problem
from .filters import FilterReport, filter_problems

logger = logging.getLogger(__name__)

DEFAULT_DATASET_REPO = "HuggingFaceH4/MATH-500"
DEFAULT_SPLIT = "test"

# Map MATH `subject` strings to the broader domain buckets
_DOMAIN_MAP = {
    "Algebra": "algebra",
    "Counting & Probability": "combinatorics",
    "Geometry": "geometry",
    "Intermediate Algebra": "algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "algebra",
    "Precalculus": "calculus_analysis",
}


def _to_problem(row: dict[str, object], idx: int) -> Problem:
    """Convert one HuggingFace row into a :class:`Problem`.

    The MATH-500 schema has `problem`, `answer`, `subject`, `level`, `unique_id` columns.
    """
    pid_raw = row.get("unique_id") or row.get("id") or row.get("idx")
    problem_id = str(pid_raw) if pid_raw is not None else f"math500_{idx:04d}"
    problem_text = str(row.get("problem") or row.get("question") or "")
    reference_answer = str(row.get("answer") or row.get("solution_final_answer") or "")
    subject = str(row.get("subject") or row.get("type") or "")
    domain = _DOMAIN_MAP.get(subject, "other")
    metadata = {
        "subject": subject,
        "level": row.get("level"),
        "source": "math500",
    }
    return Problem(
        problem_id=problem_id,
        problem_text=problem_text,
        reference_answer=reference_answer,
        domain=domain,
        metadata=metadata,
    )


def _iter_math500_from_hf(repo: str, split: str) -> Iterator[Problem]:
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover - datasets is in core deps
        raise ImportError(
            "Install with `uv sync` (datasets is in default deps)."
        ) from e
    ds = load_dataset(repo, split=split)
    for i, row in enumerate(ds):
        yield _to_problem(row, i)


def _iter_math500_from_jsonl(path: str) -> Iterator[Problem]:
    """Local fallback if HF download is not available."""
    import json
    from pathlib import Path
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield _to_problem(json.loads(line), i)


def load_math500(
    *,
    max_examples: int | None = None,
    repo: str = DEFAULT_DATASET_REPO,
    split: str = DEFAULT_SPLIT,
    jsonl_path: str | None = None,
    min_level: int | None = None,
    max_level: int | None = None,
) -> tuple[list[Problem], FilterReport]:
    """Load and filter MATH-500. Returns (kept problems, exclusion report).

    ``min_level`` / ``max_level`` filter on the MATH-500 `level` integer (1–5) before the
    standard exclusion filters run. Use ``min_level=4`` to restrict to the hard subset
    (262 raw problems, ~220 after the diagram/proof-only/non-normalisable filters).
    """
    if jsonl_path is not None:
        raw: Iterable[Problem] = _iter_math500_from_jsonl(jsonl_path)
    else:
        raw = _iter_math500_from_hf(repo, split)
    if min_level is not None or max_level is not None:
        def _level_pass(p: Problem) -> bool:
            lvl = p.metadata.get("level")
            if not isinstance(lvl, int):
                return False
            if min_level is not None and lvl < min_level:
                return False
            if max_level is not None and lvl > max_level:
                return False
            return True
        raw = (p for p in raw if _level_pass(p))
    report = filter_problems(raw, max_examples=max_examples)
    # Pre-normalise once so downstream `Problem.reference_class_label` is fast/consistent.
    for p in report.kept:
        p.metadata["normalized_reference"] = normalize_answer(p.reference_answer)
    logger.info("load_math500: %d kept / %d seen", report.n_kept, report.total_seen)
    return report.kept, report
