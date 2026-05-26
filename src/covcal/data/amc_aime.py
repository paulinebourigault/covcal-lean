"""AMC/AIME-style robustness subset loader.

We use AMC/AIME-style examples with multiple-choice labels mapped to
their mathematical answer when available, otherwise the label itself. We support two
sources:

  * a local JSONL file (any schema with `problem`, `answer`, optional `choices`/`subject`);
  * the `AI-MO/aimo-validation-aime` HF dataset, used for AIME problems specifically.

Multiple-choice mapping: if a `choices` field is present and the `answer` is a single letter
A-E, we look up the corresponding choice and use that as the reference answer; otherwise
we keep the letter and rely on the MC normalizer (`MC::A` etc.) to compare against
predictions in the same form.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path

from ..normalization import normalize_answer
from ..pipeline import Problem
from .filters import FilterReport, filter_problems

logger = logging.getLogger(__name__)

_MC_LETTER_RE = re.compile(r"^[A-Ea-e]$")


def _resolve_choice(answer: str, choices: list[str] | None) -> str:
    if not _MC_LETTER_RE.match(answer.strip()):
        return answer
    if not choices:
        return answer
    idx = ord(answer.strip().upper()) - ord("A")
    if 0 <= idx < len(choices):
        return choices[idx]
    return answer


def _to_problem(row: dict[str, object], idx: int) -> Problem:
    # Always use the row index in the ID so duplicates in the upstream `id` field can't
    # collapse problems when SplitsManifest dedupes via set(). The source ID (when
    # present) is kept in metadata for cross-reference with upstream debugging.
    source_id = row.get("id") or row.get("unique_id")
    pid = f"amc_aime_{idx:04d}" if not source_id else f"amc_aime_{idx:04d}_{source_id}"
    text = str(row.get("problem") or row.get("question") or "")
    raw_answer = str(row.get("answer") or row.get("final_answer") or "")
    choices_raw = row.get("choices") or row.get("options")
    choices = list(choices_raw) if isinstance(choices_raw, list) else None
    reference_answer = _resolve_choice(raw_answer, choices)
    return Problem(
        problem_id=str(pid),
        problem_text=text,
        reference_answer=reference_answer,
        domain=str(row.get("domain") or row.get("subject") or "other"),
        metadata={
            "source": "amc_aime",
            "source_id": source_id,
            "raw_answer": raw_answer,
            "had_choices": choices is not None,
        },
    )


def _iter_from_jsonl(path: str) -> Iterator[Problem]:
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield _to_problem(json.loads(line), i)


def _iter_from_hf(repo: str, split: str) -> Iterator[Problem]:
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install with `uv sync` (datasets is in default deps).") from e
    ds = load_dataset(repo, split=split)
    for i, row in enumerate(ds):
        yield _to_problem(row, i)


def load_amc_aime(
    *,
    max_examples: int | None = None,
    jsonl_path: str | None = None,
    repo: str | None = None,
    split: str = "train",
) -> tuple[list[Problem], FilterReport]:
    """Load and filter the AMC/AIME robustness subset.

    At least one of `jsonl_path` or `repo` must be provided. We do not default to a HF repo
    because the AMC/AIME mirror landscape is fragmented; an explicit `repo` keeps the
    provenance auditable.
    """
    if jsonl_path is not None:
        raw = _iter_from_jsonl(jsonl_path)
    elif repo is not None:
        raw = _iter_from_hf(repo, split)
    else:
        raise ValueError("provide either jsonl_path or repo")
    report = filter_problems(raw, max_examples=max_examples)
    for p in report.kept:
        p.metadata["normalized_reference"] = normalize_answer(p.reference_answer)
    logger.info("load_amc_aime: %d kept / %d seen", report.n_kept, report.total_seen)
    return report.kept, report
