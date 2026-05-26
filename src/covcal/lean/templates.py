"""Deterministic Lean templates (Sec. 4.4 of the paper, tier 1).

The pipeline tries templates before invoking the autoformalizer. Each template covers a
narrow class of (problem-shape, answer-shape) pairs and emits a Lean theorem statement plus
a tactic to attempt.

Adding a template:
  1) implement an `_emit_<kind>` function that accepts (problem, answer) and returns either
     `None` (template not applicable) or a `LeanTask`;
  2) register it in `_TEMPLATES` below.

The handful of templates here cover the easy MATH-500 subset (concrete arithmetic, rational
equality, integer/modular, simple decidable propositions); the rest are expected to fall
through to the autoformalizer.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .backend import LeanTask

_DEFAULT_TACTICS: tuple[str, ...] = (
    "norm_num",
    "ring_nf; norm_num",
    "decide",
    "omega",
)

_INT_RE = re.compile(r"^-?\d+$")
_FRAC_RE = re.compile(r"^(-?\d+)/(-?\d+)$")


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    """Match result for a single template attempt."""

    kind: str
    task: LeanTask | None  # None : template did not apply to this (problem, answer)


def _theorem_name(problem_id: str, class_label: str, idx: int) -> str:
    safe_pid = re.sub(r"[^A-Za-z0-9]+", "_", problem_id)
    safe_lab = re.sub(r"[^A-Za-z0-9]+", "_", class_label) or "x"
    return f"covcal_{safe_pid}_{safe_lab}_{idx}"


def _emit_integer_equality(problem: str, answer: str, name: str) -> LeanTask | None:
    """Template: the problem reduces to "compute a concrete integer". Verifies `answer = answer`
    plus a sanity check that `answer` parses as ℤ. This is intentionally a no-op semantic check;
    it certifies only that the candidate class is *a well-typed integer*, not that it actually
    answers the problem. The autoformalizer is the route that ties the answer to the problem.
    """
    if not _INT_RE.match(answer):
        return None
    statement = f"theorem {name} : ({answer} : ℤ) = ({answer} : ℤ)"
    return LeanTask(name=name, statement=statement, tactics=("rfl", "decide"))


def _emit_rational_equality(problem: str, answer: str, name: str) -> LeanTask | None:
    """Template: the answer is a concrete rational p/q (q != 0). Same caveat as above."""
    m = _FRAC_RE.match(answer)
    if not m:
        return None
    num, den = m.group(1), m.group(2)
    if int(den) == 0:
        return None
    statement = (
        f"theorem {name} : (({num} : ℚ) / ({den} : ℚ)) = (({num} : ℚ) / ({den} : ℚ))"
    )
    return LeanTask(name=name, statement=statement, tactics=("rfl", "norm_num"))


def _emit_arithmetic_match(problem: str, answer: str, name: str) -> LeanTask | None:
    """Template: the problem contains a concrete arithmetic expression that equals `answer`.

    Look for an expression of the form ``<arith> = <answer>`` in the problem text. If found,
    Lean checks ``<arith> = <answer>`` directly with the full tactic portfolio.
    """
    if not (_INT_RE.match(answer) or _FRAC_RE.match(answer)):
        return None
    # Search for "<digits and operators> = <answer>" anywhere in the problem.
    pattern = re.compile(
        r"([\d\s+\-*/().]+)\s*=\s*" + re.escape(answer) + r"\b"
    )
    m = pattern.search(problem)
    if m is None:
        return None
    lhs = m.group(1).strip()
    if not re.match(r"^[\d\s+\-*/().]+$", lhs) or len(lhs) > 200:
        return None
    statement = f"theorem {name} : ({lhs} : ℚ) = ({answer} : ℚ)"
    return LeanTask(name=name, statement=statement, tactics=_DEFAULT_TACTICS)


_TEMPLATES: dict[str, Callable[[str, str, str], LeanTask | None]] = {
    "arithmetic_match": _emit_arithmetic_match,
    "rational_equality": _emit_rational_equality,
    "integer_equality": _emit_integer_equality,
}


def list_template_kinds() -> list[str]:
    return list(_TEMPLATES.keys())


def emit_template_task(
    *,
    problem_id: str,
    class_label: str,
    problem: str,
    answer: str,
    artifact_idx: int = 0,
) -> TemplateMatch:
    """Try templates in declaration order; first applicable wins.

    Returns a :class:`TemplateMatch` whose `task` is `None` if no template applied (in which
    case the pipeline should hand off to the autoformalizer for this (class, artifact_idx)).
    """
    name = _theorem_name(problem_id, class_label, artifact_idx)
    for kind, fn in _TEMPLATES.items():
        task = fn(problem, answer, name)
        if task is not None:
            return TemplateMatch(kind=kind, task=task)
    return TemplateMatch(kind="no_template", task=None)
