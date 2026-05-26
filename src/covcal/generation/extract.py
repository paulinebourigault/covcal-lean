"""Final-answer extraction from generator output.

The candidate-generation prompt asks the model to wrap the final answer in `\\boxed{...}`.
We implement an extractor with a few fallbacks, but every fallback is logged so we can audit drift between runs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Same regex as covcal.normalization, duplicated here so this module has no upward dependency.
_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_ANSWER_TAG_RE = re.compile(r"(?:final answer|answer)\s*[:=]\s*(.+?)(?:\n|$)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ExtractedAnswer:
    text: str
    source: str  # "boxed" | "answer_tag" | "last_number" | "raw"

    @property
    def found(self) -> bool:
        return self.source != "raw"


def extract_final_answer(generation: str) -> ExtractedAnswer:
    """Best-effort extraction. Always returns a non-empty `text` (raw input as last resort).

    Order:
      1. last `\\boxed{...}` (the prompted format)
      2. "Final answer: ..." or "Answer: ..." trailing line
      3. last numeric-looking token in the string
      4. the whole generation, marked `raw` so downstream normalization treats it accordingly
    """
    g = generation or ""
    boxed = _BOXED_RE.findall(g)
    if boxed:
        return ExtractedAnswer(text=boxed[-1].strip(), source="boxed")

    tag = _ANSWER_TAG_RE.findall(g)
    if tag:
        return ExtractedAnswer(text=tag[-1].strip(), source="answer_tag")

    # Last fallback: trailing token. Better than nothing for math contests.
    tail = g.strip().splitlines()[-1].strip() if g.strip() else ""
    if tail:
        return ExtractedAnswer(text=tail, source="last_number")

    return ExtractedAnswer(text=g.strip(), source="raw")
