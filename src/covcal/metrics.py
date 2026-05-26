"""Evaluation metrics for selective answer selection.

Inputs are always lists of (selector_output, reference_label) pairs. The reference label
is the normalized class label produced by the same normalizer used in the pipeline.

We use the distinction between overall and accepted quantities:

* overall accuracy: ignores abstention (a selector that always abstains gets accuracy 0).
* accepted accuracy: accuracy among the *accepted* subset.
* selective risk: 1 - accepted accuracy.
* accepted fraction (a.k.a. coverage in the selective-classification sense).

The Clopper--Pearson upper bound on the selective risk is also exposed here for reporting
in Table 1 of the paper (its calibration counterpart lives in :mod:`covcal.calibration`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .calibration import clopper_pearson_upper
from .types import SelectorOutput


@dataclass(frozen=True, slots=True)
class SelectiveMetrics:
    n_total: int
    n_accepted: int
    n_correct_overall: int
    n_correct_accepted: int

    @property
    def overall_accuracy(self) -> float:
        return self.n_correct_overall / self.n_total if self.n_total else 0.0

    @property
    def accepted_accuracy(self) -> float:
        return self.n_correct_accepted / self.n_accepted if self.n_accepted else 0.0

    @property
    def selective_risk(self) -> float:
        return 1.0 - self.accepted_accuracy if self.n_accepted else 0.0

    @property
    def abstention_rate(self) -> float:
        return 1.0 - (self.n_accepted / self.n_total) if self.n_total else 0.0

    @property
    def accepted_fraction(self) -> float:
        return self.n_accepted / self.n_total if self.n_total else 0.0

    def risk_upper_bound(self, alpha: float = 0.05) -> float:
        """One-sided Clopper--Pearson UB on the selective risk."""
        if self.n_accepted == 0:
            return 1.0
        errors = self.n_accepted - self.n_correct_accepted
        return clopper_pearson_upper(errors, self.n_accepted, alpha)


def evaluate(
    outputs: Sequence[SelectorOutput], references: Sequence[str]
) -> SelectiveMetrics:
    if len(outputs) != len(references):
        raise ValueError(
            f"outputs and references must align (got {len(outputs)} vs {len(references)})"
        )
    n_total = len(outputs)
    n_accepted = 0
    n_correct_overall = 0
    n_correct_accepted = 0
    for out, ref in zip(outputs, references, strict=True):
        accepted = not out.abstained
        if accepted:
            n_accepted += 1
            if out.selected == ref:
                n_correct_accepted += 1
                n_correct_overall += 1
        # An abstained prediction never contributes to overall accuracy.
    return SelectiveMetrics(
        n_total=n_total,
        n_accepted=n_accepted,
        n_correct_overall=n_correct_overall,
        n_correct_accepted=n_correct_accepted,
    )
