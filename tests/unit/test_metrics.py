"""Tests for covcal.metrics."""

from __future__ import annotations

import pytest

from covcal.metrics import evaluate
from covcal.types import ABSTAIN, SelectorOutput


def _out(label: str) -> SelectorOutput:
    return SelectorOutput(label)


class TestSelectiveMetrics:
    def test_basic_accuracy_and_acceptance(self):
        outs = [_out("a"), _out("b"), _out(ABSTAIN), _out("c")]
        refs = ["a", "a", "b", "c"]
        m = evaluate(outs, refs)
        # Accepted = 3 (a, b, c). Correct = 2 (a, c). Overall correct = 2.
        assert m.n_total == 4
        assert m.n_accepted == 3
        assert m.n_correct_accepted == 2
        assert m.accepted_accuracy == pytest.approx(2 / 3)
        assert m.selective_risk == pytest.approx(1 / 3)
        assert m.overall_accuracy == pytest.approx(2 / 4)
        assert m.accepted_fraction == pytest.approx(3 / 4)
        assert m.abstention_rate == pytest.approx(1 / 4)

    def test_all_abstain_gives_zero_accuracy(self):
        outs = [_out(ABSTAIN), _out(ABSTAIN)]
        refs = ["a", "b"]
        m = evaluate(outs, refs)
        assert m.overall_accuracy == 0.0
        assert m.accepted_accuracy == 0.0
        assert m.selective_risk == 0.0  # no accepted ⇒ vacuous
        assert m.risk_upper_bound() == 1.0  # vacuous UB

    def test_perfect_accepted(self):
        outs = [_out("a"), _out("b"), _out("c")]
        refs = ["a", "b", "c"]
        m = evaluate(outs, refs)
        assert m.selective_risk == 0.0
        # UB for 0/3 errors at alpha=0.05 ⇒ 1 - 0.05^(1/3)
        assert m.risk_upper_bound(0.05) == pytest.approx(1 - 0.05 ** (1 / 3))

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            evaluate([_out("a")], ["a", "b"])
