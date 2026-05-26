"""Tests for covcal.diagnostics."""

from __future__ import annotations

import math

import pytest

from covcal.diagnostics import compute_diagnostics
from covcal.types import ArtifactOutcome, ClassRecord, FormalObservation, Status


def _obs(classes: list[ClassRecord]) -> FormalObservation:
    return FormalObservation(problem_id="t", classes=classes)


def _cls(label: str, weight: float, *, proved: bool = False, typed: bool = False) -> ClassRecord:
    arts: list[ArtifactOutcome] = []
    if proved:
        arts.append(ArtifactOutcome(Status.PROVED))
    elif typed:
        arts.append(ArtifactOutcome(Status.TYPECHECKED))
    return ClassRecord(label=label, weight=weight, artifacts=arts)


class TestCoverageMass:
    def test_typed_and_proved_sum(self):
        obs = _obs([
            _cls("a", 0.5, proved=True),
            _cls("b", 0.3, typed=True),
            _cls("c", 0.2),  # unformalized
        ])
        d = compute_diagnostics(obs)
        assert d.proved_coverage == pytest.approx(0.5)
        # Typed coverage = proved + typechecked + timeout. Here that's 0.5 + 0.3 = 0.8.
        assert d.typed_coverage == pytest.approx(0.8)


class TestProvedWinner:
    def test_picks_highest_weight_proved(self):
        obs = _obs([
            _cls("a", 0.4, proved=True),
            _cls("b", 0.3, proved=True),  # second proved class -> conflict, but winner is "a"
            _cls("c", 0.3),
        ])
        d = compute_diagnostics(obs)
        assert d.proved_winner == "a"
        assert d.proved_winner_weight == pytest.approx(0.4)
        assert d.conflict is True

    def test_no_proved(self):
        obs = _obs([_cls("a", 0.7, typed=True), _cls("b", 0.3)])
        d = compute_diagnostics(obs)
        assert d.proved_winner is None
        assert d.proved_coverage == 0.0
        assert math.isinf(d.margin) and d.margin < 0


class TestUnresolvedRivalAndMargin:
    def test_basic_margin(self):
        obs = _obs([
            _cls("a", 0.45, proved=True),
            _cls("b", 0.35),  # unresolved rival
            _cls("c", 0.20),
        ])
        d = compute_diagnostics(obs)
        assert d.unresolved_rival_mass == pytest.approx(0.35)
        assert d.margin == pytest.approx(0.10)

    def test_no_rival_when_everything_proved(self):
        obs = _obs([
            _cls("a", 0.6, proved=True),
            _cls("b", 0.4, proved=True),
        ])
        d = compute_diagnostics(obs)
        assert d.unresolved_rival_mass == 0.0  # both proved, no unresolved rival
        assert d.margin == pytest.approx(0.6)  # winner weight only
        assert d.conflict is True


class TestConflictSemantics:
    def test_two_proved_classes_flag_conflict(self):
        obs = _obs([
            _cls("a", 0.5, proved=True),
            _cls("b", 0.5, proved=True),
        ])
        assert compute_diagnostics(obs).conflict is True

    def test_single_proved_no_conflict(self):
        obs = _obs([
            _cls("a", 0.7, proved=True),
            _cls("b", 0.3, typed=True),
        ])
        assert compute_diagnostics(obs).conflict is False
