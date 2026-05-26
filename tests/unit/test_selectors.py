"""Tests for covcal.selectors."""

from __future__ import annotations

from covcal.selectors import (
    ConfidenceOnly,
    CovCal,
    CovCalPlusFallback,
    MarginOnly,
    ProvedCoverageOnly,
    TypedCoverageOnly,
    base_formal_selector,
    raw_lean_plus_fallback,
    self_consistency,
)
from covcal.types import (
    ABSTAIN,
    ArtifactOutcome,
    ClassRecord,
    FormalObservation,
    Status,
    Thresholds,
)


def _obs(classes: list[ClassRecord]) -> FormalObservation:
    return FormalObservation(problem_id="t", classes=classes)


def _cls(label: str, weight: float, *, proved=False, typed=False) -> ClassRecord:
    arts: list[ArtifactOutcome] = []
    if proved:
        arts.append(ArtifactOutcome(Status.PROVED))
    elif typed:
        arts.append(ArtifactOutcome(Status.TYPECHECKED))
    return ClassRecord(label=label, weight=weight, artifacts=arts)


class TestBaseFormalSelector:
    def test_picks_proved_winner(self):
        obs = _obs([_cls("a", 0.6, proved=True), _cls("b", 0.4)])
        assert base_formal_selector(obs).selected == "a"

    def test_abstains_on_no_proof(self):
        obs = _obs([_cls("a", 0.7, typed=True), _cls("b", 0.3)])
        assert base_formal_selector(obs).abstained

    def test_abstains_on_conflict(self):
        obs = _obs([_cls("a", 0.6, proved=True), _cls("b", 0.4, proved=True)])
        out = base_formal_selector(obs)
        assert out.abstained and out.reason == "conflict"


class TestSelfConsistency:
    def test_picks_top_weight(self):
        obs = _obs([_cls("a", 0.7), _cls("b", 0.3)])
        assert self_consistency(obs).selected == "a"

    def test_abstains_on_empty(self):
        obs = _obs([])
        assert self_consistency(obs).abstained


class TestConfidenceOnly:
    def test_accepts_above_threshold(self):
        obs = _obs([_cls("a", 0.8), _cls("b", 0.2)])
        assert ConfidenceOnly(0.5)(obs).selected == "a"

    def test_abstains_below_threshold(self):
        obs = _obs([_cls("a", 0.4), _cls("b", 0.35), _cls("c", 0.25)])
        assert ConfidenceOnly(0.5)(obs).abstained


class TestRawLeanFallback:
    def test_falls_back_when_no_proof(self):
        obs = _obs([_cls("a", 0.7, typed=True), _cls("b", 0.3)])
        # No proved class ⇒ fall back to self-consistency, which returns "a".
        assert raw_lean_plus_fallback(obs).selected == "a"

    def test_uses_proved_when_available(self):
        obs = _obs([_cls("a", 0.4, proved=True), _cls("b", 0.6, typed=True)])
        # Proved winner is "a" even though "b" has higher weight (only "a" is proved).
        assert raw_lean_plus_fallback(obs).selected == "a"


class TestSingleAxisCalibrated:
    def test_typed_only_abstains_below(self):
        obs = _obs([_cls("a", 0.3, proved=True), _cls("b", 0.2, typed=True), _cls("c", 0.5)])
        # C_typ = 0.3 + 0.2 = 0.5
        assert TypedCoverageOnly(0.6)(obs).abstained
        assert TypedCoverageOnly(0.5)(obs).selected == "a"

    def test_proved_only_uses_proved_coverage(self):
        obs = _obs([_cls("a", 0.4, proved=True), _cls("b", 0.6, typed=True)])
        # C_prf = 0.4
        assert ProvedCoverageOnly(0.5)(obs).abstained
        assert ProvedCoverageOnly(0.3)(obs).selected == "a"

    def test_margin_only_uses_margin(self):
        obs = _obs([_cls("a", 0.4, proved=True), _cls("b", 0.35), _cls("c", 0.25)])
        # margin = 0.4 - 0.35 = 0.05
        assert MarginOnly(0.1)(obs).abstained
        assert MarginOnly(0.0)(obs).selected == "a"


class TestCovCal:
    def test_accepts_when_all_thresholds_met(self):
        obs = _obs([_cls("a", 0.7, proved=True), _cls("b", 0.2, typed=True), _cls("c", 0.1)])
        # C_typ = 0.9, C_prf = 0.7, margin = 0.7 - 0.2 = 0.5
        sel = CovCal(Thresholds(typ=0.5, prf=0.5, margin=0.25))
        assert sel(obs).selected == "a"

    def test_rejects_when_any_threshold_violated(self):
        obs = _obs([_cls("a", 0.4, proved=True), _cls("b", 0.35), _cls("c", 0.25)])
        # margin = 0.05
        sel = CovCal(Thresholds(typ=0.0, prf=0.0, margin=0.1))
        assert sel(obs).abstained

    def test_rejects_on_conflict(self):
        obs = _obs([_cls("a", 0.5, proved=True), _cls("b", 0.5, proved=True)])
        sel = CovCal(Thresholds(0.0, 0.0, -1.0))
        out = sel(obs)
        assert out.abstained and out.reason == "conflict"


class TestCovCalPlusFallback:
    def test_falls_back_on_reject(self):
        obs = _obs([_cls("a", 0.4, proved=True), _cls("b", 0.6)])
        sel = CovCalPlusFallback(Thresholds(0.9, 0.9, 0.5))
        out = sel(obs)
        assert out.selected == "b"  # self-consistency winner
        assert out.reason.startswith("fallback_sc")

    def test_uses_covcal_when_accepted(self):
        obs = _obs([_cls("a", 0.9, proved=True), _cls("b", 0.1)])
        sel = CovCalPlusFallback(Thresholds(0.0, 0.0, 0.0))
        out = sel(obs)
        assert out.selected == "a"
        assert "accepted" in out.reason


def test_abstain_sentinel_string():
    """ABSTAIN must be a JSON-safe string so logs round-trip cleanly."""
    import json
    assert ABSTAIN == "ABSTAIN"
    assert json.loads(json.dumps(ABSTAIN)) == ABSTAIN
