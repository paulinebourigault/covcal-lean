"""Selectors and baselines (Sec. 4, 5 of the paper + Sec. 6.4 baselines list).

Every selector is a callable: ``FormalObservation -> SelectorOutput``.

The nine entries in the experimental section are:

1. self_consistency
2. confidence_only_abstention (calibrated)
3. raw_lean_plus_fallback
4. proof_existence_abstention
5. typed_coverage_only (calibrated)
6. proved_coverage_only (calibrated)
7. margin_only (calibrated)
8. covcal (calibrated joint Thresholds)
9. covcal_plus_fallback

The calibrated selectors take a single scalar or `Thresholds` value as the calibration
output. The choice of *how* to calibrate (CP + Eq. 7) lives in :mod:`covcal.calibration`;
this module is just the rule that consumes a calibrated threshold and emits a decision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .diagnostics import Diagnostics, compute_diagnostics
from .types import ABSTAIN, FormalObservation, SelectorOutput, Thresholds

Selector = Callable[[FormalObservation], SelectorOutput]


# --- Helpers ---------------------------------------------------------------------------

def _top_class_label_and_weight(obs: FormalObservation) -> tuple[str | None, float]:
    if not obs.classes:
        return None, 0.0
    top = max(obs.classes, key=lambda c: (c.weight, c.label))
    return top.label, top.weight


def base_formal_selector(obs: FormalObservation) -> SelectorOutput:
    """g_F from the paper: return c+ if it exists and there is no conflict; else abstain."""
    diag = compute_diagnostics(obs)
    if diag.conflict:
        return SelectorOutput(ABSTAIN, reason="conflict")
    if diag.proved_winner is None:
        return SelectorOutput(ABSTAIN, reason="no_proof")
    return SelectorOutput(diag.proved_winner, reason="proved_winner")


# --- 1. Self-consistency ---------------------------------------------------------------

def self_consistency(obs: FormalObservation) -> SelectorOutput:
    label, _ = _top_class_label_and_weight(obs)
    if label is None:
        return SelectorOutput(ABSTAIN, reason="no_candidates")
    return SelectorOutput(label, reason="self_consistency")


# --- 2. Confidence-only abstention -----------------------------------------------------

@dataclass(frozen=True, slots=True)
class ConfidenceOnly:
    """Accept self-consistency only if the top class weight exceeds threshold."""

    weight_threshold: float

    def __call__(self, obs: FormalObservation) -> SelectorOutput:
        label, weight = _top_class_label_and_weight(obs)
        if label is None:
            return SelectorOutput(ABSTAIN, reason="no_candidates")
        if weight < self.weight_threshold:
            return SelectorOutput(ABSTAIN, reason=f"below_conf_{self.weight_threshold:.3f}")
        return SelectorOutput(label, reason="conf_accepted")


# --- 3. Raw Lean + fallback ------------------------------------------------------------

def raw_lean_plus_fallback(obs: FormalObservation) -> SelectorOutput:
    """Choose the highest-weight proved class if any (no conflict); else self-consistency."""
    decision = base_formal_selector(obs)
    if not decision.abstained:
        return decision
    return self_consistency(obs)


# --- 4. Proof-existence abstention -----------------------------------------------------

def proof_existence_abstention(obs: FormalObservation) -> SelectorOutput:
    """Accept iff at least one class is proved and no conflict; else abstain."""
    return base_formal_selector(obs)


# --- 5-7. Single-axis calibrated thresholds -------------------------------------------

@dataclass(frozen=True, slots=True)
class TypedCoverageOnly:
    threshold: float

    def __call__(self, obs: FormalObservation) -> SelectorOutput:
        diag = compute_diagnostics(obs)
        if diag.conflict:
            return SelectorOutput(ABSTAIN, reason="conflict")
        if diag.proved_winner is None:
            return SelectorOutput(ABSTAIN, reason="no_proof")
        if diag.typed_coverage < self.threshold:
            return SelectorOutput(ABSTAIN, reason=f"typ_below_{self.threshold:.3f}")
        return SelectorOutput(diag.proved_winner, reason="typ_accepted")


@dataclass(frozen=True, slots=True)
class ProvedCoverageOnly:
    threshold: float

    def __call__(self, obs: FormalObservation) -> SelectorOutput:
        diag = compute_diagnostics(obs)
        if diag.conflict:
            return SelectorOutput(ABSTAIN, reason="conflict")
        if diag.proved_winner is None:
            return SelectorOutput(ABSTAIN, reason="no_proof")
        if diag.proved_coverage < self.threshold:
            return SelectorOutput(ABSTAIN, reason=f"prf_below_{self.threshold:.3f}")
        return SelectorOutput(diag.proved_winner, reason="prf_accepted")


@dataclass(frozen=True, slots=True)
class MarginOnly:
    threshold: float

    def __call__(self, obs: FormalObservation) -> SelectorOutput:
        diag = compute_diagnostics(obs)
        if diag.conflict:
            return SelectorOutput(ABSTAIN, reason="conflict")
        if diag.proved_winner is None:
            return SelectorOutput(ABSTAIN, reason="no_proof")
        if diag.margin < self.threshold:
            return SelectorOutput(ABSTAIN, reason=f"margin_below_{self.threshold:.3f}")
        return SelectorOutput(diag.proved_winner, reason="margin_accepted")


# --- 8. CovCal -------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CovCal:
    """Risk-controlled max-coverage selector. Acceptance rule from Sec. 5 of the paper.

    Decision rule (with thresholds tau):
      - reject if there is a conflict or no proved class;
      - reject if any of C_typ < tau.typ, C_prf < tau.prf, M < tau.margin;
      - else return c+.
    """

    thresholds: Thresholds

    def acceptance(self, obs: FormalObservation, diag: Diagnostics | None = None) -> bool:
        diag = diag if diag is not None else compute_diagnostics(obs)
        if diag.conflict or diag.proved_winner is None:
            return False
        if diag.typed_coverage < self.thresholds.typ:
            return False
        if diag.proved_coverage < self.thresholds.prf:
            return False
        return not diag.margin < self.thresholds.margin

    def __call__(self, obs: FormalObservation) -> SelectorOutput:
        diag = compute_diagnostics(obs)
        if not self.acceptance(obs, diag):
            if diag.conflict:
                return SelectorOutput(ABSTAIN, reason="conflict")
            if diag.proved_winner is None:
                return SelectorOutput(ABSTAIN, reason="no_proof")
            return SelectorOutput(ABSTAIN, reason="below_thresholds")
        # `compute_diagnostics` guarantees proved_winner is not None when accepted
        assert diag.proved_winner is not None
        return SelectorOutput(diag.proved_winner, reason="covcal_accepted")


# --- 9. CovCal + fallback --------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CovCalPlusFallback:
    """Same acceptance rule as CovCal; falls back to self-consistency on rejection.

    The paper is explicit (Sec. 5): the formal selective-risk certificate covers only the
    accepted formal predictions, **not** the fallback outputs. We expose this clearly via
    the `reason` field so downstream metrics can stratify "covered" vs. "uncovered".
    """

    thresholds: Thresholds

    def __call__(self, obs: FormalObservation) -> SelectorOutput:
        covcal = CovCal(self.thresholds)
        out = covcal(obs)
        if not out.abstained:
            return SelectorOutput(out.selected, reason="covcal_accepted_in_fallback")
        sc = self_consistency(obs)
        return SelectorOutput(sc.selected, reason=f"fallback_sc_after_{out.reason}")
