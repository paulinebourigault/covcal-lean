"""Tests for covcal.classes."""

from __future__ import annotations

import pytest

from covcal.classes import (
    aggregate_classes,
    assert_normalized_weights,
    attach_artifacts,
    self_consistency_candidates,
    top_k_by_weight,
)
from covcal.types import ArtifactOutcome, Candidate, Status


def test_self_consistency_candidates_uniform():
    cands = self_consistency_candidates(["1/2", "1/2", "0.5", "1/3"])
    assert all(c.weight == 0.25 for c in cands)
    assert [c.sample_id for c in cands] == [0, 1, 2, 3]


def test_aggregate_classes_merges_equivalent_answers():
    cands = self_consistency_candidates(["1/2", "0.5", "2/4", "1/3"])
    classes = aggregate_classes(cands)
    # All three "1/2"-equivalents merge.
    labels = [c.label for c in classes]
    assert "1/2" in labels
    assert "1/3" in labels
    half = next(c for c in classes if c.label == "1/2")
    third = next(c for c in classes if c.label == "1/3")
    assert half.weight == pytest.approx(0.75)
    assert third.weight == pytest.approx(0.25)


def test_aggregate_classes_sorted_by_weight():
    cands = self_consistency_candidates(["a", "a", "b", "c", "c", "c"])
    classes = aggregate_classes(cands)
    weights = [c.weight for c in classes]
    assert weights == sorted(weights, reverse=True)


def test_normalized_weights_check():
    cands = [Candidate("a", 0.4), Candidate("b", 0.3)]
    with pytest.raises(ValueError):
        assert_normalized_weights(aggregate_classes(cands))


def test_top_k_by_weight_stable():
    # Use numeric labels so we exercise the pure aggregator without touching MC handling.
    cands = self_consistency_candidates(["1", "1", "2", "3", "3", "3"])
    classes = aggregate_classes(cands)
    top = top_k_by_weight(classes, 2)
    assert [c.label for c in top] == ["3", "1"]


def test_attach_artifacts_links_by_label():
    cands = self_consistency_candidates(["1/2", "1/2", "1/3"])
    classes = aggregate_classes(cands)
    artifacts = {
        "1/2": [ArtifactOutcome(status=Status.PROVED, tactic_used="norm_num")],
        "1/3": [ArtifactOutcome(status=Status.TIMEOUT, tactic_used="ring_nf")],
        "missing_class_label": [ArtifactOutcome(status=Status.PROVED)],
    }
    attach_artifacts(classes, artifacts)
    half = next(c for c in classes if c.label == "1/2")
    third = next(c for c in classes if c.label == "1/3")
    assert half.proved is True
    assert third.proved is False
    assert third.typed is True  # timeout still counts as typed


def test_status_best_of_priority():
    assert Status.best_of([Status.TIMEOUT, Status.PROVED, Status.ILLTYPED]) is Status.PROVED
    assert Status.best_of([Status.ILLTYPED, Status.TYPECHECKED]) is Status.TYPECHECKED
    assert Status.best_of([]) is Status.UNFORMALIZED
