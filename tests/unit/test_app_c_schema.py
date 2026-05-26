"""Schema test: observation_to_dict + class_aux_rows match paper the structured log schema."""

from __future__ import annotations

from covcal.pipeline import (
    Problem,
    assemble_observation,
    class_aux_rows,
    observation_to_dict,
)
from covcal.types import (
    ArtifactOutcome,
    Candidate,
    ClassRecord,
    FormalObservation,
    Status,
    Thresholds,
)


def _make_obs() -> FormalObservation:
    candidates = [
        Candidate(answer_text="2", weight=0.75, sample_id=0),
        Candidate(answer_text="2", weight=0.0, sample_id=1),
        Candidate(answer_text="3", weight=0.25, sample_id=2),
    ]
    classes = [
        ClassRecord(label="2", weight=0.75,
                    artifacts=[ArtifactOutcome(Status.PROVED, "norm_num", 0.1, "ok", "template")]),
        ClassRecord(label="3", weight=0.25,
                    artifacts=[ArtifactOutcome(Status.TYPECHECKED, None, 0.05, "fail", "template")]),
    ]
    return FormalObservation(
        problem_id="t1",
        classes=classes,
        candidates=candidates,
        prover_budget_seconds=10.0,
        metadata={
            "dataset": "math500",
            "domain": "algebra",
            "problem_text": "What is 1+1?",
            "reference_answer": "2",
            "reference_class": "2",
            "included": True,
            "raw_samples": ["foo \\boxed{2}", "foo \\boxed{2}", "bar \\boxed{3}"],
            "extracted_answers": [{"text": "2", "source": "boxed"}] * 3,
            "artifacts_detail": {},
        },
    )


class TestAppCFields:
    def test_problem_fields_present(self):
        d = observation_to_dict(_make_obs())
        meta = d["metadata"]
        for key in ("dataset", "domain", "problem_text", "reference_answer",
                    "reference_class", "included"):
            assert key in meta, f"missing the structured log schema Problem field: {key}"

    def test_candidate_fields_present(self):
        d = observation_to_dict(_make_obs())
        assert "raw_samples" in d["metadata"]
        assert "extracted_answers" in d["metadata"]
        # And the per-candidate dicts include weight & answer_text & sample_id.
        c0 = d["candidates"][0]
        assert {"answer_text", "weight", "sample_id"} <= set(c0)

    def test_coverage_fields_present(self):
        d = observation_to_dict(_make_obs())
        diag = d["diagnostics"]
        for key in ("typed_coverage", "proved_coverage", "unresolved_rival_mass",
                    "margin", "conflict", "proved_winner"):
            assert key in diag
        # Numerics line up: C_typ = 1.0, C_prf = 0.75, winner = "2", R_unres = 0.25, M = 0.5
        assert diag["typed_coverage"] == 1.0
        assert diag["proved_coverage"] == 0.75
        assert diag["proved_winner"] == "2"
        assert diag["unresolved_rival_mass"] == 0.25
        assert diag["margin"] == 0.5
        assert diag["conflict"] is False

    def test_selection_fields_when_thresholds_given(self):
        obs = _make_obs()
        d = observation_to_dict(obs, covcal_thresholds=Thresholds(0.5, 0.5, 0.0))
        dec = d["decisions"]
        assert dec["self_consistency_class"] == "2"
        assert dec["covcal_decision"] == "2"
        assert dec["covcal_correct"] is True
        assert dec["fallback_decision"] == "2"
        assert dec["fallback_correct"] is True
        assert dec["thresholds"] == [0.5, 0.5, 0.0]

    def test_selection_correctness_none_when_unlabeled(self):
        obs = _make_obs()
        obs.metadata.pop("reference_class", None)
        d = observation_to_dict(obs, covcal_thresholds=Thresholds(0.0, 0.0, 0.0))
        assert d["decisions"]["covcal_correct"] is None
        assert d["decisions"]["fallback_correct"] is None

    def test_class_records_include_class_status(self):
        d = observation_to_dict(_make_obs())
        statuses = {c["label"]: c["class_status"] for c in d["classes"]}
        assert statuses["2"] == "proved"
        assert statuses["3"] == "typechecked"

    def test_class_aux_rows_per_class(self):
        rows = class_aux_rows(_make_obs())
        labels = {r["class_label"] for r in rows}
        assert labels == {"2", "3"}
        ref_row = next(r for r in rows if r["class_label"] == "2")
        assert ref_row["proved"] is True
        assert ref_row["is_reference"] is True
        assert ref_row["routes"] == ["template"]


class TestAssembleObservation:
    def test_threads_raw_samples_dataset_included(self):
        problem = Problem(
            problem_id="p1", problem_text="What is 1+1?",
            reference_answer="2", domain="algebra",
        )
        candidates = [Candidate("2", 0.5, 0), Candidate("3", 0.5, 1)]
        # Empty outcomes record so we don't have to fake a Lean run.
        from covcal.pipeline import OutcomesRecord
        outs = OutcomesRecord(problem_id="p1", outcomes_by_class={})
        obs = assemble_observation(
            problem, candidates, outs,
            prover_budget_seconds=10.0,
            raw_samples=["raw1", "raw2"],
            extracted=[{"text": "2", "source": "boxed"}, {"text": "3", "source": "boxed"}],
            dataset_name="math500",
            included=True,
        )
        assert obs.metadata["dataset"] == "math500"
        assert obs.metadata["raw_samples"] == ["raw1", "raw2"]
        assert obs.metadata["problem_text"] == "What is 1+1?"
        assert obs.metadata["included"] is True
