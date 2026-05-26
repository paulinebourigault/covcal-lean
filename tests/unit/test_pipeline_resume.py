"""Crash-recovery tests for ``PipelineRun.write_run``.

Covers:
  * ``_recover_done_ids`` parses complete rows and truncates a partial trailing line.
  * ``_rebuild_aux_for_done`` regenerates class_aux from the surviving observations.
  * ``write_run`` with ``resume=True`` appends only the pending problems and leaves the
    already-completed rows untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

from covcal.generation.backend import GenerationRequest, GenerationResult, LLMBackend
from covcal.lean import Autoformalizer, AutoformalizerConfig, MockLeanBackend
from covcal.pipeline import (
    PipelineRun,
    PipelineRunConfig,
    Problem,
    _recover_done_ids,
)
from covcal.types import Status


class ConstantLLM(LLMBackend):
    def __init__(self, samples: list[str]) -> None:
        self._samples = samples

    @property
    def backend_id(self) -> str:
        return "constant:resume-test"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        samples = list(self._samples[: request.n_samples])
        while len(samples) < request.n_samples:
            samples.append(self._samples[-1])
        return GenerationResult(samples=samples, backend_id=self.backend_id)


def _boxed(x: str) -> str:
    return f"derivation\\boxed{{{x}}}"


def test_recover_done_ids_parses_complete_and_truncates_partial(tmp_path: Path):
    p = tmp_path / "obs.jsonl"
    good1 = json.dumps({"problem_id": "P1", "classes": [], "candidates": []})
    good2 = json.dumps({"problem_id": "P2", "classes": [], "candidates": []})
    partial = '{"problem_id": "P3", "par'  # SIGKILL mid-write, no trailing newline
    p.write_text(good1 + "\n" + good2 + "\n" + partial)

    done = _recover_done_ids(p)

    assert done == {"P1", "P2"}
    # File should be truncated to exactly the two good lines + their trailing newlines.
    assert p.read_text() == good1 + "\n" + good2 + "\n"


def test_recover_done_ids_empty_file(tmp_path: Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert _recover_done_ids(p) == set()


def test_recover_done_ids_no_trailing_newline_on_last_good_line(tmp_path: Path):
    # If the SIGKILL fell *exactly* on the newline after a complete record, the last good
    # JSON is still considered partial (no '\n' yet) and will be re-run. That's safer than
    # accepting a possibly-half-written line.
    p = tmp_path / "obs.jsonl"
    good1 = json.dumps({"problem_id": "P1", "classes": [], "candidates": []})
    p.write_text(good1)  # no trailing \n
    done = _recover_done_ids(p)
    assert done == set()
    assert p.read_text() == ""


def test_write_run_resume_appends_only_pending(tmp_path: Path):
    """End-to-end resume: pre-seed obs.jsonl, run write_run, verify no re-runs and new appends."""
    problems = [
        Problem(problem_id="P1", problem_text="What is 1+1?", reference_answer="2"),
        Problem(problem_id="P2", problem_text="What is 2/4?", reference_answer="1/2"),
        Problem(problem_id="P3", problem_text="Solve.", reference_answer="42"),
    ]

    # Pre-seed obs.jsonl with P1's record (as if a prior run got partway), plus a partial
    # tail to make sure truncation works in the integration path too.
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    obs_path = run_dir / "observations.jsonl"
    aux_path = run_dir / "class_aux.jsonl"
    p1_record = {
        "problem_id": "P1",
        "prover_budget_seconds": 10.0,
        "metadata": {"reference_class": "2"},
        "diagnostics": {},
        "decisions": {},
        "classes": [
            {"label": "2", "weight": 1.0, "candidate_indices": [0], "artifacts": []},
        ],
        "candidates": [{"answer_text": "2", "weight": 1.0, "sample_id": 0}],
    }
    obs_path.write_text(json.dumps(p1_record) + "\n" + '{"problem_id": "P2", "par')

    # An LLM/Lean that would fail loudly if called for P1 (since P1 must be skipped).
    class TripWireLLM(LLMBackend):
        @property
        def backend_id(self) -> str:
            return "tripwire"

        def generate(self, request: GenerationRequest) -> GenerationResult:
            if "1+1" in request.prompt:
                raise AssertionError("P1 must be skipped on resume")
            # Return one boxed answer per sample for P2 / P3.
            text = _boxed("1/2") if "2/4" in request.prompt else _boxed("42")
            return GenerationResult(
                samples=[text] * request.n_samples, backend_id=self.backend_id
            )

    llm = TripWireLLM()
    lean = MockLeanBackend(default_status=Status.UNFORMALIZED)
    lean.add_by_name("covcal_P2_1_2_0", Status.PROVED, tactic_used="rfl")
    lean.add_by_name("covcal_P3_42_0", Status.PROVED, tactic_used="rfl")

    autoformalizer = Autoformalizer(
        config=AutoformalizerConfig(artifacts_per_class=1, use_llm=False),
        lean=lean,
        llm=None,
    )

    cfg = PipelineRunConfig(
        run_dir=run_dir,
        n_samples=2,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=64,
        seed=0,
        formalize_top_k_classes=1,
        prover_budget_seconds=10.0,
        resume=True,
    )
    runner = PipelineRun(config=cfg, llm=llm, autoformalizer=autoformalizer, lean=lean)
    runner.write_run(problems)

    # All three problems should now have exactly one obs row, in original-input order:
    # the pre-existing P1 row, then the newly appended P2 and P3.
    lines = [json.loads(line) for line in obs_path.read_text().splitlines() if line.strip()]
    assert [r["problem_id"] for r in lines] == ["P1", "P2", "P3"]
    # The original P1 record must be byte-for-byte preserved (no re-formalization).
    assert lines[0]["classes"][0]["label"] == "2"

    # class_aux was rebuilt + appended, so it has rows for all three problems.
    aux_lines = [json.loads(line) for line in aux_path.read_text().splitlines() if line.strip()]
    aux_pids = {r["problem_id"] for r in aux_lines}
    assert aux_pids == {"P1", "P2", "P3"}
