"""End-to-end pipeline test with all mocks (no LLM, no Lean).

Exercises:
  - generation via a stub LLM that emits one boxed answer per sample
  - aggregation into normalized classes
  - formalization via templates only
  - verification via MockLeanBackend with pre-registered outcomes
  - assembly into FormalObservation
  - diagnostics + selectors + calibration on the assembled observations
"""

from __future__ import annotations

from pathlib import Path

from covcal.calibration import make_grid, select_thresholds
from covcal.diagnostics import compute_diagnostics
from covcal.generation.backend import GenerationRequest, GenerationResult, LLMBackend
from covcal.lean import Autoformalizer, AutoformalizerConfig, MockLeanBackend
from covcal.metrics import evaluate
from covcal.pipeline import (
    PipelineRun,
    PipelineRunConfig,
    Problem,
    observation_to_dict,
    write_jsonl,
)
from covcal.selectors import CovCal, self_consistency
from covcal.types import Status, Thresholds


class ConstantLLM(LLMBackend):
    """Emits a fixed pattern of boxed answers per sample to drive deterministic tests."""

    def __init__(self, samples_by_prompt_suffix: dict[str, list[str]]) -> None:
        self._by = samples_by_prompt_suffix

    @property
    def backend_id(self) -> str:
        return "constant:test"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        for suffix, samples in self._by.items():
            if request.prompt.rstrip().endswith(suffix):
                samples = samples[: request.n_samples]
                # pad with last sample if requested more than provided
                while len(samples) < request.n_samples:
                    samples.append(samples[-1])
                return GenerationResult(samples=samples, backend_id=self.backend_id)
        raise AssertionError(f"no scripted samples for prompt ending in: ...{request.prompt[-60:]}")


def _boxed(x: str) -> str:
    return f"derivation\\boxed{{{x}}}"


def test_pipeline_end_to_end_with_mocks(tmp_path: Path):
    problems = [
        Problem(problem_id="P1", problem_text="What is 1+1?", reference_answer="2"),
        Problem(problem_id="P2", problem_text="What is 2/4?", reference_answer="1/2"),
        Problem(problem_id="P3", problem_text="Solve.", reference_answer="42"),
    ]
    llm = ConstantLLM({
        "What is 1+1?": [_boxed("2"), _boxed("2"), _boxed("3"), _boxed("2")],
        "What is 2/4?": [_boxed("1/2"), _boxed("0.5"), _boxed("2/4"), _boxed("1/3")],
        "Solve.": [_boxed("42"), _boxed("42"), _boxed("0"), _boxed("0")],
    })

    lean = MockLeanBackend(default_status=Status.UNFORMALIZED)
    # Pre-register: integer/rational-equality template winners pass; rivals are typechecked.
    lean.add_by_name("covcal_P1_2_0", Status.PROVED, tactic_used="rfl")
    lean.add_by_name("covcal_P1_3_0", Status.TYPECHECKED)
    lean.add_by_name("covcal_P2_1_2_0", Status.PROVED, tactic_used="rfl")
    lean.add_by_name("covcal_P2_1_3_0", Status.TYPECHECKED)
    lean.add_by_name("covcal_P3_42_0", Status.PROVED, tactic_used="rfl")
    lean.add_by_name("covcal_P3_0_0", Status.TYPECHECKED)

    autoformalizer = Autoformalizer(
        config=AutoformalizerConfig(artifacts_per_class=1, use_llm=False),
        lean=lean,
        llm=None,
    )

    cfg = PipelineRunConfig(
        run_dir=tmp_path / "runs",
        n_samples=4,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=64,
        seed=0,
        formalize_top_k_classes=2,
        prover_budget_seconds=10.0,
    )
    runner = PipelineRun(config=cfg, llm=llm, autoformalizer=autoformalizer, lean=lean)
    observations = list(runner.run_many(problems))
    assert len(observations) == 3

    # P1: classes {"2": 0.75, "3": 0.25}; both formalized; "2" proved -> winner "2"
    p1 = next(o for o in observations if o.problem_id == "P1")
    d1 = compute_diagnostics(p1)
    assert d1.proved_winner == "2"
    assert d1.proved_coverage == 0.75
    # P2: normalization merges "1/2","0.5","2/4" into one class with weight 0.75
    p2 = next(o for o in observations if o.problem_id == "P2")
    d2 = compute_diagnostics(p2)
    assert d2.proved_winner == "1/2"
    assert d2.proved_coverage == 0.75

    # Selectors
    sc_results = [self_consistency(o) for o in observations]
    refs = [o.metadata["reference_class"] for o in observations]
    m_sc = evaluate(sc_results, refs)
    assert m_sc.accepted_accuracy == 1.0  # all three self-consistency winners are correct

    # CovCal at trivial thresholds should also accept all and be correct.
    covcal = CovCal(Thresholds(typ=0.0, prf=0.0, margin=0.0))
    cov_results = [covcal(o) for o in observations]
    m_cov = evaluate(cov_results, refs)
    assert m_cov.n_accepted == 3
    assert m_cov.accepted_accuracy == 1.0

    # Calibrate over a tiny grid: expect a feasible threshold and a reported UB ≤ 1.
    grid = make_grid([0.0, 0.5], [0.0, 0.5], [-1.0, 0.0])
    counts: dict[Thresholds, tuple[int, int]] = {}
    for tau in grid:
        s = CovCal(tau)
        m, k = 0, 0
        for o, r in zip(observations, refs, strict=True):
            out = s(o)
            if out.abstained:
                continue
            m += 1
            if out.selected != r:
                k += 1
        counts[tau] = (m, k)
    # With only 3 calibration examples, the CP upper bound at 0 errors and |T|=8 is
    # 1 - (0.05/8)^(1/3) ≈ 0.82, so we need a loose epsilon for the plumbing test.
    # The statistical sharpness is exercised in the calibration unit tests.
    res = select_thresholds(grid, counts, epsilon=0.95, delta=0.05)
    assert res.selected is not None  # feasible
    assert res.risk_upper_bound <= 0.95

    # Persistence: write & re-read observations as JSONL.
    obs_path = tmp_path / "obs.jsonl"
    n = write_jsonl(obs_path, (observation_to_dict(o) for o in observations))
    assert n == 3
    text = obs_path.read_text(encoding="utf-8").splitlines()
    assert len(text) == 3
