"""``Autoformalizer.formalize_many`` batches LLM calls across classes.

The point of these tests is to pin the batching contract: the cross-class formalization
must arrive at the backend as one ``generate_batch`` call, not as a Python loop of
``generate`` calls. On llama.cpp this is a no-op (the default ``generate_batch`` is a
serial fallback), but vLLM's override is where the GPU win lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from covcal.generation.backend import GenerationRequest, GenerationResult, LLMBackend
from covcal.lean import Autoformalizer, AutoformalizerConfig, MockLeanBackend
from covcal.types import Status


@dataclass
class RecordingLLM(LLMBackend):
    """Records every backend call so tests can assert on batching."""

    canned_lean: str = (
        "theorem t1 (x : Nat) : x + 0 = x := by sorry"
    )
    batch_calls: list[list[GenerationRequest]] = field(default_factory=list)
    generate_calls: list[GenerationRequest] = field(default_factory=list)

    @property
    def backend_id(self) -> str:
        return "recording:test"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.generate_calls.append(request)
        return GenerationResult(
            samples=[self.canned_lean] * request.n_samples, backend_id=self.backend_id
        )

    def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> list[GenerationResult]:
        # Capture the batch as one event; do NOT delegate to ``generate`` so the test
        # distinguishes "real batch override" from "serial fallback".
        self.batch_calls.append(list(requests))
        return [
            GenerationResult(
                samples=[self.canned_lean] * r.n_samples, backend_id=self.backend_id
            )
            for r in requests
        ]


def test_formalize_many_issues_one_batch_for_initial_formalization():
    """4 classes that need LLM artifacts ⇒ one ``generate_batch`` call with 4 requests."""
    llm = RecordingLLM()
    # MockLeanBackend defaults to UNFORMALIZED for unknown names, which means no artifact
    # will be flagged ILLTYPED — repair pass is a no-op, isolating the test to tier 2.
    lean = MockLeanBackend(default_status=Status.UNFORMALIZED)

    af = Autoformalizer(
        # use_templates=False forces every artifact through the LLM tier; this is the
        # case the batching contract targets (it's what the autoformalizer-heavy stronger
        # run looks like in practice when templates miss).
        config=AutoformalizerConfig(artifacts_per_class=2, use_templates=False, use_llm=True),
        lean=lean,
        llm=llm,
    )

    items = [
        ("P1", "label_a", "problem text", "answer_a"),
        ("P1", "label_b", "problem text", "answer_b"),
        ("P1", "label_c", "problem text", "answer_c"),
        ("P1", "label_d", "problem text", "answer_d"),
    ]
    result = af.formalize_many(items)

    # Exactly one batched call carrying all four formalization requests.
    assert len(llm.batch_calls) == 1, (
        f"expected a single generate_batch call across classes, got {len(llm.batch_calls)}"
    )
    assert len(llm.batch_calls[0]) == 4
    # No raw .generate() calls in the initial-formalization tier.
    # (Repair-pass .generate_batch may add a call iff there are illtyped artifacts.)
    assert llm.generate_calls == []

    # Per-class artifacts produced; each gets ``artifacts_per_class`` samples (n_samples=2).
    assert set(result) == {"label_a", "label_b", "label_c", "label_d"}
    for label, arts in result.items():
        assert len(arts) == 2, label
        assert all(a.source == "autoformalizer" for a in arts)


def test_formalize_many_templates_skip_llm_when_satisfied():
    """If templates fully fill every class, no LLM call is issued."""
    llm = RecordingLLM()
    lean = MockLeanBackend(default_status=Status.UNFORMALIZED)
    # Force templates to satisfy the quota: artifacts_per_class=1 + use_templates=True.
    # The MATH integer-equality template applies for numeric answers.
    af = Autoformalizer(
        config=AutoformalizerConfig(artifacts_per_class=1, use_templates=True, use_llm=True),
        lean=lean,
        llm=llm,
    )
    items = [("P1", "2", "What is 1+1?", "2"), ("P1", "3", "What is 1+1?", "3")]
    result = af.formalize_many(items)

    # Templates can handle integer answers; LLM should be untouched.
    assert llm.batch_calls == []
    assert llm.generate_calls == []
    for arts in result.values():
        assert len(arts) == 1
        assert arts[0].source == "template"


def test_formalize_many_repair_batches_across_classes():
    """Illtyped artifacts across classes feed one batched repair generate_batch."""
    llm = RecordingLLM()
    lean = MockLeanBackend(default_status=Status.ILLTYPED)
    # Two classes through the LLM tier; both will come back ILLTYPED per MockLeanBackend,
    # so each gets a repair request — which we expect to arrive as one batch.
    af = Autoformalizer(
        config=AutoformalizerConfig(
            artifacts_per_class=1,
            use_templates=False,
            use_llm=True,
            repair_on_illtyped=True,
        ),
        lean=lean,
        llm=llm,
    )
    items = [("P1", "label_a", "p", "a"), ("P1", "label_b", "p", "b")]
    af.formalize_many(items)

    # Two batched calls total: initial formalization (1 batch of 2) and repair (1 batch of 2).
    assert len(llm.batch_calls) == 2
    assert all(len(b) == 2 for b in llm.batch_calls)
    assert llm.generate_calls == []
