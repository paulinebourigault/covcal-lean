"""Tests for the autoformalizer (templates + LLM + repair routing)."""

from __future__ import annotations

from covcal.generation.backend import GenerationRequest, GenerationResult, LLMBackend
from covcal.lean import Autoformalizer, AutoformalizerConfig, MockLeanBackend
from covcal.types import Status


class StubLLM(LLMBackend):
    """Returns scripted completions. `replies` is a list of strings, consumed in order.
    A trailing "REPAIR:" prefix marks a repair-only response that is returned for the
    first repair call only.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[GenerationRequest] = []

    @property
    def backend_id(self) -> str:
        return "stub:llm"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if not self._replies:
            text = ""
        else:
            text = self._replies.pop(0)
        # Pad to request.n_samples by repeating.
        samples = [text] * request.n_samples
        return GenerationResult(samples=samples, backend_id=self.backend_id)


class TestTemplateOnlyPath:
    def test_uses_template_skips_llm(self):
        lean = MockLeanBackend()
        llm = StubLLM(replies=["should_not_be_used"])
        af = Autoformalizer(
            config=AutoformalizerConfig(artifacts_per_class=1, use_llm=True),
            lean=lean,
            llm=llm,
        )
        arts = af.formalize(
            problem_id="p", class_label="42", problem="x", answer="42"
        )
        assert len(arts) == 1
        assert arts[0].source == "template"
        assert llm.calls == []


class TestLLMFallback:
    def test_llm_used_when_template_inapplicable(self):
        lean = MockLeanBackend()
        # Pretend autoformalizer output succeeds (so no repair is triggered).
        llm = StubLLM(replies=["```lean\ntheorem foo : True := by trivial\n```"])
        af = Autoformalizer(
            config=AutoformalizerConfig(
                artifacts_per_class=1, use_llm=True, repair_on_illtyped=False
            ),
            lean=lean,
            llm=llm,
        )
        arts = af.formalize(
            problem_id="p", class_label="x", problem="abstract", answer="UNNORMALIZED::?"
        )
        assert len(arts) == 1
        assert arts[0].source == "autoformalizer"
        assert "theorem af_p_x_1" in arts[0].task.statement
        assert llm.calls and llm.calls[0].n_samples == 1


class TestRepairPass:
    def test_repair_runs_on_illtyped(self):
        lean = MockLeanBackend()
        # First LLM artifact will be marked ILLTYPED in the mock so repair fires.
        af = Autoformalizer(
            config=AutoformalizerConfig(
                artifacts_per_class=1, use_llm=True, repair_on_illtyped=True
            ),
            lean=lean,
            llm=StubLLM(
                replies=[
                    "```lean\ntheorem foo : False := by trivial\n```",
                    "```lean\ntheorem foo : True := by trivial\n```",
                ]
            ),
        )
        # Pre-register: the first call to the mock for the LLM artifact returns ILLTYPED.
        # We don't know the salted name in advance, so use default behavior + add_by_statement
        # for the first artifact's statement-hash. The autoformalizer renames the theorem
        # head to af_p_x_1; we capture that by calling formalize and inspecting.
        # Simpler: rely on the default (UNFORMALIZED) and observe that no repair fires.
        # To explicitly exercise repair we need ILLTYPED — override default.
        lean._default = Status.ILLTYPED  # type: ignore[attr-defined]
        arts = af.formalize(
            problem_id="p", class_label="x", problem="abstract", answer="UNNORMALIZED::?"
        )
        # One LLM artifact + one repair artifact.
        sources = [a.source for a in arts]
        assert "autoformalizer" in sources
        assert "autoformalizer_repair" in sources


class TestNoLLM:
    def test_no_llm_only_templates(self):
        lean = MockLeanBackend()
        af = Autoformalizer(
            config=AutoformalizerConfig(artifacts_per_class=2, use_llm=False),
            lean=lean,
            llm=None,
        )
        # No template applies to UNNORMALIZED ⇒ no artifacts.
        arts = af.formalize(
            problem_id="p", class_label="x", problem="abstract", answer="UNNORMALIZED::?"
        )
        assert arts == []
