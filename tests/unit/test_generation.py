"""Tests for covcal.generation (backend ABC, extract, cache, prompts)."""

from __future__ import annotations

from pathlib import Path

from covcal.generation import (
    GenerationRequest,
    GenerationResult,
    LLMBackend,
    SamplingCache,
    candidate_generation_prompt,
    extract_final_answer,
    formalization_prompt,
    repair_prompt,
)


class FakeBackend(LLMBackend):
    """Deterministic stub that echoes a per-(prompt, k) seed into the sample text."""

    @property
    def backend_id(self) -> str:
        return "fake:v0"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        samples = [
            f"prompt={request.prompt!r}|seed={request.seed * 10000 + k}"
            for k in range(request.n_samples)
        ]
        return GenerationResult(samples=samples, backend_id=self.backend_id)


class TestExtract:
    def test_boxed_wins(self):
        out = extract_final_answer(r"derivation... \boxed{42}")
        assert out.text == "42"
        assert out.source == "boxed"

    def test_answer_tag_fallback(self):
        out = extract_final_answer("Some derivation.\nFinal answer: 17\n")
        assert out.text == "17"
        assert out.source == "answer_tag"

    def test_raw_when_empty(self):
        out = extract_final_answer("")
        assert out.source == "raw"

    def test_last_box_wins(self):
        out = extract_final_answer(r"\boxed{1} and then \boxed{2}")
        assert out.text == "2"


class TestCacheKey:
    def test_same_inputs_same_key(self):
        r1 = GenerationRequest("p", n_samples=2, seed=0)
        r2 = GenerationRequest("p", n_samples=2, seed=0)
        assert r1.cache_key("b1") == r2.cache_key("b1")

    def test_different_backend_different_key(self):
        r = GenerationRequest("p")
        assert r.cache_key("b1") != r.cache_key("b2")

    def test_different_seed_different_key(self):
        a = GenerationRequest("p", seed=0).cache_key("b")
        b = GenerationRequest("p", seed=1).cache_key("b")
        assert a != b


class TestSamplingCache:
    def test_roundtrip(self, tmp_path: Path):
        cache = SamplingCache(tmp_path / "c.jsonl")
        req = GenerationRequest("hello", n_samples=1, seed=0)
        backend = FakeBackend()
        assert cache.get(req, backend.backend_id) is None
        result = backend.generate(req)
        cache.put(req, result)
        again = cache.get(req, backend.backend_id)
        assert again is not None
        assert again.samples == result.samples
        # Re-instantiating reads back from disk.
        cache2 = SamplingCache(tmp_path / "c.jsonl")
        assert cache2.get(req, backend.backend_id) is not None
        assert len(cache2) == 1

    def test_distinct_keys_dont_collide(self, tmp_path: Path):
        cache = SamplingCache(tmp_path / "c.jsonl")
        backend = FakeBackend()
        for seed in range(3):
            req = GenerationRequest("hello", n_samples=1, seed=seed)
            cache.put(req, backend.generate(req))
        assert len(cache) == 3


class TestFakeBackend:
    def test_deterministic_for_fixed_seed(self):
        b = FakeBackend()
        r = GenerationRequest("p", n_samples=3, seed=42)
        a = b.generate(r).samples
        c = b.generate(r).samples
        assert a == c

    def test_n_samples_count(self):
        b = FakeBackend()
        r = GenerationRequest("p", n_samples=5, seed=0)
        assert len(b.generate(r).samples) == 5


class TestPrompts:
    def test_candidate_prompt_includes_boxed(self):
        p = candidate_generation_prompt("compute 1+1")
        assert "\\boxed" in p
        assert "compute 1+1" in p

    def test_formalization_prompt_includes_both(self):
        p = formalization_prompt("compute", "42")
        assert "Lean 4" in p and "42" in p and "compute" in p

    def test_repair_prompt_includes_error(self):
        p = repair_prompt("theorem t : ...", "type mismatch")
        assert "type mismatch" in p
