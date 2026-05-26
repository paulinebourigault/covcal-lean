"""LLM generation: candidate sampling and autoformalizer prompting.

Public entry points:

* :class:`covcal.generation.backend.LLMBackend` — ABC every backend implements.
* :class:`covcal.generation.llamacpp.LlamaCppBackend` — local GGUF inference.
* :func:`covcal.generation.extract.extract_final_answer` — `\\boxed{...}` extractor.
* :class:`covcal.generation.cache.SamplingCache` — disk-backed JSONL cache.
* prompt templates in :mod:`covcal.generation.prompts`.
"""

from .backend import GenerationRequest, GenerationResult, LLMBackend
from .cache import SamplingCache
from .extract import extract_final_answer
from .prompts import (
    candidate_generation_prompt,
    formalization_prompt,
    repair_prompt,
)

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "LLMBackend",
    "SamplingCache",
    "candidate_generation_prompt",
    "extract_final_answer",
    "formalization_prompt",
    "repair_prompt",
]
