"""Abstract LLM backend interface.

The interface is deliberately narrow so the rest of the pipeline can run unchanged
against:

* `LlamaCppBackend`: CPU / CPU+GPU GGUF via llama-cpp-python (default on the 96-core box);
* a future `VLLMBackend` for the GPU server;
* a `FakeBackend` in tests.

All backends are deterministic given a fixed seed and sampling parameters. The pipeline
relies on this for reproducibility and for the disk cache to be a sound replacement for
re-running inference.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A single prompt + sampling spec. Multiple samples per request are returned as a list.

    A request is fully determined by `(prompt, n_samples, temperature, top_p, max_new_tokens,
    seed, stop)` plus the backend identity. The cache key is a SHA-256 of the canonical JSON.
    """

    prompt: str
    n_samples: int = 1
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 2048
    seed: int = 0
    stop: tuple[str, ...] = ()

    def cache_key(self, backend_id: str) -> str:
        payload = {
            "backend_id": backend_id,
            **{k: v for k, v in asdict(self).items()},
            "stop": list(self.stop),
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class GenerationResult:
    """N completions for one request, plus backend metadata."""

    samples: list[str]
    backend_id: str
    elapsed_seconds: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)


class LLMBackend(ABC):
    """Abstract LLM backend. Implementations must be deterministic for fixed seeds."""

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """A stable identifier (e.g., 'llama_cpp:qwen2.5-math-7b-q4_k_m@<sha>')."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return `request.n_samples` completions for the given prompt."""

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Default: serial loop. Backends can override for true batching."""
        return [self.generate(r) for r in requests]

    def close(self) -> None:  # pragma: no cover - default is no-op
        """Release model resources. Optional."""
