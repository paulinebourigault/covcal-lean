"""llama.cpp / GGUF backend.

Loads a GGUF model via ``llama_cpp.Llama`` and runs sampling. Designed for a 96-core
CPU box: defaults use ``n_threads = os.cpu_count()`` and a moderate context window.

The constructor accepts either a local file path or a HF Hub repo + filename, which it
downloads via ``huggingface_hub`` the first time. The downloaded path becomes part of
``backend_id`` so cache keys are stable across runs.

This module is import-safe even when ``llama_cpp`` is not installed (the import is deferred
to construction time, so the rest of CovCal can be unit-tested without the LLM extra).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

from .backend import GenerationRequest, GenerationResult, LLMBackend

logger = logging.getLogger(__name__)


def _resolve_model_path(model_path: str | None, repo_id: str | None, filename: str | None) -> Path:
    if model_path is not None:
        p = Path(model_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"GGUF file not found: {p}")
        return p
    if repo_id is None or filename is None:
        raise ValueError("Either model_path or (repo_id, filename) must be provided.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - only triggered when extra missing
        raise ImportError(
            "huggingface_hub not installed. Install with `uv sync --extra llm`."
        ) from e
    local = hf_hub_download(repo_id=repo_id, filename=filename)
    return Path(local)


def _file_short_sha(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


class LlamaCppBackend(LLMBackend):
    """GGUF inference via llama-cpp-python.

    Construction is intentionally explicit: pass either ``model_path`` for an already-local
    file, or ``repo_id``+``filename`` for HF Hub download. Construction loads the model into
    memory; release with :meth:`close`.
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        repo_id: str | None = None,
        filename: str | None = None,
        n_ctx: int = 8192,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        chat_format: str | None = None,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - only triggered when extra missing
            raise ImportError(
                "llama-cpp-python not installed. Install with `uv sync --extra llm`."
            ) from e

        self._model_path = _resolve_model_path(model_path, repo_id, filename)
        self._file_sha = _file_short_sha(self._model_path)
        self._n_ctx = n_ctx
        self._n_threads = n_threads if n_threads is not None else (os.cpu_count() or 4)
        self._n_gpu_layers = n_gpu_layers
        self._chat_format = chat_format
        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=n_ctx,
            n_threads=self._n_threads,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
            verbose=verbose,
        )

    @property
    def backend_id(self) -> str:
        return f"llama_cpp:{self._model_path.name}@{self._file_sha}"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = time.perf_counter()
        samples: list[str] = []
        # llama.cpp doesn't natively batch K samples in a single call; we re-seed per
        # sample to keep determinism per (seed, sample_index).
        for k in range(request.n_samples):
            seed = request.seed * 10_000 + k
            out = self._llm(
                prompt=request.prompt,
                max_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=seed,
                stop=list(request.stop) if request.stop else None,
            )
            text = out["choices"][0]["text"]
            samples.append(text)
        elapsed = time.perf_counter() - t0
        return GenerationResult(
            samples=samples,
            backend_id=self.backend_id,
            elapsed_seconds=elapsed,
            extra={
                "n_threads": self._n_threads,
                "n_gpu_layers": self._n_gpu_layers,
                "n_ctx": self._n_ctx,
            },
        )

    def close(self) -> None:
        # llama_cpp.Llama doesn't expose an explicit close; rely on GC to free memory.
        self._llm = None  # type: ignore[assignment]
