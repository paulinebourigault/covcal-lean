"""On-disk JSONL cache for LLM generations.

Keyed by `GenerationRequest.cache_key(backend_id)`. One file per cache instance, append-only.
Reads are loaded into memory once per process for fast lookup. This is enough for the paper's
~200-problem * K=16 sampling load (a few thousand requests at most).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path

from .backend import GenerationRequest, GenerationResult

logger = logging.getLogger(__name__)


class SamplingCache:
    """Append-only JSONL cache: each line is ``{"key": ..., "result": {...}}``.

    Use one cache file per (project, model, sampling-spec-family). The key already includes
    the backend identity, sampling params, prompt, and seed, so distinct entries do not
    collide; the file-per-family split is purely for locality.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem: dict[str, GenerationResult] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        n = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._mem[obj["key"]] = GenerationResult(
                    samples=obj["result"]["samples"],
                    backend_id=obj["result"]["backend_id"],
                    elapsed_seconds=obj["result"].get("elapsed_seconds", 0.0),
                    extra=obj["result"].get("extra", {}),
                )
                n += 1
        logger.info("loaded %d cached generations from %s", n, self.path)

    def get(self, request: GenerationRequest, backend_id: str) -> GenerationResult | None:
        return self._mem.get(request.cache_key(backend_id))

    def put(self, request: GenerationRequest, result: GenerationResult) -> None:
        key = request.cache_key(result.backend_id)
        with self._lock:
            self._mem[key] = result
            with self.path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "key": key,
                            "request": {**asdict(request), "stop": list(request.stop)},
                            "result": {
                                "samples": result.samples,
                                "backend_id": result.backend_id,
                                "elapsed_seconds": result.elapsed_seconds,
                                "extra": result.extra,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def __contains__(self, key: str) -> bool:
        return key in self._mem

    def __len__(self) -> int:
        return len(self._mem)
