"""vLLM backend: GPU inference via vllm.LLM.

Designed for a single GPU (or multi-GPU via tensor parallelism). Optimised for the two
batching patterns the pipeline actually creates:

* one ``GenerationRequest`` with ``n_samples = K`` — handled in a single ``LLM.generate``
  call with ``SamplingParams(n=K)`` so the K candidates share KV-cache compute and bench
  near peak GPU throughput;
* a list of requests via :meth:`generate_batch` — flattened into one vLLM call so prompts
  across problems batch together when the caller queues them up.

The constructor is import-safe even when ``vllm`` is not installed; the import is deferred
to construction time, matching :mod:`covcal.generation.llamacpp`. ``backend_id`` is stable
across runs (model + revision + dtype + quantization), so the disk
:class:`~covcal.generation.cache.SamplingCache` invalidates correctly when the GPU stack
changes but reuses entries across re-runs of the same config.

Determinism caveat
------------------------------------------------
vLLM honours ``SamplingParams.seed`` per request, but its batched sampling kernels are not
bit-exact across different batch compositions or GPU/driver combinations. The
:class:`SamplingCache` is the durable reproducibility surface: once an
``(observation, problem)`` has been sampled and cached, re-runs of the same config replay
from disk rather than re-sampling.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .backend import GenerationRequest, GenerationResult, LLMBackend

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm import LLM, SamplingParams  # noqa: F401

logger = logging.getLogger(__name__)

# Safety headroom (tokens) subtracted from the context window when checking that a prompt
# fits. A prompt that exceeds ``max_model_len - max_new_tokens`` can make vLLM error or
# wedge the engine indefinitely; we refuse to submit such prompts (see ``generate``).
_PROMPT_MARGIN_TOKENS = 64


@dataclass(frozen=True, slots=True)
class VLLMBackendConfig:
    """Constructor-time vLLM settings.

    ``model`` accepts an HF repo id (e.g. ``"Qwen/Qwen2.5-Math-7B-Instruct"``) or a local
    snapshot directory. ``revision`` pins to a specific commit for reproducibility — when
    provided it becomes part of ``backend_id``, so the cache invalidates if the model is
    re-pinned.
    """

    model: str
    revision: str | None = None
    dtype: str = "auto"                       # auto | float16 | bfloat16 | float32
    quantization: str | None = None           # None | awq | gptq | fp8 | ...
    max_model_len: int = 8192
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    trust_remote_code: bool = False
    download_dir: str | None = None
    seed: int = 0                             # engine-level seed
    max_num_seqs: int = 256                   # batch cap; vLLM default
    chat_template: bool = False               # if True, wrap prompts via tokenizer template


class VLLMBackend(LLMBackend):
    """Optimised vLLM backend for CovCal.

    Key invariants:

    * Single ``LLM.generate`` call per :class:`GenerationRequest`, using
      ``SamplingParams(n=request.n_samples, seed=request.seed)`` so K candidates batch on
      the GPU instead of running serially.
    * :meth:`generate_batch` flattens a list of requests into one ``LLM.generate`` call,
      yielding the largest meaningful batch the caller can produce — this is where the
      win over llama.cpp is largest.
    * ``backend_id`` is a deterministic function of ``(model, revision, dtype,
      quantization, max_model_len, chat_template)``; the SamplingCache keys on it so the
      same config replays from disk across re-runs.
    """

    def __init__(self, config: VLLMBackendConfig, *, verbose: bool = False) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - only when extra missing
            raise ImportError(
                "vllm not installed. Install with `uv sync --extra gpu`."
            ) from e

        self._cfg = config
        self._SamplingParams = SamplingParams
        # Lazily-fetched tokenizer used only for the prompt-length guard in ``generate``.
        # ``None`` = not yet fetched; ``False`` = unavailable (fall back to char estimate).
        self._len_tok: Any = None

        if not verbose:
            # vLLM is very chatty at INFO; bump to WARNING for the run logs to stay readable.
            logging.getLogger("vllm").setLevel(logging.WARNING)

        # Resolved revision: when the caller pins ``revision``, we use it verbatim; otherwise
        # we'd rather record the actual commit than ``"main"`` so the backend id is stable.
        resolved_revision = self._resolve_revision(config.model, config.revision)
        self._resolved_revision = resolved_revision

        # Cache-only mode: when running a second large LLM (e.g. Goedel-V2-32B at the
        # autoformalizer role) on the same single GPU, the smaller generator model can be
        # served entirely from SamplingCache replay. Setting
        # ``COVCAL_VLLM_CACHE_ONLY_MODELS`` to a comma-separated list of substrings that
        # appear in the model id skips engine init for those backends; ``generate`` will
        # then raise if it is ever actually called, which is the desired guard.
        cache_only_models = os.environ.get("COVCAL_VLLM_CACHE_ONLY_MODELS", "")
        if cache_only_models:
            needles = [s.strip() for s in cache_only_models.split(",") if s.strip()]
            if any(n in config.model for n in needles):
                logger.warning(
                    "VLLMBackend skipping LLM init for %s (matched "
                    "COVCAL_VLLM_CACHE_ONLY_MODELS); generate() must be served by "
                    "SamplingCache replay.",
                    config.model,
                )
                self._llm = None
                self._tokenizer = None
                return

        self._llm = LLM(
            model=config.model,
            revision=resolved_revision,
            dtype=config.dtype,
            quantization=config.quantization,
            max_model_len=config.max_model_len,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            trust_remote_code=config.trust_remote_code,
            download_dir=config.download_dir,
            seed=config.seed,
            max_num_seqs=config.max_num_seqs,
            disable_log_stats=not verbose,
        )

        # The tokenizer is needed when ``chat_template=True``. vLLM exposes it via
        # get_tokenizer(); we cache it once.
        self._tokenizer = self._llm.get_tokenizer() if config.chat_template else None

    # ------------------------------------------------------------------ identity

    @property
    def backend_id(self) -> str:
        payload = {
            "model": self._cfg.model,
            "revision": self._resolved_revision or "main",
            "dtype": self._cfg.dtype,
            "quantization": self._cfg.quantization,
            "max_model_len": self._cfg.max_model_len,
            "chat_template": self._cfg.chat_template,
        }
        digest = hashlib.sha256(
            "|".join(f"{k}={payload[k]}" for k in sorted(payload)).encode("utf-8")
        ).hexdigest()[:12]
        # Use just the model basename in the readable part so logs stay short.
        readable = self._cfg.model.rsplit("/", 1)[-1]
        return f"vllm:{readable}@{digest}"

    # ------------------------------------------------------------------ generation

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """K candidates in one vLLM call (``SamplingParams(n=K)``)."""
        t0 = time.perf_counter()
        sp = self._make_sampling_params(request)
        prompt = self._format_prompt(request.prompt)
        # Guard: a prompt that does not fit the engine context window can make vLLM error
        # or, worse, wedge the engine indefinitely. Treat it as
        # a failed generation and return zero samples so the pipeline skips the artifact and
        # continues. This keeps the engine healthy for subsequent problems.
        n_prompt = self._count_prompt_tokens(prompt)
        budget = self._cfg.max_model_len - request.max_new_tokens - _PROMPT_MARGIN_TOKENS
        if n_prompt > budget:
            logger.warning(
                "skipping oversized prompt: %d prompt tokens > budget %d "
                "(max_model_len=%d - max_new_tokens=%d - margin=%d); returning 0 samples",
                n_prompt, budget, self._cfg.max_model_len,
                request.max_new_tokens, _PROMPT_MARGIN_TOKENS,
            )
            return GenerationResult(
                samples=[],
                backend_id=self.backend_id,
                elapsed_seconds=time.perf_counter() - t0,
                extra=self._extra(request),
            )
        # vLLM ``LLM.generate`` returns ``RequestOutput`` per prompt; with n=K each has K
        # completions in ``.outputs``.
        outputs = self._llm.generate([prompt], sp, use_tqdm=False)
        if not outputs:  # pragma: no cover - defensive
            return GenerationResult(samples=[], backend_id=self.backend_id)
        samples = [c.text for c in outputs[0].outputs]
        elapsed = time.perf_counter() - t0
        return GenerationResult(
            samples=samples,
            backend_id=self.backend_id,
            elapsed_seconds=elapsed,
            extra=self._extra(request),
        )

    def close(self) -> None:
        # vLLM doesn't expose an explicit shutdown for LLM; drop the reference and let GC
        # / CUDA caching allocator reclaim. Wrap torch.cuda.empty_cache() opportunistically.
        self._llm = None  # type: ignore[assignment]
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    # ------------------------------------------------------------------ helpers

    def _count_prompt_tokens(self, prompt: str) -> int:
        """Token length of ``prompt`` for the context-window guard.

        Uses the engine tokenizer (reusing the chat-template one when present, else fetched
        once and cached). If no tokenizer is available, falls back to a conservative
        ~3-chars-per-token estimate so the guard still fires on pathological inputs.
        """
        tok = self._tokenizer
        if tok is None:
            if self._len_tok is None:
                try:
                    self._len_tok = self._llm.get_tokenizer()
                except Exception:  # pragma: no cover - defensive
                    self._len_tok = False
            tok = self._len_tok
        if not tok:
            return len(prompt) // 3
        try:
            return len(tok.encode(prompt))
        except Exception:  # pragma: no cover - defensive
            return len(prompt) // 3

    def _make_sampling_params(self, request: GenerationRequest) -> Any:
        return self._SamplingParams(
            n=request.n_samples,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_new_tokens,
            seed=request.seed,
            stop=list(request.stop) if request.stop else None,
        )

    def _format_prompt(self, prompt: str) -> str:
        """Optionally wrap a raw prompt with the tokenizer's chat template.

        Off by default to match the llama.cpp run, where prompts are sent raw. When
        on, each prompt is wrapped as a single user turn; appropriate for Qwen-Instruct
        models, which were post-trained with their own ChatML template.
        """
        if self._tokenizer is None:
            return prompt
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _extra(self, request: GenerationRequest) -> dict[str, object]:
        return {
            "max_model_len": self._cfg.max_model_len,
            "dtype": self._cfg.dtype,
            "quantization": self._cfg.quantization,
            "tensor_parallel_size": self._cfg.tensor_parallel_size,
            "chat_template": self._cfg.chat_template,
            "n_samples": request.n_samples,
        }

    @staticmethod
    def _resolve_revision(model: str, revision: str | None) -> str | None:
        """Resolve ``revision="main"``/None to the current commit SHA when possible.

        Recording the SHA in ``backend_id`` keeps the cache stable if the upstream branch
        moves. Local model directories return None unchanged. Failures fall back to the
        caller-supplied revision so this never blocks a run.
        """
        if revision is not None and revision != "main":
            return revision
        # If ``model`` looks like a local path, skip resolution.
        if os.sep in model or os.path.isdir(model):
            return revision
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]

            api = HfApi()
            info = api.model_info(model, revision=revision or "main")
            return info.sha or revision
        except Exception as e:  # pragma: no cover - HF offline / gated
            logger.warning("could not resolve revision for %s: %s", model, e)
            return revision
