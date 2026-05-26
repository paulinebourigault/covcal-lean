"""Autoformalizer: NL (problem, answer) → Lean theorem statement(s).

Routes:

1) **template tier**: tries the deterministic templates in
   :mod:`covcal.lean.templates`. If one applies, no LLM call is made.
2) **LLM tier**: if no template applied, calls a `LLMBackend`
   with the prompt in :mod:`covcal.generation.prompts`. Generates `artifacts_per_class`
   theorems per (problem, answer) pair.
3) **repair pass**: if the first elaboration fails (status ILLTYPED), the
   error log is fed back to the LLM with the repair prompt for a single retry.

The FANS-style tier (2) is optional.

The autoformalizer is stateless: each call produces artifacts but does not verify them.
Verification is the caller's responsibility (it goes to `LeanBackend.check`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from ..generation.backend import GenerationRequest, LLMBackend
from ..generation.prompts import formalization_prompt, repair_prompt
from .backend import LeanBackend, LeanOutcome, LeanTask
from .templates import emit_template_task

logger = logging.getLogger(__name__)

ArtifactSource = Literal["template", "autoformalizer", "autoformalizer_repair"]


@dataclass(slots=True)
class FormalizedArtifact:
    """One Lean task with provenance, ready for verification."""

    task: LeanTask
    source: ArtifactSource
    template_kind: str | None = None  # set when source == "template"
    raw_llm_output: str | None = None  # set when source involves the LLM


@dataclass(slots=True)
class AutoformalizerConfig:
    artifacts_per_class: int = 2
    use_templates: bool = True
    use_llm: bool = True
    repair_on_illtyped: bool = True
    llm_temperature: float = 0.2
    llm_repair_temperature: float = 0.7
    llm_max_new_tokens: int = 2048
    llm_seed: int = 0


_CODE_BLOCK_RE = re.compile(r"```(?:lean4?|lean)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_FENCE_RE = re.compile(r"```(?:lean4?|lean)?\s*\n?", re.IGNORECASE)
_LEAN_START_RE = re.compile(
    r"(?m)^\s*(import|open|theorem|lemma|example|variable|namespace|section|noncomputable|/--|/-)"
)
_THEOREM_HEAD_RE = re.compile(r"theorem\s+\w+", re.IGNORECASE)
_DECL_HEAD_RE = re.compile(r"\b(?:theorem|lemma)\s+\w+", re.IGNORECASE)


def _extract_lean_code(text: str) -> str:
    """Pull Lean code out of a possibly-fenced LLM response.

    Robust to two failure modes seen in practice: (a) the closing ``` fence is
    stripped by the generation ``stop`` sequence (``"```\\n\\n"``), so a *complete*
    fenced block never matches and the old code returned the whole prose-prefixed
    text; (b) the model prefixes prose before the code block. We therefore fall
    back to the text after an opening fence, then strip any leading prose up to
    the first Lean keyword.
    """
    text = text or ""
    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        code = blocks[0].strip()
    else:
        m = _OPEN_FENCE_RE.search(text)
        code = (text[m.end():] if m else text)
        code = code.split("```")[0].strip()
    km = _LEAN_START_RE.search(code)
    if km:
        code = code[km.start():].strip()
    return code.strip()


def _statement_signature(code: str) -> str | None:
    """Return a *bare* ``theorem name : prop`` signature (no ``:= ...``, no preamble).

    The Lean runner elaborates ``<statement> := by sorry`` inside a namespace
    against an environment that already imports Mathlib, so the signature must be
    a single declaration. Any leading ``import``/``open``/``variable`` lines and
    the proof body are dropped; ``lemma`` is normalized to ``theorem`` so the
    runner's name-salting (which matches ``theorem``) applies. Returns ``None`` if
    no theorem/lemma head is present.
    """
    m = _DECL_HEAD_RE.search(code or "")
    if m is None:
        return None
    sig = code[m.start():]
    sig = re.sub(r"^\s*lemma\b", "theorem", sig, count=1, flags=re.IGNORECASE)
    sig = re.split(r":=\s*by\b", sig, maxsplit=1)[0]
    sig = re.split(r":=", sig, maxsplit=1)[0]
    return sig.rstrip()


@dataclass(slots=True)
class _Generation:
    code: str
    signature: str | None


def _generate_artifacts(
    llm: LLMBackend,
    *,
    problem: str,
    answer: str,
    n: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> list[_Generation]:
    """Call the LLM `n` times (single GenerationRequest with n_samples=n)."""
    req = GenerationRequest(
        prompt=formalization_prompt(problem, answer),
        n_samples=n,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seed=seed,
        stop=("```\n\n",),
    )
    res = llm.generate(req)
    out: list[_Generation] = []
    for raw in res.samples:
        code = _extract_lean_code(raw)
        out.append(_Generation(code=code, signature=_statement_signature(code)))
    return out


@dataclass(slots=True)
class Autoformalizer:
    """Orchestrates the three tiers (template / LLM / repair).

    The constructor takes a `LeanBackend` so the repair tier can check elaboration of LLM
    artifacts as it generates them. The `LLMBackend` may be `None`, in which case only the
    template tier fires.
    """

    config: AutoformalizerConfig
    lean: LeanBackend
    llm: LLMBackend | None = None
    _theorem_counter: dict[str, int] = field(default_factory=dict)

    def _name(self, problem_id: str, class_label: str) -> str:
        key = f"{problem_id}::{class_label}"
        self._theorem_counter[key] = self._theorem_counter.get(key, 0) + 1
        safe = re.sub(r"[^A-Za-z0-9]+", "_", key) or "x"
        return f"af_{safe}_{self._theorem_counter[key]}"

    def formalize(
        self,
        *,
        problem_id: str,
        class_label: str,
        problem: str,
        answer: str,
    ) -> list[FormalizedArtifact]:
        """Produce up to `artifacts_per_class` Lean tasks for one (problem, answer)."""
        artifacts: list[FormalizedArtifact] = []

        if self.config.use_templates:
            for idx in range(self.config.artifacts_per_class):
                m = emit_template_task(
                    problem_id=problem_id,
                    class_label=class_label,
                    problem=problem,
                    answer=answer,
                    artifact_idx=idx,
                )
                if m.task is None:
                    break  # template doesn't apply for this (problem, answer)
                artifacts.append(
                    FormalizedArtifact(task=m.task, source="template", template_kind=m.kind)
                )

        if not self.config.use_llm or self.llm is None:
            return artifacts

        # Top-up with LLM-generated artifacts until we hit the per-class quota.
        n_remaining = self.config.artifacts_per_class - len(artifacts)
        if n_remaining <= 0:
            return artifacts

        gens = _generate_artifacts(
            self.llm,
            problem=problem,
            answer=answer,
            n=n_remaining,
            temperature=self.config.llm_temperature,
            max_new_tokens=self.config.llm_max_new_tokens,
            seed=self.config.llm_seed,
        )

        for gen in gens:
            if gen.signature is None:
                # Unparseable LLM output ⇒ surface as UNFORMALIZED via no task.
                continue
            name = self._name(problem_id, class_label)
            tactics_portfolio = (
                "norm_num", "ring_nf; norm_num", "decide", "omega",
                "linarith", "nlinarith", "simp_all; norm_num", "aesop",
            )
            # Replace the theorem name in the signature with our salted one to avoid clashes.
            signature_renamed = _THEOREM_HEAD_RE.sub(f"theorem {name}", gen.signature, count=1)
            task = LeanTask(
                name=name, statement=signature_renamed, tactics=tactics_portfolio
            )
            artifacts.append(
                FormalizedArtifact(
                    task=task, source="autoformalizer", raw_llm_output=gen.code
                )
            )

        if self.config.repair_on_illtyped:
            artifacts = self._repair_pass(
                artifacts=artifacts,
                problem=problem,
                answer=answer,
                class_label=class_label,
                problem_id=problem_id,
            )
        return artifacts

    def formalize_many(
        self,
        items: list[tuple[str, str, str, str]],
    ) -> dict[str, list[FormalizedArtifact]]:
        """Formalize multiple ``(problem_id, class_label, problem, answer)`` items at once.

        Tier 1 (templates) runs per item — cheap, no LLM. Tier 2 (initial autoformalization)
        and Tier 3 (repair) each issue a single :meth:`LLMBackend.generate_batch` call
        spanning every item that needs an LLM, so the vLLM backend can fuse them into one
        GPU pass. Lean elaboration for the repair pass is likewise batched into one
        :meth:`LeanBackend.check` invocation.

        The return dict preserves the input order of class labels. Behavior for a single
        item matches :meth:`formalize` exactly (same template attempts, same seed scheme,
        same theorem naming via :meth:`_name`).

        Caller invariant: ``class_label`` values within ``items`` are unique (the pipeline
        guarantees this; the autoformalizer's theorem-name salting also relies on it).
        """
        by_class: dict[str, list[FormalizedArtifact]] = {}

        # Tier 1: templates per item.
        for problem_id, class_label, problem, answer in items:
            arts: list[FormalizedArtifact] = []
            if self.config.use_templates:
                for idx in range(self.config.artifacts_per_class):
                    m = emit_template_task(
                        problem_id=problem_id,
                        class_label=class_label,
                        problem=problem,
                        answer=answer,
                        artifact_idx=idx,
                    )
                    if m.task is None:
                        break
                    arts.append(
                        FormalizedArtifact(task=m.task, source="template", template_kind=m.kind)
                    )
            by_class[class_label] = arts

        if not self.config.use_llm or self.llm is None:
            return by_class

        # Tier 2: collect LLM requests for items that still need artifacts; issue as one batch.
        gen_requests: list[GenerationRequest] = []
        gen_meta: list[tuple[str, str]] = []  # (class_label, problem_id) — for naming
        for problem_id, class_label, problem, answer in items:
            n_remaining = self.config.artifacts_per_class - len(by_class[class_label])
            if n_remaining <= 0:
                continue
            gen_requests.append(
                GenerationRequest(
                    prompt=formalization_prompt(problem, answer),
                    n_samples=n_remaining,
                    temperature=self.config.llm_temperature,
                    max_new_tokens=self.config.llm_max_new_tokens,
                    seed=self.config.llm_seed,
                    stop=("```\n\n",),
                )
            )
            gen_meta.append((class_label, problem_id))

        if gen_requests:
            results = self.llm.generate_batch(gen_requests)
            tactics_portfolio = (
                "norm_num", "ring_nf; norm_num", "decide", "omega",
                "linarith", "nlinarith", "simp_all; norm_num", "aesop",
            )
            for (class_label, problem_id), res in zip(gen_meta, results, strict=True):
                for raw in res.samples:
                    code = _extract_lean_code(raw)
                    sig = _statement_signature(code)
                    if sig is None:
                        continue
                    name = self._name(problem_id, class_label)
                    sig_renamed = _THEOREM_HEAD_RE.sub(f"theorem {name}", sig, count=1)
                    by_class[class_label].append(
                        FormalizedArtifact(
                            task=LeanTask(name=name, statement=sig_renamed, tactics=tactics_portfolio),
                            source="autoformalizer",
                            raw_llm_output=code,
                        )
                    )

        # Tier 3: batched repair across all classes.
        if self.config.repair_on_illtyped:
            problem_by_label = {cls: (pid, prob, ans) for pid, cls, prob, ans in items}
            by_class = self._repair_pass_batched(by_class, problem_by_label)

        return by_class

    def _repair_pass_batched(
        self,
        by_class: dict[str, list[FormalizedArtifact]],
        problem_by_label: dict[str, tuple[str, str, str]],
    ) -> dict[str, list[FormalizedArtifact]]:
        """Cross-class repair pass: one Lean check, one LLM batch."""
        if self.llm is None:
            return by_class
        flat: list[tuple[str, FormalizedArtifact]] = []
        for label, arts in by_class.items():
            for a in arts:
                if a.source == "autoformalizer":
                    flat.append((label, a))
        if not flat:
            return by_class
        try:
            outcomes: list[LeanOutcome] = self.lean.check([a.task for _, a in flat])
        except Exception as e:  # pragma: no cover -- a Lean backend failure must not kill the pipeline
            logger.warning(
                "batched repair Lean check failed (%s); skipping repair for this problem", e
            )
            return by_class
        out_by_name = {o.name: o for o in outcomes}

        repair_requests: list[GenerationRequest] = []
        repair_meta: list[tuple[str, FormalizedArtifact]] = []
        for label, art in flat:
            outcome = out_by_name.get(art.task.name)
            if outcome is None or outcome.status.value != "illtyped":
                continue
            repair_requests.append(
                GenerationRequest(
                    prompt=repair_prompt(art.raw_llm_output or art.task.statement, outcome.log),
                    n_samples=1,
                    temperature=self.config.llm_repair_temperature,
                    max_new_tokens=self.config.llm_max_new_tokens,
                    seed=self.config.llm_seed + 7919,
                    stop=("```\n\n",),
                )
            )
            repair_meta.append((label, art))

        if not repair_requests:
            return by_class

        try:
            results = self.llm.generate_batch(repair_requests)
        except Exception as e:  # pragma: no cover — LLM failure shouldn't kill the pipeline
            logger.warning("batched repair LLM call failed: %s", e)
            return by_class

        for (label, art), res in zip(repair_meta, results, strict=True):
            if not res.samples:
                continue
            code = _extract_lean_code(res.samples[0])
            sig = _statement_signature(code)
            if sig is None:
                continue
            problem_id, _, _ = problem_by_label[label]
            name = self._name(problem_id, label) + "_repair"
            sig_renamed = _THEOREM_HEAD_RE.sub(f"theorem {name}", sig, count=1)
            by_class[label].append(
                FormalizedArtifact(
                    task=LeanTask(name=name, statement=sig_renamed, tactics=art.task.tactics),
                    source="autoformalizer_repair",
                    raw_llm_output=code,
                )
            )
        return by_class

    def _repair_pass(
        self,
        *,
        artifacts: list[FormalizedArtifact],
        problem: str,
        answer: str,
        class_label: str,
        problem_id: str,
    ) -> list[FormalizedArtifact]:
        if self.llm is None:
            return artifacts
        # Check artifacts to detect ILLTYPED ones. We only do this for autoformalizer
        # outputs; template tasks shouldn't be repaired (they are deterministic).
        llm_artifacts = [a for a in artifacts if a.source == "autoformalizer"]
        if not llm_artifacts:
            return artifacts
        try:
            outcomes: list[LeanOutcome] = self.lean.check([a.task for a in llm_artifacts])
        except Exception as e:  # pragma: no cover -- a Lean backend failure must not kill the pipeline
            logger.warning(
                "repair Lean check failed (%s); skipping repair for this problem", e
            )
            return artifacts
        out_by_name = {o.name: o for o in outcomes}

        repaired: list[FormalizedArtifact] = []
        for art in artifacts:
            if art.source != "autoformalizer":
                continue
            outcome = out_by_name.get(art.task.name)
            if outcome is None or outcome.status.value != "illtyped":
                continue
            # One repair attempt per ill-typed artifact.
            try:
                req = GenerationRequest(
                    prompt=repair_prompt(art.raw_llm_output or art.task.statement, outcome.log),
                    n_samples=1,
                    temperature=self.config.llm_repair_temperature,
                    max_new_tokens=self.config.llm_max_new_tokens,
                    seed=self.config.llm_seed + 7919,  # arbitrary distinct seed
                    stop=("```\n\n",),
                )
                res = self.llm.generate(req)
            except Exception as e:  # pragma: no cover - LLM failure shouldn't kill the pipeline
                logger.warning("repair LLM call failed: %s", e)
                continue
            code = _extract_lean_code(res.samples[0])
            sig = _statement_signature(code)
            if sig is None:
                continue
            name = self._name(problem_id, class_label) + "_repair"
            sig_renamed = _THEOREM_HEAD_RE.sub(f"theorem {name}", sig, count=1)
            tactics_portfolio = art.task.tactics
            repaired.append(
                FormalizedArtifact(
                    task=LeanTask(name=name, statement=sig_renamed, tactics=tactics_portfolio),
                    source="autoformalizer_repair",
                    raw_llm_output=code,
                )
            )
        return artifacts + repaired
