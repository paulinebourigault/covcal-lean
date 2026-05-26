"""End-to-end orchestration: problem -> candidates -> classes -> formal artifacts -> outcomes
-> per-problem record.

The pipeline writes one JSONL row per problem so that selectors and calibration can be
re-run offline without touching the LLM or Lean again.

Stages
------

1. ``generate(problems, llm, sampler)`` — sample K candidate answers per problem and write
   them to ``runs/<name>/01_candidates.jsonl``.
2. ``formalize(candidates, autoformalizer)`` — emit Lean artifacts per top-K answer class
   and write them to ``runs/<name>/02_artifacts.jsonl``.
3. ``verify(artifacts, lean_backend)`` — run the Lean wrapper and write outcomes to
   ``runs/<name>/03_outcomes.jsonl``.
4. ``assemble(candidates, artifacts, outcomes)`` — produce :class:`FormalObservation`s and
   write them to ``runs/<name>/04_observations.jsonl``.

The CLI exposes each stage separately plus a ``run`` target that does all four in sequence.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .classes import aggregate_classes, attach_artifacts, top_k_by_weight
from .diagnostics import compute_diagnostics
from .generation import GenerationRequest, LLMBackend, SamplingCache, candidate_generation_prompt
from .generation.extract import extract_final_answer
from .lean import (
    Autoformalizer,
    FormalizedArtifact,
    LeanBackend,
    LeanTask,
)
from .normalization import normalize_answer
from .selectors import CovCal, CovCalPlusFallback, self_consistency
from .types import (
    ABSTAIN,
    ArtifactOutcome,
    Candidate,
    ClassRecord,
    FormalObservation,
    Status,
    Thresholds,
)

logger = logging.getLogger(__name__)

# Per-problem wall-clock backstop (seconds). LAST-RESORT only: the Lean backend's own
# per-line idle timeout recovers wedged-runner cases (mark TIMEOUT + respawn) long before
# this fires, so this exists for an unforeseen native-code hang that escapes that path.
# Headroom must cover the worst recoverable case: a poison problem can wedge both the
# repair and the verify Lean call, each costing ~idle_timeout detection + a ~700s Mathlib
# re-boot on respawn (~25 min total) plus the one-time boot on a fresh process. 2700s (45
# min) keeps this from tripping on anything recoverable. Set <=0 to disable.
_PROBLEM_TIMEOUT_SECONDS = float(os.environ.get("COVCAL_PROBLEM_TIMEOUT_SECONDS", "2700"))


def _kill_child_processes() -> None:
    """Best-effort kill of all descendant processes (e.g. orphaned vLLM engine
    subprocesses). Without this, a child that inherited the run's stdout pipe keeps it open
    after we exit, so the surrounding ``… | tee`` never sees EOF and the shell script wedges
    instead of moving on to the next run."""
    try:
        import psutil  # type: ignore[import-not-found]

        for child in psutil.Process().children(recursive=True):
            try:
                child.kill()
            except Exception:  # noqa: BLE001 - best effort
                pass
    except Exception:  # noqa: BLE001 - psutil missing or /proc unavailable
        pass


class _PerProblemWatchdog:
    """Re-armable last-resort guard. If a single problem makes no progress for
    ``timeout_s``, dump every thread's traceback (for diagnosis), kill child processes (so an
    orphaned engine can't strand the surrounding shell), then ``os._exit`` — bounded failure
    instead of a silent multi-hour hang."""

    def __init__(self, timeout_s: float) -> None:
        self._timeout = timeout_s
        self._deadline: float | None = None
        self._lock = threading.Lock()
        self._alive = True
        self._thread = threading.Thread(
            target=self._loop, name="covcal-watchdog", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def kick(self) -> None:
        with self._lock:
            self._deadline = time.monotonic() + self._timeout

    def stop(self) -> None:
        self._alive = False

    def _loop(self) -> None:
        while self._alive:
            time.sleep(2.0)
            with self._lock:
                dl = self._deadline
            if dl is not None and time.monotonic() > dl:
                self._fire()

    def _fire(self) -> None:
        sys.stderr.write(
            f"\n[watchdog] no progress for {self._timeout:.0f}s — dumping tracebacks, "
            "killing children, exiting\n"
        )
        sys.stderr.flush()
        try:
            faulthandler.dump_traceback(all_threads=True)
        except Exception:  # noqa: BLE001 - never let diagnostics block the exit
            pass
        _kill_child_processes()
        os._exit(1)


@dataclass(slots=True)
class Problem:
    """A single benchmark example."""

    problem_id: str
    problem_text: str
    reference_answer: str  # raw answer string from the dataset; normalised to a class label
    domain: str | None = None  # e.g. "algebra", "number_theory" — for Tab. 3
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reference_class_label(self) -> str:
        return normalize_answer(self.reference_answer)


# --- Stage 1: candidate generation ----------------------------------------------------


@dataclass(slots=True)
class CandidateRecord:
    problem_id: str
    raw_samples: list[str]
    extracted: list[dict[str, str]]  # [{"text": ..., "source": ...}]
    candidates: list[dict[str, Any]]  # serialized Candidate

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def generate_candidates(
    problems: Iterable[Problem],
    llm: LLMBackend,
    *,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    cache: SamplingCache | None = None,
) -> Iterator[CandidateRecord]:
    """Stream one CandidateRecord per problem."""
    for prob in problems:
        req = GenerationRequest(
            prompt=candidate_generation_prompt(prob.problem_text),
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        result = None
        if cache is not None:
            result = cache.get(req, llm.backend_id)
        if result is None:
            result = llm.generate(req)
            if cache is not None:
                cache.put(req, result)
        extracted = [extract_final_answer(s) for s in result.samples]
        candidates = [
            Candidate(answer_text=e.text, weight=1.0 / n_samples, sample_id=i)
            for i, e in enumerate(extracted)
        ]
        yield CandidateRecord(
            problem_id=prob.problem_id,
            raw_samples=result.samples,
            extracted=[{"text": e.text, "source": e.source} for e in extracted],
            candidates=[asdict(c) for c in candidates],
        )


# --- Stage 2: formalization ----------------------------------------------------------


@dataclass(slots=True)
class ArtifactsRecord:
    problem_id: str
    artifacts_by_class: dict[str, list[dict[str, Any]]]  # class_label -> list of artifact dicts

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def formalize_candidates(
    problem: Problem,
    candidates: list[Candidate],
    autoformalizer: Autoformalizer,
    *,
    formalize_top_k_classes: int,
) -> ArtifactsRecord:
    """Formalize the top-K answer classes via a single batched LLM call.

    Issues one :meth:`Autoformalizer.formalize_many` call for the whole top-K rather than
    looping :meth:`Autoformalizer.formalize` per class. On llama.cpp the per-call
    ``generate_batch`` falls back to the serial loop, so behavior matches the pre-batch
    code path. On vLLM, all top-K formalization prompts are dispatched in one
    ``LLM.generate`` call and run in one GPU pass.
    """
    classes = aggregate_classes(candidates)
    top = top_k_by_weight(classes, formalize_top_k_classes)
    items = [
        (problem.problem_id, cls.label, problem.problem_text, cls.label)
        for cls in top
    ]
    formalized = autoformalizer.formalize_many(items)
    by_class = {label: [_artifact_to_dict(a) for a in arts] for label, arts in formalized.items()}
    return ArtifactsRecord(problem_id=problem.problem_id, artifacts_by_class=by_class)


def _artifact_to_dict(a: FormalizedArtifact) -> dict[str, Any]:
    """Serialize one FormalizedArtifact — the structured log schema "Formalization fields".

    Stores the *full* generated Lean source (`raw_llm_output` when the LLM produced it,
    or the synthesized template statement otherwise) plus the repair-round index inferred
    from `source` ("template" -> 0, "autoformalizer" -> 1, "autoformalizer_repair" -> 2).
    """
    repair_round = {"template": 0, "autoformalizer": 1, "autoformalizer_repair": 2}.get(
        a.source, 0
    )
    return {
        "source": a.source,                          # the structured log schema "route"
        "template_kind": a.template_kind,
        "repair_round": repair_round,
        "imports": ["Mathlib"],                       # always Mathlib in the minimal run
        "lean_code": a.raw_llm_output if a.raw_llm_output is not None else a.task.statement,
        "task": {
            "name": a.task.name,
            "statement": a.task.statement,           # the structured log schema "theorem statement"
            "tactics": list(a.task.tactics),
            "max_heartbeats_per_tactic": a.task.max_heartbeats_per_tactic,
        },
    }


def _dict_to_artifact(d: dict[str, Any]) -> FormalizedArtifact:
    t = d["task"]
    return FormalizedArtifact(
        task=LeanTask(
            name=t["name"],
            statement=t["statement"],
            tactics=tuple(t["tactics"]),
            max_heartbeats_per_tactic=t.get("max_heartbeats_per_tactic"),
        ),
        source=d["source"],
        template_kind=d.get("template_kind"),
        raw_llm_output=d.get("lean_code"),
    )


# --- Stage 3: verification -----------------------------------------------------------


@dataclass(slots=True)
class OutcomesRecord:
    problem_id: str
    outcomes_by_class: dict[str, list[dict[str, Any]]]  # class_label -> list of outcome dicts

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def verify_artifacts(
    artifacts: ArtifactsRecord, lean: LeanBackend
) -> OutcomesRecord:
    # Flatten all tasks across classes, preserving back-pointers for re-grouping.
    flat: list[tuple[str, FormalizedArtifact]] = []
    for label, items in artifacts.artifacts_by_class.items():
        for d in items:
            flat.append((label, _dict_to_artifact(d)))
    if not flat:
        return OutcomesRecord(problem_id=artifacts.problem_id, outcomes_by_class={})
    tasks = [a.task for _, a in flat]
    outcomes = lean.check(tasks)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for (label, art), out in zip(flat, outcomes, strict=True):
        repair_round = {"template": 0, "autoformalizer": 1, "autoformalizer_repair": 2}.get(
            art.source, 0
        )
        timed_out = out.status is Status.TIMEOUT
        grouped.setdefault(label, []).append(
            {
                "source": art.source,                       # the structured log schema "route"
                "template_kind": art.template_kind,
                "repair_round": repair_round,
                "imports": ["Mathlib"],
                "lean_code": (
                    art.raw_llm_output if art.raw_llm_output is not None else art.task.statement
                ),
                "task_name": art.task.name,
                "statement": art.task.statement,            # the structured log schema "theorem statement"
                "tactics": list(art.task.tactics),
                "status": out.status.value,
                "tactic_used": out.tactic_used,
                "elapsed_seconds": out.elapsed_seconds,
                "timeout_flag": timed_out,                  # the structured log schema "timeout flag"
                "log_excerpt": out.log[:1000],              # the structured log schema "error message"
            }
        )
    return OutcomesRecord(problem_id=artifacts.problem_id, outcomes_by_class=grouped)


# --- Stage 4: assembly into FormalObservation ----------------------------------------


def assemble_observation(
    problem: Problem,
    candidates: list[Candidate],
    outcomes: OutcomesRecord,
    *,
    prover_budget_seconds: float,
    raw_samples: list[str] | None = None,
    extracted: list[dict[str, str]] | None = None,
    dataset_name: str | None = None,
    included: bool = True,
) -> FormalObservation:
    """Assemble the full the structured log schema verification-attempt log.

    `raw_samples` and `extracted` are stored in `metadata` so they round-trip through
    `observation_to_dict`/`_load_observations` without changing the public dataclass
    surface (which would force callers to handle the larger payload).
    """
    classes = aggregate_classes(candidates)
    artifacts_by_label: dict[str, list[ArtifactOutcome]] = {}
    raw_artifact_dicts: dict[str, list[dict[str, Any]]] = {}
    for label, items in outcomes.outcomes_by_class.items():
        outs: list[ArtifactOutcome] = []
        kept: list[dict[str, Any]] = []
        for item in items:
            outs.append(
                ArtifactOutcome(
                    status=Status(item["status"]),
                    tactic_used=item.get("tactic_used"),
                    elapsed_seconds=float(item.get("elapsed_seconds", 0.0)),
                    log=item.get("log_excerpt", ""),
                    source=item.get("source", "template"),
                )
            )
            kept.append(item)
        artifacts_by_label[label] = outs
        raw_artifact_dicts[label] = kept
    attach_artifacts(classes, artifacts_by_label)
    return FormalObservation(
        problem_id=problem.problem_id,
        classes=classes,
        candidates=candidates,
        prover_budget_seconds=prover_budget_seconds,
        metadata={
            # --- the structured log schema "Problem fields" ---
            "dataset": dataset_name,
            "domain": problem.domain,
            "problem_text": problem.problem_text,
            "reference_answer": problem.reference_answer,
            "reference_class": problem.reference_class_label,
            "included": included,
            # --- the structured log schema "Candidate fields" (raw samples kept for the audit) ---
            "raw_samples": raw_samples or [],
            "extracted_answers": extracted or [],
            # --- the structured log schema "Formalization fields" (richer per-artifact dicts) ---
            "artifacts_detail": raw_artifact_dicts,
        },
    )


# --- Convenience: write/read JSONL ---------------------------------------------------


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if hasattr(r, "to_jsonl"):
                f.write(r.to_jsonl() + "\n")
            elif hasattr(r, "to_dict"):
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# --- Pipeline runner -----------------------------------------------------------------


def _recover_done_ids(obs_path: Path) -> set[str]:
    """Collect completed ``problem_id``s from ``obs_path``; truncate any partial trailing line.

    A SIGKILL between the last flush and the next ``\\n`` would leave a partial JSON object at
    the tail of the file. We split on ``\\n``, parse complete lines, and truncate the file at
    the last well-formed newline so a subsequent append starts from a clean byte offset.
    Malformed lines *before* the tail abort recovery — the caller treats this as a
    non-resumable file and the caller should rerun with ``resume=False`` after inspection.
    """
    content = obs_path.read_bytes()
    if not content:
        return set()
    parts = content.split(b"\n")
    # ``split`` produces one trailing element after the final '\n'. If the file ends with
    # a newline this element is empty (complete file); otherwise it's the partial tail.
    complete_lines = parts[:-1]
    last_good_end = 0
    done: set[str] = set()
    for line in complete_lines:
        if not line:
            last_good_end += 1  # blank line, just the '\n'
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            logger.error(
                "malformed line at byte %d of %s; refusing to silently skip earlier records",
                last_good_end, obs_path,
            )
            raise
        pid = d.get("problem_id")
        if isinstance(pid, str):
            done.add(pid)
        last_good_end += len(line) + 1  # +1 for the '\n'
    if last_good_end < len(content):
        with obs_path.open("r+b") as f:
            f.truncate(last_good_end)
        logger.warning(
            "truncated %d partial trailing bytes in %s",
            len(content) - last_good_end, obs_path,
        )
    return done


def _rebuild_aux_for_done(obs_path: Path, aux_path: Path) -> None:
    """Rewrite ``aux_path`` from scratch by replaying ``class_aux_rows`` for each obs row.

    Calibration only consumes ``observations.jsonl``; ``class_aux.jsonl`` is for the paper's
    auxiliary table. We keep them consistent on resume to avoid orphan rows from a previous
    aborted run's writes. Cheap: each row is a JSON-load + deterministic recompute, and the
    file is small (~25 KB for 45 problems).
    """
    if not obs_path.exists():
        return
    aux_path.parent.mkdir(parents=True, exist_ok=True)
    with obs_path.open("r", encoding="utf-8") as obs_f, aux_path.open("w", encoding="utf-8") as aux_f:
        for line in obs_f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            obs = _observation_from_dict(d)
            for row in class_aux_rows(obs):
                aux_f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _observation_from_dict(d: dict[str, Any]) -> FormalObservation:
    """Inverse of ``observation_to_dict`` for the fields ``class_aux_rows`` reads.

    We don't need to reconstruct ``candidates`` faithfully (they aren't read by the aux
    rebuild path); we do need ``classes`` and ``metadata`` because the aux row reports the
    reference class label and per-class diagnostics.
    """
    classes = [
        ClassRecord(
            label=c["label"],
            weight=c["weight"],
            candidate_indices=c.get("candidate_indices", []),
            artifacts=[
                ArtifactOutcome(
                    status=Status(a["status"]),
                    tactic_used=a.get("tactic_used"),
                    elapsed_seconds=a.get("elapsed_seconds", 0.0),
                    log=a.get("log", ""),
                    source=a.get("source", "template"),
                )
                for a in c.get("artifacts", [])
            ],
        )
        for c in d["classes"]
    ]
    return FormalObservation(
        problem_id=d["problem_id"],
        classes=classes,
        candidates=[Candidate(**c) for c in d.get("candidates", [])],
        prover_budget_seconds=d.get("prover_budget_seconds", 0.0),
        metadata=d.get("metadata", {}),
    )


@dataclass(slots=True)
class PipelineRunConfig:
    run_dir: Path
    n_samples: int
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int
    formalize_top_k_classes: int
    prover_budget_seconds: float
    dataset_name: str | None = None
    # Optional acceptance thresholds; if provided, each observation is annotated with
    # the corresponding CovCal accept/reject decision so the the structured log schema "Selection fields"
    # can be filled without rerunning the selector offline.
    covcal_thresholds: Thresholds | None = None
    # When True (default), an existing ``observations.jsonl`` is opened in append mode and
    # already-completed problem_ids are skipped. Setting False truncates and re-runs the
    # whole pipeline, matching pre-resume behavior.
    resume: bool = True


@dataclass(slots=True)
class PipelineRun:
    config: PipelineRunConfig
    llm: LLMBackend
    autoformalizer: Autoformalizer
    lean: LeanBackend
    sampling_cache: SamplingCache | None = None

    def run_one(self, problem: Problem) -> FormalObservation:
        """Run all four stages for a single problem and return its observation."""
        # Stage 1: candidates (raw samples + extracted answers kept for the structured log schema audit)
        records = list(
            generate_candidates(
                [problem],
                self.llm,
                n_samples=self.config.n_samples,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
                seed=self.config.seed,
                cache=self.sampling_cache,
            )
        )
        cand_rec = records[0]
        candidates = [Candidate(**c) for c in cand_rec.candidates]
        # Stage 2: artifacts
        arts_rec = formalize_candidates(
            problem,
            candidates,
            self.autoformalizer,
            formalize_top_k_classes=self.config.formalize_top_k_classes,
        )
        # Stage 3: outcomes
        out_rec = verify_artifacts(arts_rec, self.lean)
        # Stage 4: assemble — includes raw samples, dataset tag, included flag for the structured log schema
        return assemble_observation(
            problem, candidates, out_rec,
            prover_budget_seconds=self.config.prover_budget_seconds,
            raw_samples=cand_rec.raw_samples,
            extracted=cand_rec.extracted,
            dataset_name=self.config.dataset_name,
            included=True,
        )

    def run_many(self, problems: Iterable[Problem]) -> Iterator[FormalObservation]:
        run_dir = self.config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        # Last-resort backstop, re-armed per problem (see _PerProblemWatchdog). The Lean
        # backend's per-line idle timeout handles wedged-runner cases on its own; this only
        # fires for an unforeseen hang that escapes every inner timeout.
        wd = (
            _PerProblemWatchdog(_PROBLEM_TIMEOUT_SECONDS)
            if _PROBLEM_TIMEOUT_SECONDS > 0
            else None
        )
        if wd is not None:
            wd.start()
        try:
            for i, prob in enumerate(problems):
                if wd is not None:
                    wd.kick()
                t0 = time.perf_counter()
                obs = self.run_one(prob)
                elapsed = time.perf_counter() - t0
                logger.info("[%d] %s in %.1fs", i, prob.problem_id, elapsed)
                yield obs
        finally:
            if wd is not None:
                wd.stop()
        logger.info("pipeline finished in %.1fs", time.perf_counter() - start)

    def write_run(self, problems: Iterable[Problem]) -> tuple[Path, Path]:
        """Run the pipeline and write both required logs:

        * ``observations.jsonl``: one row per problem with the full the structured log schema.
        * ``class_aux.jsonl``: one row per (problem, answer class) for the auxiliary table.

        Returns (observations_path, aux_path).

        Crash recovery
        --------------
        With ``config.resume = True`` (default), an existing ``observations.jsonl`` from a
        prior aborted run is read once at start, malformed trailing bytes are truncated,
        and problems whose ``problem_id`` is already present are skipped. ``class_aux.jsonl``
        is rebuilt from the surviving observations to keep the two files in sync. New
        rows are appended; each row is flushed before the loop advances so an OS-level
        SIGKILL can lose at most the in-flight problem.
        """
        obs_path = self.config.run_dir / "observations.jsonl"
        aux_path = self.config.run_dir / "class_aux.jsonl"
        obs_path.parent.mkdir(parents=True, exist_ok=True)

        done_ids: set[str] = set()
        if self.config.resume and obs_path.exists() and obs_path.stat().st_size > 0:
            done_ids = _recover_done_ids(obs_path)
            _rebuild_aux_for_done(obs_path, aux_path)
            logger.info(
                "resume: %d problems already in %s; appending new rows",
                len(done_ids),
                obs_path,
            )
            mode = "a"
        else:
            mode = "w"

        problems = list(problems)
        pending = [p for p in problems if p.problem_id not in done_ids]
        if done_ids:
            logger.info(
                "resume: %d/%d pending after skip", len(pending), len(problems)
            )

        n_obs = 0
        n_aux = 0
        with (
            obs_path.open(mode, encoding="utf-8") as obs_f,
            aux_path.open(mode, encoding="utf-8") as aux_f,
        ):
            for obs in self.run_many(pending):
                obs_f.write(
                    json.dumps(
                        observation_to_dict(obs, covcal_thresholds=self.config.covcal_thresholds),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                obs_f.flush()
                n_obs += 1
                for row in class_aux_rows(obs):
                    aux_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_aux += 1
                aux_f.flush()
        logger.info("wrote %d new observations + %d class rows", n_obs, n_aux)
        return obs_path, aux_path


def observation_to_dict(
    obs: FormalObservation,
    *,
    covcal_thresholds: Thresholds | None = None,
) -> dict[str, Any]:
    """Serialize one FormalObservation as the the structured log schema verification-attempt-log row.

    Includes computed diagnostics and selector decisions so the log is the
    "sufficient statistic for all coverage diagnostics, calibration, and failure
    analysis" the paper requires.
    """
    diag = compute_diagnostics(obs)
    sc_out = self_consistency(obs)
    ref = obs.metadata.get("reference_class")
    decisions: dict[str, Any] = {
        "self_consistency_class": sc_out.selected if not sc_out.abstained else ABSTAIN,
    }
    if covcal_thresholds is not None:
        cc = CovCal(covcal_thresholds)(obs)
        ccf = CovCalPlusFallback(covcal_thresholds)(obs)
        decisions.update(
            {
                "thresholds": list(covcal_thresholds.as_tuple()),
                "covcal_decision": cc.selected,            # may be ABSTAIN
                "covcal_reason": cc.reason,
                "covcal_correct": (
                    None if cc.abstained or ref is None else (cc.selected == ref)
                ),
                "fallback_decision": ccf.selected,
                "fallback_correct": (
                    None if ref is None else (ccf.selected == ref)
                ),
            }
        )
    return {
        "problem_id": obs.problem_id,
        "prover_budget_seconds": obs.prover_budget_seconds,
        "metadata": obs.metadata,
        # --- the structured log schema "Coverage fields" ---
        "diagnostics": {
            "typed_coverage": diag.typed_coverage,
            "proved_coverage": diag.proved_coverage,
            "proved_winner": diag.proved_winner,
            "proved_winner_weight": diag.proved_winner_weight,
            "unresolved_rival_mass": diag.unresolved_rival_mass,
            "margin": None if diag.margin == float("-inf") else diag.margin,
            "conflict": diag.conflict,
        },
        # --- the structured log schema "Selection fields" (decisions; correctness when ref known) ---
        "decisions": decisions,
        "classes": [_class_record_to_dict(c) for c in obs.classes],
        "candidates": [asdict(c) for c in obs.candidates],
    }


def class_aux_rows(obs: FormalObservation) -> list[dict[str, Any]]:
    """Flat per-(problem, class) rows for the auxiliary table the paper requests.

    "A flat auxiliary table has one row per answer class and records Q_c, status,
    correctness, and formalization route."
    """
    ref = obs.metadata.get("reference_class")
    rows: list[dict[str, Any]] = []
    for c in obs.classes:
        # Class-level status: best across artifacts (proved > typechecked > timeout > ...).
        cls_status = c.best_status.value if c.artifacts else "unformalized"
        routes = sorted({a.source for a in c.artifacts}) if c.artifacts else []
        rows.append(
            {
                "problem_id": obs.problem_id,
                "class_label": c.label,
                "weight": c.weight,
                "class_status": cls_status,
                "proved": c.proved,
                "typed": c.typed,
                "is_reference": ref is not None and ref == c.label,
                "routes": routes,
                "n_artifacts": len(c.artifacts),
            }
        )
    return rows


def _class_record_to_dict(c: ClassRecord) -> dict[str, Any]:
    return {
        "label": c.label,
        "weight": c.weight,
        "candidate_indices": c.candidate_indices,
        "class_status": c.best_status.value if c.artifacts else "unformalized",  # the structured log schema
        "proved": c.proved,
        "typed": c.typed,
        "artifacts": [
            {
                "status": a.status.value,
                "tactic_used": a.tactic_used,
                "elapsed_seconds": a.elapsed_seconds,
                "log": a.log,
                "source": a.source,
            }
            for a in c.artifacts
        ],
    }
