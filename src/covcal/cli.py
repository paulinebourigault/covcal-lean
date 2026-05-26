"""CovCal command-line interface (typer).

Subcommands
-----------

* ``covcal info``       — print versions, paths, and tool availability.
* ``covcal pipeline``   — run all four pipeline stages from a YAML config.
* ``covcal calibrate``  — read an observations JSONL, run Eq. (7) on the calibration split,
                          report the selected thresholds and the certified risk upper bound.
* ``covcal evaluate``   — read observations + thresholds, run all 9 selectors on the test
                          split, write per-method metrics as JSON.

The point of separating these from ``pipeline`` is what the paper makes explicit:
generation/formalization/verification are heavy and frozen; selection is cheap and is
re-run for every method and every ablation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from . import __version__
from .calibration import make_grid, select_thresholds
from .data import (
    load_amc_aime,
    load_math500,
    load_splits,
    make_splits,
    write_splits,
)
from .diagnostics import compute_diagnostics
from .metrics import evaluate as _eval_selectors
from .pipeline import (
    PipelineRun,
    PipelineRunConfig,
    Problem,
)
from .selectors import (
    ConfidenceOnly,
    CovCal,
    CovCalPlusFallback,
    MarginOnly,
    ProvedCoverageOnly,
    TypedCoverageOnly,
    proof_existence_abstention,
    raw_lean_plus_fallback,
    self_consistency,
)
from .types import (
    ABSTAIN,
    ArtifactOutcome,
    Candidate,
    ClassRecord,
    FormalObservation,
    SelectorOutput,
    Status,
    Thresholds,
)

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(
    name="covcal",
    help="Coverage-calibrated formal verification (CovCal).",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )


# --- info -------------------------------------------------------------------------------


@app.command()
def info() -> None:
    """Print versions, paths, and tool availability."""
    import shutil
    import sys
    rows = [
        ("covcal version", __version__),
        ("python", sys.version.split()[0]),
        ("lake on PATH", shutil.which("lake") or "<not found>"),
        ("lean on PATH", shutil.which("lean") or "<not found>"),
    ]
    try:
        import llama_cpp  # type: ignore[import-not-found]
        rows.append(("llama_cpp", llama_cpp.__version__))
    except ImportError:
        rows.append(("llama_cpp", "<not installed; install with `uv sync --extra llm`>"))
    try:
        import vllm  # type: ignore[import-not-found]
        rows.append(("vllm", vllm.__version__))
    except ImportError:
        rows.append(("vllm", "<not installed; install with `uv sync --extra gpu`>"))
    t = Table(title="CovCal environment")
    t.add_column("key"); t.add_column("value")
    for k, v in rows:
        t.add_row(k, str(v))
    console.print(t)


# --- dataset / splits ------------------------------------------------------------------


def _load_problems_from_config(cfg: dict[str, object]) -> list[Problem]:
    """Read `cfg["dataset"]` and dispatch to the right loader."""
    ds = cfg["dataset"]  # type: ignore[index]
    name = ds["name"]  # type: ignore[index]
    max_examples = ds.get("max_examples")  # type: ignore[union-attr]
    if name == "math500":
        problems, report = load_math500(
            max_examples=max_examples,
            jsonl_path=ds.get("jsonl_path"),  # type: ignore[union-attr]
            min_level=ds.get("min_level"),  # type: ignore[union-attr]
            max_level=ds.get("max_level"),  # type: ignore[union-attr]
        )
    elif name == "amc_aime":
        problems, report = load_amc_aime(
            max_examples=max_examples,
            jsonl_path=ds.get("jsonl_path"),  # type: ignore[union-attr]
            repo=ds.get("repo"),  # type: ignore[union-attr]
            split=ds.get("split", "train"),  # type: ignore[union-attr]
        )
    else:
        raise typer.BadParameter(f"unknown dataset: {name!r}")
    logger.info("dataset exclusion report: %s", report.as_dict())
    return problems


@app.command(name="split")
def split_cmd(
    config: Annotated[Path, typer.Option(..., help="YAML config providing dataset + splits.")],
    out: Annotated[Path, typer.Option(..., help="Output JSON for the splits manifest.")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate a deterministic dev/cal/test splits manifest from the dataset config."""
    _setup_logging(verbose)
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    problems = _load_problems_from_config(cfg)
    ds_cfg = cfg["dataset"]
    fractions = ds_cfg["split_fractions"]
    seed = int(ds_cfg.get("split_seed", 0))
    manifest = make_splits(
        (p.problem_id for p in problems),
        name=str(cfg.get("name", "<unnamed>")),
        seed=seed,
        fractions=fractions,
    )
    write_splits(manifest, out)
    console.print(f"[green]split done[/] → {out} ({manifest.n_total} ids)")
    for k in sorted(fractions):
        console.print(f"  {k}: {len(manifest.splits[k])}")


# --- pipeline runner -------------------------------------------------------------------


def _build_lean_backend(cfg: dict[str, object]) -> object:
    """Instantiate the lean backend selected by `cfg['lean']['backend']`."""
    lcfg = cfg["lean"]  # type: ignore[index]
    backend = str(lcfg.get("backend", "mock"))
    if backend == "mock":
        from .lean import MockLeanBackend
        return MockLeanBackend()
    if backend == "subprocess":
        from .lean import SubprocessLeanBackend
        from .lean.subprocess_backend import SubprocessLeanConfig
        project_dir = Path(lcfg.get("project_dir", "lean")).resolve()  # type: ignore[union-attr]
        timeout = float(lcfg.get("timeout_seconds", 10.0))  # type: ignore[union-attr]
        return SubprocessLeanBackend(
            SubprocessLeanConfig(
                project_dir=project_dir,
                per_task_timeout_seconds=timeout,
                # Rough conversion: ~200k heartbeats ≈ 1s of CPU on commodity hardware.
                max_heartbeats_per_tactic=int(timeout * 200_000),
            )
        )
    raise typer.BadParameter(f"unknown lean backend: {backend!r}")


def _build_llm_backend(spec: dict[str, object] | None) -> object | None:
    """Instantiate an LLM backend from a `{backend, model_id, model_file, ...}` dict.

    Supported backends:

    * ``llama_cpp`` (default): GGUF inference via llama-cpp-python. Uses
      ``model_id`` + ``model_file`` for HF Hub download, or ``model_path`` for local.
    * ``vllm``: GPU inference via vLLM. Uses ``model_id`` as the HF repo (or local path);
      no ``model_file`` is needed since vLLM loads multi-file safetensors automatically.
      Pass ``revision``, ``dtype``, ``quantization``, ``max_model_len``,
      ``tensor_parallel_size``, ``gpu_memory_utilization``, ``enforce_eager``,
      ``trust_remote_code``, ``download_dir``, ``max_num_seqs``, ``chat_template``,
      and ``seed`` to override the defaults.
    """
    if spec is None:
        return None
    backend = str(spec.get("backend", "llama_cpp"))
    if backend == "llama_cpp":
        from .generation.llamacpp import LlamaCppBackend
        return LlamaCppBackend(
            repo_id=spec.get("model_id"),  # type: ignore[arg-type]
            filename=spec.get("model_file"),  # type: ignore[arg-type]
            model_path=spec.get("model_path"),  # type: ignore[arg-type]
            n_ctx=int(spec.get("n_ctx", 8192)),  # type: ignore[arg-type]
            n_threads=spec.get("n_threads"),  # type: ignore[arg-type]
            n_gpu_layers=int(spec.get("n_gpu_layers", 0)),  # type: ignore[arg-type]
        )
    if backend == "vllm":
        from .generation.vllm_backend import VLLMBackend, VLLMBackendConfig
        model = spec.get("model_id") or spec.get("model_path")
        if model is None:
            raise typer.BadParameter("vllm backend requires `model_id` (HF repo) or `model_path`.")
        vcfg = VLLMBackendConfig(
            model=str(model),
            revision=spec.get("revision"),                              # type: ignore[arg-type]
            dtype=str(spec.get("dtype", "auto")),
            quantization=spec.get("quantization"),                      # type: ignore[arg-type]
            max_model_len=int(spec.get("max_model_len", 8192)),         # type: ignore[arg-type]
            tensor_parallel_size=int(spec.get("tensor_parallel_size", 1)),  # type: ignore[arg-type]
            gpu_memory_utilization=float(spec.get("gpu_memory_utilization", 0.90)),  # type: ignore[arg-type]
            enforce_eager=bool(spec.get("enforce_eager", False)),
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
            download_dir=spec.get("download_dir"),                      # type: ignore[arg-type]
            seed=int(spec.get("vllm_seed", 0)),                         # type: ignore[arg-type]
            max_num_seqs=int(spec.get("max_num_seqs", 256)),            # type: ignore[arg-type]
            chat_template=bool(spec.get("chat_template", False)),
        )
        return VLLMBackend(vcfg)
    raise typer.BadParameter(f"unknown llm backend: {backend!r}")


@app.command()
def pipeline(
    config: Annotated[Path, typer.Option(..., help="YAML config (e.g. configs/minimal.yaml).")],
    splits: Annotated[Path, typer.Option(..., help="Splits manifest from `covcal split`.")],
    only_split: Annotated[
        str, typer.Option(help="Restrict to one split (dev/cal/test/all).")
    ] = "all",
    limit: Annotated[int, typer.Option(help="Process at most N problems (0 = no limit).")] = 0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run all four pipeline stages: generate → formalize → verify → assemble.

    Writes per-problem `FormalObservation` rows to `<run_dir>/observations.jsonl`.
    """
    _setup_logging(verbose)
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    problems = _load_problems_from_config(cfg)
    by_id = {p.problem_id: p for p in problems}

    manifest = load_splits(splits)
    if only_split == "all":
        wanted: list[str] = []
        for k in manifest.splits:
            wanted.extend(manifest.splits[k])
    else:
        wanted = manifest.splits.get(only_split, [])
        if not wanted:
            raise typer.BadParameter(
                f"split {only_split!r} not in manifest; have: {list(manifest.splits)}"
            )
    selected_problems = [by_id[pid] for pid in wanted if pid in by_id]
    if limit > 0:
        selected_problems = selected_problems[:limit]
    console.print(f"[yellow]pipeline:[/] {len(selected_problems)} problems")

    gen_cfg = cfg["generation"]
    form_cfg = cfg["formalization"]
    lean_cfg = cfg["lean"]
    run_dir = Path(cfg["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    from .generation.cache import SamplingCache
    from .lean import Autoformalizer, AutoformalizerConfig

    llm = _build_llm_backend(gen_cfg)
    lean_backend = _build_lean_backend(cfg)
    afmt_spec = form_cfg.get("autoformalizer") if form_cfg.get("use_autoformalizer") else None
    af_llm = _build_llm_backend(afmt_spec) if afmt_spec else None
    autoformalizer = Autoformalizer(
        config=AutoformalizerConfig(
            artifacts_per_class=int(form_cfg.get("artifacts_per_class", 2)),
            use_templates=bool(form_cfg.get("use_templates", True)),
            use_llm=bool(form_cfg.get("use_autoformalizer", False)),
        ),
        lean=lean_backend,  # type: ignore[arg-type]
        llm=af_llm,  # type: ignore[arg-type]
    )

    pipe_cfg = PipelineRunConfig(
        run_dir=run_dir,
        n_samples=int(gen_cfg["n_samples"]),
        temperature=float(gen_cfg["temperature"]),
        top_p=float(gen_cfg["top_p"]),
        max_new_tokens=int(gen_cfg["max_new_tokens"]),
        seed=int(gen_cfg["seed"]),
        formalize_top_k_classes=int(form_cfg["formalize_top_k_classes"]),
        prover_budget_seconds=float(lean_cfg["timeout_seconds"]),
        dataset_name=str(cfg["dataset"].get("name", "<unknown>")),
        # CovCal thresholds get filled in by `covcal calibrate`; if a threshold file is
        # present beside the run dir we read it eagerly so observations carry decisions.
        covcal_thresholds=_load_calibrated_thresholds(run_dir / "thresholds.json"),
    )
    cache = SamplingCache(run_dir / "samples_cache.jsonl")
    runner = PipelineRun(
        config=pipe_cfg,
        llm=llm,  # type: ignore[arg-type]
        autoformalizer=autoformalizer,
        lean=lean_backend,  # type: ignore[arg-type]
        sampling_cache=cache,
    )

    # Per-run metadata: pin Lean version / Mathlib commit / hardware / git sha / config.
    from .run_metadata import finalize, make_run_metadata, write_metadata
    repo_root = Path(__file__).resolve().parents[2]
    lean_dir = repo_root / "lean"
    meta = make_run_metadata(
        name=str(cfg.get("name", "<unnamed>")),
        repo_root=repo_root,
        lean_dir=lean_dir,
        config_snapshot=cfg,
    )
    meta_path = run_dir / "metadata.json"
    write_metadata(meta, meta_path)  # partial info at start

    obs_path, aux_path = runner.write_run(selected_problems)

    # Re-write with finalised timing + summary counts.
    summary = {
        "n_problems_requested": len(selected_problems),
        "splits_used": only_split,
        "thresholds_from": str(run_dir / "thresholds.json") if (run_dir / "thresholds.json").exists() else None,
    }
    finalize(meta, pipeline_summary=summary)
    write_metadata(meta, meta_path)
    console.print(f"[green]pipeline done[/] → {obs_path} (+ {aux_path}, {meta_path})")


def _load_calibrated_thresholds(path: Path) -> Thresholds | None:
    """Pre-fill CovCal acceptance thresholds if a prior `covcal calibrate` was run."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        sel = d.get("selected")
        if sel is None:
            return None
        return Thresholds(typ=float(sel[0]), prf=float(sel[1]), margin=float(sel[2]))
    except (KeyError, ValueError, TypeError):
        return None


# --- calibrate --------------------------------------------------------------------------


def _load_observations(path: Path) -> list[FormalObservation]:
    """Reverse of `pipeline.observation_to_dict`."""
    obs_list: list[FormalObservation] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
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
            candidates = [Candidate(**c) for c in d.get("candidates", [])]
            obs_list.append(
                FormalObservation(
                    problem_id=d["problem_id"],
                    classes=classes,
                    candidates=candidates,
                    prover_budget_seconds=d.get("prover_budget_seconds", 0.0),
                    metadata=d.get("metadata", {}),
                )
            )
    return obs_list


def _split_by_id(
    observations: list[FormalObservation], split_path: Path
) -> dict[str, list[FormalObservation]]:
    """Apply a splits manifest: {"dev": [...ids...], "cal": [...], "test": [...]}."""
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    # SplitsManifest.to_dict mixes metadata keys (name/seed/fractions/n_total) with the
    # per-split id lists at the top level — only the list values are id buckets.
    bucket = {k: set(v) for k, v in manifest.items() if isinstance(v, list)}
    out: dict[str, list[FormalObservation]] = {k: [] for k in bucket}
    for obs in observations:
        for split, ids in bucket.items():
            if obs.problem_id in ids:
                out[split].append(obs)
                break
    return out


def _grid_from_config(cfg: dict[str, object]) -> list[Thresholds]:
    cal = cfg["calibration"]  # type: ignore[index]
    grid_cfg = cal["threshold_grid"]  # type: ignore[index]
    return make_grid(grid_cfg["typ"], grid_cfg["prf"], grid_cfg["margin"])


@app.command()
def calibrate(
    observations: Annotated[Path, typer.Option(..., help="JSONL of pipeline observations.")],
    splits: Annotated[Path, typer.Option(..., help="splits manifest (dev/cal/test).")],
    config: Annotated[Path, typer.Option(..., help="YAML config providing the grid + ε,δ.")],
    out: Annotated[Path, typer.Option(..., help="Output JSON with the selected thresholds.")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run Eq. (7) on the calibration split and persist the selected thresholds.

    Implements the paper's risk-controlled selector: among all predeclared coverage rules
    whose Clopper–Pearson upper bound is ≤ ε, pick the one accepting the most calibration
    examples. The selected rule and bound are written to `out` as JSON.
    """
    _setup_logging(verbose)
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cal_cfg = cfg["calibration"]
    epsilon = float(cal_cfg["epsilon"])
    delta = float(cal_cfg["delta"])
    grid = _grid_from_config(cfg)

    obs_list = _load_observations(observations)
    by_split = _split_by_id(obs_list, splits)
    cal_obs = by_split.get("cal", [])
    if not cal_obs:
        raise typer.BadParameter("calibration split is empty")

    accepted: dict[Thresholds, tuple[int, int]] = {}
    for tau in grid:
        sel = CovCal(tau)
        m, k = 0, 0
        for obs in cal_obs:
            out_sel = sel(obs)
            if out_sel.abstained:
                continue
            m += 1
            ref = obs.metadata.get("reference_class")
            if ref is not None and out_sel.selected != ref:
                k += 1
        accepted[tau] = (m, k)

    res = select_thresholds(grid, accepted, epsilon=epsilon, delta=delta)
    payload = {
        "selected": None if res.selected is None else res.selected.as_tuple(),
        "epsilon": res.epsilon,
        "delta": res.delta,
        "grid_size": res.grid_size,
        "per_threshold_alpha": res.per_threshold_alpha,
        "risk_upper_bound": res.risk_upper_bound,
        "accepted_count": res.accepted_count,
        "accepted_errors": res.accepted_errors,
        "calibration_size": len(cal_obs),
        "reject_all": res.reject_all,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[green]calibrate done[/] → {out}")
    console.print(payload)


# --- evaluate ---------------------------------------------------------------------------


def _build_baselines(
    *, calibrated_thresholds: Thresholds | None, conf_threshold: float
) -> dict[str, object]:
    sels: dict[str, object] = {
        "self_consistency": self_consistency,
        "confidence_only": ConfidenceOnly(conf_threshold),
        "raw_lean_plus_fallback": raw_lean_plus_fallback,
        "proof_existence": proof_existence_abstention,
    }
    if calibrated_thresholds is not None:
        sels["typed_coverage_only"] = TypedCoverageOnly(calibrated_thresholds.typ)
        sels["proved_coverage_only"] = ProvedCoverageOnly(calibrated_thresholds.prf)
        sels["margin_only"] = MarginOnly(calibrated_thresholds.margin)
        sels["covcal"] = CovCal(calibrated_thresholds)
        sels["covcal_plus_fallback"] = CovCalPlusFallback(calibrated_thresholds)
    return sels


@app.command()
def evaluate(
    observations: Annotated[Path, typer.Option(..., help="JSONL of pipeline observations.")],
    splits: Annotated[Path, typer.Option(..., help="splits manifest.")],
    thresholds: Annotated[Path, typer.Option(..., help="JSON from `covcal calibrate`.")],
    out: Annotated[Path, typer.Option(..., help="Output JSON with per-method metrics.")],
    conf_threshold: Annotated[
        float, typer.Option(help="Confidence threshold for the confidence-only baseline.")
    ] = 0.5,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run all selectors on the test split and dump per-method metrics."""
    _setup_logging(verbose)
    obs_list = _load_observations(observations)
    by_split = _split_by_id(obs_list, splits)
    test_obs = by_split.get("test", [])
    if not test_obs:
        raise typer.BadParameter("test split is empty")

    calib = json.loads(thresholds.read_text(encoding="utf-8"))
    tau: Thresholds | None = None
    if not calib.get("reject_all", False) and calib.get("selected") is not None:
        t = calib["selected"]
        tau = Thresholds(typ=float(t[0]), prf=float(t[1]), margin=float(t[2]))

    selectors = _build_baselines(calibrated_thresholds=tau, conf_threshold=conf_threshold)
    references = [obs.metadata.get("reference_class", ABSTAIN) for obs in test_obs]

    rows: dict[str, dict[str, float]] = {}
    for name, fn in selectors.items():
        outputs: list[SelectorOutput] = [fn(o) for o in test_obs]  # type: ignore[operator]
        m = _eval_selectors(outputs, references)
        rows[name] = {
            "overall_accuracy": m.overall_accuracy,
            "accepted_accuracy": m.accepted_accuracy,
            "selective_risk": m.selective_risk,
            "abstention_rate": m.abstention_rate,
            "accepted_fraction": m.accepted_fraction,
            "n_total": float(m.n_total),
            "n_accepted": float(m.n_accepted),
            "risk_upper_bound_95": m.risk_upper_bound(0.05),
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"thresholds": calib, "methods": rows}, indent=2), encoding="utf-8")
    console.print(f"[green]evaluate done[/] → {out}")
    t = Table(title="Per-method metrics")
    t.add_column("method")
    t.add_column("overall acc", justify="right")
    t.add_column("accepted acc", justify="right")
    t.add_column("accepted frac", justify="right")
    t.add_column("sel-risk UB", justify="right")
    for name, vals in rows.items():
        t.add_row(
            name,
            f"{vals['overall_accuracy']:.3f}",
            f"{vals['accepted_accuracy']:.3f}",
            f"{vals['accepted_fraction']:.3f}",
            f"{vals['risk_upper_bound_95']:.3f}",
        )
    console.print(t)


# --- diagnose --------------------------------------------------------------------------


@app.command()
def diagnose(
    observations: Annotated[Path, typer.Option(..., help="JSONL of pipeline observations.")],
    out: Annotated[Path, typer.Option(..., help="Output JSONL with per-problem diagnostics.")],
) -> None:
    """Dump per-problem coverage diagnostics (C_typ, C_prf, M, conflict) as JSONL."""
    obs_list = _load_observations(observations)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for obs in obs_list:
            d = compute_diagnostics(obs)
            f.write(json.dumps({
                "problem_id": obs.problem_id,
                "typed_coverage": d.typed_coverage,
                "proved_coverage": d.proved_coverage,
                "proved_winner": d.proved_winner,
                "proved_winner_weight": d.proved_winner_weight,
                "unresolved_rival_mass": d.unresolved_rival_mass,
                "margin": d.margin if d.margin != float("-inf") else None,
                "conflict": d.conflict,
            }) + "\n")
    console.print(f"[green]diagnose done[/] → {out} ({len(obs_list)} problems)")


if __name__ == "__main__":  # pragma: no cover
    app()
