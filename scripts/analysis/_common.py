"""Shared helpers for offline post-hoc analyses.

These scripts read a frozen run directory (observations.jsonl + splits.json +
metadata.json) and produce new metrics without invoking any GPU. They reuse the
core CovCal modules for selectors and calibration so the offline analysis stays
consistent with the on-line pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from covcal.calibration import (
    CalibrationResult,
    clopper_pearson_upper,
    make_grid,
    select_thresholds,
)
from covcal.metrics import evaluate as _eval_selectors
from covcal.selectors import (
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
from covcal.types import ABSTAIN, FormalObservation, SelectorOutput, Thresholds
from covcal.data.splits import make_splits

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_observations(run_dir: Path) -> list[FormalObservation]:
    """Read observations.jsonl into a list of `FormalObservation`."""
    from covcal.pipeline import _observation_from_dict
    out: list[FormalObservation] = []
    with (run_dir / "observations.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(_observation_from_dict(json.loads(line)))
    return out


def load_splits_dict(splits_path: Path) -> dict[str, list[str]]:
    """Read splits.json and return `{dev, cal, test}` lists of problem_ids."""
    s = json.loads(splits_path.read_text(encoding="utf-8"))
    if "splits" in s and isinstance(s["splits"], dict):
        return {k: list(v) for k, v in s["splits"].items()}
    return {k: list(s[k]) for k in ("dev", "cal", "test") if k in s}


def reshuffle_splits(
    problem_ids: list[str], seed: int, fractions: dict[str, float]
) -> dict[str, list[str]]:
    """Build a new dev/cal/test partition deterministically from `seed`."""
    m = make_splits(problem_ids, name=f"seed{seed}", seed=seed, fractions=fractions)
    return dict(m.splits)


def split_observations(
    obs_list: list[FormalObservation], split_ids: dict[str, list[str]]
) -> dict[str, list[FormalObservation]]:
    by_id = {o.problem_id: o for o in obs_list}
    return {k: [by_id[pid] for pid in ids if pid in by_id] for k, ids in split_ids.items()}


def _accepted_table(
    obs: list[FormalObservation],
    grid: list[Thresholds],
) -> dict[Thresholds, tuple[int, int]]:
    """For each cell tau in the grid, count (m_tau, k_tau) on a set of observations."""
    accepted: dict[Thresholds, tuple[int, int]] = {}
    for tau in grid:
        sel = CovCal(tau)
        m = k = 0
        for o in obs:
            out_sel = sel(o)
            if out_sel.abstained:
                continue
            m += 1
            ref = o.metadata.get("reference_class")
            if ref is not None and out_sel.selected != ref:
                k += 1
        accepted[tau] = (m, k)
    return accepted


def calibrate_in_memory(
    cal_obs: list[FormalObservation],
    grid: list[Thresholds],
    epsilon: float,
    delta: float,
) -> dict:
    """Replicate `covcal calibrate` purely in-process (Bonferroni regime)."""
    accepted = _accepted_table(cal_obs, grid)
    res = select_thresholds(grid, accepted, epsilon=epsilon, delta=delta)
    return {
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
        "regime": "bonferroni",
    }


def calibrate_in_memory_dev_then_cal(
    dev_obs: list[FormalObservation],
    cal_obs: list[FormalObservation],
    grid: list[Thresholds],
    epsilon: float,
    delta: float,
    epsilon_dev: float | None = None,
) -> dict:
    """Theorem-1 (dev-then-cal) certificate, no Bonferroni penalty.

    Stage 1 picks tau_hat on dev (most permissive cell with k_dev/m_dev <= epsilon_dev;
    lex tie-break). Stage 2 certifies on cal with a single CP bound at alpha = delta.
    """
    from covcal.calibration import select_thresholds_dev_then_cal
    dev_accepted = _accepted_table(dev_obs, grid)
    cal_accepted = _accepted_table(cal_obs, grid)
    res = select_thresholds_dev_then_cal(
        grid, dev_accepted, cal_accepted,
        epsilon=epsilon, delta=delta, epsilon_dev=epsilon_dev,
    )
    return {
        "selected": None if res.selected is None else res.selected.as_tuple(),
        "epsilon": res.epsilon,
        "delta": res.delta,
        "grid_size": res.grid_size,
        "per_threshold_alpha": res.per_threshold_alpha,
        "risk_upper_bound": res.risk_upper_bound,
        "accepted_count": res.accepted_count,
        "accepted_errors": res.accepted_errors,
        "calibration_size": len(cal_obs),
        "dev_size": len(dev_obs),
        "reject_all": res.reject_all,
        "regime": "dev-then-cal",
        "epsilon_dev": epsilon_dev if epsilon_dev is not None else epsilon,
    }


def evaluate_in_memory(
    test_obs: list[FormalObservation],
    calib: dict,
    conf_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Replicate `covcal evaluate` purely in-process; returns the methods dict."""
    tau: Thresholds | None = None
    if not calib.get("reject_all", False) and calib.get("selected") is not None:
        t = calib["selected"]
        tau = Thresholds(typ=float(t[0]), prf=float(t[1]), margin=float(t[2]))
    selectors: dict[str, object] = {
        "self_consistency": self_consistency,
        "confidence_only": ConfidenceOnly(conf_threshold),
        "raw_lean_plus_fallback": raw_lean_plus_fallback,
        "proof_existence": proof_existence_abstention,
    }
    if tau is not None:
        selectors["typed_only"] = TypedCoverageOnly(tau.typ)
        selectors["proved_only"] = ProvedCoverageOnly(tau.prf)
        selectors["margin_only"] = MarginOnly(tau.margin)
        selectors["covcal"] = CovCal(tau)
        selectors["covcal_plus_fallback"] = CovCalPlusFallback(tau)
    refs = [obs.metadata.get("reference_class", ABSTAIN) for obs in test_obs]
    rows: dict[str, dict[str, float]] = {}
    for name, fn in selectors.items():
        outputs: list[SelectorOutput] = [fn(o) for o in test_obs]  # type: ignore[operator]
        m = _eval_selectors(outputs, refs)
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
    return rows


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson two-sided confidence interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    from scipy.stats import norm  # type: ignore
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def normal_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi) under a normal approximation; returns (mean, mean, mean) if n<2."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return (mean, mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    from scipy.stats import t  # type: ignore
    crit = t.ppf(1 - alpha / 2, n - 1)
    return (mean, mean - crit * se, mean + crit * se)


def default_grid_125() -> list[Thresholds]:
    return make_grid(
        typ=[0.0, 0.25, 0.5, 0.75, 0.9],
        prf=[0.0, 0.1, 0.25, 0.5, 0.75],
        margin=[-0.5, 0.0, 0.1, 0.25, 0.5],
    )


def coarse_grid_27() -> list[Thresholds]:
    """3x3x3 grid for the coarse-grid ablation: alpha/cell = 0.05/27 ≈ 1.85e-3."""
    return make_grid(
        typ=[0.0, 0.5, 0.9],
        prf=[0.0, 0.25, 0.75],
        margin=[-0.5, 0.0, 0.5],
    )
