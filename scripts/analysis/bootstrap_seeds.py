"""Multi-seed bootstrap for all paper tables.

Replays calibrate+evaluate on each run dir for K seeds, reshuffling the dev/cal/test
partition each time. The observations themselves don't change, only the split, so this
captures calibration-set variability without re-running any inference.

Usage:
    python scripts/analysis/bootstrap_seeds.py \\
        --run-dir runs/main --k-seeds 5

Outputs:
    runs/<run_dir>/analysis/bootstrap_seeds.json     mean ± 95% CI per (selector, metric)
    runs/<run_dir>/analysis/bootstrap_seeds.tex      LaTeX table
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from _common import (
    calibrate_in_memory,
    calibrate_in_memory_dev_then_cal,
    coarse_grid_27,
    default_grid_125,
    evaluate_in_memory,
    load_observations,
    load_splits_dict,
    normal_ci,
    reshuffle_splits,
    split_observations,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Which selectors we report rows for (in this order)
SELECTOR_ORDER = [
    "self_consistency",
    "confidence_only",
    "raw_lean_plus_fallback",
    "proof_existence",
    "typed_only",
    "proved_only",
    "margin_only",
    "covcal",
    "covcal_plus_fallback",
]
METRIC_KEYS = [
    "overall_accuracy",
    "accepted_accuracy",
    "accepted_fraction",
    "risk_upper_bound_95",
]


def read_run_config(run_dir: Path) -> dict:
    """Pull eps/delta/fractions from metadata.json + splits.json (no YAML required)."""
    md = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    sp = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    cfg = md.get("config", md)
    cal = cfg.get("calibration", {})
    return {
        "epsilon": float(cal.get("epsilon", 0.15)),
        "delta": float(cal.get("delta", 0.05)),
        "fractions": dict(sp.get("fractions", {"dev": 0.2, "cal": 0.4, "test": 0.4})),
    }


def run_one_seed(
    obs_list, seed: int, fractions: dict[str, float], epsilon: float, delta: float,
    regime: str = "bonferroni",
) -> tuple[dict, dict[str, dict[str, float]]]:
    pids = [o.problem_id for o in obs_list]
    splits = reshuffle_splits(pids, seed=seed, fractions=fractions)
    parts = split_observations(obs_list, splits)
    dev_obs = parts.get("dev", [])
    cal_obs = parts.get("cal", [])
    test_obs = parts.get("test", [])
    if not cal_obs or not test_obs:
        return {"reject_all": True}, {}
    if regime == "dev-then-cal":
        if not dev_obs:
            calib = {
                "selected": None,
                "reject_all": True,
                "epsilon": epsilon, "delta": delta,
                "grid_size": len(default_grid_125()),
                "per_threshold_alpha": delta,
                "risk_upper_bound": 1.0,
                "accepted_count": 0, "accepted_errors": 0,
                "calibration_size": len(cal_obs),
                "dev_size": 0,
                "regime": "dev-then-cal",
                "note": "dev split empty; dev-then-cal not applicable",
            }
            return calib, evaluate_in_memory(test_obs, calib)
        calib = calibrate_in_memory_dev_then_cal(
            dev_obs, cal_obs, default_grid_125(), epsilon, delta,
        )
    else:
        calib = calibrate_in_memory(cal_obs, default_grid_125(), epsilon, delta)
    metrics = evaluate_in_memory(test_obs, calib)
    return calib, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--k-seeds", type=int, default=5)
    ap.add_argument(
        "--regime", choices=["bonferroni", "dev-then-cal"], default="bonferroni",
        help="Certificate regime: Bonferroni union-bound over the grid, or two-stage "
             "dev-then-cal (Theorem 1, no union bound).",
    )
    args = ap.parse_args()

    rd = args.run_dir.resolve()
    cfg = read_run_config(rd)
    print(f"[{rd.name}] eps={cfg['epsilon']} delta={cfg['delta']} fractions={cfg['fractions']}")

    obs_list = load_observations(rd)
    print(f"[{rd.name}] loaded {len(obs_list)} observations")

    per_seed_metrics: dict[int, dict[str, dict[str, float]]] = {}
    per_seed_calib: dict[int, dict] = {}
    for seed in range(args.k_seeds):
        calib, metrics = run_one_seed(
            obs_list, seed=seed, fractions=cfg["fractions"],
            epsilon=cfg["epsilon"], delta=cfg["delta"],
            regime=args.regime,
        )
        per_seed_calib[seed] = calib
        per_seed_metrics[seed] = metrics
        n_cal = calib.get("calibration_size", "?")
        verdict = "REJECT_ALL" if calib.get("reject_all") else f"sel={calib.get('selected')}"
        print(f"  seed={seed}  n_cal={n_cal}  {verdict}  UB={calib.get('risk_upper_bound'):.3f}")

    # Aggregate per (selector, metric)
    agg: dict[str, dict[str, dict[str, float]]] = {}
    for name in SELECTOR_ORDER:
        agg[name] = {}
        for metric in METRIC_KEYS:
            values: list[float] = []
            for s in range(args.k_seeds):
                m = per_seed_metrics[s].get(name)
                if m is None:
                    continue
                v = m.get(metric)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                values.append(float(v))
            if not values:
                agg[name][metric] = {"mean": float("nan"), "lo": float("nan"),
                                     "hi": float("nan"), "n_seeds_present": 0}
            else:
                mean, lo, hi = normal_ci(values, alpha=0.05)
                agg[name][metric] = {
                    "mean": mean, "lo": lo, "hi": hi,
                    "n_seeds_present": len(values),
                }

    # Calibration stability across seeds
    selections = [tuple(per_seed_calib[s].get("selected") or ("REJ",))
                  for s in range(args.k_seeds)]
    n_reject = sum(1 for c in per_seed_calib.values() if c.get("reject_all"))
    print(f"\ncalibration stability: {n_reject}/{args.k_seeds} reject_all; "
          f"distinct selections: {len(set(selections))}")

    out_dir = rd / "analysis"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "run_dir": str(rd),
        "k_seeds": args.k_seeds,
        "regime": args.regime,
        "config": cfg,
        "aggregate": agg,
        "per_seed_calibration": {str(s): per_seed_calib[s] for s in per_seed_calib},
        "per_seed_metrics": {str(s): per_seed_metrics[s] for s in per_seed_metrics},
        "calibration_stability": {
            "n_reject_all": n_reject,
            "n_distinct_selections": len(set(selections)),
            "selections": [list(s) if s != ("REJ",) else None for s in selections],
        },
    }
    suffix = "" if args.regime == "bonferroni" else "_devcal"
    out_json = out_dir / f"bootstrap_seeds{suffix}.json"
    out_tex = out_dir / f"bootstrap_seeds{suffix}.tex"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_json}")

    # LaTeX table
    method_disp = {
        "self_consistency": "Self-consistency",
        "confidence_only": "Confidence-only abst.",
        "raw_lean_plus_fallback": "Raw Lean + fallback",
        "proof_existence": "Proof-existence abst.",
        "typed_only": "Typed-coverage only",
        "proved_only": "Proved-coverage only",
        "margin_only": "Margin-only",
        "covcal": "\\methodname{}",
        "covcal_plus_fallback": "\\methodname{}+fallback",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Overall & Accepted & Acc. frac. & Risk UB \\\\",
        "\\midrule",
    ]
    for name in SELECTOR_ORDER:
        row = agg[name]
        if row["overall_accuracy"]["n_seeds_present"] == 0:
            lines.append(f"{method_disp[name]} & -- & -- & -- & -- \\\\")
            continue

        def fmt(key: str) -> str:
            d = row[key]
            if math.isnan(d["mean"]):
                return "--"
            return f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
        lines.append(
            f"{method_disp[name]} & {fmt('overall_accuracy')} & {fmt('accepted_accuracy')} & "
            f"{fmt('accepted_fraction')} & {fmt('risk_upper_bound_95')} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        f"\\caption{{Bootstrap aggregation of Table~\\ref{{tab:main}} over $K\\!=\\!{args.k_seeds}$ "
        f"seeds for the dev/cal/test partition (observations frozen). "
        f"Each cell is {{mean}} $[$lower, upper$]$ at the $95\\%$ normal-approximation level. "
        f"$n_{{\\mathrm{{cal}}}}\\!=\\!{per_seed_calib[0].get('calibration_size','?')}$, "
        f"$\\epsilon\\!=\\!{cfg['epsilon']}$, $\\delta\\!=\\!{cfg['delta']}$. "
        f"{n_reject}/{args.k_seeds} seeds returned reject-all; "
        f"{len(set(selections))} distinct threshold cells were selected across the seeds where the "
        f"certificate was feasible.}}",
        "\\label{tab:bootstrap}",
        "\\end{table}",
    ]
    out_tex.write_text("\n".join(lines))
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
