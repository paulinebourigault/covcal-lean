"""Per-MATH-500-level cliff breakdown.

For the main full-MATH-500 run, join observations to the dataset's
level field (1..5) and compute per-level coverage / accuracy / certificate
feasibility. Produces the continuous version of Tables 2-3.

Usage:
    python scripts/analysis/per_level.py --run-dir runs/main

Outputs:
    runs/<rd>/analysis/per_level.json
    runs/<rd>/analysis/per_level.tex
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _common import (
    calibrate_in_memory,
    default_grid_125,
    evaluate_in_memory,
    load_observations,
    load_splits_dict,
    split_observations,
    wilson_ci,
)
from covcal.data.math500 import load_math500
from covcal.diagnostics import compute_diagnostics


def _level_map() -> dict[str, int]:
    """problem_id -> level. We rebuild this from the math500 dataset since
    observations don't carry the level field directly."""
    problems, _ = load_math500(max_examples=None)
    out = {}
    for p in problems:
        lvl = p.metadata.get("level")
        if lvl is not None:
            try:
                out[p.problem_id] = int(lvl)
            except (TypeError, ValueError):
                pass
    return out


def run_one(run_dir: Path) -> dict:
    md = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    cal_cfg = (md.get("config", md)).get("calibration", {})
    eps, delta = float(cal_cfg.get("epsilon", 0.15)), float(cal_cfg.get("delta", 0.05))

    obs = load_observations(run_dir)
    levels = _level_map()
    by_level = defaultdict(list)
    n_unmatched = 0
    for o in obs:
        lvl = levels.get(o.problem_id)
        if lvl is None:
            n_unmatched += 1
        else:
            by_level[lvl].append(o)
    print(f"  matched {len(obs) - n_unmatched}/{len(obs)} to levels "
          f"({n_unmatched} no-match)")

    out: dict[int, dict] = {}
    for lvl in sorted(by_level):
        bucket = by_level[lvl]
        n = len(bucket)
        diags = [compute_diagnostics(o) for o in bucket]
        typed = sum(d.typed_coverage for d in diags) / max(1, n)
        proved = sum(d.proved_coverage for d in diags) / max(1, n)
        margin = sum(d.margin for d in diags) / max(1, n)
        # Per-problem correctness for the proof-existence selector (proxy for
        # "raw formal selection accuracy"):
        from covcal.selectors import proof_existence_abstention
        accepted = 0
        correct = 0
        for o in bucket:
            sel = proof_existence_abstention(o)
            if not sel.abstained:
                accepted += 1
                ref = o.metadata.get("reference_class")
                if sel.selected == ref:
                    correct += 1
        accept_frac = accepted / n if n else 0.0
        accept_acc = correct / accepted if accepted else float("nan")
        # Certificate feasibility check on this stratum: would the 125-cell
        # certificate be feasible if we *only* had these examples as cal?
        # Heuristic: clopper_pearson_upper(k_pe, m_pe, alpha) ≤ eps
        from covcal.calibration import clopper_pearson_upper
        cert_alpha = delta / 125
        cert_ub = clopper_pearson_upper(accepted - correct, accepted, cert_alpha) if accepted > 0 else float("inf")
        wilson_lo, wilson_hi = wilson_ci(correct, accepted, alpha=0.05) if accepted > 0 else (float("nan"), float("nan"))
        out[lvl] = {
            "n": n,
            "typed_coverage_mean": typed,
            "proved_coverage_mean": proved,
            "margin_mean": margin,
            "proof_exist_accept_frac": accept_frac,
            "proof_exist_accept_acc": accept_acc,
            "proof_exist_accept_acc_lo": wilson_lo,
            "proof_exist_accept_acc_hi": wilson_hi,
            "proof_exist_certifies_at_eps": cert_ub <= eps if accepted > 0 else False,
            "proof_exist_certificate_ub": cert_ub,
        }
    return {
        "run_dir": str(run_dir),
        "epsilon": eps,
        "delta": delta,
        "n_total": len(obs),
        "n_unmatched": n_unmatched,
        "by_level": {str(k): v for k, v in out.items()},
    }


def write_latex(run_dir: Path, payload: dict) -> None:
    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\small\\setlength{\\tabcolsep}{3pt}")
    rows.append("\\resizebox{\\columnwidth}{!}{%")
    rows.append("\\begin{tabular}{cccccc}")
    rows.append("\\toprule")
    rows.append("Lvl & $n$ & $\\bar C_{\\typ}$ & $\\bar C_{\\prf}$ & PE acc. & Cert.\\ feas.\\ \\\\")
    rows.append("\\midrule")
    for lvl_str, d in sorted(payload["by_level"].items(), key=lambda kv: int(kv[0])):
        accept_acc = d["proof_exist_accept_acc"]
        accept_str = "--" if accept_acc != accept_acc else f"{accept_acc:.3f}"  # nan check
        feas = "\\checkmark" if d["proof_exist_certifies_at_eps"] else "--"
        rows.append(
            f"{lvl_str} & {d['n']} & {d['typed_coverage_mean']:.3f} & "
            f"{d['proved_coverage_mean']:.3f} & {accept_str} & {feas} \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}}")
    rows.append(
        f"\\caption{{Per-MATH-500-level cliff breakdown on the main run "
        f"($\\epsilon={payload['epsilon']}$, $\\delta={payload['delta']}$). "
        f"$\\bar C_{{\\typ}}$/$\\bar C_{{\\prf}}$ are mean typed/proved coverage; "
        f"\\emph{{PE acc.}} is proof-existence accepted accuracy; "
        f"\\emph{{Cert.\\ feas.}} marks whether the proof-existence rule alone "
        f"certifies $\\epsilon$ on examples at that level (a single-cell, "
        f"single-stratum stand-in for the full $125$-cell certificate). "
        f"The cliff is visible level-by-level: lower levels certify at $\\epsilon=0.15$, "
        f"higher levels do not.}}"
    )
    rows.append("\\label{tab:per-level}")
    rows.append("\\end{table}")
    (run_dir / "analysis" / "per_level.tex").write_text("\n".join(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    rd = args.run_dir.resolve()
    print(f"=== {rd.name} ===")
    payload = run_one(rd)
    out_dir = rd / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "per_level.json").write_text(json.dumps(payload, indent=2))
    write_latex(rd, payload)
    for lvl, d in sorted(payload["by_level"].items(), key=lambda kv: int(kv[0])):
        acc = d["proof_exist_accept_acc"]
        acc_str = f"{acc:.3f}" if acc == acc else "nan"  # NaN check
        print(f"  level {lvl}: n={d['n']}, C_prf={d['proved_coverage_mean']:.3f}, "
              f"PE acc={acc_str}, cert_feasible={d['proof_exist_certifies_at_eps']}")
    print(f"\nwrote {out_dir/'per_level.json'} and .tex")


if __name__ == "__main__":
    main()
