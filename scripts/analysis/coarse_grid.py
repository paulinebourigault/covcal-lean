"""Recalibrate with the coarser 3x3x3 threshold grid.

The default 5x5x5=125 grid eats a Bonferroni penalty alpha/cell = 0.05/125 = 4e-4.
A 3x3x3=27 grid gets alpha/cell = 0.05/27 ≈ 1.85e-3, which is ~4.6x less stringent
per cell. This may flip some reject_all verdicts to certifies, at the cost of
coarser threshold resolution.

Usage:
    python scripts/analysis/coarse_grid.py \\
        --run-dir runs/main [--run-dir runs/goedel8b ...]

Outputs (per run dir):
    runs/<rd>/analysis/coarse_grid.json
    runs/<rd>/analysis/coarse_grid.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import (
    calibrate_in_memory,
    coarse_grid_27,
    default_grid_125,
    evaluate_in_memory,
    load_observations,
    load_splits_dict,
    split_observations,
)


def read_eps_delta(run_dir: Path) -> tuple[float, float]:
    md = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    cal = (md.get("config", md)).get("calibration", {})
    return float(cal.get("epsilon", 0.15)), float(cal.get("delta", 0.05))


def run_one(run_dir: Path) -> dict:
    eps, delta = read_eps_delta(run_dir)
    obs = load_observations(run_dir)
    splits = load_splits_dict(run_dir / "splits.json")
    parts = split_observations(obs, splits)
    cal, test = parts.get("cal", []), parts.get("test", [])

    # Re-run both grids so the comparison is matched (in case the existing
    # thresholds.json reflects an earlier grid definition).
    calib_125 = calibrate_in_memory(cal, default_grid_125(), eps, delta)
    metrics_125 = evaluate_in_memory(test, calib_125)
    calib_27 = calibrate_in_memory(cal, coarse_grid_27(), eps, delta)
    metrics_27 = evaluate_in_memory(test, calib_27)

    return {
        "epsilon": eps,
        "delta": delta,
        "n_cal": len(cal),
        "n_test": len(test),
        "grid_125": {"calibrate": calib_125, "metrics": metrics_125},
        "grid_27": {"calibrate": calib_27, "metrics": metrics_27},
    }


def write_latex(run_dir: Path, payload: dict) -> None:
    rd_name = run_dir.name
    eps = payload["epsilon"]
    c125 = payload["grid_125"]["calibrate"]
    c27 = payload["grid_27"]["calibrate"]
    m125 = payload["grid_125"]["metrics"]
    m27 = payload["grid_27"]["metrics"]

    def verdict(calib: dict) -> str:
        if calib.get("reject_all"):
            return "reject-all"
        sel = calib.get("selected")
        return (f"$\\hat\\tau=({sel[0]},{sel[1]},{sel[2]})$, "
                f"UB$={calib['risk_upper_bound']:.3f}$")

    rows = []
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\small\\setlength{\\tabcolsep}{3pt}")
    rows.append("\\resizebox{\\columnwidth}{!}{%")
    rows.append("\\begin{tabular}{lcccc}")
    rows.append("\\toprule")
    rows.append("Grid & $|\\calT|$ & $\\alpha/$cell & Calibrate & "
                "CovCal acc. frac. \\\\")
    rows.append("\\midrule")
    cov125 = m125.get("covcal", {}).get("accepted_fraction")
    cov27 = m27.get("covcal", {}).get("accepted_fraction")
    rows.append(f"$5\\times 5\\times 5$ (main) & {c125['grid_size']} & "
                f"{c125['per_threshold_alpha']:.2g} & {verdict(c125)} & "
                f"{cov125 if cov125 is None else f'{cov125:.3f}'} \\\\")
    rows.append(f"$3\\times 3\\times 3$ (coarse) & {c27['grid_size']} & "
                f"{c27['per_threshold_alpha']:.2g} & {verdict(c27)} & "
                f"{cov27 if cov27 is None else f'{cov27:.3f}'} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}}")
    rows.append(f"\\caption{{Threshold-grid density sensitivity for the {rd_name} run "
                f"($n_{{\\mathrm{{cal}}}}={payload['n_cal']}$, "
                f"$n_{{\\mathrm{{test}}}}={payload['n_test']}$, $\\epsilon={eps}$). "
                f"A coarser grid reduces the Bonferroni penalty per cell, sometimes "
                f"flipping reject-all to a feasible cell, at the cost of cruder "
                f"threshold resolution.}}")
    rows.append(f"\\label{{tab:grid-{rd_name}}}")
    rows.append("\\end{table}")
    (run_dir / "analysis" / "coarse_grid.tex").write_text("\n".join(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, action="append", required=True,
                    help="Repeat for each run dir; e.g. --run-dir runs/main ...")
    args = ap.parse_args()

    for rd in args.run_dir:
        rd = rd.resolve()
        print(f"\n=== {rd.name} ===")
        payload = run_one(rd)
        out_dir = rd / "analysis"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "coarse_grid.json").write_text(json.dumps(payload, indent=2))
        c125 = payload["grid_125"]["calibrate"]
        c27 = payload["grid_27"]["calibrate"]
        print(f"  125-cell: {'REJECT' if c125['reject_all'] else c125['selected']}  "
              f"UB={c125['risk_upper_bound']:.3f}")
        print(f"   27-cell: {'REJECT' if c27['reject_all'] else c27['selected']}  "
              f"UB={c27['risk_upper_bound']:.3f}")
        write_latex(rd, payload)
        print(f"  → {out_dir/'coarse_grid.json'} + .tex")


if __name__ == "__main__":
    main()
