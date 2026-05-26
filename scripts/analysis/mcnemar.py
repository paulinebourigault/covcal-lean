"""McNemar paired significance tests.

For two selectors evaluated on the same test items, McNemar gives a paired test
on the difference in correctness. We report it for the three comparisons:

  - CovCal vs proof-existence  (on the union of accepted predictions)
  - CovCal+fallback vs self-consistency  (on overall correctness)
  - confidence-only vs CovCal  (on accepted-set risk; restrict to intersection)

We use the exact binomial form of McNemar (no continuity correction) for the
small discordant counts likely on n_test ≈ 150.

Usage:
    python scripts/analysis/mcnemar.py --run-dir runs/main

Outputs:
    runs/<rd>/analysis/mcnemar.json
    runs/<rd>/analysis/mcnemar.tex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _common import (
    calibrate_in_memory,
    default_grid_125,
    evaluate_in_memory,
    load_observations,
    load_splits_dict,
    split_observations,
)
from covcal.selectors import (
    ConfidenceOnly,
    CovCal,
    CovCalPlusFallback,
    proof_existence_abstention,
    self_consistency,
)
from covcal.types import Thresholds


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b, c are the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    from scipy.stats import binomtest  # type: ignore
    return float(binomtest(min(b, c), n, p=0.5, alternative="two-sided").pvalue)


def _correct_for(selector, obs_list, restrict_accept: bool = False) -> list[int | None]:
    """Per-problem correctness (1/0) or None (abstained, only if restrict_accept)."""
    out: list[int | None] = []
    for o in obs_list:
        sel = selector(o)
        ref = o.metadata.get("reference_class")
        if sel.abstained:
            out.append(None if restrict_accept else 0)
        else:
            out.append(1 if sel.selected == ref else 0)
    return out


def mcnemar_for_pair(a_corr: list[int | None], b_corr: list[int | None], restrict_both_accept: bool = False) -> dict:
    n = len(a_corr)
    pairs = []
    for x, y in zip(a_corr, b_corr, strict=False):
        if restrict_both_accept and (x is None or y is None):
            continue
        pairs.append((0 if x is None else x, 0 if y is None else y))
    # 2x2 table: a=both correct, b=A correct B wrong, c=A wrong B correct, d=both wrong
    a = sum(1 for x, y in pairs if x == 1 and y == 1)
    b = sum(1 for x, y in pairs if x == 1 and y == 0)
    c = sum(1 for x, y in pairs if x == 0 and y == 1)
    d = sum(1 for x, y in pairs if x == 0 and y == 0)
    p = exact_mcnemar_p(b, c)
    return {
        "n_paired": len(pairs),
        "a_both_correct": a,
        "b_A_correct_B_wrong": b,
        "c_A_wrong_B_correct": c,
        "d_both_wrong": d,
        "p_value": p,
        "discordant_total": b + c,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    rd = args.run_dir.resolve()
    print(f"=== {rd.name} ===")

    obs = load_observations(rd)
    splits = load_splits_dict(rd / "splits.json")
    parts = split_observations(obs, splits)
    test_obs = parts.get("test", [])

    # Rebuild the same thresholds the on-line calibrate produced.
    md = json.loads((rd / "metadata.json").read_text(encoding="utf-8"))
    cal_cfg = (md.get("config", md)).get("calibration", {})
    eps, delta = float(cal_cfg.get("epsilon", 0.15)), float(cal_cfg.get("delta", 0.05))
    calib = calibrate_in_memory(parts.get("cal", []), default_grid_125(), eps, delta)
    if calib.get("reject_all") or not calib.get("selected"):
        print("  calibrate reject_all → CovCal predictions are all abstain;\n"
              "  McNemar comparisons involving CovCal will mostly count CovCal as 'wrong'.")
        tau = None
    else:
        t = calib["selected"]
        tau = Thresholds(typ=float(t[0]), prf=float(t[1]), margin=float(t[2]))

    if tau is None:
        sel_covcal = lambda o: type("X", (), {"selected": None, "abstained": True})()
        sel_covcal_fb = self_consistency  # falls back to SC for the +fallback variant
    else:
        sel_covcal = CovCal(tau)
        sel_covcal_fb = CovCalPlusFallback(tau)

    corr_sc = _correct_for(self_consistency, test_obs)
    corr_co = _correct_for(ConfidenceOnly(0.5), test_obs)
    corr_pe = _correct_for(proof_existence_abstention, test_obs)
    corr_cc = _correct_for(sel_covcal, test_obs, restrict_accept=False)
    corr_ccfb = _correct_for(sel_covcal_fb, test_obs)

    results = {
        "CovCal vs Proof-existence (overall correctness)": mcnemar_for_pair(corr_cc, corr_pe),
        "CovCal+fallback vs Self-consistency (overall)": mcnemar_for_pair(corr_ccfb, corr_sc),
        "Confidence-only vs CovCal (overall correctness)": mcnemar_for_pair(corr_co, corr_cc),
    }

    out_dir = rd / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "mcnemar.json").write_text(json.dumps({
        "run_dir": str(rd), "n_test": len(test_obs),
        "calib_reject_all": calib.get("reject_all", False),
        "calib_selected": calib.get("selected"),
        "comparisons": results,
    }, indent=2))

    # LaTeX
    lines = ["\\begin{table}[t]", "\\centering\\small\\setlength{\\tabcolsep}{3pt}",
             "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lcccc}",
             "\\toprule",
             "Comparison & $b$ & $c$ & $b+c$ & $p$ (exact) \\\\",
             "\\midrule"]
    for name, r in results.items():
        p = r["p_value"]
        p_str = f"$<10^{{-3}}$" if p < 1e-3 else f"{p:.3f}"
        lines.append(f"{name} & {r['b_A_correct_B_wrong']} & {r['c_A_wrong_B_correct']} & "
                     f"{r['discordant_total']} & {p_str} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}",
              f"\\caption{{Exact McNemar tests on $n_{{\\mathrm{{test}}}}={len(test_obs)}$ paired "
              f"predictions. $b$ is the count where the first selector is correct and the second "
              f"is not; $c$ is the reverse. $p$ is the two-sided exact binomial probability of the "
              f"observed discordance under the null of no difference.}}",
              "\\label{tab:mcnemar}", "\\end{table}"]
    (out_dir / "mcnemar.tex").write_text("\n".join(lines))

    for name, r in results.items():
        print(f"  {name}: b={r['b_A_correct_B_wrong']} c={r['c_A_wrong_B_correct']} "
              f"p={r['p_value']:.4f}")
    print(f"\nwrote {out_dir/'mcnemar.json'} and .tex")


if __name__ == "__main__":
    main()
