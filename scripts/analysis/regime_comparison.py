"""Regime comparison.

Reads both bootstrap_seeds.json (Bonferroni) and bootstrap_seeds_devcal.json
(dev-then-cal) for every run under runs/ that has both, produces a summary + a table.

Usage:
    python scripts/analysis/regime_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ORDER = [
    ("main",       "Qwen2.5-Coder-7B (main)",  151),
    ("goedel8b",   "Goedel-Prover-V2-8B",      151),
    ("goedel32b",  "Goedel-Prover-V2-32B",     151),
]


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(j: dict | None) -> dict:
    if j is None:
        return {"available": False}
    sels = j["per_seed_calibration"]
    n_rej = sum(1 for v in sels.values() if v.get("reject_all"))
    ubs = [v.get("risk_upper_bound") for v in sels.values() if not v.get("reject_all")]
    if ubs:
        return {
            "available": True,
            "K": len(sels),
            "n_reject_all": n_rej,
            "ub_min": min(ubs),
            "ub_max": max(ubs),
            "ub_med": sorted(ubs)[len(ubs) // 2],
        }
    return {"available": True, "K": len(sels), "n_reject_all": n_rej,
            "ub_min": None, "ub_max": None, "ub_med": None}


def main() -> None:
    rows = []
    for run_dir, label, n_cal in RUN_ORDER:
        rd = REPO_ROOT / "runs" / run_dir
        bonf = _read(rd / "analysis" / "bootstrap_seeds.json")
        devcal = _read(rd / "analysis" / "bootstrap_seeds_devcal.json")
        rows.append({"run": run_dir, "label": label, "n_cal": n_cal,
                     "bonf": _stats(bonf), "devcal": _stats(devcal)})

    # markdown
    print("Run | n_cal | Bonferroni reject@K=20 | dev-then-cal reject@K=20 | Bonf UB med | DTC UB med")
    print(":-- | --:| --: | --: | --: | --:")
    for r in rows:
        b = r["bonf"]; d = r["devcal"]
        def fmt(s, key):
            if not s.get("available"):
                return "--"
            if s[key] is None:
                return "--"
            return f"{s[key]:.3f}" if isinstance(s[key], float) else str(s[key])
        bonf_rej = f"{b['n_reject_all']}/{b['K']}" if b.get("available") else "--"
        dtc_rej = f"{d['n_reject_all']}/{d['K']}" if d.get("available") else "--"
        print(f"{r['label']} | {r['n_cal']} | {bonf_rej} | {dtc_rej} | "
              f"{fmt(b, 'ub_med')} | {fmt(d, 'ub_med')}")

    # LaTeX
    out_dir = REPO_ROOT / "docs" / "paper"
    tex_lines = [
        "\\begin{table*}[t]",
        "\\centering\\small\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\multirow{2}{*}{Run} & \\multirow{2}{*}{$n_{\\text{cal}}$} "
        "& \\multicolumn{2}{c}{Bonferroni ($\\alpha=\\delta/|\\calT|$)} "
        "& \\multicolumn{2}{c}{Dev-then-cal ($\\alpha=\\delta$)} & UB ratio \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "& & rej@$K\\!=\\!20$ & med UB & rej@$K\\!=\\!20$ & med UB & (med) \\\\",
        "\\midrule",
    ]
    for r in rows:
        b = r["bonf"]; d = r["devcal"]
        def fmt_med(s):
            if not s.get("available") or s.get("ub_med") is None:
                return "--"
            return f"{s['ub_med']:.3f}"
        bonf_rej = f"{b['n_reject_all']}/{b['K']}" if b.get("available") else "--"
        dtc_rej = f"{d['n_reject_all']}/{d['K']}" if d.get("available") else "n/a"
        ratio = "--"
        if (b.get("ub_med") not in (None, 0) and d.get("ub_med") not in (None, 0)
                and b.get("available") and d.get("available")):
            ratio = f"{d['ub_med'] / b['ub_med']:.2f}\\(\\times\\)"
        tex_lines.append(
            f"{r['label']} & {r['n_cal']} & {bonf_rej} & {fmt_med(b)} "
            f"& {dtc_rej} & {fmt_med(d)} & {ratio} \\\\"
        )
    tex_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Side-by-side comparison of the two valid certificate regimes "
        "across all eight runs ($K\\!=\\!20$ bootstrap of the dev/cal/test partition). "
        "Bonferroni applies $\\alpha=\\delta/|\\calT|=4\\!\\times\\!10^{-4}$ over the "
        "$125$-cell grid -- valid under arbitrary dependence between cells. "
        "Dev-then-cal applies $\\alpha=\\delta=0.05$ via Theorem~\\ref{thm:devcal} -- "
        "valid because $\\hat\\tau$ depends only on the dev split. Med UB is the "
        "median selective-risk upper bound across the seeds where the certificate is "
        "feasible. AMC has $\\mathrm{dev\\,frac}\\!=\\!0$ in its original split so "
        "dev-then-cal is not applicable (`n/a').}",
        "\\label{tab:regime-comparison}",
        "\\end{table*}",
    ]
    out_tex = REPO_ROOT / "runs" / "regime_comparison.tex"
    out_tex.write_text("\n".join(tex_lines))
    print(f"\nwrote {out_tex}")


if __name__ == "__main__":
    main()
