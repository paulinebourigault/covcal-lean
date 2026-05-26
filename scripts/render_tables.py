#!/usr/bin/env python
"""
Inputs (in a run directory):
  - observations.jsonl   : produced by `covcal pipeline`
  - metrics.json         : produced by `covcal evaluate` (after `covcal calibrate`)
  - splits.json          : the splits manifest used to bound the test set

Outputs (written to <run_dir>/tables/):
  - tab_main.tex         : main answer-selection results
  - tab_cliff.tex        : coverage-cliff diagnostic
  - tab_domain.tex       : per-domain coverage
  - tab_taxonomy.tex     : manual failure audit; placeholder if no audit JSON


Usage:
  uv run python scripts/render_tables.py --run-dir runs/minimal
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# --- helpers ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _latex_escape(s: object) -> str:
    s = str(s)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _fmt(x: float | None, *, pct: bool = False, digits: int = 3) -> str:
    if x is None:
        return "--"
    if pct:
        return f"{100 * x:.{digits - 1 if digits > 0 else 0}f}\\%"
    return f"{x:.{digits}f}"


# --- Table 1: main results -------------------------------------------------------------

_MAIN_METHOD_ORDER = [
    ("self_consistency", "Self-consistency"),
    ("confidence_only", "Confidence-only abst."),
    ("raw_lean_plus_fallback", "Raw Lean + fallback"),
    ("proof_existence", "Proof-existence abst."),
    ("typed_coverage_only", "Typed-coverage only"),
    ("proved_coverage_only", "Proved-coverage only"),
    ("margin_only", "Margin-only"),
    ("covcal", r"\methodname{}"),
    ("covcal_plus_fallback", r"\methodname{}+fallback"),
]


def render_table_main(metrics: dict[str, Any]) -> str:
    methods = metrics.get("methods", {})
    rows = []
    for key, label in _MAIN_METHOD_ORDER:
        m = methods.get(key)
        if m is None:
            rows.append(f"{label} & -- & -- & -- & -- \\\\")
            continue
        rows.append(
            f"{label} & {_fmt(m['overall_accuracy'])} & "
            f"{_fmt(m['accepted_accuracy'])} & "
            f"{_fmt(m['abstention_rate'])} & "
            f"{_fmt(m['risk_upper_bound_95'])} \\\\"
        )
    return _wrap_table(
        "Main answer-selection results",
        ["Method", "Overall", "Accepted", "Abstain", "Risk UB"],
        rows,
        "tab:main",
    )


# --- Table 2: coverage cliff -----------------------------------------------------------

_PRF_CLIFF_BINS = [
    (0.00, 0.25, "$C_{\\prf} < 0.25$"),
    (0.25, 0.50, "$0.25 \\le C_{\\prf} < 0.50$"),
    (0.50, 0.75, "$0.50 \\le C_{\\prf} < 0.75$"),
    (0.75, 1.01, "$C_{\\prf} \\ge 0.75$"),
]

_MARGIN_CLIFF_BINS = [
    (float("-inf"), 0.0, "$M < 0$ or undefined"),
    (0.0, 0.25, "$0 \\le M < 0.25$"),
    (0.25, 0.5, "$0.25 \\le M < 0.5$"),
    (0.5, float("inf"), "$M \\ge 0.5$"),
]


def _bin(value: float, bins: list[tuple[float, float, str]]) -> str | None:
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    return None


def _proved_coverage(obs: dict[str, Any]) -> float:
    # Prefer the precomputed diagnostic if the structured log is present.
    d = obs.get("diagnostics") or {}
    if "proved_coverage" in d:
        return float(d["proved_coverage"])
    return sum(c["weight"] for c in obs["classes"]
               if any(a["status"] == "proved" for a in c.get("artifacts", [])))


def _margin(obs: dict[str, Any]) -> float:
    d = obs.get("diagnostics") or {}
    if "margin" in d and d["margin"] is not None:
        return float(d["margin"])
    # Fallback compute: winner weight - max unresolved class weight (or -inf if no proof).
    proved = [c for c in obs["classes"]
              if any(a["status"] == "proved" for a in c.get("artifacts", []))]
    if not proved:
        return float("-inf")
    proved.sort(key=lambda c: (-c["weight"], c["label"]))
    winner = proved[0]
    rivals = [c["weight"] for c in obs["classes"]
              if c["label"] != winner["label"]
              and not any(a["status"] == "proved" for a in c.get("artifacts", []))]
    return float(winner["weight"]) - (max(rivals) if rivals else 0.0)


def _render_cliff(
    observations: list[dict[str, Any]],
    references: dict[str, str],
    bins: list[tuple[float, float, str]],
    value_fn,
    *,
    caption: str,
    label: str,
    bin_header: str,
) -> str:
    rows_data: dict[str, dict[str, int]] = defaultdict(lambda: {
        "n": 0, "proved": 0, "raw_err": 0, "covcal_acc": 0, "covcal_n": 0
    })
    for obs in observations:
        bin_label = _bin(value_fn(obs), bins)
        if bin_label is None:
            continue
        rd = rows_data[bin_label]
        rd["n"] += 1
        proved_class = _proved_winner_label(obs)
        if proved_class is not None:
            rd["proved"] += 1
            ref = references.get(obs["problem_id"])
            if ref is not None and proved_class != ref:
                rd["raw_err"] += 1
            rd["covcal_n"] += 1
            if ref is not None and proved_class == ref:
                rd["covcal_acc"] += 1
    rows = []
    for _, _, lbl in bins:
        rd = rows_data[lbl]
        n = rd["n"]
        if n == 0:
            rows.append(f"{lbl} & 0 & -- & -- & -- \\\\")
            continue
        prove_rate = rd["proved"] / n
        raw_err_rate = rd["raw_err"] / rd["proved"] if rd["proved"] else None
        cc_acc = rd["covcal_acc"] / rd["covcal_n"] if rd["covcal_n"] else None
        rows.append(
            f"{lbl} & {n} & {_fmt(prove_rate)} & {_fmt(raw_err_rate)} & {_fmt(cc_acc)} \\\\"
        )
    return _wrap_table(
        caption,
        [bin_header, "Ex.", "Proof", "Raw err.", r"\methodname{} acc."],
        rows,
        label,
    )


def render_table_cliff(observations: list[dict[str, Any]], references: dict[str, str]) -> str:
    """C_prf-binned cliff (Table 2)."""
    return _render_cliff(
        observations, references, _PRF_CLIFF_BINS, _proved_coverage,
        caption="Coverage-cliff diagnostic, binned by proved coverage.",
        label="tab:cliff_prf",
        bin_header="Coverage bin",
    )


def render_table_cliff_margin(
    observations: list[dict[str, Any]], references: dict[str, str]
) -> str:
    """Margin-binned cliff variant."""
    return _render_cliff(
        observations, references, _MARGIN_CLIFF_BINS, _margin,
        caption="Coverage-cliff diagnostic, binned by formal margin $M$.",
        label="tab:cliff_margin",
        bin_header="Margin bin",
    )


def _proved_winner_label(obs: dict[str, Any]) -> str | None:
    proved = [
        c for c in obs["classes"]
        if any(a["status"] == "proved" for a in c.get("artifacts", []))
    ]
    if not proved:
        return None
    proved.sort(key=lambda c: (-c["weight"], c["label"]))
    return proved[0]["label"]


# --- Table 3: per-domain coverage ------------------------------------------------------


def render_table_domain(
    observations: list[dict[str, Any]],
    references: dict[str, str],
) -> str:
    by_domain: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "typed_sum": 0.0, "proved_sum": 0.0,
                 "correct": 0, "fail_reason": Counter()}
    )
    for obs in observations:
        domain = (obs.get("metadata") or {}).get("domain") or "other"
        d = by_domain[domain]
        d["n"] += 1
        d["typed_sum"] += sum(
            c["weight"] for c in obs["classes"]
            if any(a["status"] in ("proved", "typechecked", "timeout")
                   for a in c.get("artifacts", []))
        )
        d["proved_sum"] += sum(
            c["weight"] for c in obs["classes"]
            if any(a["status"] == "proved" for a in c.get("artifacts", []))
        )
        winner = _proved_winner_label(obs)
        ref = references.get(obs["problem_id"])
        if winner is not None and ref is not None and winner == ref:
            d["correct"] += 1
        # tally dominant per-class failure reason (heuristic)
        for c in obs["classes"]:
            for a in c.get("artifacts", []):
                if a["status"] != "proved":
                    d["fail_reason"][a["status"]] += 1

    rows = []
    for domain in sorted(by_domain):
        d = by_domain[domain]
        n = d["n"] or 1
        top_fail = d["fail_reason"].most_common(1)
        top_fail_label = _latex_escape(top_fail[0][0]) if top_fail else "--"
        rows.append(
            f"{_latex_escape(domain)} & {_fmt(d['typed_sum'] / n)} & "
            f"{_fmt(d['proved_sum'] / n)} & {_fmt(d['correct'] / n)} & "
            f"{top_fail_label} \\\\"
        )
    return _wrap_table(
        "Domain-level formal coverage",
        ["Domain", "Typed", "Proved", "Acc.", "Top failure"],
        rows,
        "tab:domain",
    )


# --- Table 4: failure taxonomy (from optional manual audit file) -----------------------


def render_table_taxonomy(audit_path: Path | None) -> str:
    if audit_path is None or not audit_path.exists():
        rows = [
            "Bad autoformalization & -- & -- \\\\",
            "Ill-typed Lean statement & -- & -- \\\\",
            "Missing library / import & -- & -- \\\\",
            "Proof-search timeout & -- & -- \\\\",
            "Semantic mismatch & -- & -- \\\\",
            "Actually wrong answer & -- & -- \\\\",
        ]
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        rows = []
        for kind, entry in audit.items():
            count = entry.get("count", 0) if isinstance(entry, dict) else int(entry)
            note = entry.get("note", "") if isinstance(entry, dict) else ""
            rows.append(f"{_latex_escape(kind)} & {count} & {_latex_escape(note)} \\\\")
    return _wrap_table(
        "Manual low-coverage failure audit",
        ["Failure mode", "Count", "Note"],
        rows,
        "tab:taxonomy",
    )


# --- LaTeX wrapper ---------------------------------------------------------------------


def _wrap_table(caption: str, headers: list[str], rows: list[str], label: str) -> str:
    n_cols = len(headers)
    align = "l" + "c" * (n_cols - 1)
    header_line = " & ".join(headers) + " \\\\"
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{align}}}\n"
        "\\toprule\n"
        f"{header_line}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


# --- main ------------------------------------------------------------------------------


def _references_from_observations(
    observations: Iterable[dict[str, Any]]
) -> dict[str, str]:
    refs: dict[str, str] = {}
    for obs in observations:
        ref = (obs.get("metadata") or {}).get("reference_class")
        if ref:
            refs[obs["problem_id"]] = ref
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, default=None,
                        help="Optional manual-audit JSON for Table 4.")
    args = parser.parse_args()

    run_dir = args.run_dir
    obs_path = run_dir / "observations.jsonl"
    metrics_path = run_dir / "metrics.json"
    out_dir = run_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    observations = _read_jsonl(obs_path) if obs_path.exists() else []
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    refs = _references_from_observations(observations)

    files = {
        "tab_main.tex": render_table_main(metrics),
        "tab_cliff_prf.tex": render_table_cliff(observations, refs),
        "tab_cliff_margin.tex": render_table_cliff_margin(observations, refs),
        "tab_domain.tex": render_table_domain(observations, refs),
        "tab_taxonomy.tex": render_table_taxonomy(args.audit_json),
    }
    for name, body in files.items():
        (out_dir / name).write_text(body, encoding="utf-8")
        print(f"wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
