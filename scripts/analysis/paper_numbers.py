#!/usr/bin/env python3
"""Summary of a run's formal signal.

Computes, from observations.jsonl alone:
  * artifact status totals,
  * proved/typed coverage and the C_prf distribution over the threshold grid,
  * proved-winner precision (does the proof-winning class == the gold class?),
  * CLASS-LEVEL DISCRIMINATION: the decisive "is the proof signal real or a
    second vacuity" test: for each problem, which answer classes Lean proved, and
    whether proofs fire only for the correct answer (clean), for correct+wrong
    (ambiguous), or only for a wrong answer (misleading),
  * self-consistency accuracy baseline for comparison.

Usage:
  python paper_numbers.py --obs runs/<run>/observations.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else f"{n}/0 (—)"


def proved_classes(row: dict) -> set[str]:
    out: set[str] = set()
    for cls, arts in row.get("metadata", {}).get("artifacts_detail", {}).items():
        if any(a.get("status") == "proved" for a in arts):
            out.add(str(cls))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.obs) if l.strip()]
    n = len(rows)

    # ---- artifact-level status totals ----
    st = Counter()
    src = Counter()
    for r in rows:
        for cls, arts in r.get("metadata", {}).get("artifacts_detail", {}).items():
            for a in arts:
                st[a.get("status")] += 1
                src[a.get("source")] += 1
    n_art = sum(st.values())

    # ---- coverage diagnostics ----
    cprf = [float(r.get("diagnostics", {}).get("proved_coverage") or 0.0) for r in rows]
    ctyp = [float(r.get("diagnostics", {}).get("typed_coverage") or 0.0) for r in rows]
    n_cprf_pos = sum(1 for x in cprf if x > 0)
    n_ctyp_pos = sum(1 for x in ctyp if x > 0)

    # ---- proved-winner precision ----
    winner_problems = 0
    winner_correct = 0
    for r in rows:
        w = r.get("diagnostics", {}).get("proved_winner")
        ref = r.get("metadata", {}).get("reference_class")
        if w is not None:
            winner_problems += 1
            if ref is not None and str(w) == str(ref):
                winner_correct += 1

    # ---- CLASS-LEVEL DISCRIMINATION ----
    cat = Counter()  # no_proof / clean / ambiguous / misleading
    n_any_wrong_proved = 0
    for r in rows:
        ref = r.get("metadata", {}).get("reference_class")
        pc = proved_classes(r)
        if not pc:
            cat["no_proof"] += 1
            continue
        ref = str(ref) if ref is not None else None
        wrong = {c for c in pc if c != ref}
        if wrong:
            n_any_wrong_proved += 1
        if ref in pc and not wrong:
            cat["clean (only correct class proved)"] += 1
        elif ref in pc and wrong:
            cat["ambiguous (correct + wrong proved)"] += 1
        else:  # ref not proved, but some wrong class is
            cat["misleading (only wrong class proved)"] += 1
    n_proof = n - cat["no_proof"]

    # ---- self-consistency accuracy baseline ----
    sc_correct = sc_total = 0
    for r in rows:
        ref = r.get("metadata", {}).get("reference_class")
        sc = r.get("decisions", {}).get("self_consistency_class")
        if ref is not None and sc is not None:
            sc_total += 1
            sc_correct += int(str(sc) == str(ref))

    # ----------------------------------------------------------------- report
    P = print
    P("=" * 72)
    P(f"RUN-1 PAPER NUMBERS  —  {n} problems, {n_art} artifacts")
    P("=" * 72)
    P("\n## Artifact status totals")
    for k, v in st.most_common():
        P(f"   {k:14s} {pct(v, n_art)}")
    P(f"   sources: {dict(src)}")

    P("\n## Coverage (per problem)")
    P(f"   typed_coverage  > 0 : {pct(n_ctyp_pos, n)}   mean={statistics.mean(ctyp):.3f} median={statistics.median(ctyp):.3f}")
    P(f"   proved_coverage > 0 : {pct(n_cprf_pos, n)}   mean={statistics.mean(cprf):.3f} median={statistics.median(cprf):.3f}")

    P("\n## C_prf distribution over the grid thresholds")
    for thr in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]:
        m = sum(1 for x in cprf if x >= thr) if thr > 0 else sum(1 for x in cprf if x > 0)
        label = f">={thr}" if thr > 0 else ">0"
        P(f"   C_prf {label:6s}: {pct(m, n)}")

    P("\n## Proved-winner precision  (does the proof-winning class == gold?)")
    P(f"   problems with a proved_winner : {pct(winner_problems, n)}")
    P(f"   of those, winner == gold class: {pct(winner_correct, winner_problems)}")

    P("\n## CLASS-LEVEL DISCRIMINATION  (decisive: do proofs fire only for the correct answer?)")
    for k in ["clean (only correct class proved)",
              "ambiguous (correct + wrong proved)",
              "misleading (only wrong class proved)",
              "no_proof"]:
        P(f"   {k:38s} {pct(cat[k], n)}")
    if n_proof:
        clean = cat["clean (only correct class proved)"]
        P(f"   --> among problems WITH any proof ({n_proof}): clean = {pct(clean, n_proof)}")
        P(f"   --> problems where a WRONG class got proved: {pct(n_any_wrong_proved, n)}")

    P("\n## Baseline")
    P(f"   self-consistency accuracy: {pct(sc_correct, sc_total)}")
    P("=" * 72)


if __name__ == "__main__":
    main()
