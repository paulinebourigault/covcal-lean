"""Reliability diagrams.

Bins predictions by confidence and plots empirical accuracy vs predicted confidence
with 95% Wilson CIs. We produce three curves on the same axes:

- self-consistency (binned by top-class weight Q_c)
- confidence-only abstention (same Q_c)
- CovCal selector (binned by C_prf * (1 + M)/2 — a coverage-weighted score)

Usage:
    python scripts/analysis/reliability.py --run-dir runs/main [--n-bins 8]

Outputs:
    runs/<rd>/analysis/reliability.json
    runs/<rd>/analysis/reliability.pdf  + .png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from _common import (
    load_observations,
    load_splits_dict,
    split_observations,
    wilson_ci,
)
from covcal.diagnostics import compute_diagnostics


def _top_class(obs) -> tuple[str | None, float, float, float, float]:
    """Return (top_class_label, top_weight Q_c, C_typ, C_prf, M) for a problem."""
    if not obs.classes:
        return None, 0.0, 0.0, 0.0, 0.0
    top = max(obs.classes, key=lambda c: c.weight)
    d = compute_diagnostics(obs)
    return (
        top.label,
        float(top.weight),
        float(d.typed_coverage),
        float(d.proved_coverage),
        float(d.margin),
    )


def reliability_bins(scores: list[float], correct: list[int], n_bins: int) -> list[dict]:
    """Equal-width binning over [0, 1] (or the actual range) with Wilson CIs."""
    if not scores:
        return []
    lo, hi = 0.0, max(1.0, max(scores) + 1e-9)
    edges = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]
    bins = []
    for i in range(n_bins):
        in_bin = [(s, c) for s, c in zip(scores, correct, strict=False)
                  if edges[i] <= s < edges[i + 1] or (i == n_bins - 1 and s == edges[i + 1])]
        if not in_bin:
            bins.append({"low": edges[i], "high": edges[i + 1], "n": 0,
                         "score_mean": float("nan"), "acc": float("nan"),
                         "acc_lo": float("nan"), "acc_hi": float("nan")})
            continue
        s_mean = sum(s for s, _ in in_bin) / len(in_bin)
        n_c = sum(c for _, c in in_bin)
        acc = n_c / len(in_bin)
        ci_lo, ci_hi = wilson_ci(n_c, len(in_bin))
        bins.append({
            "low": edges[i], "high": edges[i + 1], "n": len(in_bin),
            "score_mean": s_mean, "acc": acc, "acc_lo": ci_lo, "acc_hi": ci_hi,
        })
    return bins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--n-bins", type=int, default=8)
    args = ap.parse_args()
    rd = args.run_dir.resolve()
    obs = load_observations(rd)
    splits = load_splits_dict(rd / "splits.json")
    parts = split_observations(obs, splits)
    test_obs = parts.get("test", obs)
    print(f"  using {len(test_obs)} test observations")

    sc_scores, sc_correct = [], []
    co_scores, co_correct = [], []
    cc_scores, cc_correct = [], []
    for o in test_obs:
        label, q, c_typ, c_prf, M = _top_class(o)
        ref = o.metadata.get("reference_class")
        is_correct = int(label is not None and label == ref)
        if label is not None:
            sc_scores.append(q)
            sc_correct.append(is_correct)
            co_scores.append(q)
            co_correct.append(is_correct)
        cc_score = c_prf * (1.0 + max(0.0, M)) / 2.0
        cc_scores.append(cc_score)
        cc_correct.append(is_correct)

    bins_sc = reliability_bins(sc_scores, sc_correct, args.n_bins)
    bins_co = reliability_bins(co_scores, co_correct, args.n_bins)
    bins_cc = reliability_bins(cc_scores, cc_correct, args.n_bins)

    out_dir = rd / "analysis"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "run_dir": str(rd),
        "n_bins": args.n_bins,
        "self_consistency": bins_sc,
        "confidence_only": bins_co,
        "covcal_score": bins_cc,
    }
    (out_dir / "reliability.json").write_text(json.dumps(payload, indent=2))

    # Plot
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="perfect calibration")
    for bins, color, label in [
        (bins_sc, "C0", "Self-consistency (top $Q_c$)"),
        (bins_cc, "C1", "CovCal $C_{\\mathrm{prf}}\\cdot(1+M)/2$"),
    ]:
        xs = [b["score_mean"] for b in bins if b["n"] > 0]
        ys = [b["acc"] for b in bins if b["n"] > 0]
        lo = [b["acc"] - b["acc_lo"] for b in bins if b["n"] > 0]
        hi = [b["acc_hi"] - b["acc"] for b in bins if b["n"] > 0]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=color, label=label,
                    capsize=3, lw=1.2, ms=4)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Reliability diagram — {rd.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.pdf")
    fig.savefig(out_dir / "reliability.png", dpi=150)
    print(f"  → {out_dir/'reliability.json'} + .pdf + .png")


if __name__ == "__main__":
    main()
