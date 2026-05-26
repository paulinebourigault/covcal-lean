"""Smoke test for scripts/render_tables.py.

We invoke the module's `main()` via subprocess so the test exercises the same entry
point a user would. Inputs are a tiny synthetic observations.jsonl + metrics.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_render_tables_smoke(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    obs = [
        {
            "problem_id": "P1",
            "metadata": {"domain": "algebra", "reference_class": "2"},
            "prover_budget_seconds": 10.0,
            "classes": [
                {"label": "2", "weight": 0.8, "candidate_indices": [0, 1, 2, 3],
                 "artifacts": [{"status": "proved", "source": "template"}]},
                {"label": "3", "weight": 0.2, "candidate_indices": [4],
                 "artifacts": [{"status": "typechecked", "source": "template"}]},
            ],
            "candidates": [],
        },
        {
            "problem_id": "P2",
            "metadata": {"domain": "number_theory", "reference_class": "42"},
            "prover_budget_seconds": 10.0,
            "classes": [
                {"label": "42", "weight": 0.6, "candidate_indices": [0, 1, 2],
                 "artifacts": [{"status": "proved", "source": "template"}]},
                {"label": "0", "weight": 0.4, "candidate_indices": [3, 4],
                 "artifacts": [{"status": "illtyped", "source": "autoformalizer"}]},
            ],
            "candidates": [],
        },
    ]
    (run_dir / "observations.jsonl").write_text("\n".join(json.dumps(o) for o in obs))
    metrics = {
        "thresholds": {"selected": [0.5, 0.5, 0.0], "epsilon": 0.1, "delta": 0.05},
        "methods": {
            "self_consistency": {"overall_accuracy": 0.9, "accepted_accuracy": 0.9,
                                 "abstention_rate": 0.0, "selective_risk": 0.1,
                                 "accepted_fraction": 1.0, "n_total": 100, "n_accepted": 100,
                                 "risk_upper_bound_95": 0.15},
            "covcal": {"overall_accuracy": 0.6, "accepted_accuracy": 0.95,
                       "abstention_rate": 0.4, "selective_risk": 0.05,
                       "accepted_fraction": 0.6, "n_total": 100, "n_accepted": 60,
                       "risk_upper_bound_95": 0.10},
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "render_tables.py"
    res = subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )
    assert res.returncode == 0, f"stderr: {res.stderr}"
    out_dir = run_dir / "tables"
    for name in ("tab_main.tex", "tab_cliff_prf.tex", "tab_cliff_margin.tex",
                 "tab_domain.tex", "tab_taxonomy.tex"):
        p = out_dir / name
        assert p.exists()
        body = p.read_text(encoding="utf-8")
        assert "\\begin{table}" in body and "\\end{table}" in body
    # Sanity: main table mentions the methods.
    main_body = (out_dir / "tab_main.tex").read_text(encoding="utf-8")
    assert "Self-consistency" in main_body
    assert "methodname" in main_body
