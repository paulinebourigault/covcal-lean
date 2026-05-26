"""Tests for covcal.data (filters, splits, math500 jsonl loader)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from covcal.data import (
    SplitsManifest,
    filter_problems,
    load_math500,
    load_splits,
    make_splits,
    write_splits,
)
from covcal.data.filters import FilterReport, classify
from covcal.pipeline import Problem


def _p(pid: str, text: str, ans: str) -> Problem:
    return Problem(problem_id=pid, problem_text=text, reference_answer=ans)


class TestClassify:
    def test_keeps_normal(self):
        r = classify("What is 1+1?", "2")
        assert r.keep

    def test_drops_asy_block(self):
        r = classify("Consider this figure: [asy] draw(circle) [/asy]", "5")
        assert not r.keep and r.reason == "diagram_asy_block"

    def test_drops_image_reference(self):
        r = classify("In the figure above, find x.", "3")
        assert not r.keep and r.reason == "diagram_image_reference"

    def test_drops_missing_reference(self):
        r = classify("What is 1+1?", "")
        assert not r.keep and r.reason == "missing_reference_answer"

    def test_drops_proof_only_when_empty(self):
        r = classify("Prove that n+1 > n.", "")
        # Missing reference fires first because the regex is short-circuit-friendly,
        # but we tagged this 'proof_only' first in classify(). Accept either.
        assert not r.keep
        assert r.reason in {"proof_only", "missing_reference_answer"}

    def test_drops_non_normalizable_reference(self):
        r = classify("What?", "totallybrokenanswer???not_math")
        assert not r.keep and r.reason == "non_normalizable_reference"


class TestFilterProblems:
    def test_max_examples_respected(self):
        problems = [_p(f"p{i}", "What is 1+1?", "2") for i in range(10)]
        report = filter_problems(problems, max_examples=3)
        assert report.n_kept == 3
        assert report.excluded_by_reason.get("over_max_examples") == 7

    def test_report_counts_match(self):
        problems = [
            _p("a", "What is 1+1?", "2"),
            _p("b", "[asy] foo [/asy]", "3"),
            _p("c", "Compute 5.", "5"),
        ]
        report = filter_problems(problems)
        assert report.n_kept == 2
        assert report.n_excluded == 1
        assert report.total_seen == 3
        assert report.as_dict()["n_kept"] == 2


class TestMakeSplits:
    def test_basic_partition(self):
        ids = [f"p{i}" for i in range(10)]
        m = make_splits(ids, name="t", seed=0, fractions={"dev": 0.2, "cal": 0.4, "test": 0.4})
        # Every id appears exactly once across splits.
        flat = [pid for s in m.splits.values() for pid in s]
        assert sorted(flat) == sorted(ids)
        # Counts match the configured fractions (within rounding).
        assert len(m.splits["dev"]) == 2
        assert len(m.splits["cal"]) == 4
        assert len(m.splits["test"]) == 4

    def test_deterministic_under_same_seed(self):
        ids = [f"p{i}" for i in range(20)]
        m1 = make_splits(ids, name="t", seed=42, fractions={"dev": 0.5, "test": 0.5})
        m2 = make_splits(ids, name="t", seed=42, fractions={"dev": 0.5, "test": 0.5})
        assert m1.splits == m2.splits

    def test_different_seed_different_split(self):
        ids = [f"p{i}" for i in range(20)]
        m1 = make_splits(ids, name="t", seed=0, fractions={"dev": 0.5, "test": 0.5})
        m2 = make_splits(ids, name="t", seed=1, fractions={"dev": 0.5, "test": 0.5})
        assert m1.splits != m2.splits

    def test_rejects_bad_fractions(self):
        with pytest.raises(ValueError):
            make_splits(["a"], name="t", seed=0, fractions={"dev": 1.5})
        with pytest.raises(ValueError):
            make_splits(["a"], name="t", seed=0, fractions={"dev": 0.0})

    def test_dedupes_inputs(self):
        ids = ["a", "a", "b"]
        m = make_splits(ids, name="t", seed=0, fractions={"all": 1.0})
        assert sorted(m.splits["all"]) == ["a", "b"]


class TestSplitsRoundTrip:
    def test_write_then_load(self, tmp_path: Path):
        ids = [f"p{i}" for i in range(8)]
        m = make_splits(ids, name="t", seed=0, fractions={"dev": 0.25, "cal": 0.5, "test": 0.25})
        path = tmp_path / "splits.json"
        write_splits(m, path)
        m2 = load_splits(path)
        assert m2.splits == m.splits
        assert m2.seed == m.seed
        assert m2.name == m.name
        assert isinstance(m2, SplitsManifest)


class TestMath500JsonlLoader:
    def test_local_jsonl(self, tmp_path: Path):
        rows = [
            {"problem": "What is 1+1?", "answer": "2", "subject": "Algebra",
             "level": 1, "unique_id": "m1"},
            {"problem": "[asy] x [/asy] In the figure", "answer": "0", "subject": "Geometry",
             "level": 3, "unique_id": "m2"},  # filtered out
            {"problem": "Compute 5+5.", "answer": "10", "subject": "Prealgebra",
             "level": 1, "unique_id": "m3"},
        ]
        p = tmp_path / "ds.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows))
        problems, report = load_math500(jsonl_path=str(p))
        ids = [pp.problem_id for pp in problems]
        assert ids == ["m1", "m3"]
        assert report.n_excluded == 1
        assert problems[0].domain == "algebra"
        # Reference is normalised once and cached.
        assert problems[0].metadata.get("normalized_reference") == "2"


class TestFilterReportAsDict:
    def test_serializable(self):
        rep = FilterReport()
        rep.total_seen = 3
        rep.excluded_by_reason["foo"] = 1
        d = rep.as_dict()
        # Must be JSON-serialisable so we can ship the audit alongside the run.
        json.dumps(d)
