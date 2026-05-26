"""Tests for the mock Lean backend and templates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from covcal.lean import (
    LeanTask,
    MockLeanBackend,
    emit_template_task,
    list_template_kinds,
)
from covcal.lean.mock import task_content_hash
from covcal.types import Status


class TestMockBackend:
    def test_unknown_task_returns_default(self):
        backend = MockLeanBackend()
        task = LeanTask(name="t1", statement="theorem t : 1 = 1", tactics=("rfl",))
        [out] = backend.check([task])
        assert out.status is Status.UNFORMALIZED
        assert out.name == "t1"

    def test_lookup_by_name(self):
        backend = MockLeanBackend()
        backend.add_by_name("t1", Status.PROVED, tactic_used="rfl")
        [out] = backend.check([
            LeanTask(name="t1", statement="theorem t : 1 = 1", tactics=("rfl",))
        ])
        assert out.status is Status.PROVED
        assert out.tactic_used == "rfl"

    def test_lookup_by_statement_hash(self):
        backend = MockLeanBackend()
        stmt = "theorem t : 2 + 2 = 4"
        backend.add_by_statement(stmt, ("norm_num",), Status.PROVED)
        # The name is irrelevant for hash-based lookup.
        [out] = backend.check([
            LeanTask(name="whatever", statement=stmt, tactics=("norm_num",))
        ])
        assert out.status is Status.PROVED

    def test_preserves_order(self):
        backend = MockLeanBackend()
        backend.add_by_name("a", Status.PROVED)
        backend.add_by_name("b", Status.ILLTYPED)
        backend.add_by_name("c", Status.TIMEOUT)
        out = backend.check([
            LeanTask("a", "theorem a : True", ("trivial",)),
            LeanTask("b", "theorem b : False", ("trivial",)),
            LeanTask("c", "theorem c : 1 = 1", ("rfl",)),
        ])
        assert [o.name for o in out] == ["a", "b", "c"]
        assert [o.status for o in out] == [Status.PROVED, Status.ILLTYPED, Status.TIMEOUT]

    def test_history_records_calls(self):
        backend = MockLeanBackend()
        t1 = LeanTask("t1", "theorem t : 1 = 1", ("rfl",))
        t2 = LeanTask("t2", "theorem u : 2 = 2", ("rfl",))
        backend.check([t1, t2])
        backend.check([t1])
        assert len(backend.history) == 3

    def test_fixture_loading(self, tmp_path: Path):
        fixture = tmp_path / "lean_outcomes.json"
        stmt = "theorem foo : 1 = 1"
        h = task_content_hash(LeanTask(name="_", statement=stmt, tactics=("rfl",)))
        fixture.write_text(
            json.dumps(
                [
                    {"name": "ok", "status": "proved", "tactic_used": "norm_num"},
                    {"name": "bad", "status": "illtyped"},
                    {"statement_hash": h, "status": "proved"},
                ]
            )
        )
        backend = MockLeanBackend()
        backend.load_fixture(fixture)
        out_named = backend.check([
            LeanTask("ok", "theorem t : True", ("trivial",)),
            LeanTask("bad", "theorem b : False", ("trivial",)),
        ])
        assert [o.status for o in out_named] == [Status.PROVED, Status.ILLTYPED]
        out_hashed = backend.check([
            LeanTask("anything", statement=stmt, tactics=("rfl",))
        ])
        assert out_hashed[0].status is Status.PROVED

    def test_fixture_missing_key_errors(self, tmp_path: Path):
        fixture = tmp_path / "bad.json"
        fixture.write_text(json.dumps([{"status": "proved"}]))
        backend = MockLeanBackend()
        with pytest.raises(ValueError):
            backend.load_fixture(fixture)


class TestTemplates:
    def test_integer_equality_applies(self):
        m = emit_template_task(
            problem_id="p1", class_label="42", problem="What is the answer?", answer="42"
        )
        assert m.kind == "arithmetic_match" or m.kind == "integer_equality"
        assert m.task is not None
        assert "theorem" in m.task.statement

    def test_rational_equality_applies(self):
        m = emit_template_task(
            problem_id="p2", class_label="1/2", problem="Compute the value.", answer="1/2"
        )
        # Either arithmetic_match (if the problem contained "= 1/2") or rational_equality fires.
        assert m.task is not None

    def test_garbage_answer_no_template(self):
        m = emit_template_task(
            problem_id="p3", class_label="X", problem="Solve.", answer="UNNORMALIZED::foo"
        )
        assert m.task is None
        assert m.kind == "no_template"

    def test_arithmetic_match_when_problem_contains_equality(self):
        m = emit_template_task(
            problem_id="p4",
            class_label="7",
            problem="Compute 3 + 4 = 7 directly.",
            answer="7",
        )
        assert m.task is not None
        # The lhs from the problem should be in the statement.
        assert "3 + 4" in m.task.statement or "3+4" in m.task.statement.replace(" ", "")

    def test_template_kinds_nonempty(self):
        assert len(list_template_kinds()) >= 1


def test_task_content_hash_stable():
    t1 = LeanTask("a", "theorem t : 1 = 1", ("rfl", "norm_num"))
    t2 = LeanTask("b", "theorem t : 1 = 1", ("rfl", "norm_num"))
    t3 = LeanTask("a", "theorem t : 2 = 2", ("rfl", "norm_num"))
    assert task_content_hash(t1) == task_content_hash(t2)
    assert task_content_hash(t1) != task_content_hash(t3)


def test_outcome_roundtrip():
    from covcal.lean.backend import LeanOutcome
    o = LeanOutcome(name="x", status=Status.PROVED, tactic_used="rfl",
                    elapsed_seconds=0.1, log="ok")
    d = o.to_dict()
    assert d["status"] == "proved"
    o2 = LeanOutcome.from_runner_dict({**d})
    assert o2.status is Status.PROVED
    assert o2.tactic_used == "rfl"
