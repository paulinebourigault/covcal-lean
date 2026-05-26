"""Tests for covcal.run_metadata."""

from __future__ import annotations

import json
from pathlib import Path

from covcal.run_metadata import (
    collect_git,
    collect_host,
    collect_lean,
    finalize,
    make_run_metadata,
    write_metadata,
)


class TestCollectors:
    def test_host_info_populated(self):
        h = collect_host()
        assert h.hostname
        assert h.platform
        assert h.python_version
        assert h.uname["system"]

    def test_lean_info_reads_toolchain(self, tmp_path: Path):
        (tmp_path / "lean-toolchain").write_text("leanprover/lean4:v4.21.0\n")
        (tmp_path / "lake-manifest.json").write_text(
            json.dumps({"packages": [
                {"name": "mathlib", "rev": "abc123def"},
                {"name": "batteries", "rev": "xyz"},
            ]})
        )
        info = collect_lean(tmp_path)
        assert info.toolchain == "leanprover/lean4:v4.21.0"
        assert info.mathlib_rev == "abc123def"

    def test_lean_info_handles_missing_toolchain(self, tmp_path: Path):
        info = collect_lean(tmp_path)
        assert info.toolchain is None
        assert info.mathlib_rev is None

    def test_git_info_handles_non_git_dir(self, tmp_path: Path):
        info = collect_git(tmp_path)
        # In an empty tmp dir we shouldn't have a sha, but the call shouldn't raise.
        assert info.sha is None or isinstance(info.sha, str)


class TestRoundTrip:
    def test_write_then_read(self, tmp_path: Path):
        meta = make_run_metadata(
            name="t",
            repo_root=tmp_path,
            lean_dir=tmp_path / "lean",
            config_snapshot={"foo": 1},
        )
        out = tmp_path / "metadata.json"
        write_metadata(meta, out)
        assert out.exists()
        d = json.loads(out.read_text(encoding="utf-8"))
        assert d["name"] == "t"
        assert d["host"]["hostname"]
        assert d["config_snapshot"] == {"foo": 1}
        # `finished_at` is None until finalize() runs.
        assert d["finished_at"] is None

    def test_finalize_sets_timing(self, tmp_path: Path):
        meta = make_run_metadata(
            name="t",
            repo_root=tmp_path,
            lean_dir=tmp_path / "lean",
            config_snapshot={},
        )
        finalize(meta, pipeline_summary={"n_problems": 3})
        assert meta.finished_at is not None
        assert meta.elapsed_seconds is not None and meta.elapsed_seconds >= 0
        assert meta.pipeline_summary == {"n_problems": 3}
