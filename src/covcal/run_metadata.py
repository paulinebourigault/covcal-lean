"""Per-run metadata: pinned versions, hardware, git provenance, config snapshot.

This module has no I/O beyond capturing what it can without side effects, plus an
explicit ``write_metadata`` helper.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HostInfo:
    hostname: str
    platform: str
    cpu_count: int | None
    mem_total_kb: int | None
    python_version: str
    uname: dict[str, str]


@dataclass(slots=True)
class GitInfo:
    sha: str | None
    branch: str | None
    dirty: bool


@dataclass(slots=True)
class LeanInfo:
    toolchain: str | None  # contents of lean/lean-toolchain
    lean_version: str | None
    lake_version: str | None
    mathlib_rev: str | None  # from lake-manifest.json
    runner_binary_sha256: str | None


@dataclass(slots=True)
class RunMetadata:
    name: str
    started_at: str  # ISO-8601 UTC
    finished_at: str | None
    elapsed_seconds: float | None
    host: HostInfo
    git: GitInfo
    lean: LeanInfo
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    pipeline_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- collectors --------------------------------------------------------------------------


def _read_proc_meminfo() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def collect_host() -> HostInfo:
    u = platform.uname()
    return HostInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        mem_total_kb=_read_proc_meminfo(),
        python_version=platform.python_version(),
        uname={"system": u.system, "release": u.release, "machine": u.machine,
               "version": u.version, "processor": u.processor},
    )


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True,
            timeout=5.0,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def collect_git(repo_root: Path) -> GitInfo:
    sha = _git(["rev-parse", "HEAD"], repo_root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    status = _git(["status", "--porcelain"], repo_root)
    return GitInfo(sha=sha, branch=branch, dirty=bool(status))


def _file_sha256(path: Path, n: int = 16) -> str | None:
    if not path.exists():
        return None
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _read_lean_toolchain(lean_dir: Path) -> str | None:
    p = lean_dir / "lean-toolchain"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip()


def _read_mathlib_rev(lean_dir: Path) -> str | None:
    manifest = lean_dir / "lake-manifest.json"
    if not manifest.exists():
        return None
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for pkg in m.get("packages", []):
        if pkg.get("name") == "mathlib":
            return pkg.get("rev") or pkg.get("inputRev")
    return None


def _tool_version(tool: str, *args: str) -> str | None:
    bin_path = shutil.which(tool)
    if not bin_path:
        return None
    try:
        out = subprocess.run([bin_path, *args], check=False, capture_output=True,
                             text=True, timeout=10.0)
        text = (out.stdout or "") + (out.stderr or "")
        return text.strip().splitlines()[0] if text.strip() else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def collect_lean(lean_dir: Path) -> LeanInfo:
    runner_bin = lean_dir / ".lake" / "build" / "bin" / "CovCalRunner"
    return LeanInfo(
        toolchain=_read_lean_toolchain(lean_dir),
        lean_version=_tool_version("lean", "--version"),
        lake_version=_tool_version("lake", "--version"),
        mathlib_rev=_read_mathlib_rev(lean_dir),
        runner_binary_sha256=_file_sha256(runner_bin),
    )


# --- public API --------------------------------------------------------------------------


def make_run_metadata(
    *,
    name: str,
    repo_root: Path,
    lean_dir: Path,
    config_snapshot: dict[str, Any],
) -> RunMetadata:
    """Build a RunMetadata with everything known at run start. `finished_at` is None."""
    return RunMetadata(
        name=name,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        finished_at=None,
        elapsed_seconds=None,
        host=collect_host(),
        git=collect_git(repo_root),
        lean=collect_lean(lean_dir),
        config_snapshot=config_snapshot,
    )


def finalize(
    metadata: RunMetadata,
    *,
    pipeline_summary: dict[str, Any] | None = None,
) -> RunMetadata:
    """Set `finished_at`, compute elapsed, attach summary counts."""
    metadata.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        start = time.strptime(metadata.started_at, "%Y-%m-%dT%H:%M:%SZ")
        end = time.strptime(metadata.finished_at, "%Y-%m-%dT%H:%M:%SZ")
        metadata.elapsed_seconds = max(0.0, time.mktime(end) - time.mktime(start))
    except (ValueError, OverflowError):
        metadata.elapsed_seconds = None
    if pipeline_summary is not None:
        metadata.pipeline_summary = pipeline_summary
    return metadata


def write_metadata(metadata: RunMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2, default=str), encoding="utf-8")
    logger.info("wrote run metadata to %s", path)
