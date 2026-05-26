"""Subprocess-based wrapper around the `CovCalRunner` Lean executable.

The runner is launched via ``lake env <lean_project>/.lake/build/bin/CovCalRunner
<bootstrap.lean>``. After printing ``{"ready":true}`` it accepts one JSON task per line on
stdin and emits one JSON outcome per line on stdout. We multiplex an arbitrary batch of
tasks across a single long-lived runner process.

Wall-clock safety net: each batch is bounded by ``per_task_timeout_seconds * len(tasks) +
startup_slack_seconds``. If the runner exceeds that, the subprocess is killed and the
remaining tasks are filled in as TIMEOUT.

The runner binary path and bootstrap path are inferred from ``project_dir`` by default; you
can also pass them explicitly.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import select
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..types import Status
from .backend import LeanBackend, LeanOutcome, LeanTask

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SubprocessLeanConfig:
    project_dir: Path
    runner_bin: Path | None = None  # default: project_dir/.lake/build/bin/CovCalRunner
    bootstrap_path: Path | None = None  # default: project_dir/_mathlib_bootstrap.lean
    per_task_timeout_seconds: float = 10.0
    startup_slack_seconds: float = 180.0  # one-shot Mathlib boot is slow on first run
    max_heartbeats_per_tactic: int = 2_000_000  # ≈10s at default heartbeat rate

    def resolve(self) -> SubprocessLeanConfig:
        runner = self.runner_bin or (
            self.project_dir / ".lake" / "build" / "bin" / "CovCalRunner"
        )
        bootstrap = self.bootstrap_path or (self.project_dir / "_mathlib_bootstrap.lean")
        return SubprocessLeanConfig(
            project_dir=self.project_dir,
            runner_bin=runner,
            bootstrap_path=bootstrap,
            per_task_timeout_seconds=self.per_task_timeout_seconds,
            startup_slack_seconds=self.startup_slack_seconds,
            max_heartbeats_per_tactic=self.max_heartbeats_per_tactic,
        )


def _ensure_built(cfg: SubprocessLeanConfig) -> None:
    if cfg.runner_bin is None or not cfg.runner_bin.exists():
        raise FileNotFoundError(
            f"CovCalRunner binary not found at {cfg.runner_bin}. "
            f"Run `bash scripts/setup_lean.sh` (or `cd lean && lake build CovCalRunner`) first."
        )
    if cfg.bootstrap_path is None or not cfg.bootstrap_path.exists():
        raise FileNotFoundError(
            f"Bootstrap file not found: {cfg.bootstrap_path}. "
            f"Expected `import Mathlib` at {cfg.project_dir}/_mathlib_bootstrap.lean."
        )


class SubprocessLeanBackend(LeanBackend):
    """A LeanBackend that drives `CovCalRunner` as a persistent subprocess.

    The process is started lazily on the first :meth:`check` call. Re-use the same backend
    instance across many batches to amortise the Mathlib boot cost.
    """

    def __init__(self, config: SubprocessLeanConfig) -> None:
        self._cfg = config.resolve()
        self._proc: subprocess.Popen[str] | None = None

    def _start(self) -> subprocess.Popen[str]:
        _ensure_built(self._cfg)
        if shutil.which("lake") is None:
            raise FileNotFoundError("`lake` not on PATH. Install elan first.")
        cmd = [
            "lake",
            "env",
            str(self._cfg.runner_bin),
            str(self._cfg.bootstrap_path),
        ]
        logger.info("starting CovCalRunner: %s (cwd=%s)", " ".join(cmd), self._cfg.project_dir)
        proc = subprocess.Popen(
            cmd,
            cwd=str(self._cfg.project_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        # Wait for the runner's ready line. It may be preceded by warnings on stderr.
        assert proc.stdout is not None
        ready_line = proc.stdout.readline()
        if not ready_line:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"CovCalRunner exited before ready: {stderr}")
        try:
            payload = json.loads(ready_line.strip())
            if not payload.get("ready"):
                raise ValueError(payload)
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"unexpected runner ready line: {ready_line!r} ({e})") from e
        return proc

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = self._start()
        return self._proc

    def check(self, tasks: list[LeanTask]) -> list[LeanOutcome]:
        if not tasks:
            return []
        proc = self._ensure_proc()
        assert proc.stdin is not None and proc.stdout is not None

        # Inject the heartbeat budget if the task didn't set its own.
        prepared = [
            t if t.max_heartbeats_per_tactic is not None
            else LeanTask(
                name=t.name,
                statement=t.statement,
                tactics=t.tactics,
                max_heartbeats_per_tactic=self._cfg.max_heartbeats_per_tactic,
            )
            for t in tasks
        ]

        # Push all tasks first; read all outcomes after. The runner echoes one outcome per line.
        for t in prepared:
            proc.stdin.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
        proc.stdin.flush()

        outcomes: list[LeanOutcome] = []
        # Per-LINE idle timeout (reset on every received outcome) rather than one big
        # whole-batch deadline. A healthy runner emits an outcome per task within this
        # budget; a runner wedged mid-tactic emits nothing, so we detect the wedge in
        # ~idle_timeout seconds (and restart) instead of waiting out a huge batch deadline.
        # 180s exceeds any single healthy task (10 tactics x ~10s heartbeat cap)
        # while still catching a wedge quickly.
        idle_timeout = max(self._cfg.per_task_timeout_seconds * 6.0, 180.0)
        for t in prepared:
            line = self._read_line_with_timeout(proc, timeout=idle_timeout)
            if line is None:
                # Runner died or timed out mid-batch. Mark remaining as TIMEOUT.
                outcomes.append(
                    LeanOutcome(
                        name=t.name,
                        status=Status.TIMEOUT,
                        tactic_used=None,
                        elapsed_seconds=idle_timeout,
                        log="wrapper: runner did not respond before deadline",
                    )
                )
                # All later tasks will also be missing.
                for u in prepared[len(outcomes):]:
                    outcomes.append(
                        LeanOutcome(
                            name=u.name,
                            status=Status.TIMEOUT,
                            tactic_used=None,
                            elapsed_seconds=0.0,
                            log="wrapper: runner died before this task",
                        )
                    )
                self._kill_proc()
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                outcomes.append(
                    LeanOutcome(
                        name=t.name,
                        status=Status.UNFORMALIZED,
                        tactic_used=None,
                        elapsed_seconds=0.0,
                        log=f"wrapper: could not parse runner line: {line!r}",
                    )
                )
                continue
            outcomes.append(LeanOutcome.from_runner_dict(payload))
        return outcomes

    @staticmethod
    def _read_line_with_timeout(
        proc: subprocess.Popen[str], *, timeout: float
    ) -> str | None:
        assert proc.stdout is not None
        end = time.monotonic() + timeout
        while True:
            if proc.poll() is not None:
                # Runner exited — drain whatever is buffered.
                tail = proc.stdout.readline()
                return tail.rstrip("\n") if tail else None
            remaining = end - time.monotonic()
            if remaining <= 0.0:
                return None
            try:
                readable, _, _ = select.select([proc.stdout], [], [], min(remaining, 1.0))
            except (OSError, ValueError):  # fd closed mid-read
                return None
            if not readable:
                continue  # nothing yet — loop and re-check the deadline
            try:
                line = proc.stdout.readline()
            except ValueError:  # closed mid-read
                return None
            if line:
                return line.rstrip("\n")
            return None  # readable but empty => EOF / runner closed stdout

    def _kill_proc(self) -> None:
        if self._proc is None:
            return
        with contextlib.suppress(OSError):
            self._proc.kill()
        self._proc = None

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None and not self._proc.stdin.closed:
                self._proc.stdin.close()
            self._proc.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            self._kill_proc()
        self._proc = None
