"""Lean backends: real subprocess wrapper and deterministic mock.

The two implementations satisfy the same ABC :class:`covcal.lean.backend.LeanBackend`,
so the rest of the pipeline does not care which one is in use. The mock is the workhorse
for unit / integration tests because building Mathlib is expensive (~30 minutes the first
time); the real wrapper is used end-to-end on the experiment runs.
"""

from .autoformalize import (
    Autoformalizer,
    AutoformalizerConfig,
    FormalizedArtifact,
)
from .backend import LeanBackend, LeanOutcome, LeanTask
from .mock import MockLeanBackend
from .subprocess_backend import SubprocessLeanBackend
from .templates import emit_template_task, list_template_kinds

__all__ = [
    "Autoformalizer",
    "AutoformalizerConfig",
    "FormalizedArtifact",
    "LeanBackend",
    "LeanOutcome",
    "LeanTask",
    "MockLeanBackend",
    "SubprocessLeanBackend",
    "emit_template_task",
    "list_template_kinds",
]
