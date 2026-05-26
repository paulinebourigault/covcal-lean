#!/usr/bin/env bash
# Build the CovCal Lean project. Requires elan/lake on PATH.
set -euo pipefail

cd "$(dirname "$0")/../lean"

if ! command -v lake >/dev/null 2>&1; then
  echo "lake not found; install elan first: https://leanprover-community.github.io/install/" >&2
  exit 1
fi

echo "[setup_lean] fetching Mathlib cache (this can take a few minutes the first time)..."
lake exe cache get || echo "[setup_lean] cache fetch failed; will compile from source"

echo "[setup_lean] building CovCal Runner..."
lake build CovCalRunner

echo "[setup_lean] done."
