#!/usr/bin/env bash
# Run the full CovCal pipeline end to end for a given config:
#   split -> pipeline (generate + formalize + Lean) -> calibrate -> evaluate -> diagnose
#   -> render tables + selective-risk figure.
# Usage: bash scripts/run_minimal.sh configs/main.yaml
# Assumes the Lean project is built (see scripts/setup_lean.sh) and `uv sync` has run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"

CONFIG="${1:-configs/main.yaml}"
RUN_DIR=$(uv run python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output']['run_dir'])")
echo "[run] config=$CONFIG run_dir=$RUN_DIR"
mkdir -p "$RUN_DIR"

LOG="$RUN_DIR/run.log"
echo "[run] streaming logs to $LOG"

{
  echo "==== covcal split ===="
  uv run covcal split --config "$CONFIG" --out "$RUN_DIR/splits.json" --verbose

  echo "==== covcal pipeline (this is the long one) ===="
  uv run covcal pipeline --config "$CONFIG" --splits "$RUN_DIR/splits.json" --verbose

  echo "==== covcal calibrate ===="
  uv run covcal calibrate \
    --observations "$RUN_DIR/observations.jsonl" \
    --splits "$RUN_DIR/splits.json" \
    --config "$CONFIG" \
    --out "$RUN_DIR/thresholds.json" --verbose

  echo "==== covcal evaluate ===="
  uv run covcal evaluate \
    --observations "$RUN_DIR/observations.jsonl" \
    --splits "$RUN_DIR/splits.json" \
    --thresholds "$RUN_DIR/thresholds.json" \
    --out "$RUN_DIR/metrics.json" --verbose

  echo "==== covcal diagnose ===="
  uv run covcal diagnose \
    --observations "$RUN_DIR/observations.jsonl" \
    --out "$RUN_DIR/diagnostics.jsonl"

  echo "==== render tables ===="
  uv run python scripts/render_tables.py --run-dir "$RUN_DIR"

  echo "==== render selective-risk figure ===="
  uv run python scripts/render_figure_selective_risk.py \
    --run-dir "$RUN_DIR" \
    --observations "$RUN_DIR/observations.jsonl" \
    --splits "$RUN_DIR/splits.json" \
    --config "$CONFIG" \
    --thresholds "$RUN_DIR/thresholds.json"
} 2>&1 | tee "$LOG"

echo "[run] done. Tables in $RUN_DIR/tables/, figure in $RUN_DIR/figures/."
