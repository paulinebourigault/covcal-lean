# CovCal: Risk-Controlled Lean-as-Judge

[![arXiv](https://img.shields.io/badge/arXiv-2605.28365-b31b1b.svg)](https://arxiv.org/abs/2605.28365v1)
[![Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://haithamb.github.io/covcal/)

Code and experiments for *Risk-Controlled Lean-as-Judge for Natural-Language Mathematical
Reasoning*.

![CovCal overview](docs/overview.png)

CovCal is a **selective wrapper** around a Lean-based answer selector for natural-language
math problems. It does not train a theorem prover. For each problem it samples candidate
solutions, groups them into normalized **answer classes**, autoformalizes the top classes into
Lean, and runs proof search. From the resulting Lean trace it computes coverage **diagnostics**,
that is typed coverage `C_typ`, proved coverage `C_prf`, unresolved rival mass `R_unres`, and a formal
margin `M`, and then either accepts the proved answer or abstains. A finite-sample certificate
chooses the least-conservative coverage rule whose **selective risk** is provably bounded.

The central empirical finding is a **coverage cliff**: a proved answer is highly reliable when
formal coverage is high and uninformative when it is low. Whether a risk certificate is feasible
at all is governed by the autoformalizer. A general code model leaves the proved signal too sparse
for the conservative (Bonferroni) certificate, while a prover-specialized formalizer raises
coverage enough to make it feasible. See the paper for the full results.

## Repository layout

```
CovCal/
├── src/covcal/            Python implementation
│   ├── types.py           Status enum, dataclasses
│   ├── normalization.py   answer → normalized class
│   ├── classes.py         class-weight aggregation
│   ├── diagnostics.py     C_typ, C_prf, R_unres, M
│   ├── selectors.py       baselines + CovCal selector
│   ├── calibration.py     Clopper–Pearson + grid certificate
│   ├── metrics.py         selective risk, accepted accuracy
│   ├── generation/        LLM backends (vLLM / llama.cpp) + sampling cache + prompts
│   ├── lean/              Lean wrapper + autoformalizer + mock backend
│   ├── pipeline.py        end-to-end orchestration
│   └── cli.py             CLI entry points
├── lean/                  Lean 4 Lake project (CovCalRunner)
├── configs/               experiment configs (see below)
├── scripts/               pipeline runner, setup, analysis
└── tests/                 unit + integration tests
```

## Setup

Requires [`uv`](https://github.com/astral-sh/uv) (Python) and, for Lean verification,
[`elan`](https://github.com/leanprover/elan)/`lake`.

```bash
uv sync --extra dev --extra llm --extra viz   # Python environment
bash scripts/setup_lean.sh                     # build the Lean project (slow first time: Mathlib cache)
```

## Configs

| Config | Experiment |
|---|---|
| `configs/dev.yaml` | Tiny no-GPU smoke test (mock Lean backend; verifies the pipeline wiring). |
| `configs/main.yaml` | Main MATH-500 run: Qwen2.5-Math-7B generator + Qwen2.5-Coder-7B autoformalizer. |
| `configs/goedel8b.yaml` | Autoformalizer ablation: Goedel-Prover-V2-8B. |
| `configs/goedel32b.yaml` | Autoformalizer ablation: Goedel-Prover-V2-32B. |
| `configs/amc.yaml` | AMC/AIME robustness subset. |

All seeds, sampling parameters, the threshold grid, splits, and the Lean tactic budget are
pinned in the YAML. The main runs use Lean 4.21.0 with Mathlib commit `308445d7`.

## Running

```bash
# Quick wiring check (no GPU; downloads a small ~1 GB CPU GGUF for generation):
bash scripts/run_minimal.sh configs/dev.yaml

# A full experiment (needs a GPU + the two models):
bash scripts/run_minimal.sh configs/main.yaml
```

`run_minimal.sh` runs the full chain — `split → pipeline (generate + formalize + Lean) →
calibrate → evaluate → diagnose → render tables + figure` — and writes everything under the
config's `run_dir` (`runs/<name>/`). Per-problem outputs are JSONL, so selectors, metrics, and
the offline analyses can be recomputed without re-running generation or Lean.

The CLI stages can also be invoked individually (`uv run covcal --help`).

### Analysis

`scripts/analysis/` contains the offline post-hoc analyses used for the paper's tables and
figures, each taking one or more `--run-dir runs/<name>` arguments: K-seed bootstraps
(`bootstrap_seeds.py`), the two-regime comparison (`regime_comparison.py`), per-level and
coarse-grid cliff breakdowns, McNemar tests, reliability diagrams, the proof-faithfulness
classifier (`proof_faithfulness.py`), and the headline-number summary (`paper_numbers.py`).

## Reproducibility notes

- The threshold grid `T` and the target risk `ε`, `δ` are fixed *before* calibration labels are
  inspected; do not change `dataset.split_seed` after inspecting a split, or the finite-sample
  certificate's precondition is violated.
- Each run records Lean version, Mathlib commit, model ids, hardware, and the full config in
  `metadata.json`.
- `runs/` (model outputs and caches) is gitignored; reproduce it from the configs above.
- The AMC config reads `data/amc_aime_combined.jsonl` (included: 173 problems, 83 AMC + 90
  AIME), derived from the open `AI-MO/aimo-validation-amc` and `AI-MO/aimo-validation-aime`
  datasets, with each row's AoPS `url` retained for provenance.

## Tests

```bash
make test        # fast unit tests (no Lean, no LLM)
make test-all    # unit + integration (mock-based pipeline smoke tests)
make lint        # ruff
```

## Citation

```bibtex
@unpublished{bourigault2026covcal,
  title   = {{CovCal}: Risk-Controlled Lean-as-Judge for Natural-Language Mathematical Reasoning},
  author  = {Bourigault, Pauline and Ji, Xiaotong and Zimmer, Matthieu and Tutunov, Rasul and Bou-Ammar, Haitham},
  year    = {2026},
  eprint  = {2605.28365},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2605.28365v1}
}
```
