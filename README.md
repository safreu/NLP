# Quick Start

[![CI](https://github.com/safreu/NLP/actions/workflows/workflow.yml/badge.svg?branch=dev)](https://github.com/safreu/NLP/actions/workflows/workflow.yml)
[![Tests](https://img.shields.io/badge/tests-GitHub%20Actions-blue?logo=github)](https://github.com/safreu/NLP/actions/workflows/workflow.yml)
[![Coverage](https://img.shields.io/badge/coverage-see%20Actions%20summary-blueviolet?logo=github)](https://github.com/safreu/NLP/actions/workflows/workflow.yml)

```bash
git clone git@gitlab.uni-ulm.de:slb51/nlp.git
cd nlp
uv sync --dev
uv run nox
```

---

# Project Structure

This repository deliberately uses a `src` layout with top-level import names. The project is installed in editable mode by `uv sync`, so imports such as `from config import TrainingConfig` and `from metrics.f1 import compute_f1` work without setting `PYTHONPATH`.

- `src/main.py` -> application entry point used by the `src` command
- `src/config.py` and `src/prompts.py` -> top-level modules
- `src/data/`, `src/evaluation/`, `src/metrics/`, `src/pipeline/`, `src/preprocessing/`, `src/storage/`, `src/training/` -> top-level packages
- `tests/` -> test suite

## Local outputs and generated files

Experiment outputs are local artifacts and are ignored by Git by default:

- `runs/` stores run-specific predictions, scores, metadata, and model outputs.
- `results/` stores ad-hoc evaluation outputs.
- `models/` stores local model checkpoints.
- `data/**/*.pkl` stores generated dataset caches.

Keep reproducible source data, code, and documentation in Git. Do not commit generated run outputs unless they are intentionally curated examples for the report.
---

# Development Setup

Dependencies are split into two groups:

- **Runtime dependencies**: required to run the project (e.g. `transformers`, `datasets`)
- **Development dependencies**: required for development (e.g. `pytest`, `jupyter`, `ruff`, `mypy`)

## Install runtime dependencies

```bash
uv sync
```

## Install development dependencies

```bash
uv sync --dev
```

## Add a new runtime dependency

```bash
uv add package_name
```

## Add a new development dependency

```bash
uv add --dev package_name
```

---

# Commands

## Run tests

```bash
uv run pytest
```

Or run the nox test session:

```bash
uv run nox -s tests
```

## Run the linter

```bash
uv run nox -s lint
```

## Run the type checker

```bash
uv run nox -s typecheck
```

## Run all checks (recommended)

```bash
uv run nox
```

---

# CI and Reports

- CI runs on every push and pull request.
- The Test job generates a summary with total tests, failures, errors, skipped, and line coverage.
- You can view these directly in the GitHub Actions run summary without downloading artifacts.

Where to see the numbers:

- Open the latest CI run in GitHub Actions.
- Click the Test job.
- The run summary shows test counts and line coverage percentage.

Coverage and test reports are still uploaded as artifacts for deeper inspection when needed.

---

# Running the Project

## Run the default experiment

```bash
uv run src
```

This creates the next `runs/run_XXX` directory, writes `config.json`, and reproduces the previous `src/main.py` behavior:

| Pipeline | Defaults |
| --- | --- |
| `onestop` | `TrainingConfig()` with checkpoint evaluation |
| `wikilarge` | `epochs=3`, `max_target_length=128`, `max_train_samples=10000`, `max_eval_samples=2000`, checkpoint evaluation |

The command is configured in `pyproject.toml`:

```toml
[project.scripts]
src = "main:main"
```

Explanation:

- `src` := command name  
- `main` := module name, resolved from `src/main.py` by the package layout in `pyproject.toml`  
- `main` := function that will be executed  

## Run a quick experiment

```bash
uv run src --dataset wikilarge --epochs 1 --batch-size 4 --wikilarge-max-train-samples 100 --wikilarge-max-eval-samples 20 --output-path runs/quick_wikilarge
```

Why: use this shape for a small CLI sanity check before running longer training jobs.

## Run a full WikiLarge experiment

```bash
uv run src --dataset wikilarge --wikilarge-max-train-samples 0 --wikilarge-max-eval-samples 0 --output-path runs/full_wikilarge
```

Why: `0` disables the WikiLarge sample cap and uses the full train, validation, and test splits.

## Common experiment flags

| Flag | Purpose |
| --- | --- |
| `--dataset all\|onestop\|wikilarge` | Selects which pipeline to run |
| `--model-name MODEL` | Overrides `TrainingConfig.model_name` |
| `--epochs N` | Overrides the selected dataset defaults |
| `--batch-size N` | Overrides train and evaluation batch size |
| `--evaluation-mode final_model\|checkpoints` | Selects final-model or checkpoint evaluation |
| `--output-path PATH` | Writes the run to a specific directory instead of the next `runs/run_XXX` |

Every run writes the resolved configuration to `config.json` in the run directory.

## Evaluate simple baselines

Run copy and rule-based baselines before interpreting trained model scores:

```bash
uv run evaluate-baselines --dataset wikilarge --max-examples 100 --output-path runs/baselines_quick
uv run aggregate-results runs/baselines_quick
```

The baseline runner writes one pipeline directory per dataset and baseline, for example `runs/baselines_quick/wikilarge_copy/scores.json`. The `copy` baseline returns the source text unchanged after prompt removal. The `punctuation_split` baseline is a deliberately simple rule-based baseline that splits on semicolons, colons, dashes, and a few clause boundaries.

## Zero-shot LLM baseline

Run a **local, open-weights** instruction-tuned Gemma model as a
zero-shot simplification baseline on ASSET.

### Prompt template

The single documented zero-shot prompt lives in `src/prompts.py` as
`zero_shot_simplify_messages(text)`. It returns chat messages for
`tokenizer.apply_chat_template`.

```text
Rewrite the following English sentence so it is easier to read. Use simpler
words and shorter sentences while keeping the original meaning. Reply with only
the simplified sentence and nothing else.

Sentence: <sentence>
```

### Hugging Face token (gated download only)

Gemma weights are license-gated. Accept the license on the model's Hub page,
then provide a read token via the environment. Copy `.env.example` to `.env`
(gitignored) and set `HF_TOKEN`:

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=hf_...
```

The token is used **only** to download the weights and is never written to any
prediction or score file. No API secrets are stored in the repo.

### Two independent steps

Generation and scoring are separate subcommands so that re-scoring never
re-runs generation:

```bash
# 1) GENERATE: load ASSET, run the local model on GPU, write predictions JSON
uv run llm-zero-shot generate \
  --model-name google/gemma-4-12b-it \
  --revision <pinned-commit-hash> \
  --split test --max-examples 0 \
  --predictions-path results/asset_llm_zero_shot_predictions.json

# 2) SCORE: compute ASSET SARI on the saved predictions
uv run llm-zero-shot score \
  --predictions-path results/asset_llm_zero_shot_predictions.json \
  --score-path results/asset_llm_zero_shot_score.json \
  --model-name google/gemma-4-12b-it --revision <pinned-commit-hash> \
  --split test --max-examples 0
```

Decoding is deterministic (`do_sample=False`, greedy, fixed seed) and the
`generation_config` is logged into `score.json` alongside `model_name`,
`revision`, and the prompt template.

- **Default model:** `google/gemma-4-12b-it`
- **Lighter alternative:** `google/gemma-4-e4b-it` — pass it via
  `--model-name google/gemma-4-e4b-it` for smaller GPUs.
- Pin `--revision` to a commit hash for fully reproducible downloads.

## Aggregate experiment scores

Create a Markdown table that can be copied into the report:

```bash
uv run aggregate-results runs/run_001 runs/run_002
```

Write CSV or JSON instead:

```bash
uv run aggregate-results runs/run_001 runs/run_002 --format csv --output results/score_summary.csv
uv run aggregate-results runs/run_001 runs/run_002 --format json --output results/score_summary.json
```

The table includes run, pipeline, model, checkpoint, SARI, FKGL, BERTScore precision/recall/F1, BLEU, ROUGE-L, and token F1. Missing metrics are written as `NA` in Markdown and CSV, and as `null` in JSON.

## Visualize readability-preservation trade-offs

Generate a reproducible plot from aggregated results:

```bash
uv run aggregate-results runs/run_001 runs/run_002 --format json --output results/score_summary.json
uv run visualize-results results/score_summary.json --output-dir results/visualizations
```

This writes `readability_preservation_tradeoff.png`, `readability_preservation_tradeoff.csv`, and `readability_preservation_tradeoff.md`. The PNG combines a SARI-vs-FKGL scatter plot with a mean BERTScore F1 comparison per model.

## Run a module command

```bash
uv run python -m pipeline.sari_asset_pipeline --help
```

---

# Understanding the CI/CD pipeline

- On every push to any branch:
  - the **linter** runs
  - the **tests** run

- On merge requests targeting `main`:
  - the **typechecker** runs in addition

The typechecker is configured in **strict mode**, which means:

- all functions must have type annotations  
- type errors will fail the pipeline  

Will fail:
```python
def main():
    print("setting up")

if __name__ == "__main__":
    main()
```
Will succeed:
```python
def main() -> None:
    print("setting up")


if __name__ == "__main__":
    main()
```


If the typecheck fails, the merge request cannot be merged.

---

## Merges to main

It is recommended to create a merge request describing:

- what changes were made  
- how they were implemented  

## Commit messages

It is recommended to use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
e.g.:

```bash
git commit -m "feat(transformer): implement new transformer library"
```
