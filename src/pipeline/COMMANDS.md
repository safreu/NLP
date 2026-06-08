# ASSET SARI Pipeline Commands

This directory contains runnable evaluation pipelines.

The most important file here is `sari_asset_pipeline.py`. It evaluates a trained sequence-to-sequence simplification model on the ASSET dataset and computes SARI.

## What The Pipeline Does

The pipeline runs these steps:

| Step | What Happens |
| --- | --- |
| 1 | Parse command line arguments |
| 2 | Resolve which model path to use |
| 3 | Load the ASSET dataset from Hugging Face |
| 4 | Load tokenizer and model |
| 5 | Add the selected prompt to each source sentence |
| 6 | Generate model predictions |
| 7 | Compute SARI with `metrics.metric_sari.compute_sari()` |
| 8 | Write predictions and score metadata to JSON files |

## Setup

Run this from the repository root before pipeline commands:

```powershell
uv sync --dev
```

Why: `uv sync` installs the project in editable mode from `src`, so the top-level imports used by these modules do not require `PYTHONPATH`.

## Show All Available Options

```powershell
uv run python -m pipeline.sari_asset_pipeline --help
```

Why: use this when you want to see every supported command line option.

## Run The Default Quick Evaluation

```powershell
uv run python -m pipeline.sari_asset_pipeline
```

Why: this is the fastest sanity check. It evaluates 20 examples from the ASSET validation split.

Default behavior:

| Setting | Default |
| --- | --- |
| Dataset | `facebook/asset` |
| Config | `simplification` |
| Split | `validation` |
| Max examples | `20` |
| Prompt level | `elementary` |
| Predictions file | `results/asset_sari_predictions.json` |
| Score file | `results/asset_sari_score.json` |

## Evaluate A Small Number Of Examples

```powershell
uv run python -m pipeline.sari_asset_pipeline --max-examples 100
```

Why: use this for a faster check before running the full ASSET split.

## Evaluate The Full ASSET Validation Split

```powershell
uv run python -m pipeline.sari_asset_pipeline --max-examples 0 --predictions-path results/asset_sari_full/asset_sari_predictions_full.json --score-path results/asset_sari_full/asset_sari_score_full.json
```

Why: `--max-examples 0` means no limit. This evaluates the complete validation split.

This is the command shape used for the full ASSET validation run.

## Evaluate The Full ASSET Test Split

```powershell
uv run python -m pipeline.sari_asset_pipeline --split test --max-examples 0 --predictions-path results/asset_sari_test_full/asset_sari_predictions_test_full.json --score-path results/asset_sari_test_full/asset_sari_score_test_full.json
```

Why: use this when you want a final score on the ASSET test split.

Use the validation split for development. Use the test split only for final reporting.

## Evaluate A Specific Model

```powershell
uv run python -m pipeline.sari_asset_pipeline --model-path runs/run_003/wikilarge/model --max-examples 0
```

Why: use this to avoid accidentally evaluating the wrong model through `runs/latest.txt`.

The pipeline chooses a model in this order:

| Priority | Model Source |
| --- | --- |
| 1 | `--model-path` argument |
| 2 | `--pipeline-name` plus `runs/latest.txt`, for example `runs/run_003/wikilarge/model` |
| 3 | The latest run, if it contains exactly one model directory |
| 4 | `TrainingConfig.model_name` from `src/config.py` |

If the latest run contains multiple model directories, pass `--model-path` or `--pipeline-name`
explicitly so the script does not evaluate the wrong model.

## Compare Two Models

```powershell
uv run python -m pipeline.sari_asset_pipeline --model-path runs/run_003/wikilarge/model --max-examples 0 --predictions-path results/model_a_predictions.json --score-path results/model_a_score.json
```

```powershell
uv run python -m pipeline.sari_asset_pipeline --model-path path/to/other/model --max-examples 0 --predictions-path results/model_b_predictions.json --score-path results/model_b_score.json
```

Why: use this when two models were trained on different datasets and you want a fair comparison on the same ASSET split.

Keep these settings identical for both models:

| Setting | Why It Should Match |
| --- | --- |
| `--split` | Same evaluation data |
| `--max-examples` | Same number of examples |
| `--prompt-level` | Same input prompt style |
| `--device` | Optional, only affects runtime |

## Run Without A Prompt

```powershell
uv run python -m pipeline.sari_asset_pipeline --prompt-level none --max-examples 0
```

Why: use this to test whether the prompt helps or hurts model output.

Input shape:

```text
Original sentence only.
```

## Run With The Elementary Prompt

```powershell
uv run python -m pipeline.sari_asset_pipeline --prompt-level elementary --max-examples 0
```

Why: use this when the target output should be simpler English for elementary readers.

Input shape:

```text
rewrite this in simpler English for elementary readers. Use short sentences and simple words: <source>
```

## Run With The Intermediate Prompt

```powershell
uv run python -m pipeline.sari_asset_pipeline --prompt-level intermediate --max-examples 0
```

Why: use this when the target output should be clearer, but not necessarily as aggressively simplified as elementary output.

Input shape:

```text
rewrite this in simpler English for intermediate readers. Keep the meaning but use clearer language: <source>
```

## Force CPU Execution

```powershell
uv run python -m pipeline.sari_asset_pipeline --device cpu --max-examples 100
```

Why: use this when GPU execution is unavailable, unstable, or not needed for a small run.

## Force CUDA Execution

```powershell
uv run python -m pipeline.sari_asset_pipeline --device cuda --max-examples 100
```

Why: use this when an NVIDIA GPU with CUDA is available.

## Force Apple Silicon MPS Execution

```bash
uv run python -m pipeline.sari_asset_pipeline --device mps --max-examples 100
```

Why: use this on macOS with Apple Silicon if PyTorch MPS is available.

## Write Outputs To Custom Files

```powershell
uv run python -m pipeline.sari_asset_pipeline --max-examples 100 --predictions-path results/custom_predictions.json --score-path results/custom_score.json
```

Why: use this to keep results from different runs separate.

## Create A Named Result Folder For One Model

```powershell
uv run python -m pipeline.sari_asset_pipeline --model-path runs/run_003/wikilarge/model --max-examples 0 --predictions-path results/run_003_asset/asset_sari_predictions.json --score-path results/run_003_asset/asset_sari_score.json
```

Why: use this when you want each model evaluation to have its own result directory.

## Supported Command Line Arguments

| Argument | Values | Purpose |
| --- | --- | --- |
| `--model-path` | Local model path or Hugging Face model name | Selects the model to evaluate |
| `--pipeline-name` | Latest-run subdirectory, for example `wikilarge` | Selects a model from `runs/latest.txt` when `--model-path` is omitted |
| `--split` | `validation`, `test` | Selects the ASSET split |
| `--max-examples` | Integer, `0` for full split | Limits the number of examples |
| `--prompt-level` | `elementary`, `intermediate`, `none` | Selects the prompt style |
| `--device` | `cpu`, `cuda`, `mps` | Selects the Torch device |
| `--predictions-path` | File path | Writes sources, predictions, and references |
| `--score-path` | File path | Writes SARI score metadata |

## Output Files

The predictions file contains rows like this:

```json
{
    "source": "Original sentence.",
    "prediction": "Model output.",
    "references": [
        "Human simplification 1.",
        "Human simplification 2."
    ]
}
```

Why: inspect this file to see whether the model really simplifies text or mostly copies the source.

The score file contains metadata like this:

```json
{
    "metric": "sari",
    "score": 54.63,
    "dataset": "facebook/asset",
    "dataset_config": "simplification",
    "split": "validation",
    "max_examples": null,
    "model_path": "runs/run_003/wikilarge/model",
    "prompt_level": "elementary",
    "configured_model_name": "google/flan-t5-base",
    "generation_config": {
        "max_new_tokens": 256,
        "do_sample": false,
        "num_beams": 4,
        "length_penalty": 0.9,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1
    }
}
```

Why: use this file to report the exact model, dataset split, prompt style, and SARI score.

---

# Zero-Shot LLM Baseline (`llm_zero_shot_pipeline.py`)

`llm_zero_shot_pipeline.py` runs a **local, open-weights** instruction-tuned
Gemma model (no hosted API, no secrets) as a zero-shot simplification baseline on
ASSET. It writes predictions in the same `{source, prediction, references}`
format as `sari_asset_pipeline.py`, and SARI is computed on the *saved* outputs.

The pipeline is split into two independent steps so that re-scoring never
triggers a new (expensive) generation pass.

## Prompt Template

The single documented zero-shot prompt is `zero_shot_simplify_messages(text)` in
`src/prompts.py`. It returns chat messages for `tokenizer.apply_chat_template`.
Gemma chat templates support only the `user`/`model` roles (no `system` role), so
the full instruction is one `user` turn:

```text
Rewrite the following English sentence so it is easier to read. Use simpler
words and shorter sentences while keeping the original meaning. Reply with only
the simplified sentence and nothing else.

Sentence: <source>
```

## Hugging Face Token (gated download only)

Gemma weights are license-gated. Accept the license on the model's Hub page,
then copy `.env.example` to `.env` (gitignored) and set `HF_TOKEN`. The token is
read from the environment for the download only and is never written to any
output file. No inference API is used.

## 1) Generate Predictions

```bash
uv run python -m pipeline.llm_zero_shot_pipeline generate \
  --model-name google/gemma-4-12b-it \
  --revision <pinned-commit-hash> \
  --split test --max-examples 0 \
  --predictions-path results/asset_llm_zero_shot_predictions.json
```

Why: loads ASSET, loads the model locally with `AutoModelForCausalLM` +
`AutoTokenizer`, builds the prompt via `apply_chat_template`, generates
deterministically (`do_sample=False`, fixed seed), decodes only the newly
generated tokens, and writes predictions JSON.

## 2) Score Saved Predictions

```bash
uv run python -m pipeline.llm_zero_shot_pipeline score \
  --predictions-path results/asset_llm_zero_shot_predictions.json \
  --score-path results/asset_llm_zero_shot_score.json \
  --model-name google/gemma-4-12b-it --revision <pinned-commit-hash> \
  --split test --max-examples 0
```

Why: reads the saved predictions JSON and computes SARI with
`metrics.metric_sari.compute_sari()`, without reloading the model. `score.json`
records `model_name`, `revision`, the prompt template, the seed, and the
`generation_config`.

The `llm-zero-shot` console script is equivalent, e.g.
`uv run llm-zero-shot generate --help`.

## Models

| Setting | Value |
| --- | --- |
| Default model | `google/gemma-4-12b-it` (one A100, bfloat16, `device_map="auto"`) |
| Lighter alternative | `google/gemma-4-e4b-it` (pass via `--model-name`) |
| Reproducibility | pin `--revision <commit-hash>`; decoding is greedy with a fixed seed |

## Supported Command Line Arguments

| Step | Argument | Values | Purpose |
| --- | --- | --- | --- |
| both | `--model-name` | HF model id | Instruction-tuned, open-weights causal LM |
| both | `--revision` | Commit hash or tag | Pinned model revision (recorded in score) |
| both | `--split` | `validation`, `test` | ASSET split |
| both | `--max-examples` | Integer, `0` for full split | Limits the number of examples |
| generate | `--max-new-tokens` | Integer | Max newly generated tokens per example |
| generate | `--device` | `cpu`, `cuda` | Torch device (auto-detected when omitted) |
| generate | `--predictions-path` | File path | Where predictions JSON is written |
| score | `--predictions-path` | File path | Predictions JSON written by `generate` |
| score | `--score-path` | File path | Where SARI score metadata is written |

On the bwUniCluster, `scripts/run_llm_zero_shot.slurm` runs both steps as a
single GPU job (`sbatch scripts/run_llm_zero_shot.slurm`).
