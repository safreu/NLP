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

## Windows PowerShell Setup

Run this from the repository root before pipeline commands:

```powershell
$env:PYTHONPATH = "src"
```

Why: the project imports modules from the `src` directory directly.

## macOS/Linux Setup

Run this from the repository root before pipeline commands:

```bash
export PYTHONPATH=src
```

Why: the project imports modules from the `src` directory directly.

## Show All Available Options

```powershell
python -m pipeline.sari_asset_pipeline --help
```

Why: use this when you want to see every supported command line option.

## Run The Default Quick Evaluation

```powershell
python -m pipeline.sari_asset_pipeline
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

## Run With uv

```powershell
uv run python -m pipeline.sari_asset_pipeline
```

Why: this uses the project environment managed by `uv`, which is useful when dependencies are installed through `uv.lock`.

## Evaluate A Small Number Of Examples

```powershell
python -m pipeline.sari_asset_pipeline --max-examples 100
```

Why: use this for a faster check before running the full ASSET split.

## Evaluate The Full ASSET Validation Split

```powershell
python -m pipeline.sari_asset_pipeline --max-examples 0 --predictions-path results/asset_sari_full/asset_sari_predictions_full.json --score-path results/asset_sari_full/asset_sari_score_full.json
```

Why: `--max-examples 0` means no limit. This evaluates the complete validation split.

This is the command shape used for the full ASSET validation run.

## Evaluate The Full ASSET Test Split

```powershell
python -m pipeline.sari_asset_pipeline --split test --max-examples 0 --predictions-path results/asset_sari_test_full/asset_sari_predictions_test_full.json --score-path results/asset_sari_test_full/asset_sari_score_test_full.json
```

Why: use this when you want a final score on the ASSET test split.

Use the validation split for development. Use the test split only for final reporting.

## Evaluate A Specific Model

```powershell
python -m pipeline.sari_asset_pipeline --model-path runs/run_003/model --max-examples 0
```

Why: use this to avoid accidentally evaluating the wrong model through `runs/latest.txt`.

The pipeline chooses a model in this order:

| Priority | Model Source |
| --- | --- |
| 1 | `--model-path` argument |
| 2 | `runs/latest.txt` plus `/model` |
| 3 | `MODEL_OUTPUT_DIR` from `src/config.py` |

If multiple teammates trained different models, always pass `--model-path` explicitly.

## Compare Two Models

```powershell
python -m pipeline.sari_asset_pipeline --model-path runs/run_003/model --max-examples 0 --predictions-path results/model_a_predictions.json --score-path results/model_a_score.json
```

```powershell
python -m pipeline.sari_asset_pipeline --model-path path/to/other/model --max-examples 0 --predictions-path results/model_b_predictions.json --score-path results/model_b_score.json
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
python -m pipeline.sari_asset_pipeline --prompt-level none --max-examples 0
```

Why: use this to test whether the prompt helps or hurts model output.

Input shape:

```text
Original sentence only.
```

## Run With The Elementary Prompt

```powershell
python -m pipeline.sari_asset_pipeline --prompt-level elementary --max-examples 0
```

Why: use this when the target output should be simpler English for elementary readers.

Input shape:

```text
rewrite this in simpler English for elementary readers. Use short sentences and simple words: <source>
```

## Run With The Intermediate Prompt

```powershell
python -m pipeline.sari_asset_pipeline --prompt-level intermediate --max-examples 0
```

Why: use this when the target output should be clearer, but not necessarily as aggressively simplified as elementary output.

Input shape:

```text
rewrite this in simpler English for intermediate readers. Keep the meaning but use clearer language: <source>
```

## Force CPU Execution

```powershell
python -m pipeline.sari_asset_pipeline --device cpu --max-examples 100
```

Why: use this when GPU execution is unavailable, unstable, or not needed for a small run.

## Force CUDA Execution

```powershell
python -m pipeline.sari_asset_pipeline --device cuda --max-examples 100
```

Why: use this when an NVIDIA GPU with CUDA is available.

## Force Apple Silicon MPS Execution

```bash
python -m pipeline.sari_asset_pipeline --device mps --max-examples 100
```

Why: use this on macOS with Apple Silicon if PyTorch MPS is available.

## Write Outputs To Custom Files

```powershell
python -m pipeline.sari_asset_pipeline --max-examples 100 --predictions-path results/custom_predictions.json --score-path results/custom_score.json
```

Why: use this to keep results from different runs separate.

## Create A Named Result Folder For One Model

```powershell
python -m pipeline.sari_asset_pipeline --model-path runs/run_003/model --max-examples 0 --predictions-path results/run_003_asset/asset_sari_predictions.json --score-path results/run_003_asset/asset_sari_score.json
```

Why: use this when you want each model evaluation to have its own result directory.

## Supported Command Line Arguments

| Argument | Values | Purpose |
| --- | --- | --- |
| `--model-path` | Local model path or Hugging Face model name | Selects the model to evaluate |
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
    "model_path": "runs/run_003/model",
    "prompt_level": "elementary"
}
```

Why: use this file to report the exact model, dataset split, prompt style, and SARI score.
