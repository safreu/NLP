# SARI Metric Commands

This directory contains metric helpers used by the text simplification evaluation code.

The most important file here is `metric_sari.py`. It wraps the Hugging Face `evaluate` implementation of SARI and makes sure references are formatted correctly.

## What SARI Is For

SARI is a text simplification metric. It compares three things:

| Input | Meaning |
| --- | --- |
| `sources` | The original complex sentences |
| `predictions` | The model outputs |
| `references` | Human simplifications |

SARI is useful because text simplification is not only about matching a reference. It also checks whether the model kept useful words, deleted unnecessary words, and added good simplifications.

## Setup

Run this from the repository root before using imports such as `from metrics.metric_sari import compute_sari`:

```powershell
uv sync --dev
```

Why: `uv sync` installs the project in editable mode from `src`, so top-level imports do not require `PYTHONPATH`.

## Run The Built-In SARI Demo

```powershell
uv run python -m metrics.metric_sari
```

Why: this checks that the `evaluate` package can load the SARI metric and compute a score on a tiny example.

This command does not evaluate your trained model. It only runs the demo example inside `metric_sari.py`.

## Use `compute_sari` In Python

```python
from metrics.metric_sari import compute_sari

sources = ["The physician administered medication to the patient."]
predictions = ["The doctor gave medicine to the patient."]
references = [["The doctor gave medicine to the patient."]]

score = compute_sari(sources, predictions, references)
print(score)
```

Why: use this when another script already has sources, model predictions, and references in memory.

## Use Multiple References Per Example

```python
from metrics.metric_sari import compute_sari

sources = ["The physician administered medication to the patient."]
predictions = ["The doctor gave medicine to the patient."]
references = [
    [
        "The doctor gave medicine to the patient.",
        "The doctor gave the patient medicine.",
    ]
]

score = compute_sari(sources, predictions, references)
print(score)
```

Why: datasets such as ASSET provide several human simplifications per original sentence. SARI can use all of them.

## Use One Reference Per Example

```python
from metrics.metric_sari import compute_sari

sources = ["The physician administered medication to the patient."]
predictions = ["The doctor gave medicine to the patient."]
references = ["The doctor gave medicine to the patient."]

score = compute_sari(sources, predictions, references)
print(score)
```

Why: `format_references()` converts plain string references into the list-of-lists format required by Hugging Face SARI.

## Required Data Shape

The lists must have the same number of examples:

```python
len(sources) == len(predictions) == len(references)
```

Correct multi-reference format:

```python
references = [
    ["reference 1 for example 1", "reference 2 for example 1"],
    ["reference 1 for example 2", "reference 2 for example 2"],
]
```

Correct single-reference format:

```python
references = [
    "reference for example 1",
    "reference for example 2",
]
```

Why: Hugging Face SARI expects references as a list of lists, but this wrapper accepts both formats to make other pipeline code easier to write.

## What The Functions Do

| Function | Purpose |
| --- | --- |
| `format_references()` | Converts references into the list-of-lists format required by SARI |
| `compute_sari()` | Computes one float SARI score for sources, predictions, and references |
| `main()` | Runs a tiny local demo example |

## When To Use This Directory Directly

Use `metric_sari.py` directly when you want to test only the metric logic.

Use the pipeline in `src/pipeline/sari_asset_pipeline.py` when you want to load a model, generate predictions, write JSON files, and compute SARI on ASSET.
