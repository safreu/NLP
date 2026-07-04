# SimplePPDB++ vs WikiLarge Dictionary Ablation Handoff

## Purpose

This document summarizes an ablation study for the classical lexical simplification
pipeline. The goal is to compare two replacement-dictionary sources while keeping
the classifier family, dataset split, features, and evaluation setup fixed.

The two dictionary conditions are:

- **A. WikiLarge-derived dictionary**: replacements extracted from aligned
  WikiLarge training sentence pairs using the existing alignment logic.
- **B. SimplePPDB++ dictionary**: replacements extracted from SimplePPDB++ lexical
  paraphrase rules and then scored by the same classical feature extractor.

No hybrid dictionary was used.

## Dataset Reference

SimplePPDB++ source:

- Repository: https://github.com/mounicam/lexical_simplification
- Data file used by the pipeline:
  https://media.githubusercontent.com/media/mounicam/lexical_simplification/master/SimplePPDBpp/simpleppdbpp_xl.tsv.gz
- Format: `phrase1`, `phrase2`, relative complexity score, PPDB 2.0 score.
- Direction rule: if the relative complexity score is positive, `phrase1` is
  treated as more complex than `phrase2`; if negative, the direction is reversed.
- Citation: Maddela, Mounica and Xu, Wei. 2018. *A Word-Complexity Lexicon and A
  Neural Readability Ranking Model for Lexical Simplification*. EMNLP.

## Code Added

- `src/preprocessing/simpleppdb.py`
  - downloads the SimplePPDB++ gzip file when needed
  - streams the file instead of loading it all into memory
  - extracts lexical single-word complex-to-simple candidates
  - restricts candidates to the training vocabulary
  - keeps the top 5 candidates per source word by absolute readability score

- `src/training/classical_trainer.py`
  - now supports `replacement_source="wikilarge"` and
    `replacement_source="simpleppdb"`
  - the default remains `wikilarge`, so existing behavior is preserved

- `src/pipeline/simpleppdb_ablation_pipeline.py`
  - new command-line experiment runner
  - runs Logistic Regression, Linear SVM, and Random Forest
  - writes model artifacts, prediction JSON/CSV files, metric summaries, and
    preservation summaries

- `simpleppdb-ablation` console script added in `pyproject.toml`.

## Commands Run

SimplePPDB++ condition:

```bash
uv run simpleppdb-ablation \
  --dataset wikilarge \
  --replacement-source simpleppdb \
  --model-type all \
  --run-id simpleppdb_wikilarge_20260704 \
  --max-train-samples 10000 \
  --max-eval-samples 2000 \
  --run-preservation
```

WikiLarge-derived baseline condition:

```bash
uv run simpleppdb-ablation \
  --dataset wikilarge \
  --replacement-source wikilarge \
  --model-type all \
  --run-id wikilarge_dictionary_baseline_20260704 \
  --max-train-samples 10000 \
  --max-eval-samples 2000 \
  --run-preservation
```

Focused tests:

```bash
uv run pytest tests/test_simpleppdb_replacements.py tests/test_import_smoke.py tests/test_classical_ml_pipeline.py
```

Result: `23 passed`.

## Shared Setup

- Dataset: WikiLarge via `an-atlas/wikilarge`
- Train cap: 10,000 sentence pairs
- Validation split size available after loader cap: 494 pairs
- Test split size available after loader cap: 191 pairs
- Classifiers:
  - Logistic Regression
  - Linear SVM
  - Random Forest
- Feature extractor: unchanged classical feature extractor
- Metrics: validation classifier accuracy/macro-F1, plus test SARI, BLEU,
  ROUGE-L, Flesch-Kincaid proxy, BERTScore/F1 metrics through the existing
  generation metric stack
- Preservation metrics: entity preservation and number preservation

## SimplePPDB++ Extraction Details

For the SimplePPDB++ runs:

- Rules streamed from SimplePPDB++ XL file: 13,159,398
- Usable lexical rules after filtering and training-vocabulary restriction:
  893,400
- Training vocabulary size: 29,630
- Replacement source words retained: 14,326
- Replacement candidates retained after top-5 pruning: 53,947
- Minimum absolute complexity score: 0.0
- Max candidates per source: 5

The SimplePPDB++ readability score was converted into an integer support weight
so the existing replacement-count and replacement-frequency features could still
be computed without changing the classifier interface.

## Main Results

| Dictionary | Classifier | Validation Accuracy | Validation Macro-F1 | Test SARI | Test BLEU | Test ROUGE-L | Entity Preservation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WikiLarge-derived | Logistic Regression | 0.7068 | 0.4772 | 32.8428 | 0.1979 | 0.4911 | 0.3990 |
| WikiLarge-derived | SVM | 0.8848 | 0.6196 | 39.1540 | 0.3285 | 0.6129 | 0.7876 |
| WikiLarge-derived | Random Forest | 0.6693 | 0.4359 | 31.3322 | 0.1700 | 0.4678 | 0.5026 |
| SimplePPDB++ | Logistic Regression | 0.9199 | 0.6703 | 40.3195 | 0.3614 | 0.6447 | 0.8238 |
| SimplePPDB++ | SVM | 0.9872 | 0.8985 | 53.2448 | 0.4442 | 0.6914 | 0.9585 |
| SimplePPDB++ | Random Forest | 0.9953 | 0.9578 | 54.5330 | 0.4454 | 0.6943 | 0.9948 |

## Interpretation Notes

The SimplePPDB++ replacement source substantially improved the metrics in this
10k WikiLarge-sample setting. The likely reason is that the external paraphrase
resource provides cleaner and broader lexical candidate coverage than the noisy
alignment-derived WikiLarge dictionary.

The number preservation metric is not informative for this specific WikiLarge
test slice because `sentences_with_numbers = 0` and `total_numbers_in_original = 0`
for all runs.

Logistic Regression emitted a standard sklearn convergence warning at
`max_iter=1000` in both dictionary conditions. The runs still completed and
produced predictions/metrics, but this should be mentioned as a limitation or
future tuning point.

## Output Locations

SimplePPDB++ run:

- `results/simpleppdb_ablation/simpleppdb_wikilarge_20260704/summary.csv`
- `results/simpleppdb_ablation/simpleppdb_wikilarge_20260704/summary.json`
- `results/simpleppdb_ablation/simpleppdb_wikilarge_20260704/handoff_for_chatgpt.md`

WikiLarge dictionary baseline run:

- `results/simpleppdb_ablation/wikilarge_dictionary_baseline_20260704/summary.csv`
- `results/simpleppdb_ablation/wikilarge_dictionary_baseline_20260704/summary.json`
- `results/simpleppdb_ablation/wikilarge_dictionary_baseline_20260704/handoff_for_chatgpt.md`

Each classifier folder contains:

- `model/model.pkl`
- `model/replacement_dictionary.json`
- `model/replacement_metadata.json`
- `validation_predictions.json`
- `test_predictions.json`
- `test_predictions.csv`
- `scores.json`
- `preservation/number_preservation.json`
- `preservation/entity_preservation_summary.json`
- `preservation/entity_preservation_by_type.json`
- `preservation/entity_preservation_examples.json`

## Suggested Report Framing

This ablation isolates the effect of the replacement dictionary source in a
classical lexical simplification pipeline. The classifier and feature extractor
remain the same; only the source of candidate replacements changes.

The results suggest that replacement candidate quality is a major bottleneck for
the classical system. Compared with the dictionary mined from WikiLarge sentence
alignments, SimplePPDB++ produced higher simplification quality and higher entity
preservation across all three classifier families in this run.
