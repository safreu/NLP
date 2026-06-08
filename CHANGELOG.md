# Changelog

## Unreleased

### Added
- Added smoke tests for core module imports so broken pipeline imports are caught early.
- Added focused tests for latest-run resolution and pipeline path wiring after the RunPaths refactor.

### Fixed
- Fixed the ASSET SARI pipeline after the configuration refactor by using `TrainingConfig` instead of removed module-level constants.
- Aligned trainer path annotations with the `Path` values passed by the pipeline.

### Changed
- Updated ASSET SARI model resolution documentation for latest-run pipeline model directories and `TrainingConfig.model_name` fallback behavior.
- Deferred loading of the Hugging Face SARI metric until SARI is computed, keeping imports fast and side-effect free.
- Centralized run path construction through `RunPaths.for_runs_root()` and moved run-store defaults to that factory.

### CI
- CI lint now runs in check-only mode, tests fail on real failures, and the workflow posts a test/coverage summary into the GitHub Actions run summary.
- Reformatted Python files with Ruff to satisfy the new lint gate.

### Removed
- Removed generated run outputs, test reports, OS metadata, and local dataset cache files from Git tracking.

## 2026-06-05 - Branch Cleanup and Repository Consolidation

### Initial State
- All remotes and tags were updated with `git fetch --all --prune --tags`.
- The local worktree was not clean and was saved before the merges: `pre-branch-cleanup-2026-06-05`.
- `origin/main` was the default branch, but `origin/dev` contained most of the current development work.
- `origin/main` and `origin/dev` had diverged; `origin/main` contained a revert of old `SimplePPDB` demo code, while `origin/dev` contained newer pipeline and training work.

### Planned Procedure
- Integrate `main` into `dev` and resolve the expected conflict in `src/main.py`.
- Integrate the remaining open branches into `dev`: `feature/asset-sari-trained-model-validation`, `fix-copy-rate`, `t5-pipeline`.
- Run the available checks.
- Merge `dev` into `main`.
- Delete fully integrated local and remote branches.

### Completed
- `origin/main` was merged into `dev`.
- The conflict in `src/main.py` was resolved in favor of the current `dev` pipeline code; the old `SimplePPDB` demo code from `main` remains removed.
- `origin/feature/asset-sari-trained-model-validation` was merged into `dev`.
- `origin/fix-copy-rate` was merged into `dev`.
- The conflict in `src/main.py` was resolved in favor of the newer `TrainingPipeline` entry-point logic.
- `origin/t5-pipeline` was merged into `dev`.
- The run artifacts already tracked by `t5-pipeline` under `runs/` were kept.
- Ruff lint issues from the integrated branches were fixed after the merges.
- The T5 script now uses the current `storage` layer instead of the removed `evaluation.file_writer` file.
- Checks passed: `uv run pytest`, `uv run ruff check .`.
- `dev` was pushed to `origin/dev`.
- `dev` was merged into `main` and pushed to `origin/main`.
- Deleted remote branches: `apply-BLEU`, `apply-rouge`, `f1-metric`, `feature/asset-sari-trained-model-validation`, `feature/sari-asset-pipeline`, `feature/sari-metric`, `fix-copy-rate`, `fleschkincaid-metric`, `import-OneStopEnglishCorpus`, `import-asset`, `import-simple_ppdb`, `import-wikilarge`, `import-wikismall`, `metric_BERTScore`, `pipeline_wiki_large`, `revert-8-apply-BLEU`, `t5-pipeline`, `train-pipeline-OneStopEnglishCorpus`.
- Deleted local branches: `feature/asset-sari-trained-model-validation`, `feature/sari-asset-pipeline`, `feature/sari-metric`, `import-asset`.
- The previous local worktree remains saved in the stash `pre-branch-cleanup-2026-06-05`. It was not applied automatically because it contains large training artifacts and untracked `runs/` files that may collide with files that are now tracked.
