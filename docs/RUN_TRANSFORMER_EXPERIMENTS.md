# Running the Transformer Experiments

## Setup

```bash
git clone https://github.com/safreu/NLP.git
cd NLP
git switch codex/dual-decoder-transformer
uv sync --dev
```

WikiLarge is loaded through `data.wikilarge_loader.WikiLargeLoader` from
`an-atlas/wikilarge`. The first run downloads it through Hugging Face; later runs use the local
cache. No API token is required for WikiLarge.

## Verification before training

Run focused tests:

```bash
uv run pytest tests/test_dual_decoder_transformer.py tests/test_transformer_experiment.py -q
uv run ruff check src/dual_decoder_transformer.py \
  src/training/transformer_experiment.py \
  src/pipeline/transformer_experiments.py \
  tests/test_dual_decoder_transformer.py \
  tests/test_transformer_experiment.py
```

Run the CPU smoke test. It intentionally interrupts after epoch 1, loads the last checkpoint, and
finishes epoch 2, thereby testing forward/backward, generation, checkpoint save/load, and resume:

```bash
uv run transformer-experiments smoke --overwrite
```

## Train one experiment

Corrected baseline:

```bash
uv run transformer-experiments train \
  --model corrected \
  --experiment-name E1_corrected_baseline \
  --embed-size 256 \
  --heads 8 \
  --activation relu \
  --position-encoding learned \
  --reconstruction-weight 0 \
  --epochs 5 \
  --batch-size 8 \
  --seed 42
```

Dual decoder at lambda 0.50:

```bash
uv run transformer-experiments train \
  --model dual_decoder \
  --experiment-name E3_dual_decoder_lambda_050 \
  --embed-size 256 \
  --heads 8 \
  --activation relu \
  --position-encoding learned \
  --reconstruction-weight 0.5 \
  --epochs 5 \
  --batch-size 8 \
  --seed 42
```

Completed runs are skipped. Use `--resume` to continue `checkpoint_last.pt`. Use `--overwrite`
only when intentionally replacing a completed run.

## Run E1-E9 in dependency order

```bash
uv run transformer-experiments run-all \
  --output-root results/transformer_experiments \
  --device auto \
  --resume
```

The runner completes E1-E4, selects lambda, completes E5-E8, selects the final component
combination, and then runs E9. It terminates after E9 and does not retry code/configuration errors.

## Slurm server execution

From the repository root:

```bash
bash scripts/slurm/submit_transformer_pipeline.sh
bash scripts/slurm/check_transformer_jobs.sh
```

The default job requests one GPU, four CPU cores, 24 GB RAM, and eight hours. Override scheduler
resources with normal `sbatch` flags passed through the submit script:

```bash
bash scripts/slurm/submit_transformer_pipeline.sh \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=24G \
  --time=08:00:00
```

The job prints the hostname, CPU, memory, GPU, Python, PyTorch, and CUDA information before the
smoke test. It writes the job ID to
`results/transformer_experiments/pipeline_job_id.txt` and refuses to submit a duplicate active job.

To request cancellation:

```bash
bash scripts/slurm/cancel_transformer_jobs.sh
```

The trainer checkpoints every epoch and records an interrupted status when the scheduler sends a
termination signal. Resubmit the same pipeline to resume incomplete runs.

## tmux, screen, and nohup fallback

When Slurm is unavailable:

```bash
bash scripts/run_transformer_pipeline_background.sh
bash scripts/check_transformer_pipeline.sh
```

The launcher selects tmux first, then screen, then nohup. Session commands are:

```bash
tmux ls
tmux attach -t transformer_experiments
screen -ls
screen -r transformer_experiments
```

The nohup fallback writes
`results/transformer_experiments/pipeline.pid`; all fallbacks write
`results/transformer_experiments/pipeline.log`. Duplicate active sessions/processes are rejected.

Stop a fallback run gracefully:

```bash
bash scripts/stop_transformer_pipeline.sh
```

## Status and logs

```bash
uv run transformer-experiments status
tail -f results/transformer_experiments/pipeline.log
tail -f results/transformer_experiments/E3_dual_decoder_lambda_050/run.log
```

Status includes experiment state, PID or scheduler job ID when available, latest epoch/losses, log
paths, GPU availability, and GPU memory from `nvidia-smi`.

## Resume an interrupted experiment

```bash
uv run transformer-experiments train \
  --model dual_decoder \
  --experiment-name E3_dual_decoder_lambda_050 \
  --embed-size 256 \
  --heads 8 \
  --activation relu \
  --position-encoding learned \
  --reconstruction-weight 0.5 \
  --epochs 5 \
  --batch-size 8 \
  --seed 42 \
  --resume
```

`checkpoint_last.pt` contains model, optimizer, scheduler, epoch, best validation score,
configuration, vocabulary, and random states.

## Evaluate a checkpoint

```bash
uv run transformer-experiments evaluate \
  --checkpoint results/transformer_experiments/E9_final_combined/checkpoint_best.pt \
  --output-dir results/transformer_experiments/E9_final_combined
```

To verify generation without loading SARI/BERTScore, add `--skip-heavy-metrics`.

## Generate one simplification

```bash
uv run transformer-experiments generate \
  --checkpoint results/transformer_experiments/E9_final_combined/checkpoint_best.pt \
  --text "The physician administered medication to the patient." \
  --max-length 128
```

Dual checkpoints print both simplification and diagnostic reconstruction. Normal inference uses
only the simplification decoder.

## Rebuild comparison outputs

```bash
uv run transformer-experiments build-comparison \
  --output-root results/transformer_experiments
```

Outputs are under `results/transformer_experiments/comparison/`: CSV/JSON results, Markdown and
LaTeX tables, qualitative comparisons, selection evidence, analysis, and plots.

## Run directory contents

Every completed E1-E9 directory contains:

- `config.json` and `environment.json`;
- `metrics.json` and `metrics.csv`;
- `training_history.csv` and `predictions.csv`;
- `checkpoint_best.pt` and `checkpoint_last.pt`;
- `run.log` and `status.json`;
- `vocabulary.json`.

Checkpoints are intentionally ignored by Git because each controlled checkpoint is hundreds of
megabytes. Compact metrics, tables, plots, documentation, and selected prediction evidence are the
appropriate repository artifacts.

## Troubleshooting

### CUDA out of memory

Reduce batch size while preserving other controlled settings, then document the deviation:

```bash
uv run transformer-experiments run-all --device cuda --resume
```

Edit `configs/transformer_experiments.json` and the runner defaults together only if the controlled
design is intentionally changed.

### Dataset cache or network error

Confirm that the server can access Hugging Face once, or copy the standard Hugging Face datasets
cache to the server. Do not commit WikiLarge into the repository.

### RoPE validation error

Choose heads so that `embed_size % heads == 0` and `head_dim` is even. The supported 256-dimensional
settings (4, 8, and 16 heads) satisfy both conditions.

### Completed run is skipped

This protects results. Use `--overwrite` only for an intentional rerun. Use `--resume` for an
incomplete run.

### Full repository tests fail before the new tests run

The historical Transformer branch contains unrelated stale imports in some legacy tests. The
focused Transformer tests and smoke command are the authoritative verification for this feature;
repository-wide failures must be separated into pre-existing versus introduced failures.
