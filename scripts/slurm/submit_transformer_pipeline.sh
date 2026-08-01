#!/usr/bin/env bash
set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; use scripts/run_transformer_pipeline_background.sh" >&2
  exit 1
fi

project_root=$(git rev-parse --show-toplevel)
cd "$project_root"
mkdir -p results/transformer_experiments

if [[ -f results/transformer_experiments/pipeline_job_id.txt ]]; then
  existing_job=$(<results/transformer_experiments/pipeline_job_id.txt)
  if squeue -h -j "$existing_job" 2>/dev/null | grep -q .; then
    echo "Transformer pipeline is already active as Slurm job $existing_job" >&2
    exit 1
  fi
fi

submission=$(sbatch "$@" scripts/slurm/run_transformer_pipeline.sbatch)
job_id=${submission##* }
printf '%s\n' "$job_id" > results/transformer_experiments/pipeline_job_id.txt
echo "Submitted Transformer pipeline job $job_id"
echo "Logs: results/transformer_experiments/slurm-$job_id.out and slurm-$job_id.err"
