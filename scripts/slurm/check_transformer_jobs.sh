#!/usr/bin/env bash
set -euo pipefail

project_root=$(git rev-parse --show-toplevel)
cd "$project_root"
job_file=results/transformer_experiments/pipeline_job_id.txt
if [[ -f "$job_file" ]]; then
  job_id=$(<"$job_file")
  squeue -j "$job_id" || true
fi
uv run transformer-experiments status
