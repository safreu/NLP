#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../.." && pwd)
cd "$project_root"
job_file=results/transformer_experiments/pipeline_job_id.txt
if [[ ! -f "$job_file" ]]; then
  echo "No recorded Transformer pipeline job"
  exit 0
fi
job_id=$(<"$job_file")
scancel "$job_id"
echo "Cancellation requested for Slurm job $job_id"
