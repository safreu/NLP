#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../.." && pwd)
cd "$project_root"
job_file=results/transformer_experiments/pipeline_job_id.txt
if [[ -f "$job_file" ]]; then
  job_id=$(<"$job_file")
  squeue -j "$job_id" || true
fi
if command -v uv >/dev/null 2>&1; then
  uv run transformer-experiments status
else
  echo "Detailed experiment status is available in results/transformer_experiments/*/status.json"
fi
