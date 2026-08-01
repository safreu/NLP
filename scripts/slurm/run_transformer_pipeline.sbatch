#!/usr/bin/env bash
#SBATCH --job-name=dual-decoder-transformer
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=results/transformer_experiments/slurm-%j.out
#SBATCH --error=results/transformer_experiments/slurm-%j.err

set -euo pipefail

PROJECT_ROOT="${NLP_PROJECT_ROOT:-$(pwd)}"
OUTPUT_ROOT="${TRANSFORMER_OUTPUT_ROOT:-results/transformer_experiments}"
mkdir -p "$PROJECT_ROOT/$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

STATUS_FILE="$OUTPUT_ROOT/pipeline_status.json"
finish() {
  exit_code=$?
  PIPELINE_EXIT_CODE="$exit_code" PIPELINE_STATUS_FILE="$STATUS_FILE" uv run python -c \
    'import json, os; from datetime import datetime, timezone; p=os.environ["PIPELINE_STATUS_FILE"]; json.dump({"status": "completed" if os.environ["PIPELINE_EXIT_CODE"] == "0" else "failed", "job_id": os.environ.get("SLURM_JOB_ID"), "end_time": datetime.now(timezone.utc).isoformat(), "exit_code": int(os.environ["PIPELINE_EXIT_CODE"])}, open(p, "w"), indent=2)'
}
trap finish EXIT TERM INT

hostname
date --iso-8601=seconds
lscpu | sed -n '1,20p'
free -h
nvidia-smi || true
uv run python -c 'import platform, torch; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda)'

PIPELINE_STATUS_FILE="$STATUS_FILE" uv run python -c \
  'import json, os; from datetime import datetime, timezone; json.dump({"status": "running", "job_id": os.environ.get("SLURM_JOB_ID"), "start_time": datetime.now(timezone.utc).isoformat()}, open(os.environ["PIPELINE_STATUS_FILE"], "w"), indent=2)'

uv run transformer-experiments smoke --output-root "$OUTPUT_ROOT"
uv run transformer-experiments run-all --output-root "$OUTPUT_ROOT" --device auto --resume
