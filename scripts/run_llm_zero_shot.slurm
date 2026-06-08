#!/bin/bash
#SBATCH --job-name=llm-zero-shot
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/llm_zero_shot_%j.out
#SBATCH --error=runs/llm_zero_shot_%j.err

set -euo pipefail
module load devel/cuda/12.8 || true
cd "$SLURM_SUBMIT_DIR"
[ -f .env ] && { set -a; source .env; set +a; }
uv sync --dev

uv run python -m pipeline.llm_zero_shot_pipeline generate \
    --model-name "${MODEL_NAME:-google/gemma-4-12B-it}" \
    --split "${SPLIT:-test}" \
    --max-examples "${MAX_EXAMPLES:-0}" \
    --predictions-path results/asset_llm_zero_shot_predictions.json

uv run python -m pipeline.llm_zero_shot_pipeline score \
    --predictions-path results/asset_llm_zero_shot_predictions.json \
    --score-path results/asset_llm_zero_shot_score.json \
    --model-name "${MODEL_NAME:-google/gemma-4-12B-it}" \
    --split "${SPLIT:-test}" \
    --max-examples "${MAX_EXAMPLES:-0}"
