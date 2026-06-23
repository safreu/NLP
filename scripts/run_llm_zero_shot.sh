set -euo pipefail
cd "$(dirname "$0")/.."
module load devel/cuda/12.8 2>/dev/null || true
# load hf token
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

MODEL_NAME="${MODEL_NAME:-google/gemma-4-12B-it}"
SPLIT="${SPLIT:-test}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"   # 0 = full split
PREDICTIONS_PATH="results/asset_llm_zero_shot_predictions.json"
SCORE_PATH="results/asset_llm_zero_shot_score.json"

mkdir -p results

# generate model result
uv run python -m pipeline.llm_zero_shot_pipeline generate \
    --model-name "$MODEL_NAME" \
    --split "$SPLIT" \
    --max-examples "$MAX_EXAMPLES" \
    --predictions-path "$PREDICTIONS_PATH"

# calculate SARI score″
uv run python -m pipeline.llm_zero_shot_pipeline score \
    --predictions-path "$PREDICTIONS_PATH" \
    --score-path "$SCORE_PATH" \
    --model-name "$MODEL_NAME" \
    --split "$SPLIT" \
    --max-examples "$MAX_EXAMPLES"

echo "Fertig. Score: $SCORE_PATH"

