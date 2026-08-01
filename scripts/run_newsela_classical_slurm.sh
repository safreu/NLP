#!/usr/bin/env bash
#SBATCH --job-name=newsela-classical
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=newsela-classical-%j.out

set -euo pipefail

: "${NEWSELA_CACHE:?Set NEWSELA_CACHE to the encrypted Newsela .pkl.enc file}"
: "${NEWSELA_ENV_FILE:?Set NEWSELA_ENV_FILE to the file containing NEWSELA_CACHE_KEY}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH to the run output directory}"

uv sync --frozen
uv run python -m pipeline.newsela_classical_pipeline \
  --encrypted-cache "${NEWSELA_CACHE}" \
  --env-file "${NEWSELA_ENV_FILE}" \
  --output-path "${OUTPUT_PATH}"
