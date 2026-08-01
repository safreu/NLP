#!/usr/bin/env bash
#SBATCH --job-name=newsela-classical
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=newsela-classical-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/NLP2}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/project-data-science-2026/nvidia+pytorch+26.04-py3.sqsh}"

test -d "$PROJECT_DIR"
test -f "$CONTAINER_IMAGE"
test -x "$PROJECT_DIR/.venv-newsela/bin/python"
test -f "$PROJECT_DIR/private_data/_env"
test -f "$PROJECT_DIR/private_data/newsela_articles_20150302.aligned.sents.pkl.enc"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

apptainer exec \
  --nv \
  --bind "$PROJECT_DIR:/work" \
  --pwd /work \
  "$CONTAINER_IMAGE" \
  /usr/bin/bash -c '
    export PATH="/work/.venv-newsela/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
    export LD_LIBRARY_PATH="/.singularity.d/libs"
    cd /work
    python -m pipeline.newsela_classical_pipeline \
      --encrypted-cache /work/private_data/newsela_articles_20150302.aligned.sents.pkl.enc \
      --env-file /work/private_data/_env \
      --output-path /work/runs/newsela_classical_full
  '
