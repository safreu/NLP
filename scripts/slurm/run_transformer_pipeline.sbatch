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

run_python() {
  if command -v uv >/dev/null 2>&1; then
    PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" uv run python "$@"
    return
  fi

  runtime_root="${UC3_RUNTIME_ROOT:-$HOME/project-data-science-2026}"
  image="${UC3_APPTAINER_IMAGE:-$runtime_root/nvidia+pytorch+26.04-py3.sqsh}"
  python_bin="$runtime_root/.venv/bin/python"
  if ! command -v apptainer >/dev/null 2>&1 || [[ ! -f "$image" ]] || [[ ! -x "$python_bin" ]]; then
    echo "Neither uv nor the UC3 Apptainer runtime is available." >&2
    echo "Expected image: $image" >&2
    echo "Expected Python: $python_bin" >&2
    return 127
  fi

  mkdir -p "$runtime_root/.cuda-lib"
  ln -sfn /usr/local/cuda/compat/lib/libcuda.so.1 "$runtime_root/.cuda-lib/libcuda.so"

  apptainer exec \
    --nv \
    --bind "$PROJECT_ROOT:/workspace" \
    --bind "$runtime_root:/work" \
    --pwd /workspace \
    "$image" \
    env \
      JAVA_HOME=/work/.jdk/temurin-21 \
      PATH=/work/.jdk/temurin-21/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin \
      LD_LIBRARY_PATH=/work/.cuda-lib:/.singularity.d/libs:/usr/local/cuda/compat/lib:/work/.jdk/temurin-21/lib/server \
      OPENAI_API_KEY=unused-placeholder \
      PYTHONPATH=/work/.transformer-deps:/workspace/src \
      /work/.venv/bin/python "$@"
}

STATUS_FILE="$OUTPUT_ROOT/pipeline_status.json"
finish() {
  exit_code=$?
  PIPELINE_EXIT_CODE="$exit_code" PIPELINE_STATUS_FILE="$STATUS_FILE" run_python -c \
    'import json, os; from datetime import datetime, timezone; p=os.environ["PIPELINE_STATUS_FILE"]; json.dump({"status": "completed" if os.environ["PIPELINE_EXIT_CODE"] == "0" else "failed", "job_id": os.environ.get("SLURM_JOB_ID"), "end_time": datetime.now(timezone.utc).isoformat(), "exit_code": int(os.environ["PIPELINE_EXIT_CODE"])}, open(p, "w"), indent=2)'
}
trap finish EXIT TERM INT

hostname
date --iso-8601=seconds
lscpu | sed -n '1,20p'
free -h
nvidia-smi || true
run_python -c 'import platform, torch; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))'

PIPELINE_STATUS_FILE="$STATUS_FILE" run_python -c \
  'import json, os; from datetime import datetime, timezone; json.dump({"status": "running", "job_id": os.environ.get("SLURM_JOB_ID"), "start_time": datetime.now(timezone.utc).isoformat()}, open(os.environ["PIPELINE_STATUS_FILE"], "w"), indent=2)'

run_python -m pipeline.transformer_experiments smoke --output-root "$OUTPUT_ROOT"
run_python -m pipeline.transformer_experiments run-all --output-root "$OUTPUT_ROOT" --device auto --resume
