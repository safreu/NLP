#!/usr/bin/env bash
set -euo pipefail

project_root=$(git rev-parse --show-toplevel)
cd "$project_root"
if command -v tmux >/dev/null 2>&1; then tmux ls 2>/dev/null || true; fi
if command -v screen >/dev/null 2>&1; then screen -ls 2>/dev/null || true; fi
if [[ -f results/transformer_experiments/pipeline.pid ]]; then
  pipeline_pid=$(<results/transformer_experiments/pipeline.pid)
  ps -p "$pipeline_pid" -o pid,etime,%cpu,%mem,command || true
fi
uv run transformer-experiments status
