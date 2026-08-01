#!/usr/bin/env bash
set -euo pipefail

project_root=$(git rev-parse --show-toplevel)
cd "$project_root"
if command -v tmux >/dev/null 2>&1 && tmux has-session -t transformer_experiments 2>/dev/null; then
  tmux send-keys -t transformer_experiments C-c
  echo "Sent interrupt to tmux session transformer_experiments"
elif command -v screen >/dev/null 2>&1 && screen -ls | grep -q '[.]transformer_experiments'; then
  screen -S transformer_experiments -X stuff $'\003'
  echo "Sent interrupt to screen session transformer_experiments"
elif [[ -f results/transformer_experiments/pipeline.pid ]]; then
  pipeline_pid=$(<results/transformer_experiments/pipeline.pid)
  kill -INT "$pipeline_pid"
  echo "Sent interrupt to PID $pipeline_pid"
else
  echo "No recorded Transformer pipeline was found"
fi
