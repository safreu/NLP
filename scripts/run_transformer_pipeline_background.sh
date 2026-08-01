#!/usr/bin/env bash
set -euo pipefail

project_root=$(git rev-parse --show-toplevel)
cd "$project_root"
output_root=results/transformer_experiments
mkdir -p "$output_root"
pid_file="$output_root/pipeline.pid"

if [[ -f "$pid_file" ]]; then
  existing_pid=$(<"$pid_file")
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "Transformer pipeline already runs as PID $existing_pid" >&2
    exit 1
  fi
fi

command="cd '$project_root' && uv run transformer-experiments smoke --output-root '$output_root' && uv run transformer-experiments run-all --output-root '$output_root' --device auto --resume"
if command -v tmux >/dev/null 2>&1; then
  session=transformer_experiments
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" "$command 2>&1 | tee '$output_root/pipeline.log'"
  echo "Started tmux session $session"
  echo "Attach: tmux attach -t $session"
elif command -v screen >/dev/null 2>&1; then
  session=transformer_experiments
  if screen -ls | grep -q "[.]$session"; then
    echo "screen session $session already exists" >&2
    exit 1
  fi
  screen -dmS "$session" bash -lc "$command > '$output_root/pipeline.log' 2>&1"
  echo "Started screen session $session"
  echo "Attach: screen -r $session"
else
  nohup bash -lc "$command" > "$output_root/pipeline.log" 2>&1 &
  pipeline_pid=$!
  printf '%s\n' "$pipeline_pid" > "$pid_file"
  echo "Started nohup process $pipeline_pid"
fi
echo "Log: $output_root/pipeline.log"
