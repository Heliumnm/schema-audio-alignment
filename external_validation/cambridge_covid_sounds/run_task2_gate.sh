#!/usr/bin/env bash
# One-command, model-blind Cambridge Task-2 feasibility audit.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash run_task2_gate.sh TASK2_CSV ALL_METADATA_DIR TASK2_AUDIO_DIR OUTPUT_DIR"
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
OUTPUT_DIR=$(mkdir -p "$4" && cd "$4" && pwd)
CONFIG="$OUTPUT_DIR/config.local.json"

python "$HERE/src/configure_task2.py" \
  --task2-csv "$1" \
  --metadata-root "$2" \
  --audio-root "$3" \
  --output-root "$OUTPUT_DIR" \
  --write "$CONFIG" \
  --force

bash "$HERE/run_once.sh" "$CONFIG" gate
