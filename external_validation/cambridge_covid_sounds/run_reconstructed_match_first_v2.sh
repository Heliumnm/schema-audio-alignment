#!/usr/bin/env bash
# One-command model-blind Task-2 reconstruction and target-first matching sensitivity gate.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash run_reconstructed_match_first_v2.sh ALL_METADATA_DIR TASK2_AUDIO_ROOT OUTPUT_DIR TASK1_AUDIO_ROOT"
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
OUTPUT_DIR=$(mkdir -p "$3" && cd "$3" && pwd)
CONFIG="$OUTPUT_DIR/config.local.json"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

python "$HERE/src/prepare_reconstructed_cohort.py" \
  --metadata-root "$1" \
  --audio-root "$2" \
  --output-root "$OUTPUT_DIR" \
  --write-config "$CONFIG" \
  --split-strategy match_first_v2

bash "$HERE/run_once.sh" "$CONFIG" gate

# This report is deliberately generated after the real-WAV data gate: it checks the frozen
# train/validation/matched manifests and reuses the gate's decoded-PCM hashes. Task 1 is
# inventoried only for cross-task provenance; it is never used for model fitting.
python "$HERE/src/overlap_report.py" \
  --task1-root "$4" \
  --task2-root "$2" \
  --output-root "$OUTPUT_DIR"
python "$HERE/src/package_results.py" --config "$CONFIG"
