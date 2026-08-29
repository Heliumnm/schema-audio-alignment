#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 DATA_ROOT OUTPUT_DIR" >&2
  exit 2
fi

DATA_ROOT=$1
OUTPUT_DIR=$2
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUTPUT_DIR"
python3 "$SCRIPT_DIR/src/data_gate.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR"

