#!/usr/bin/env bash
# Clean external rerun dispatcher for the 2026-09-14 locked protocol.
#
# This script intentionally reuses existing dataset runners but writes to the output_root
# specified by each private config. Use fresh output roots; do not overwrite archived runs.
#
# Usage:
#   bash scripts/run_clean_external_rerun_20260914.sh coda /private/coda_config.json all
#   bash scripts/run_clean_external_rerun_20260914.sh cambridge /private/cambridge_config.json all
#   bash scripts/run_clean_external_rerun_20260914.sh coswara /private/coswara_config.json all
#
# Environment:
#   ALIGN_PYTHON   Python executable for PyTorch/alignment code. Defaults to python.
#   TEXT_PYTHON    Python executable pinned to the discovery text-cache stack.
#   HEAR_PYTHON    Python executable for HeAR/TensorFlow extraction. Defaults to ALIGN_PYTHON.
#   PYTHON_BIN     Required by the Cambridge runner; this dispatcher sets it from ALIGN_PYTHON.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/run_clean_external_rerun_20260914.sh {coda|cambridge|coswara} CONFIG [stage|all]" >&2
  exit 2
fi

DATASET="$1"
CONFIG="$2"
STAGE="${3:-all}"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ALIGN_PYTHON="${ALIGN_PYTHON:-python}"
TEXT_PYTHON="${TEXT_PYTHON:-$ALIGN_PYTHON}"
HEAR_PYTHON="${HEAR_PYTHON:-$ALIGN_PYTHON}"

if [[ "$ALIGN_PYTHON" == /* ]]; then
  ALIGN_BIN_DIR=$(cd "$(dirname "$ALIGN_PYTHON")" && pwd)
  export PATH="$ALIGN_BIN_DIR:$PATH"
fi

if [[ "$CONFIG" != /* ]]; then
  CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 2
fi

case "$DATASET" in
  coda|cambridge|coswara) ;;
  *) echo "Dataset must be coda, cambridge, or coswara" >&2; exit 2 ;;
esac

is_hear_enabled() {
  "$ALIGN_PYTHON" - "$CONFIG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(str(c.get("models", {}).get("hear", {}).get("enabled", False)).lower())
PY
}

run_coda_all() {
  local runner="${REPO_ROOT}/external_validation/coda_tb/run_formal_models.sh"
  for stage in prepare s0 text extract s1 train audit-train evaluate; do
    ALIGN_PYTHON="$ALIGN_PYTHON" TEXT_PYTHON="$TEXT_PYTHON" bash "$runner" "$CONFIG" "$stage"
  done
  if [[ "$(is_hear_enabled)" == "true" ]]; then
    ALIGN_PYTHON="$ALIGN_PYTHON" TEXT_PYTHON="$TEXT_PYTHON" HEAR_PYTHON="$HEAR_PYTHON" \
      bash "${REPO_ROOT}/external_validation/coda_tb/run_hear_extension.sh" "$CONFIG" all
  fi
}

run_cambridge_all() {
  PYTHON_BIN="$ALIGN_PYTHON" TEXT_PYTHON="$TEXT_PYTHON" HEAR_PYTHON="$HEAR_PYTHON" \
    bash "${REPO_ROOT}/external_validation/cambridge_covid_sounds/run_once.sh" "$CONFIG" all
}

run_coswara_all() {
  local runner="${REPO_ROOT}/external_validation/coswara/run_formal_models.sh"
  local stages=(prepare text s0 extract-ast extract-opera_ct extract-hear s1-ast s1-opera_ct s1-hear train-ast train-opera_ct train-hear audit evaluate-ast evaluate-opera_ct evaluate-hear)
  for stage in "${stages[@]}"; do
    ALIGN_PYTHON="$ALIGN_PYTHON" TEXT_PYTHON="$TEXT_PYTHON" HEAR_PYTHON="$HEAR_PYTHON" bash "$runner" "$CONFIG" "$stage"
  done
}

case "$DATASET:$STAGE" in
  coda:all) run_coda_all ;;
  cambridge:all) run_cambridge_all ;;
  coswara:all) run_coswara_all ;;
  coda:*) ALIGN_PYTHON="$ALIGN_PYTHON" TEXT_PYTHON="$TEXT_PYTHON" bash "${REPO_ROOT}/external_validation/coda_tb/run_formal_models.sh" "$CONFIG" "$STAGE" ;;
  cambridge:*) PYTHON_BIN="$ALIGN_PYTHON" TEXT_PYTHON="$TEXT_PYTHON" HEAR_PYTHON="$HEAR_PYTHON" bash "${REPO_ROOT}/external_validation/cambridge_covid_sounds/run_once.sh" "$CONFIG" "$STAGE" ;;
  coswara:*) ALIGN_PYTHON="$ALIGN_PYTHON" TEXT_PYTHON="$TEXT_PYTHON" HEAR_PYTHON="$HEAR_PYTHON" bash "${REPO_ROOT}/external_validation/coswara/run_formal_models.sh" "$CONFIG" "$STAGE" ;;
esac
