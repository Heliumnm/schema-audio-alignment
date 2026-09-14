#!/usr/bin/env bash
# One-shot execution of the locked 2026-09-14 external panel.
# Order is frozen: CODA TB -> Cambridge COVID-19 Sounds -> Coswara.
# Each dataset config must use a fresh output_root.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash scripts/run_locked_external_panel_20260914.sh CODA_CONFIG CAMBRIDGE_CONFIG COSWARA_CONFIG RUN_ROOT" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DISPATCHER="${SCRIPT_DIR}/run_clean_external_rerun_20260914.sh"
CODA_CONFIG="$1"
CAMBRIDGE_CONFIG="$2"
COSWARA_CONFIG="$3"
RUN_ROOT="$4"
ALIGN_PYTHON="${ALIGN_PYTHON:?set ALIGN_PYTHON to the absolute PyTorch Python executable}"
HEAR_PYTHON="${HEAR_PYTHON:?set HEAR_PYTHON to the absolute HeAR TensorFlow Python executable}"

for path in "$CODA_CONFIG" "$CAMBRIDGE_CONFIG" "$COSWARA_CONFIG"; do
  [[ "$path" == /* && -f "$path" ]] || { echo "missing absolute config: $path" >&2; exit 2; }
done
[[ "$RUN_ROOT" == /* ]] || { echo "RUN_ROOT must be absolute" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

exec 9>"$RUN_ROOT/panel.lock"
if command -v flock >/dev/null 2>&1; then
  flock -n 9 || { echo "another locked panel is using $RUN_ROOT" >&2; exit 3; }
fi

rm_marker="$RUN_ROOT/LOCKED_EXTERNAL_PANEL_FAILED.txt"
complete_marker="$RUN_ROOT/LOCKED_EXTERNAL_PANEL_COMPLETE.txt"
if [[ -f "$complete_marker" ]]; then
  echo "locked external panel is already complete: $complete_marker"
  exit 0
fi

on_failure() {
  status=$?
  {
    echo "status=FAILED"
    echo "exit_code=$status"
    echo "failed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$rm_marker"
  exit "$status"
}
trap on_failure ERR

{
  echo "protocol=locked-external-rerun-20260914-v2"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "dispatcher_sha256=$(sha256sum "$DISPATCHER" | awk '{print $1}')"
  echo "coda_config_sha256=$(sha256sum "$CODA_CONFIG" | awk '{print $1}')"
  echo "cambridge_config_sha256=$(sha256sum "$CAMBRIDGE_CONFIG" | awk '{print $1}')"
  echo "coswara_config_sha256=$(sha256sum "$COSWARA_CONFIG" | awk '{print $1}')"
  echo "primary_backbone=ast"
  echo "robustness_backbones=opera_ct,hear"
  echo "arms=raw_audio,correct,within_label,within_label_sex,global"
  echo "primary_contrast=C-Wys"
  echo "primary_metric=matched_target_participant_AUROC"
} > "$RUN_ROOT/run_manifest.txt"

for dataset in coda cambridge coswara; do
  case "$dataset" in
    coda) config="$CODA_CONFIG" ;;
    cambridge) config="$CAMBRIDGE_CONFIG" ;;
    coswara) config="$COSWARA_CONFIG" ;;
  esac
  {
    echo "dataset=$dataset"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$RUN_ROOT/${dataset}.running"
  ALIGN_PYTHON="$ALIGN_PYTHON" HEAR_PYTHON="$HEAR_PYTHON" \
    bash "$DISPATCHER" "$dataset" "$config" all
  {
    echo "dataset=$dataset"
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$RUN_ROOT/${dataset}.complete"
done

{
  echo "status=COMPLETE"
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "primary_contrast=C-Wys"
  echo "datasets=CODA_TB,Cambridge_COVID19_Sounds,Coswara"
} > "$complete_marker"
rm -f "$rm_marker"
trap - ERR
echo "locked external panel complete: $complete_marker"
