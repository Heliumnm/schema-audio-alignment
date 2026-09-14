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
TEXT_PYTHON="${TEXT_PYTHON:?set TEXT_PYTHON to the absolute frozen text-cache Python executable}"
HEAR_PYTHON="${HEAR_PYTHON:?set HEAR_PYTHON to the absolute HeAR TensorFlow Python executable}"

for executable in "$ALIGN_PYTHON" "$TEXT_PYTHON" "$HEAR_PYTHON"; do
  [[ "$executable" == /* && -x "$executable" ]] || {
    echo "required Python must be an executable absolute path: $executable" >&2
    exit 2
  }
done

for path in "$CODA_CONFIG" "$CAMBRIDGE_CONFIG" "$COSWARA_CONFIG"; do
  [[ "$path" == /* && -f "$path" ]] || { echo "missing absolute config: $path" >&2; exit 2; }
done
[[ "$RUN_ROOT" == /* ]] || { echo "RUN_ROOT must be absolute" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

"$ALIGN_PYTHON" - "$CODA_CONFIG" "$CAMBRIDGE_CONFIG" "$COSWARA_CONFIG" <<'PY'
import json, pathlib, sys
protocol = "locked-external-rerun-20260914-v3"
for path_text in sys.argv[1:]:
    path = pathlib.Path(path_text)
    config = json.load(open(path))
    if config.get("protocol", {}).get("rerun_protocol") != protocol:
        raise SystemExit(f"{path}: protocol.rerun_protocol must be {protocol!r}")
    alignment = config.get("models", {}).get("alignment", {})
    if alignment.get("seeds") != [0, 1, 2, 3, 4] or alignment.get("epochs") != 500:
        raise SystemExit(f"{path}: formal schedule must be seeds 0..4 and 500 epochs")
    for backbone in ("ast", "opera_ct", "hear"):
        if not config.get("models", {}).get(backbone, {}).get("enabled", False):
            raise SystemExit(f"{path}: locked panel requires enabled backbone {backbone}")
print("locked config preflight PASS")
PY

TEXT_TRANSFORMERS_VERSION=$("$TEXT_PYTHON" -c 'import transformers; print(transformers.__version__)')
if [[ "$TEXT_TRANSFORMERS_VERSION" != "4.56.0" ]]; then
  echo "TEXT_PYTHON must use transformers 4.56.0; found $TEXT_TRANSFORMERS_VERSION" >&2
  exit 2
fi

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
  echo "protocol=locked-external-rerun-20260914-v3"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "dispatcher_sha256=$(sha256sum "$DISPATCHER" | awk '{print $1}')"
  echo "alignment_python=$ALIGN_PYTHON"
  echo "text_python=$TEXT_PYTHON"
  echo "text_transformers_version=$TEXT_TRANSFORMERS_VERSION"
  echo "hear_python=$HEAR_PYTHON"
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
  ALIGN_PYTHON="$ALIGN_PYTHON" TEXT_PYTHON="$TEXT_PYTHON" HEAR_PYTHON="$HEAR_PYTHON" \
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
