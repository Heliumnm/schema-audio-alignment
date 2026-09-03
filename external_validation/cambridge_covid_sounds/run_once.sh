#!/usr/bin/env bash
# One entry point for the controlled-access collaborator. No raw data leave output/private.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CONFIG=${1:-$HERE/config.local.json}
STAGE=${2:-all}
PYTHON_BIN=${PYTHON_BIN:?set PYTHON_BIN to the absolute formal Python executable}

if [[ "$PYTHON_BIN" != /* || ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN must be an executable absolute path: $PYTHON_BIN" >&2
  exit 2
fi
CONFIG=$("$PYTHON_BIN" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$CONFIG")

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG"
  echo "Use run_reconstructed_gate.sh when the split CSV is absent, run_task2_gate.sh for the official raw release, or copy config.example.json for a canonical table."
  exit 2
fi
if [[ "$STAGE" != "gate" && "$STAGE" != "formal" && "$STAGE" != "all" ]]; then
  echo "Stage must be gate, formal, or all"; exit 2
fi

OUTROOT=$("$PYTHON_BIN" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]).resolve(); c=json.load(open(p)); q=pathlib.Path(c["output_root"]); print((q if q.is_absolute() else p.parent/q).resolve())' "$CONFIG")
mkdir -p "$OUTROOT"
exec 9>"$OUTROOT/run.lock"
if command -v flock >/dev/null 2>&1; then
  flock -n 9 || { echo "Another Cambridge audit is already using $OUTROOT"; exit 3; }
else
  LOCKDIR="$OUTROOT/.run_lock_dir"
  mkdir "$LOCKDIR" 2>/dev/null || { echo "Another Cambridge audit may already be using $OUTROOT"; exit 3; }
  trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
fi

"$PYTHON_BIN" -m py_compile "$HERE"/src/*.py
"$PYTHON_BIN" "$HERE/src/environment_report.py" --config "$CONFIG"
"$PYTHON_BIN" "$HERE/src/data_gate.py" --config "$CONFIG"
VERDICT=$("$PYTHON_BIN" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]).resolve(); c=json.load(open(p)); q=pathlib.Path(c["output_root"]); root=(q if q.is_absolute() else p.parent/q).resolve(); print(json.load(open(root/"public"/"data_gate.json"))["verdict"])' "$CONFIG")
if [[ "$VERDICT" != "GO" ]]; then
  echo "Data gate returned $VERDICT. Formal model execution is intentionally closed."
  "$PYTHON_BIN" "$HERE/src/package_results.py" --config "$CONFIG"
  exit 0
fi
if [[ "$STAGE" == "gate" ]]; then
  echo "Data gate GO. Review public/data_gate.json, then set the formal unlock token."
  "$PYTHON_BIN" "$HERE/src/package_results.py" --config "$CONFIG"
  exit 0
fi

"$PYTHON_BIN" "$HERE/src/cache_text.py" --config "$CONFIG"
for BACKBONE in ast opera_ct; do
  ENABLED=$("$PYTHON_BIN" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(str(c["models"][sys.argv[2]].get("enabled",False)).lower())' "$CONFIG" "$BACKBONE")
  [[ "$ENABLED" == "true" ]] || continue
  "$PYTHON_BIN" "$HERE/src/extract_embeddings.py" --config "$CONFIG" --backbone "$BACKBONE"
  "$PYTHON_BIN" "$HERE/src/train_alignment.py" --config "$CONFIG" --backbone "$BACKBONE"
  "$PYTHON_BIN" "$HERE/src/evaluate.py" --config "$CONFIG" --backbone "$BACKBONE"
done
"$PYTHON_BIN" "$HERE/src/package_results.py" --config "$CONFIG"
echo "External audit complete. Return only output/public/PUBLIC_RESULTS.zip."
