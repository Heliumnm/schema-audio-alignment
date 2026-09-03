#!/usr/bin/env bash
# Post-hoc HeAR third-backbone extension.  It never changes AST/OPERA artefacts.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ENGINE="$REPO/external_validation/cambridge_covid_sounds/src"
CONFIG=${1:?usage: run_hear_extension.sh CONFIG [extract|s1|train|audit|evaluate|all]}
STAGE=${2:-all}
HEAR_PYTHON=${HEAR_PYTHON:-python}
ALIGN_PYTHON=${ALIGN_PYTHON:-python}

case "$STAGE" in
  extract|s1|train|audit|evaluate|all) ;;
  *) echo "stage must be extract|s1|train|audit|evaluate|all" >&2; exit 2 ;;
esac

OUTROOT=$($ALIGN_PYTHON -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]).resolve(); c=json.load(open(p)); q=pathlib.Path(c["output_root"]); print(q if q.is_absolute() else (p.parent/q).resolve())' "$CONFIG")
for REQUIRED in \
  "$OUTROOT/public/formal_results_ast.json" \
  "$OUTROOT/public/formal_results_opera_ct.json" \
  "$OUTROOT/public/formal_training_audit.json" \
  "$OUTROOT/private/participant_manifest.csv" \
  "$OUTROOT/models/text_embeddings.npz"; do
  [[ -f "$REQUIRED" ]] || { echo "missing completed primary artefact: $REQUIRED" >&2; exit 3; }
done

$ALIGN_PYTHON -m py_compile "$REPO/src/train_metadata_alignment.py" \
  "$ENGINE/train_alignment.py" "$HERE/src/evaluate_formal.py" \
  "$HERE/src/audit_hear_training.py"
$HEAR_PYTHON -m py_compile "$ENGINE/extract_embeddings.py"

run_extract() {
  "$HEAR_PYTHON" "$ENGINE/extract_embeddings.py" --config "$CONFIG" --backbone hear
}

run_s1() {
  local DEST="$OUTROOT/private/s1/hear"
  mkdir -p "$DEST"
  "$ALIGN_PYTHON" "$REPO/src/train_metadata_alignment.py" \
    --cohort "$OUTROOT/private/participant_manifest.csv" \
    --texts "$OUTROOT/private/participant_manifest.csv" \
    --emb "$OUTROOT/models/hear/raw_embeddings.npz" \
    --text_emb "$OUTROOT/models/text_embeddings.npz" \
    --out_dir "$DEST" --seeds 0 --epochs 50 --log_every 10 --device cuda
  "$ALIGN_PYTHON" - "$DEST/manifest.json" <<'PY'
import json, math, sys
p = json.load(open(sys.argv[1]))
assert p["audio_input_dim"] == 512 and p["epochs"] == 50 and p["seeds"] == [0]
assert set(p["runs"]) == {"correct_seed0", "within_label_seed0", "global_seed0"}
for key, run in p["runs"].items():
    assert math.isfinite(run["loss_first"]) and math.isfinite(run["loss_last"])
    assert run["loss_last"] < run["loss_first"], (key, run)
print("HeAR S1 technical PASS")
PY
}

run_train() {
  "$ALIGN_PYTHON" "$ENGINE/train_alignment.py" --config "$CONFIG" --backbone hear
}

run_audit() {
  "$ALIGN_PYTHON" "$HERE/src/audit_hear_training.py" --config "$CONFIG"
}

run_evaluate() {
  "$ALIGN_PYTHON" "$HERE/src/evaluate_formal.py" --config "$CONFIG" --backbone hear
}

case "$STAGE" in
  extract) run_extract ;;
  s1) run_s1 ;;
  train) run_train ;;
  audit) run_audit ;;
  evaluate) run_evaluate ;;
  all)
    run_extract
    run_s1
    run_train
    run_audit
    run_evaluate
    ;;
esac

