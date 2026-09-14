#!/usr/bin/env bash
# Frozen Coswara post-hoc three-backbone stress-test runner.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ENGINE="$REPO/external_validation/cambridge_covid_sounds/src"
CONFIG=${1:?usage: run_formal_models.sh CONFIG STAGE}
STAGE=${2:-prepare}
ALIGN_PYTHON=${ALIGN_PYTHON:-python}
HEAR_PYTHON=${HEAR_PYTHON:-python}
HEAR_CUDNN_LIB=${HEAR_CUDNN_LIB:-$(dirname "$(dirname "$HEAR_PYTHON")")/lib/python3.11/site-packages/nvidia/cudnn/lib}
if [[ -d "$HEAR_CUDNN_LIB" ]]; then
  export LD_LIBRARY_PATH="$HEAR_CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$ENGINE:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

case "$STAGE" in
  prepare|text|s0|extract-ast|extract-opera_ct|extract-hear|s1-ast|s1-opera_ct|s1-hear|train-ast|train-opera_ct|train-hear|audit|evaluate-ast|evaluate-opera_ct|evaluate-hear) ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

OUTROOT=$($ALIGN_PYTHON -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]).resolve(); c=json.load(open(p)); q=pathlib.Path(c["output_root"]); print(q if q.is_absolute() else (p.parent/q).resolve())' "$CONFIG")

case "$STAGE" in
  prepare)
    "$ALIGN_PYTHON" "$HERE/src/prepare_formal_inputs.py" --config "$CONFIG" ;;
  text)
    "$ALIGN_PYTHON" "$ENGINE/cache_text.py" --config "$CONFIG" ;;
  s0)
    "$ALIGN_PYTHON" "$REPO/external_validation/coda_tb/src/smoke_s0_coda.py" --config "$CONFIG" ;;
  extract-ast)
    "$ALIGN_PYTHON" "$ENGINE/extract_embeddings.py" --config "$CONFIG" --backbone ast ;;
  extract-opera_ct)
    "$ALIGN_PYTHON" "$ENGINE/extract_embeddings.py" --config "$CONFIG" --backbone opera_ct ;;
  extract-hear)
    "$HEAR_PYTHON" "$ENGINE/extract_embeddings.py" --config "$CONFIG" --backbone hear ;;
  s1-ast|s1-opera_ct|s1-hear)
    backbone=${STAGE#s1-}
    destination="$OUTROOT/private/s1/$backbone"
    mkdir -p "$destination"
    "$ALIGN_PYTHON" "$REPO/src/train_metadata_alignment.py" \
      --cohort "$OUTROOT/private/participant_manifest.csv" \
      --texts "$OUTROOT/private/participant_manifest.csv" \
      --emb "$OUTROOT/models/$backbone/raw_embeddings.npz" \
      --text_emb "$OUTROOT/models/text_embeddings.npz" \
      --out_dir "$destination" --seeds 0 --epochs 50 --log_every 10 --device cuda \
      --include-within-label-sex --sex-column sex ;;
  train-ast)
    "$ALIGN_PYTHON" "$ENGINE/train_alignment.py" --config "$CONFIG" --backbone ast ;;
  train-opera_ct)
    "$ALIGN_PYTHON" "$ENGINE/train_alignment.py" --config "$CONFIG" --backbone opera_ct ;;
  train-hear)
    "$ALIGN_PYTHON" "$ENGINE/train_alignment.py" --config "$CONFIG" --backbone hear ;;
  audit)
    "$ALIGN_PYTHON" "$HERE/src/audit_training.py" --config "$CONFIG" ;;
  evaluate-ast|evaluate-opera_ct|evaluate-hear)
    backbone=${STAGE#evaluate-}
    "$ALIGN_PYTHON" "$ENGINE/evaluate.py" --config "$CONFIG" --backbone "$backbone" ;;
esac
