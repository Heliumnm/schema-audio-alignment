#!/usr/bin/env bash
# CODA formal model runner. The preregistration commit must predate this code and all runs.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ENGINE="$REPO/external_validation/cambridge_covid_sounds/src"
CONFIG=${1:-$HERE/config.formal.local.json}
STAGE=${2:-prepare}

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG (copy config.formal.example.json and fill paths)" >&2
  exit 2
fi
case "$STAGE" in
  prepare|text|extract|s0|s1|train|evaluate) ;;
  *) echo "stage must be prepare|text|extract|s0|s1|train|evaluate" >&2; exit 2 ;;
esac

export PYTHONPATH="$ENGINE${PYTHONPATH:+:$PYTHONPATH}"
python -m py_compile "$HERE"/src/*.py "$ENGINE"/*.py

if [[ "$STAGE" == "prepare" ]]; then
  python "$HERE/src/prepare_formal_inputs.py" --config "$CONFIG"
  exit 0
fi

OUTROOT=$(python -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]).resolve(); c=json.load(open(p)); q=pathlib.Path(c["output_root"]); print(q if q.is_absolute() else (p.parent/q).resolve())' "$CONFIG")
MANIFEST="$OUTROOT/private/participant_manifest.csv"
[[ -f "$MANIFEST" ]] || { echo "Run prepare first" >&2; exit 3; }

if [[ "$STAGE" == "text" ]]; then
  python "$ENGINE/cache_text.py" --config "$CONFIG"
  exit 0
fi

if [[ "$STAGE" == "extract" ]]; then
  for backbone in ast opera_ct; do
    python "$ENGINE/extract_embeddings.py" --config "$CONFIG" --backbone "$backbone"
  done
  exit 0
fi

if [[ "$STAGE" == "s0" ]]; then
  python "$HERE/src/smoke_s0_coda.py" --config "$CONFIG"
  exit 0
fi

if [[ "$STAGE" == "s1" ]]; then
  for backbone in ast opera_ct; do
    mkdir -p "$OUTROOT/private/s1/$backbone"
    python "$REPO/src/train_metadata_alignment.py" \
      --cohort "$MANIFEST" --texts "$MANIFEST" \
      --emb "$OUTROOT/models/$backbone/raw_embeddings.npz" \
      --text_emb "$OUTROOT/models/text_embeddings.npz" \
      --out_dir "$OUTROOT/private/s1/$backbone" --seeds 0 --epochs 50 \
      --log_every 10 --device cuda
  done
  exit 0
fi

if [[ "$STAGE" == "train" ]]; then
  for backbone in ast opera_ct; do
    python "$ENGINE/train_alignment.py" --config "$CONFIG" --backbone "$backbone"
  done
  exit 0
fi

if [[ "$STAGE" == "evaluate" ]]; then
  for backbone in ast opera_ct; do
    python "$HERE/src/evaluate_formal.py" --config "$CONFIG" --backbone "$backbone"
  done
  exit 0
fi
