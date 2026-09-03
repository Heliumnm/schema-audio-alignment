#!/usr/bin/env bash
# Post-hoc HeAR replication of the frozen UKCOVID pairing-controlled main audit.
set -euo pipefail

ROOT=${1:-/mnt/hd/data_heliu/audio_provenance}
STAGE=${2:-all}
HEAR_PYTHON=${HEAR_PYTHON:-/mnt/hd/data_heliu/venvs/hear_tf218/bin/python}
ALIGN_PYTHON=${ALIGN_PYTHON:-/home/heliu/anaconda3/bin/python}
HEAR_MODEL=${HEAR_MODEL:-/mnt/hd/data_heliu/hf_models/google_hear_9b2eb285}
UKCOVID_AUDIO=${UKCOVID_AUDIO:-/mnt/hd/data_heliu/resp_datasets/ukcovid/audio/audio}
N_SHARDS=${N_SHARDS:-6}
cd "$ROOT"
mkdir -p results/hear_logs results/hear_shards

case "$STAGE" in
  extract|train|evaluate|all) ;;
  *) echo "stage must be extract|train|evaluate|all" >&2; exit 2 ;;
esac

for REQUIRED in \
  results/ast_embeddings.npz results/opera_ct_embeddings.npz \
  results/alignment/manifest.json results/alignment_opera_ct/manifest.json \
  results/ukcovid_audio_cohort.csv results/metadata_texts.csv \
  results/metadata_text_embeddings.npz; do
  [[ -f "$REQUIRED" ]] || { echo "missing frozen primary artefact: $REQUIRED" >&2; exit 3; }
done

run_extract() {
  "$HEAR_PYTHON" src/extract_hear_ukcovid.py --self-test
  CUDA_VISIBLE_DEVICES=0 "$HEAR_PYTHON" src/extract_hear_ukcovid.py \
    --preflight --cohort results/ukcovid_audio_cohort.csv \
    --audio-root "$UKCOVID_AUDIO" --model "$HEAR_MODEL" \
    --preflight-out results/hear_preflight.json
  local pids=()
  local shard gpu
  for shard in $(seq 0 $((N_SHARDS - 1))); do
    gpu=$((shard % 6))
    CUDA_VISIBLE_DEVICES="$gpu" "$HEAR_PYTHON" src/extract_hear_ukcovid.py \
      --extract-shard --cohort results/ukcovid_audio_cohort.csv \
      --audio-root "$UKCOVID_AUDIO" --model "$HEAR_MODEL" \
      --preflight-out results/hear_preflight.json \
      --shard-dir results/hear_shards --num-shards "$N_SHARDS" \
      --shard-index "$shard" \
      >"results/hear_logs/extract_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]] || { tail -n 60 results/hear_logs/extract_shard_*.log; return 4; }
  "$HEAR_PYTHON" src/extract_hear_ukcovid.py --merge \
    --cohort results/ukcovid_audio_cohort.csv --model "$HEAR_MODEL" \
    --shard-dir results/hear_shards --num-shards "$N_SHARDS" \
    --merge-out results/hear_embeddings.npz
}

run_train() {
  "$ALIGN_PYTHON" src/train_metadata_alignment.py \
    --cohort results/ukcovid_audio_cohort.csv --texts results/metadata_texts.csv \
    --emb results/hear_embeddings.npz --text_emb results/metadata_text_embeddings.npz \
    --rehearsal --device cuda
  "$ALIGN_PYTHON" src/train_metadata_alignment.py \
    --cohort results/ukcovid_audio_cohort.csv --texts results/metadata_texts.csv \
    --emb results/hear_embeddings.npz --text_emb results/metadata_text_embeddings.npz \
    --out_dir results/alignment_hear --seeds 0 1 2 3 4 --epochs 500 \
    --log_every 50 --device cuda
}

run_evaluate() {
  "$ALIGN_PYTHON" src/eval_metadata_alignment.py \
    --data /mnt/hd/data_heliu/resp_datasets/ukcovid \
    --cohort results/ukcovid_audio_cohort.csv \
    --ast-emb results/hear_embeddings.npz --alignment-dir results/alignment_hear \
    --out-dir results/alignment_hear_eval --variants raw \
    --run-tag hear-raw-disease-probes --evaluate-tests
  "$ALIGN_PYTHON" src/audit_profile_retrieval.py \
    --preflight --backbone HeAR --audio-emb results/hear_embeddings.npz \
    --alignment-dir results/alignment_hear --out-dir results/profile_retrieval_hear
  "$ALIGN_PYTHON" src/audit_profile_retrieval.py \
    --run --backbone HeAR --audio-emb results/hear_embeddings.npz \
    --alignment-dir results/alignment_hear --out-dir results/profile_retrieval_hear
  "$ALIGN_PYTHON" src/audit_information_channels.py \
    --preflight --backbone HeAR --audio-emb results/hear_embeddings.npz \
    --alignment-dir results/alignment_hear --out-dir results/information_channels_hear
  "$ALIGN_PYTHON" src/audit_information_channels.py \
    --fit --backbone HeAR --audio-emb results/hear_embeddings.npz \
    --alignment-dir results/alignment_hear --out-dir results/information_channels_hear
  "$ALIGN_PYTHON" src/audit_information_channels.py \
    --score-tests --backbone HeAR --out-dir results/information_channels_hear
}

case "$STAGE" in
  extract) run_extract ;;
  train) run_train ;;
  evaluate) run_evaluate ;;
  all) run_extract; run_train; run_evaluate ;;
esac
