#!/usr/bin/env bash
# Execute the frozen Route-A follow-up exactly once on the UKCOVID discovery cohort.
set -euo pipefail

ROOT=${1:-/mnt/hd/data_heliu/audio_provenance}
cd "$ROOT"
mkdir -p results/route_a_logs
exec 9>results/route_a_once.lock
flock -n 9 || { echo "another Route-A run holds results/route_a_once.lock"; exit 1; }

source ~/anaconda3/etc/profile.d/conda.sh
conda activate qwen-audio
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
ROUTE_A_GPU=${ROUTE_A_GPU:-2}

MASTER_LOG=results/route_a_logs/master.log
exec >>"$MASTER_LOG" 2>&1

phase() {
  echo
  echo "[$(date -u +%FT%TZ)] $*"
}

wait_all() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  test "$failed" -eq 0
}

phase "environment and frozen-code smoke tests"
python --version
python -m py_compile \
  src/audit_profile_retrieval.py \
  src/audit_information_channels.py \
  src/eval_probability_transport.py \
  src/eval_fusion_followup.py \
  src/synthetic_correspondence_transfer.py \
  src/build_synthetic_phi2_profiles.py \
  src/eval_fixed_mlp_readout.py
python src/audit_profile_retrieval.py --self-test
python src/audit_information_channels.py --self-test
python src/eval_probability_transport.py --self-test
python src/eval_fusion_followup.py --self-test
python src/synthetic_correspondence_transfer.py --self-test
python src/build_synthetic_phi2_profiles.py --self-test
python src/eval_fixed_mlp_readout.py --self-test
python src/synthetic_correspondence_transfer.py \
  --mode smoke --device cpu \
  --out-dir results/synthetic_correspondence_transfer_smoke

phase "E3a: probability transport for already frozen audio-only predictions"
python src/eval_probability_transport.py \
  --source ast_audio=results/alignment_eval/metadata_alignment_predictions__raw_disease_primary.npz \
  --source opera_audio=results/alignment_opera_ct_eval/metadata_alignment_predictions__opera-ct-raw-disease.npz \
  --out-dir results/probability_transport/audio_only \
  --evaluate-tests

phase "E1: OPERA direct and raw-preserving fusion"
python src/eval_fusion_followup.py \
  --backbone opera \
  --audio-emb results/opera_ct_embeddings.npz \
  --alignment-dir results/alignment_opera_ct \
  --alignment-manifest results/alignment_opera_ct/manifest.json \
  --old-direct-preds results/direct_fusion_matched/predictions.npz \
  --out-dir results/fusion_followup/opera \
  --evaluate-tests

phase "E1: AST missing raw-preserving fusion arms"
python src/eval_fusion_followup.py \
  --backbone ast \
  --audio-emb results/ast_embeddings.npz \
  --alignment-dir results/alignment \
  --alignment-manifest results/alignment/manifest.json \
  --old-direct-preds results/direct_fusion_matched/predictions.npz \
  --common-preds results/fusion_followup/opera/predictions.npz \
  --reuse-existing-backbone-arms \
  --out-dir results/fusion_followup/ast \
  --evaluate-tests

phase "E3b: probability transport for the frozen fusion predictions"
python src/eval_probability_transport.py \
  --source ast_fusion=results/fusion_followup/ast/predictions.npz \
  --source opera_fusion=results/fusion_followup/opera/predictions.npz \
  --out-dir results/probability_transport/fusion \
  --evaluate-tests

phase "A: formal profile-retrieval manipulation checks"
python src/audit_profile_retrieval.py \
  --preflight --out-dir results/profile_retrieval_ast
python src/audit_profile_retrieval.py \
  --run --out-dir results/profile_retrieval_ast
python src/audit_profile_retrieval.py \
  --preflight --backbone OPERA-CT \
  --audio-emb results/opera_ct_embeddings.npz \
  --alignment-dir results/alignment_opera_ct \
  --out-dir results/profile_retrieval_opera_ct
python src/audit_profile_retrieval.py \
  --run --backbone OPERA-CT \
  --audio-emb results/opera_ct_embeddings.npz \
  --alignment-dir results/alignment_opera_ct \
  --out-dir results/profile_retrieval_opera_ct

phase "B: fit the AST and OPERA information-channel matrices before test scoring"
python src/audit_information_channels.py \
  --preflight --out-dir results/information_channels_ast
python src/audit_information_channels.py \
  --preflight --backbone OPERA-CT \
  --audio-emb results/opera_ct_embeddings.npz \
  --alignment-dir results/alignment_opera_ct \
  --out-dir results/information_channels_opera_ct
python src/audit_information_channels.py \
  --fit --out-dir results/information_channels_ast \
  >results/route_a_logs/information_channels_ast_fit.log 2>&1 &
pid_info_ast=$!
python src/audit_information_channels.py \
  --fit --backbone OPERA-CT \
  --audio-emb results/opera_ct_embeddings.npz \
  --alignment-dir results/alignment_opera_ct \
  --out-dir results/information_channels_opera_ct \
  >results/route_a_logs/information_channels_opera_fit.log 2>&1 &
pid_info_opera=$!
wait_all "$pid_info_ast" "$pid_info_opera"

phase "B: explicitly unlock both frozen information-channel prediction archives"
python src/audit_information_channels.py \
  --score-tests --out-dir results/information_channels_ast \
  >results/route_a_logs/information_channels_ast_score.log 2>&1 &
pid_info_ast=$!
python src/audit_information_channels.py \
  --score-tests --backbone OPERA-CT \
  --out-dir results/information_channels_opera_ct \
  >results/route_a_logs/information_channels_opera_score.log 2>&1 &
pid_info_opera=$!
wait_all "$pid_info_ast" "$pid_info_opera"

phase "E4: four-shard controlled synthetic mechanism sweep"
SYNTH=results/synthetic_correspondence_transfer
CUDA_VISIBLE_DEVICES="$ROUTE_A_GPU" python src/synthetic_correspondence_transfer.py \
  --mode formal --device cuda --out-dir "$SYNTH" \
  --num-shards 4 --shard-index 0 --skip-summary \
  >results/route_a_logs/synthetic_shard0.log 2>&1 &
pid_s0=$!
for _ in $(seq 1 120); do
  test -f "$SYNTH/config.json" && break
  sleep 1
done
test -f "$SYNTH/config.json"
CUDA_VISIBLE_DEVICES="$ROUTE_A_GPU" python src/synthetic_correspondence_transfer.py \
  --mode formal --device cuda --out-dir "$SYNTH" \
  --num-shards 4 --shard-index 1 --skip-summary \
  >results/route_a_logs/synthetic_shard1.log 2>&1 &
pid_s1=$!
CUDA_VISIBLE_DEVICES="$ROUTE_A_GPU" python src/synthetic_correspondence_transfer.py \
  --mode formal --device cuda --out-dir "$SYNTH" \
  --num-shards 4 --shard-index 2 --skip-summary \
  >results/route_a_logs/synthetic_shard2.log 2>&1 &
pid_s2=$!
CUDA_VISIBLE_DEVICES="$ROUTE_A_GPU" python src/synthetic_correspondence_transfer.py \
  --mode formal --device cuda --out-dir "$SYNTH" \
  --num-shards 4 --shard-index 3 --skip-summary \
  >results/route_a_logs/synthetic_shard3.log 2>&1 &
pid_s3=$!
wait_all "$pid_s0" "$pid_s1" "$pid_s2" "$pid_s3"
python src/synthetic_correspondence_transfer.py \
  --mode formal --out-dir "$SYNTH" --summarize-only

phase "E4 sensitivity: build frozen Phi-2 profile cache"
CUDA_VISIBLE_DEVICES="$ROUTE_A_GPU" python src/build_synthetic_phi2_profiles.py \
  --out results/synthetic_phi2_profiles.npz

phase "E4 sensitivity: one high-confounding Phi-2 text cell"
SYNTH_TEXT=results/synthetic_correspondence_transfer_phi2
CUDA_VISIBLE_DEVICES="$ROUTE_A_GPU" python src/synthetic_correspondence_transfer.py \
  --mode formal --device cuda \
  --language-profile-cache results/synthetic_phi2_profiles.npz \
  --out-dir "$SYNTH_TEXT" --num-shards 2 --shard-index 0 --skip-summary \
  >results/route_a_logs/synthetic_phi2_shard0.log 2>&1 &
pid_t0=$!
for _ in $(seq 1 120); do
  test -f "$SYNTH_TEXT/config.json" && break
  sleep 1
done
test -f "$SYNTH_TEXT/config.json"
CUDA_VISIBLE_DEVICES="$ROUTE_A_GPU" python src/synthetic_correspondence_transfer.py \
  --mode formal --device cuda \
  --language-profile-cache results/synthetic_phi2_profiles.npz \
  --out-dir "$SYNTH_TEXT" --num-shards 2 --shard-index 1 --skip-summary \
  >results/route_a_logs/synthetic_phi2_shard1.log 2>&1 &
pid_t1=$!
wait_all "$pid_t0" "$pid_t1"
python src/synthetic_correspondence_transfer.py \
  --mode formal --language-profile-cache results/synthetic_phi2_profiles.npz \
  --out-dir "$SYNTH_TEXT" --summarize-only

phase "E2: fixed nonlinear readout, AST and OPERA in parallel"
CUDA_VISIBLE_DEVICES="$ROUTE_A_GPU" python src/eval_fixed_mlp_readout.py \
  --fit --device cuda --backbone ast \
  --ast-emb results/ast_embeddings.npz \
  --alignment-dir results/alignment \
  --out-dir results/mlp_ast --include-common \
  >results/route_a_logs/mlp_ast_fit.log 2>&1 &
pid_mlp_ast=$!
CUDA_VISIBLE_DEVICES="$ROUTE_A_GPU" python src/eval_fixed_mlp_readout.py \
  --fit --device cuda --backbone opera_ct \
  --ast-emb results/opera_ct_embeddings.npz \
  --alignment-dir results/alignment_opera_ct \
  --out-dir results/mlp_opera_ct --include-common \
  >results/route_a_logs/mlp_opera_fit.log 2>&1 &
pid_mlp_opera=$!
wait_all "$pid_mlp_ast" "$pid_mlp_opera"

phase "E2: score only after both complete MLP manifests exist"
python src/eval_fixed_mlp_readout.py \
  --score --device cpu --backbone ast \
  --out-dir results/mlp_ast --include-common
python src/eval_fixed_mlp_readout.py \
  --score --device cpu --backbone opera_ct \
  --out-dir results/mlp_opera_ct --include-common

phase "all frozen Route-A experiments completed"
find \
  results/probability_transport \
  results/fusion_followup \
  results/profile_retrieval_ast \
  results/profile_retrieval_opera_ct \
  results/information_channels_ast \
  results/information_channels_opera_ct \
  results/synthetic_correspondence_transfer \
  results/synthetic_correspondence_transfer_phi2 \
  results/mlp_ast results/mlp_opera_ct \
  -type f \( -name 'results.json' -o -name 'metrics.json' -o -name 'summary.json' \) \
  -print0 | sort -z | xargs -0 sha256sum >results/route_a_logs/final_result_hashes.sha256
date -u +%FT%TZ >results/route_a_logs/COMPLETE_UTC
echo "COMPLETE"
