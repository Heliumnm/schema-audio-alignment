#!/usr/bin/env bash
set -euo pipefail

# Post-hoc UKCOVID HeAR completion of the frozen E1--E3 pairing audit.
# Usage:
#   bash scripts/run_ukcovid_hear_pairing_followup.sh \
#     /mnt/hd/data_heliu/audio_provenance \
#     /mnt/hd/data_heliu/resp_datasets/ukcovid \
#     /home/heliu/anaconda3/envs/qwen-audio/bin/python

REPO_ROOT="${1:?repository root required}"
DATA_ROOT="${2:?UKCOVID data root required}"
PY_BIN="${3:?Python executable required}"
CPUSET="${FOLLOWUP_CPUSET:-0-39}"
THREADS="${FOLLOWUP_THREADS:-40}"

RESULT_ROOT="${REPO_ROOT}/results"
FOLLOW_ROOT="${RESULT_ROOT}/pairing_followup"
LOG_ROOT="${FOLLOW_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

run_limited() {
  taskset -c "${CPUSET}" "${PY_BIN}" "$@"
}

# Code and pairing feasibility checks. These do not train or read matched/test labels.
run_limited "${REPO_ROOT}/src/train_within_label_sex_control.py" --self-test
run_limited "${REPO_ROOT}/src/eval_within_label_sex_retrieval.py" --self-test
run_limited "${REPO_ROOT}/src/eval_within_label_sex_control.py" --self-test
run_limited "${REPO_ROOT}/src/eval_target_assisted_readout.py" --self-test
run_limited "${REPO_ROOT}/src/train_within_label_sex_control.py" --preflight --device cpu \
  --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
  --texts "${RESULT_ROOT}/metadata_texts.csv" \
  --emb "${RESULT_ROOT}/hear_embeddings.npz" \
  --text-emb "${RESULT_ROOT}/metadata_text_embeddings.npz" \
  --reference-alignment-dir "${RESULT_ROOT}/alignment_hear" \
  --out-dir "${FOLLOW_ROOT}/e2_alignment_hear"

# E1 may already exist because it is also the input bank for E2 retrieval.
if [[ ! -s "${FOLLOW_ROOT}/e1_hear/metrics.json" ]]; then
  run_limited "${REPO_ROOT}/src/eval_conditional_profile_retrieval.py" --run \
    --data "${DATA_ROOT}" \
    --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
    --texts "${RESULT_ROOT}/metadata_texts.csv" \
    --text-embeddings "${RESULT_ROOT}/metadata_text_embeddings.npz" \
    --audio-emb "${RESULT_ROOT}/hear_embeddings.npz" \
    --alignment-dir "${RESULT_ROOT}/alignment_hear" \
    --backbone HeAR \
    --out-dir "${FOLLOW_ROOT}/e1_hear"
fi

test ! -e "${FOLLOW_ROOT}/e2_alignment_hear/manifest.json"
run_limited "${REPO_ROOT}/src/train_within_label_sex_control.py" --device cpu \
  --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
  --texts "${RESULT_ROOT}/metadata_texts.csv" \
  --emb "${RESULT_ROOT}/hear_embeddings.npz" \
  --text-emb "${RESULT_ROOT}/metadata_text_embeddings.npz" \
  --reference-alignment-dir "${RESULT_ROOT}/alignment_hear" \
  --out-dir "${FOLLOW_ROOT}/e2_alignment_hear" \
  > "${LOG_ROOT}/e2_hear.log" 2>&1

run_limited "${REPO_ROOT}/src/eval_within_label_sex_retrieval.py" \
  --data "${DATA_ROOT}" \
  --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
  --texts "${RESULT_ROOT}/metadata_texts.csv" \
  --text-embeddings "${RESULT_ROOT}/metadata_text_embeddings.npz" \
  --e1-dir "${FOLLOW_ROOT}/e1_hear" \
  --e2-alignment-dir "${FOLLOW_ROOT}/e2_alignment_hear" \
  --backbone HeAR \
  --out-dir "${FOLLOW_ROOT}/e2_retrieval_hear" \
  > "${LOG_ROOT}/e2_retrieval_hear.log" 2>&1

run_limited "${REPO_ROOT}/src/eval_within_label_sex_control.py" --fit \
  --data "${DATA_ROOT}" \
  --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
  --artefacts "${RESULT_ROOT}/artefact_features.csv" \
  --e2-alignment-dir "${FOLLOW_ROOT}/e2_alignment_hear" \
  --original-predictions "${RESULT_ROOT}/information_channels_hear/predictions.npz" \
  --backbone HeAR \
  --out-dir "${FOLLOW_ROOT}/e2_eval_hear" \
  > "${LOG_ROOT}/e2_eval_hear_fit.log" 2>&1

run_limited "${REPO_ROOT}/src/eval_within_label_sex_control.py" --score-tests \
  --data "${DATA_ROOT}" \
  --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
  --artefacts "${RESULT_ROOT}/artefact_features.csv" \
  --e2-alignment-dir "${FOLLOW_ROOT}/e2_alignment_hear" \
  --original-predictions "${RESULT_ROOT}/information_channels_hear/predictions.npz" \
  --backbone HeAR \
  --out-dir "${FOLLOW_ROOT}/e2_eval_hear" \
  > "${LOG_ROOT}/e2_eval_hear_score.log" 2>&1

run_limited "${REPO_ROOT}/src/eval_target_assisted_readout.py" --fit \
  --data "${DATA_ROOT}" \
  --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
  --ast-emb "${RESULT_ROOT}/hear_embeddings.npz" \
  --alignment-dir "${RESULT_ROOT}/alignment_hear" \
  --e2-alignment-dir "${FOLLOW_ROOT}/e2_alignment_hear" \
  --backbone HeAR \
  --out-dir "${FOLLOW_ROOT}/e3_target_hear" \
  > "${LOG_ROOT}/e3_target_hear_fit.log" 2>&1

run_limited "${REPO_ROOT}/src/eval_target_assisted_readout.py" --score-tests \
  --data "${DATA_ROOT}" \
  --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
  --ast-emb "${RESULT_ROOT}/hear_embeddings.npz" \
  --alignment-dir "${RESULT_ROOT}/alignment_hear" \
  --e2-alignment-dir "${FOLLOW_ROOT}/e2_alignment_hear" \
  --backbone HeAR \
  --out-dir "${FOLLOW_ROOT}/e3_target_hear" \
  > "${LOG_ROOT}/e3_target_hear_score.log" 2>&1

date -u +%Y-%m-%dT%H:%M:%SZ > "${FOLLOW_ROOT}/HEAR_E1_E2_E3_COMPLETE.txt"

