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
DEVICE="${FOLLOWUP_DEVICE:-auto}"
PARALLEL_CPUSET_A="${FOLLOWUP_PARALLEL_CPUSET_A:-0-31}"
PARALLEL_CPUSET_B="${FOLLOWUP_PARALLEL_CPUSET_B:-32-63}"
PARALLEL_THREADS="${FOLLOWUP_PARALLEL_THREADS:-32}"

RESULT_ROOT="${REPO_ROOT}/results"
FOLLOW_ROOT="${RESULT_ROOT}/pairing_followup"
LOG_ROOT="${FOLLOW_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

if [[ "${DEVICE}" == "auto" ]]; then
  if "${PY_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    DEVICE=cuda
  else
    DEVICE=cpu
  fi
fi
if [[ "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" ]]; then
  echo "FOLLOWUP_DEVICE must be auto, cpu, or cuda" >&2
  exit 2
fi
echo "selected_projector_device=${DEVICE}"

run_limited() {
  taskset -c "${CPUSET}" "${PY_BIN}" "$@"
}

run_parallel_limited() {
  local cpuset="$1"
  shift
  taskset -c "${cpuset}" env \
    OMP_NUM_THREADS="${PARALLEL_THREADS}" \
    MKL_NUM_THREADS="${PARALLEL_THREADS}" \
    OPENBLAS_NUM_THREADS="${PARALLEL_THREADS}" \
    NUMEXPR_NUM_THREADS="${PARALLEL_THREADS}" \
    "${PY_BIN}" "$@"
}

wait_both() {
  local first_pid="$1"
  local second_pid="$2"
  local failed=0
  if ! wait "${first_pid}"; then failed=1; fi
  if ! wait "${second_pid}"; then failed=1; fi
  return "${failed}"
}

# Code and pairing feasibility checks. These do not train or read matched/test labels.
run_limited "${REPO_ROOT}/src/train_within_label_sex_control.py" --self-test
run_limited "${REPO_ROOT}/src/eval_within_label_sex_retrieval.py" --self-test
run_limited "${REPO_ROOT}/src/eval_within_label_sex_control.py" --self-test
run_limited "${REPO_ROOT}/src/eval_target_assisted_readout.py" --self-test
run_limited "${REPO_ROOT}/src/train_within_label_sex_control.py" --preflight --device "${DEVICE}" \
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
run_limited "${REPO_ROOT}/src/train_within_label_sex_control.py" --device "${DEVICE}" \
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

# E2 and E3 fits use the same frozen representations but are otherwise independent.
# Run them on disjoint CPU sets; the combined hard cap is 64 logical CPUs.
run_parallel_limited "${PARALLEL_CPUSET_A}" \
  "${REPO_ROOT}/src/eval_within_label_sex_control.py" --fit \
  --data "${DATA_ROOT}" \
  --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
  --artefacts "${RESULT_ROOT}/artefact_features.csv" \
  --e2-alignment-dir "${FOLLOW_ROOT}/e2_alignment_hear" \
  --original-predictions "${RESULT_ROOT}/information_channels_hear/predictions.npz" \
  --backbone HeAR \
  --out-dir "${FOLLOW_ROOT}/e2_eval_hear" \
  > "${LOG_ROOT}/e2_eval_hear_fit.log" 2>&1 &
E2_FIT_PID=$!

run_parallel_limited "${PARALLEL_CPUSET_B}" \
  "${REPO_ROOT}/src/eval_target_assisted_readout.py" --fit \
  --data "${DATA_ROOT}" \
  --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
  --ast-emb "${RESULT_ROOT}/hear_embeddings.npz" \
  --alignment-dir "${RESULT_ROOT}/alignment_hear" \
  --e2-alignment-dir "${FOLLOW_ROOT}/e2_alignment_hear" \
  --backbone HeAR \
  --out-dir "${FOLLOW_ROOT}/e3_target_hear" \
  > "${LOG_ROOT}/e3_target_hear_fit.log" 2>&1 &
E3_FIT_PID=$!

wait_both "${E2_FIT_PID}" "${E3_FIT_PID}"

run_parallel_limited "${PARALLEL_CPUSET_A}" \
  "${REPO_ROOT}/src/eval_within_label_sex_control.py" --score-tests \
  --data "${DATA_ROOT}" \
  --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
  --artefacts "${RESULT_ROOT}/artefact_features.csv" \
  --e2-alignment-dir "${FOLLOW_ROOT}/e2_alignment_hear" \
  --original-predictions "${RESULT_ROOT}/information_channels_hear/predictions.npz" \
  --backbone HeAR \
  --out-dir "${FOLLOW_ROOT}/e2_eval_hear" \
  > "${LOG_ROOT}/e2_eval_hear_score.log" 2>&1 &
E2_SCORE_PID=$!

run_parallel_limited "${PARALLEL_CPUSET_B}" \
  "${REPO_ROOT}/src/eval_target_assisted_readout.py" --score-tests \
  --data "${DATA_ROOT}" \
  --cohort "${RESULT_ROOT}/ukcovid_audio_cohort.csv" \
  --ast-emb "${RESULT_ROOT}/hear_embeddings.npz" \
  --alignment-dir "${RESULT_ROOT}/alignment_hear" \
  --e2-alignment-dir "${FOLLOW_ROOT}/e2_alignment_hear" \
  --backbone HeAR \
  --out-dir "${FOLLOW_ROOT}/e3_target_hear" \
  > "${LOG_ROOT}/e3_target_hear_score.log" 2>&1 &
E3_SCORE_PID=$!

wait_both "${E2_SCORE_PID}" "${E3_SCORE_PID}"

date -u +%Y-%m-%dT%H:%M:%SZ > "${FOLLOW_ROOT}/HEAR_E1_E2_E3_COMPLETE.txt"
