#!/usr/bin/env bash
# One-command Cambridge formal audit for the collaborator's CSD3 layout.
# Run from this directory with: sbatch run.sh
#
# IMPORTANT: paths inside a CSD3/Slurm job are normal absolute paths.  The
# ``csd3:`` prefix is only for scp/rsync commands issued from another host.
#SBATCH -J cambridge_formal
#SBATCH -A MASCOLO-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
# Four hours is a queue-conscious upper bound for the first run (downloads included).
# A locally submitted two-hour job may finish, but is intentionally considered tight.
#SBATCH --time=04:00:00
#SBATCH --output=/home/yl809/projects/covid/external_validation/cambridge_covid_sounds/cambridge_formal_%j.out
#SBATCH --error=/home/yl809/projects/covid/external_validation/cambridge_covid_sounds/cambridge_formal_%j.err

set -eo pipefail

REPO_ROOT=/home/yl809/projects/covid
HERE=$REPO_ROOT/external_validation/cambridge_covid_sounds
DATA_ROOT=/home/yl809/rds/hpc-work/datasets/covid19
OUTPUT_ROOT=$HERE/cambridge_audit_output_webfix_v2
CONFIG=$OUTPUT_ROOT/config.local.json
MODEL_ROOT=/home/yl809/rds/hpc-work/models/schema_audio_alignment
FORMAL_VENV=/home/yl809/rds/hpc-work/venvs/cambridge_formal
AST_ROOT=$MODEL_ROOT/ast-finetuned-audioset-10-10-0.4593
PHI_ROOT=$MODEL_ROOT/phi-2
OPERA_ROOT=$MODEL_ROOT/OPERA
OPERA_CKPT=$OPERA_ROOT/cks/model/encoder-operaCT.ckpt
PHI_REVISION=810d367871c1d460086d9f82db8696f2e0a0fcd0
OPERA_SHA256=83c35b435518ad5f395bf4d34e552caa088faf9e63f6b8058d5288e9abb350ae

echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname) slurm_job_id=${SLURM_JOB_ID:-none}"

source "$HOME/.bashrc"
conda activate env_chao
cd "$HERE"

# env_chao is read only: use its already-tested torch/CUDA combination to select
# the exact official PyTorch wheels, then install torch and every other package
# into a fully isolated RDS venv. Nothing is pip-installed into env_chao.
BASE_PYTHON=$(command -v python)
BASE_TORCH_VERSION=$(python - <<'PY'
import torch
print(torch.__version__.split("+")[0])
PY
)
BASE_TORCH_CUDA=$(python - <<'PY'
import torch
print(torch.version.cuda or "cpu")
PY
)

case "$BASE_TORCH_CUDA" in
  11.8*) TORCH_WHEEL_CHANNEL=cu118 ;;
  12.4*) TORCH_WHEEL_CHANNEL=cu124 ;;
  12.6*) TORCH_WHEEL_CHANNEL=cu126 ;;
  cpu)   TORCH_WHEEL_CHANNEL=cpu ;;
  *)
    echo "Unsupported env_chao torch CUDA build: $BASE_TORCH_CUDA"
    echo "Add an explicit official PyTorch wheel channel before changing any package."
    exit 2
    ;;
esac

mkdir -p "$(dirname "$FORMAL_VENV")"
exec 8>"${FORMAL_VENV}.setup.lock"
if command -v flock >/dev/null 2>&1; then
  flock 8
fi

# The preceding revision used --system-site-packages. Preserve such an old venv
# as a timestamped backup, then replace it with a genuinely isolated venv.
if [[ -f "$FORMAL_VENV/pyvenv.cfg" ]] && \
   grep -Eqi '^include-system-site-packages[[:space:]]*=[[:space:]]*true' \
     "$FORMAL_VENV/pyvenv.cfg"; then
  LEGACY_VENV="${FORMAL_VENV}.legacy_shared_$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$FORMAL_VENV" "$LEGACY_VENV"
  echo "preserved_legacy_shared_venv=$LEGACY_VENV"
fi
if [[ ! -x "$FORMAL_VENV/bin/python" ]]; then
  "$BASE_PYTHON" -m venv "$FORMAL_VENV"
fi
source "$FORMAL_VENV/bin/activate"
set -u
echo "base_python=$BASE_PYTHON"
echo "base_torch=$BASE_TORCH_VERSION base_torch_cuda=$BASE_TORCH_CUDA"
echo "official_torch_wheel_channel=$TORCH_WHEEL_CHANNEL"
echo "isolated_formal_python=$(command -v python)"
[[ "$(command -v python)" == "$FORMAL_VENV/bin/python" ]] || {
  echo "Failed to activate isolated formal venv: $FORMAL_VENV"; exit 2;
}

[[ -d "$DATA_ROOT" ]] || { echo "Missing data root: $DATA_ROOT"; exit 2; }
[[ -f "$CONFIG" ]] || {
  echo "Missing frozen gate config: $CONFIG"
  echo "Do not recreate the cohort. Re-run the same model-blind gate only if this file was moved."
  exit 2
}
[[ -f "$OUTPUT_ROOT/public/data_gate.json" ]] || {
  echo "Missing reviewed data gate: $OUTPUT_ROOT/public/data_gate.json"; exit 2;
}
[[ -f "$OUTPUT_ROOT/public/reconstruction_report.json" ]] || {
  echo "Missing reconstruction report: $OUTPUT_ROOT/public/reconstruction_report.json"; exit 2;
}

python - "$OUTPUT_ROOT/public/data_gate.json" "$OUTPUT_ROOT/public/reconstruction_report.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
reconstruction = json.load(open(sys.argv[2]))
if reconstruction.get("format_version") != "cambridge-reconstruction-v2":
    raise SystemExit(
        "Formal execution forbidden: this is the superseded Web-collapsed Cambridge gate. "
        "Re-run run_reconstructed_match_first_v2.sh with the current code and review its "
        "new aggregate output before formal training."
    )
if gate.get("verdict") != "GO":
    raise SystemExit(f"Formal execution forbidden: data gate is {gate.get('verdict')!r}, not GO")
print("REVIEWED DATA GATE GO")
PY

# Install torch and torchaudio explicitly inside FORMAL_VENV, using the version
# and CUDA wheel family already known to work in env_chao on this cluster.
python -m pip install --upgrade pip
python -m pip install \
  --index-url "https://download.pytorch.org/whl/$TORCH_WHEEL_CHANNEL" \
  "torch==$BASE_TORCH_VERSION" "torchaudio==$BASE_TORCH_VERSION"

# All remaining dependencies are also isolated inside FORMAL_VENV.
python -m pip install -r requirements.txt
python -m pip install \
  "huggingface_hub>=0.23" \
  pytorch-lightning efficientnet-pytorch torchlibrosa timm matplotlib seaborn

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the isolated Cambridge formal venv")
try:
    import torchaudio
except Exception as exc:
    raise SystemExit(
        "torchaudio is missing or incompatible with torch after the isolated "
        "installation. Check the selected PyTorch wheel channel and cluster network. "
        f"Original error: {exc!r}"
    )
torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
audio_mm = ".".join(torchaudio.__version__.split("+")[0].split(".")[:2])
if torch_mm != audio_mm:
    raise SystemExit(
        f"torch {torch.__version__} and torchaudio {torchaudio.__version__} do not match"
    )
print("CUDA/PYTORCH PASS", torch.__version__, torchaudio.__version__,
      torch.cuda.get_device_name(0), flush=True)
PY

mkdir -p "$MODEL_ROOT"
export HF_HOME=$MODEL_ROOT/.hf_cache
export TOKENIZERS_PARALLELISM=false

# Hugging Face downloads are resumable.  Ignore duplicate legacy weight formats
# and cache only the safetensors/config/tokenizer files needed by Transformers.
python - "$AST_ROOT" "$PHI_ROOT" "$PHI_REVISION" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

ast_root, phi_root, phi_revision = map(str, sys.argv[1:])
snapshot_download(
    repo_id="MIT/ast-finetuned-audioset-10-10-0.4593",
    local_dir=ast_root,
    ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.onnx"],
)
snapshot_download(
    repo_id="microsoft/phi-2",
    revision=phi_revision,
    local_dir=phi_root,
    ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.onnx"],
)
for value in (ast_root, phi_root):
    if not Path(value, "config.json").is_file():
        raise SystemExit(f"Incomplete Hugging Face model directory: {value}")
print("AST/PHI-2 DOWNLOAD CHECK PASS", flush=True)
PY

# Freeze the first downloaded OPERA source tree rather than pulling a moving
# main branch on every resubmission.  The exact commit is printed to the log.
if [[ ! -d "$OPERA_ROOT/.git" ]]; then
  if [[ -e "$OPERA_ROOT" ]]; then
    echo "OPERA path exists but is not a git clone: $OPERA_ROOT"; exit 2
  fi
  git clone https://github.com/evelyn0414/OPERA.git "$OPERA_ROOT"
fi
mkdir -p "$OPERA_ROOT/cks/model"
python - "$OPERA_ROOT/cks/model" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="evelyn0414/OPERA",
    filename="encoder-operaCT.ckpt",
    local_dir=sys.argv[1],
)
PY

ACTUAL_OPERA_SHA=$(sha256sum "$OPERA_CKPT" | awk '{print $1}')
if [[ "$ACTUAL_OPERA_SHA" != "$OPERA_SHA256" ]]; then
  echo "Wrong OPERA-CT checkpoint SHA-256: $ACTUAL_OPERA_SHA"; exit 2
fi
echo "opera_source_commit=$(git -C "$OPERA_ROOT" rev-parse HEAD)"
echo "opera_checkpoint_sha256=$ACTUAL_OPERA_SHA"

# Check imports before touching the private config.  Inserting OPERA at index 0
# avoids collision with this audit package's own src/ directory.
python - "$OPERA_ROOT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import numpy, pandas, scipy, sklearn, soundfile, librosa, transformers, torch
import pytorch_lightning, efficientnet_pytorch, torchlibrosa, torchaudio
from src.model.models_cola import Cola
print("FORMAL DEPENDENCY IMPORT PASS", flush=True)
PY

# Modify only local model locations and the reviewed unlock token.  Scientific
# protocol fields are asserted, never silently repaired.
python - "$CONFIG" "$AST_ROOT" "$OPERA_ROOT" "$OPERA_CKPT" "$PHI_ROOT" "$PHI_REVISION" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
ast_root, opera_root, opera_ckpt, phi_root, phi_revision = sys.argv[2:]
config = json.loads(path.read_text())

assert config.get("format_version") == "cambridge-external-v1"
assert config["matching"]["min_pairs"] == 100
assert config["matching"]["max_abs_smd"] == 0.12
assert config["matching"]["max_fine_balance_difference"] == 0.08
assert config["models"]["ast"]["layers"] == 6
assert config["models"]["alignment"]["seeds"] == [0, 1, 2, 3, 4]
assert config["models"]["alignment"]["epochs"] == 500
assert str(config["models"]["text"]["revision"]) == phi_revision
assert config["protocol"]["identity_namespace_version"] == \
    "cambridge-task2-official-loader-v1"

config["models"]["device"] = "cuda"
config["models"]["ast"]["model_path"] = ast_root
config["models"]["opera_ct"]["opera_root"] = opera_root
config["models"]["opera_ct"]["checkpoint_path"] = opera_ckpt
config["models"]["text"]["model_path"] = phi_root
config["protocol"]["formal_unlock"] = "FROZEN_PROTOCOL_AND_DATA_GATE_GO"

temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(config, indent=2) + "\n")
os.replace(temporary, path)
print("PRIVATE CONFIG PATHS/UNLOCK UPDATED; FROZEN PROTOCOL ASSERTIONS PASS", flush=True)
PY

# The existing one-entry workflow reruns the model-blind gate as an integrity
# check, then caches Phi-2, extracts both backbones, trains all frozen arms and
# packages only aggregate public results.  Completed text/extraction/full-backbone
# artefacts are reusable; an alignment backbone interrupted before its final
# manifest is written restarts that backbone's 15 runs.
bash run_once.sh "$CONFIG" formal

echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Return only: $OUTPUT_ROOT/public/PUBLIC_RESULTS.zip"
