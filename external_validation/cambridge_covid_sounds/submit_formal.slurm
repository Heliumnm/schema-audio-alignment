#!/bin/bash
#SBATCH --job-name=cambridge_formal
#SBATCH --output=cambridge_formal_%j.out
#SBATCH --error=cambridge_formal_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch submit_formal.slurm /ABSOLUTE/PATH/config.local.json"
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
CONFIG=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
VENV=${CAMBRIDGE_VENV:-$HERE/.venv}
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing environment: $VENV"
  echo "Set CAMBRIDGE_VENV to the formal environment or install requirements.txt in $HERE/.venv"
  exit 2
fi
source "$VENV/bin/activate"
cd "$HERE"

python - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1]).resolve()
config = json.loads(path.read_text())
if config.get("format_version") != "cambridge-external-v1":
    raise SystemExit("config format_version must be cambridge-external-v1")
if config.get("protocol", {}).get("formal_unlock") != "FROZEN_PROTOCOL_AND_DATA_GATE_GO":
    raise SystemExit("formal_unlock is not set; review the GO gate first")
required = {
    "AST": config["models"]["ast"]["model_path"],
    "OPERA root": config["models"]["opera_ct"]["opera_root"],
    "OPERA checkpoint": config["models"]["opera_ct"]["checkpoint_path"],
    "Phi-2": config["models"]["text"]["model_path"],
}
missing = [f"{name}: {value}" for name, value in required.items()
           if not Path(value).expanduser().exists()]
if missing:
    raise SystemExit("Missing frozen model paths:\n" + "\n".join(missing))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm job")
print("FORMAL PREFLIGHT PASS", torch.cuda.get_device_name(0), flush=True)
PY

bash run_once.sh "$CONFIG" formal
