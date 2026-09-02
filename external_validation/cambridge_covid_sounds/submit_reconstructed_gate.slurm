#!/bin/bash
#SBATCH --job-name=cambridge_gate
#SBATCH --output=cambridge_gate_%j.out
#SBATCH --error=cambridge_gate_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: sbatch submit_reconstructed_gate.slurm ALL_METADATA_DIR AUDIO_ROOT OUTPUT_DIR"
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
if [[ ! -f "$HERE/.venv/bin/activate" ]]; then
  echo "Missing environment: $HERE/.venv"
  echo "Create it first and install requirements-gate.txt as described in SLURM_GATE_GUIDE_ZH.txt"
  exit 2
fi

source "$HERE/.venv/bin/activate"
cd "$HERE"

bash run_reconstructed_gate.sh "$1" "$2" "$3"
