#!/bin/bash
#SBATCH --job-name=cambridge_match_v2
#SBATCH --output=cambridge_match_v2_%j.out
#SBATCH --error=cambridge_match_v2_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: sbatch submit_reconstructed_match_first_v2.slurm ALL_METADATA_DIR TASK2_AUDIO_ROOT OUTPUT_DIR TASK1_AUDIO_ROOT"
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

bash run_reconstructed_match_first_v2.sh "$1" "$2" "$3" "$4"
