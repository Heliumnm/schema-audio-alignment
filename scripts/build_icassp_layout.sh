#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/ICASSP_template_directory" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kit_dir="$(cd "$1" && pwd)"

for required in spconf.sty IEEEbib.bst; do
  if [[ ! -f "$kit_dir/$required" ]]; then
    echo "Missing $required in template directory: $kit_dir" >&2
    exit 2
  fi
done

command -v latexmk >/dev/null 2>&1 || {
  echo "latexmk is required" >&2
  exit 2
}

mkdir -p "$repo_root/tmp/pdfs" "$repo_root/output/pdf"
cd "$repo_root/paper"

TEXINPUTS="$kit_dir//:" \
BSTINPUTS="$kit_dir//:" \
latexmk \
  -pdf \
  -interaction=nonstopmode \
  -halt-on-error \
  -auxdir=../tmp/pdfs \
  -outdir=../output/pdf \
  icassp2027_draft.tex

echo "Built $repo_root/output/pdf/icassp2027_draft.pdf"
