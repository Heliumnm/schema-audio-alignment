#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
build_dir="$repo_root/tmp/pdfs/icassp2027"
output_dir="$repo_root/output/pdf"

mkdir -p "$build_dir" "$output_dir"
pdf_python="/Users/heliumnm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
if [[ ! -x "$pdf_python" ]]; then
  pdf_python="python3"
fi
"$pdf_python" "$script_dir/make_figures.py"

cd "$script_dir"
latexmk \
  -pdf \
  -interaction=nonstopmode \
  -halt-on-error \
  -auxdir="$build_dir" \
  -outdir="$output_dir" \
  -jobname=icassp2027_pairing_audit \
  main.tex

echo "Built $output_dir/icassp2027_pairing_audit.pdf"
