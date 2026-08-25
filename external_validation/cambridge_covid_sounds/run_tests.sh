#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
python -m py_compile "$HERE"/src/*.py
python -m unittest discover -s "$HERE/tests" -v
