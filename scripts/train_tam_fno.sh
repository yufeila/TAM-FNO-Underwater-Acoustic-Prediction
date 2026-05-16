#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$ROOT/experiments/tam_fno/train.py" "$@"
