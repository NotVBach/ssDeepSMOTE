#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"
python -m stages.stage0 --config configs/stage0.yaml "$@" 2>&1 | tee -a "$LOG_DIR/stage0.log"
