#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"
python -m stages.stage3 --config configs/stage3.yaml "$@" 2>&1 | tee -a "$LOG_DIR/stage3.log"
