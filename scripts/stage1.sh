#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"
python -m stages.stage1 --config configs/stage1.yaml "$@" 2>&1 | tee -a "$LOG_DIR/stage1.log"
