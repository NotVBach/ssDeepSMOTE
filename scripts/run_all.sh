#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for stage in 0 1 2 3 4; do
  bash "scripts/stage${stage}.sh" "$@"
done
