#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if conda env list | grep -qE "^ssdeepsmote[[:space:]]"; then
  conda env update -n ssdeepsmote -f environment.yml --prune
else
  conda env create -f environment.yml
fi
conda run -n ssdeepsmote python -c "import torch, lpips; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
