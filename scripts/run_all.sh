#!/usr/bin/env bash
# setup -> prepare -> smoke -> core -> extra -> plot; stops at the first failure.
#   export IMAGENET_ROOT=/path/to/imagenet
#   nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
echo "[$(date '+%F %T')] run_all start (commit $(git rev-parse --short HEAD 2>/dev/null || echo ?))"
bash scripts/setup_env.sh
if [ ! -f data/in100/meta.json ]; then
  [ -n "${IMAGENET_ROOT:-}" ] || { echo "IMAGENET_ROOT is not set"; exit 1; }
  .venv/bin/python stage1/prepare_in100.py --root "$IMAGENET_ROOT" --out data/in100
fi
bash scripts/smoke_test.sh
cat logs/estimate.txt
bash scripts/run_stage1.sh core
bash scripts/run_stage1.sh extra
.venv/bin/python stage1/plot_stage1.py --results results --out results/plots
echo "[$(date '+%F %T')] run_all finished; summary: results/summary.csv, figures: results/plots/"
