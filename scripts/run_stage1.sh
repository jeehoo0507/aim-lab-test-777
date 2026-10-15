#!/usr/bin/env bash
# Usage: bash scripts/run_stage1.sh core|extra|all   (resumable: re-run the same command after a stop)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export PYTHONWARNINGS=ignore
DATA=${DATA_DIR:-data/in100}
case "${1:-all}" in
  core) groups="core_s0 core_s1" ;;
  extra) groups="extra_s0" ;;
  all) groups="core_s0 core_s1 extra_s0" ;;
  *) echo "usage: $0 core|extra|all"; exit 1 ;;
esac
mkdir -p logs
COMPILE=$(cat logs/compile_flag 2>/dev/null || echo auto)
for g in $groups; do
  extra=()
  bs=$("$PY" -c "import json,sys;print(json.load(open('logs/batch_override.json')).get('$g',''))" 2>/dev/null || true)
  [ -n "$bs" ] && extra+=(--batch_size "$bs")
  echo "[$(date '+%F %T')] start $g (compile=$COMPILE ${extra[*]:-})"
  "$PY" stage1/group_train.py --config "stage1/configs/$g.yaml" --data_dir "$DATA" --out results \
      --compile "$COMPILE" --resume "${extra[@]}" 2>&1 | tee -a "logs/$g.log"
  echo "[$(date '+%F %T')] done $g"
done
"$PY" stage1/plot_stage1.py --results results --out results/plots
