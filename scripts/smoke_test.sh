#!/usr/bin/env bash
# Smoke test (section 8). Prints PASS/FAIL per check, exits non-zero if any FAIL.
# DATA_DIR (default data/in100). SMOKE_SYNTHETIC=1 uses generated data (teacher/monotone checks -> SKIP).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export PYTHONWARNINGS=ignore
DATA=${DATA_DIR:-data/in100}
if [ "${SMOKE_SYNTHETIC:-0}" = "1" ]; then
  DATA=data/synth
  [ -f "$DATA/meta.json" ] || "$PY" stage1/prepare_in100.py --synthetic --out "$DATA" --train-per-class 20 --val-per-class 4
fi
[ -f "$DATA/meta.json" ] || { echo "no dataset at $DATA (run prepare_in100.py first)"; exit 1; }
OUT=results_smoke
rm -rf "$OUT"; mkdir -p logs "$OUT"; rm -f logs/smoke_report.txt
echo "# smoke test $(date '+%F %T') data=$DATA" > logs/smoke_report.txt
fail=0
chk() { "$PY" stage1/smoke_checks.py "$@" --data_dir "$DATA" --out "$OUT" || fail=1; }

chk equiv_orig       # 1. mask idx + KD loss == original DistillationLoss path (fp32)
chk equiv_group      # 2. first-step loss: solo == in group
chk teacher          # 3. teacher top-1 @160 >= 75%
chk monotone         # 4. keep 100 -> 50 -> 30 lowers masked-teacher top-1
echo "[smoke] training the smoke group (all criteria, 2 epochs)"
if "$PY" stage1/group_train.py --config stage1/configs/smoke.yaml --data_dir "$DATA" --out "$OUT" > logs/smoke_train.log 2>&1; then
  echo "CHECK smoke_train  PASS $(grep -c 'epoch .* done' logs/smoke_train.log) epochs" | tee -a logs/smoke_report.txt
else
  echo "CHECK smoke_train  FAIL see logs/smoke_train.log" | tee -a logs/smoke_report.txt; tail -20 logs/smoke_train.log; fail=1
fi
compiled=$(grep -o '"compiled": [a-z]*' "$OUT"/smoke/none/config.json 2>/dev/null | awk '{print $2}')
if [ "$compiled" = "true" ]; then echo on > logs/compile_flag; else echo off > logs/compile_flag; fi
echo "CHECK compile      INFO torch.compile works: ${compiled:-unknown} -> main runs use --compile $(cat logs/compile_flag)" | tee -a logs/smoke_report.txt
chk tau_match        # 5. entropy matching error < 0.01 nats
chk outputs          # 6. no NaN/Inf, all files, plots render
chk throughput       # 7. 50 steps per real group, peak memory, batch auto-reduce
chk estimate

echo "================ SMOKE SUMMARY ================"
grep '^CHECK' logs/smoke_report.txt
if [ $fail -ne 0 ]; then echo "SMOKE: FAIL"; exit 1; fi
echo "SMOKE: PASS"
