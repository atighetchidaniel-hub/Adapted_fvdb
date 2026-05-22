#!/usr/bin/env bash
set -eo pipefail

# Status helper for the 200-epoch d=16 fVDB/spconv benchmark.

RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_D16_200EP}"

echo "Result root: $RESULT_ROOT"
echo

echo "=== Running training processes ==="
ps -u "$USER" -o pid,etime,cmd | grep -E "train.py|run_may19_200ep|run_may19_100ep" | grep -v grep || echo "No matching process found"

echo
echo "=== Expected 200ep d16 run status ==="
for R in r30 r60 r90; do
  for B in fvdb spconv; do
    LOG="$RESULT_ROOT/logs/synthetic_may19_${R}_${B}_d16_b3_depth3_200ep.log"
    if grep -q "Training finished" "$LOG" 2>/dev/null; then
      echo "DONE        $R d=16 $B"
    elif [ -f "$LOG" ]; then
      EPOCH="$(grep -o "epoch: [0-9]*" "$LOG" | tail -1 | awk '{print $2}')"
      STEP="$(grep -o "Step [0-9]*" "$LOG" | tail -1 | awk '{print $2}')"
      BATCH="$(grep "Average batch time" "$LOG" | tail -1 | awk '{print $4}')"
      echo "RUN/PARTIAL $R d=16 $B epoch=${EPOCH:-unknown} step=${STEP:-unknown} avg_batch=${BATCH:-unknown}s"
    else
      echo "MISSING     $R d=16 $B"
    fi
  done
done

echo
echo "=== Disk space ==="
df -hT "$RESULT_ROOT" 2>/dev/null || df -hT /var/tmp

echo
echo "=== Latest log tail ==="
LAST_LOG="$(ls -t "$RESULT_ROOT/logs"/*.log 2>/dev/null | head -1 || true)"
if [ -n "$LAST_LOG" ]; then
  echo "$LAST_LOG"
  tail -40 "$LAST_LOG"
else
  echo "No logs found yet."
fi
