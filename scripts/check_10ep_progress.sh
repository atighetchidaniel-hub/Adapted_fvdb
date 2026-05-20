#!/usr/bin/env bash
set -eo pipefail

# Print progress and partial metrics for the May19 10-epoch alternating run.
#
# Typical use:
#   cd /home/atighedl/Adapted_fvdb
#   bash scripts/check_10ep_progress.sh

RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_10EP_ALTERNATING}"
LOG_DIR="$RESULT_ROOT/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Result root:"
echo "$RESULT_ROOT"
echo

if [ ! -d "$LOG_DIR" ]; then
  echo "ERROR: log folder does not exist:"
  echo "$LOG_DIR"
  exit 1
fi

echo "Currently running:"
ps -u "$USER" -o pid,etime,cmd | grep -E "train.py|run_may19_10ep|tmux" | grep -v grep || echo "No matching process found."
echo

echo "Finished runs:"
if compgen -G "$LOG_DIR/*.log" >/dev/null; then
  grep -R "Training finished" "$LOG_DIR"/*.log 2>/dev/null \
    | sed 's|.*/||; s|.log:Training finished||' \
    | sort || true
else
  echo "No log files yet."
fi
echo

echo "Latest log tail:"
latest_log="$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1 || true)"
if [ -n "$latest_log" ]; then
  echo "$latest_log"
  tail -40 "$latest_log"
else
  echo "No log files yet."
fi
echo

echo "Summary table for completed eval logs:"
python "$SCRIPT_DIR/summarize_synthetic_logs.py" "$LOG_DIR" "*.log" || true
