#!/usr/bin/env bash
set -eo pipefail

# Queue the two long 100-epoch jobs:
#   1. finish May19 remaining configs + infer + upload
#   2. train Falcor all configs + infer + upload
#
# The second job starts only if the first exits successfully.
#
# Typical unattended use:
#   cd /var/tmp/${USER}_repos/Adapted_fvdb_BACKENDOPTIMIZATION
#   git pull origin main
#   nohup bash scripts/run_may19_then_falcor_100ep_queue.sh \
#     > /var/tmp/${USER}_runs/may19_then_falcor_100ep_queue.out 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

QUEUE_LOG="${QUEUE_LOG:-/var/tmp/${USER}_runs/may19_then_falcor_100ep_queue_status.log}"

mkdir -p "$(dirname "$QUEUE_LOG")"

log_queue() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"
}

log_queue "============================================================"
log_queue "START MAY19 -> FALCOR 100EP QUEUE"
log_queue "repo=$REPO_ROOT"
log_queue "queue_log=$QUEUE_LOG"
log_queue "============================================================"

cd "$REPO_ROOT"

log_queue "Refreshing repo before long run."
if git remote get-url origin >/dev/null 2>&1; then
  git pull --rebase --autostash origin main || log_queue "WARNING: git pull failed; continuing with current checkout."
elif git remote get-url backendopt >/dev/null 2>&1; then
  git pull --rebase --autostash backendopt main || log_queue "WARNING: git pull failed; continuing with current checkout."
fi

log_queue "STEP 1/2: May19 remaining 100ep train + infer + upload."
bash "$SCRIPT_DIR/run_remaining_100ep_train_infer_push.sh"
log_queue "STEP 1/2 complete."

log_queue "STEP 2/2: Falcor 100ep train + infer + upload."
bash "$SCRIPT_DIR/run_falcor_100ep_train_infer_push.sh"
log_queue "STEP 2/2 complete."

log_queue "ALL DONE."
