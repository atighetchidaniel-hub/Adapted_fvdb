#!/usr/bin/env bash
set -eo pipefail

# 200-epoch r30 thesis benchmark queue.
#
# Order:
#   1. r30 d16: train fvdb, train spconv, infer all r30 scenes with timing, push
#   2. r30 d8:  train fvdb, train spconv, infer all r30 scenes with timing, push
#   3. r30 d32: train fvdb, train spconv, infer all r30 scenes with timing, push
#
# Typical use on Linux:
#   cd /var/tmp/${USER}_repos/Adapted_fvdb_BACKENDOPTIMIZATION
#   git pull origin main
#   nohup bash scripts/run_may19_200ep_r30_all_d_train_infer_push_by_pair.sh \
#     > /var/tmp/${USER}_runs/may19_200ep_r30_all_d_by_pair.out 2>&1 &
#
# Progress:
#   tail -f /var/tmp/${USER}_runs/may19_200ep_r30_all_d_by_pair.out
#
# Lightweight artifacts are pushed after every completed d-pair. Checkpoints,
# predicted PVVs, and raw inference folders stay on Linux under RESULT_ROOT and
# FINAL_RESULT_ROOT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

default_source_root() {
  local recovered="/var/tmp/${USER}_runs/recovered_may19_scene_generation"
  local original="/home/${USER}/May9_lunch_1000framescodexparamters/SceneGeneration"
  if [ -d "$recovered/r30/gv" ] && [ -d "$recovered/r30/pvv" ]; then
    echo "$recovered"
  else
    echo "$original"
  fi
}

SOURCE_ROOT="${SOURCE_ROOT:-$(default_source_root)}"
TRAIN_ROOT="${TRAIN_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_R30_ALLD_200EP}"
FINAL_RESULT_ROOT="${FINAL_RESULT_ROOT:-/var/tmp/${USER}_runs/final_all_inference_r30_all_d_200ep}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/var/tmp/${USER}_runs/canonical_data/FINAL_GVPVV}"

FVDB_TRAIN_REPO="${FVDB_TRAIN_REPO:-/home/${USER}/Adapted_fvdb}"
FVDB_INFER_REPO="${FVDB_INFER_REPO:-$REPO_ROOT}"
SPCONV_REPO="${SPCONV_REPO:-/home/${USER}/neuralpvs}"

FVDB_ENV="${FVDB_ENV:-fvdb_rc}"
SPCONV_ENV="${SPCONV_ENV:-cuda128}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

D_ORDER="${D_ORDER:-16 8 32}"
RADIUS="${RADIUS:-r30}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
THRESHOLD="${THRESHOLD:-3000}"
Z_SIZE="${Z_SIZE:-256}"
TEST_FRACTION="${TEST_FRACTION:-0.05}"
LR="${LR:-0.001}"

N_FRAMES="${N_FRAMES:-0}"
RUN_TIMING="${RUN_TIMING:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
SCENES="${SCENES:-}"

PUSH_TO_GITHUB="${PUSH_TO_GITHUB:-1}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-results/final_all_inference_r30_all_d_200ep_by_pair}"

STAMP="$(date +%Y%m%d_%H%M%S)"
STATUS_DIR="$TRAIN_ROOT/summaries"
STATUS_FILE="$STATUS_DIR/r30_all_d_200ep_by_pair_${STAMP}_status.txt"

mkdir -p "$STATUS_DIR"

log_status() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS_FILE"
}

require_path() {
  local path="$1"
  local label="$2"
  if [ ! -e "$path" ]; then
    echo "ERROR: $label not found: $path" >&2
    exit 1
  fi
}

newest_original_summary_csv() {
  local result_root="$1"
  find "$result_root/summaries" \
    -maxdepth 1 \
    -type f \
    -name "final_all_inference_${EPOCHS}ep_*.csv" \
    ! -name "*_compact.csv" \
    -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-
}

copy_pair_artifacts_to_repo() {
  local d="$1"
  local summary_csv="$2"
  local nice_md="$3"
  local compact_csv="$4"
  local plots_dir="$5"
  local pair_result_root="$6"
  local repo_artifacts="$REPO_ROOT/$ARTIFACT_ROOT/d${d}"

  if [ -z "$summary_csv" ] || [ ! -f "$summary_csv" ]; then
    echo "ERROR: summary CSV missing for d${d}, cannot copy artifacts." >&2
    exit 1
  fi

  rm -rf "$repo_artifacts"
  mkdir -p "$repo_artifacts"

  cp "$summary_csv" "$repo_artifacts/"
  [ -f "$nice_md" ] && cp "$nice_md" "$repo_artifacts/"
  [ -f "$compact_csv" ] && cp "$compact_csv" "$repo_artifacts/"
  [ -d "$plots_dir" ] && cp -R "$plots_dir" "$repo_artifacts/plots"

  cp "$STATUS_FILE" "$repo_artifacts/status_until_d${d}.txt"

  cat > "$repo_artifacts/README.md" <<EOF
# r30 d${d} 200ep fVDB vs spconv

Generated: ${STAMP}

This folder contains lightweight artifacts for the completed r30 d${d} pair:

- final all-scene inference CSV
- compact CSV
- Markdown summary
- SVG plots
- run status/config

Large files are intentionally not committed: checkpoints, predicted PVV folders,
and raw inference output remain on the Linux machine.

Training root:

\`${TRAIN_ROOT}\`

Pair inference root:

\`${pair_result_root}\`
EOF
}

push_artifacts() {
  local d="$1"
  local artifact_dir="$ARTIFACT_ROOT/d${d}"

  if [ "$PUSH_TO_GITHUB" != "1" ]; then
    log_status "PUSH_TO_GITHUB=0, skipping git push for d${d}."
    return
  fi

  cd "$REPO_ROOT"

  if ! git remote get-url "$GIT_REMOTE" >/dev/null 2>&1; then
    if git remote get-url origin >/dev/null 2>&1; then
      GIT_REMOTE="origin"
    elif git remote get-url backendopt >/dev/null 2>&1; then
      GIT_REMOTE="backendopt"
    else
      echo "ERROR: no usable git remote found for upload." >&2
      exit 1
    fi
  fi

  if ! git config user.email >/dev/null; then
    git config user.email "${USER:-auto}@$(hostname 2>/dev/null || echo linux)"
  fi
  if ! git config user.name >/dev/null; then
    git config user.name "${USER:-auto}"
  fi

  git add "$artifact_dir"

  if git diff --cached --quiet -- "$artifact_dir"; then
    log_status "No artifact changes to commit for d${d}."
    return
  fi

  git commit -m "Add r30 d${d} 200ep inference artifacts ${STAMP}"

  if git push "$GIT_REMOTE" "$GIT_BRANCH"; then
    log_status "Uploaded d${d} artifacts to $GIT_REMOTE/$GIT_BRANCH."
    return
  fi

  log_status "Main push failed for d${d}. Trying rebase + retry."
  if git pull --rebase --autostash "$GIT_REMOTE" "$GIT_BRANCH" && git push "$GIT_REMOTE" "$GIT_BRANCH"; then
    log_status "Uploaded d${d} artifacts to $GIT_REMOTE/$GIT_BRANCH after rebase."
    return
  fi

  log_status "Main push still failed for d${d}. Trying fallback branch upload."
  local fallback_branch="auto-upload-r30-d${d}-200ep-${STAMP}"
  if git push "$GIT_REMOTE" "HEAD:refs/heads/$fallback_branch"; then
    log_status "Uploaded d${d} artifacts to fallback branch: $GIT_REMOTE/$fallback_branch"
    return
  fi

  local bundle="$TRAIN_ROOT/summaries/git_upload_failed_r30_d${d}_200ep_${STAMP}.bundle"
  git bundle create "$bundle" HEAD
  log_status "ERROR: GitHub upload failed for d${d}. Local git bundle written to: $bundle"
  exit 1
}

require_path "$CONDA_SH" "conda activation script"
require_path "$SOURCE_ROOT/$RADIUS/gv" "$SOURCE_ROOT/$RADIUS/gv"
require_path "$SOURCE_ROOT/$RADIUS/pvv" "$SOURCE_ROOT/$RADIUS/pvv"
require_path "$CANONICAL_ROOT" "CANONICAL_ROOT"
require_path "$FVDB_TRAIN_REPO/train.py" "fVDB train.py"
require_path "$FVDB_INFER_REPO/infer.py" "optimized fVDB infer.py"
require_path "$SPCONV_REPO/train.py" "spconv train.py"
require_path "$SPCONV_REPO/infer.py" "spconv infer.py"

log_status "============================================================"
log_status "START r30 ALL-d 200EP TRAIN -> INFER -> PUSH BY PAIR"
log_status "============================================================"
log_status "SOURCE_ROOT=$SOURCE_ROOT"
log_status "TRAIN_ROOT=$TRAIN_ROOT"
log_status "FINAL_RESULT_ROOT=$FINAL_RESULT_ROOT"
log_status "CANONICAL_ROOT=$CANONICAL_ROOT"
log_status "FVDB_TRAIN_REPO=$FVDB_TRAIN_REPO"
log_status "FVDB_INFER_REPO=$FVDB_INFER_REPO"
log_status "SPCONV_REPO=$SPCONV_REPO"
log_status "D_ORDER=$D_ORDER"
log_status "RADIUS=$RADIUS EPOCHS=$EPOCHS BATCH=$BATCH DEPTH=$DEPTH THRESHOLD=$THRESHOLD"
log_status "N_FRAMES=$N_FRAMES RUN_METRICS=$RUN_METRICS RUN_TIMING=$RUN_TIMING SCENES=${SCENES:-ALL}"
log_status "ARTIFACT_ROOT=$ARTIFACT_ROOT"
log_status "PUSH_TO_GITHUB=$PUSH_TO_GITHUB remote=$GIT_REMOTE branch=$GIT_BRANCH"
log_status "status_file=$STATUS_FILE"

for d in $D_ORDER; do
  pair_stamp="$(date +%Y%m%d_%H%M%S)"
  pair_result_root="$FINAL_RESULT_ROOT/d${d}_${pair_stamp}"

  log_status ""
  log_status "============================================================"
  log_status "PAIR START: r30 d${d} 200ep"
  log_status "============================================================"

  RESULT_ROOT="$TRAIN_ROOT" \
  SOURCE_ROOT="$SOURCE_ROOT" \
  FVDB_REPO="$FVDB_TRAIN_REPO" \
  SPCONV_REPO="$SPCONV_REPO" \
  FVDB_ENV="$FVDB_ENV" \
  SPCONV_ENV="$SPCONV_ENV" \
  CONDA_SH="$CONDA_SH" \
  RUN_SPECS="${RADIUS}:${d}" \
  EPOCHS="$EPOCHS" \
  BATCH="$BATCH" \
  DEPTH="$DEPTH" \
  THRESHOLD="$THRESHOLD" \
  Z_SIZE="$Z_SIZE" \
  TEST_FRACTION="$TEST_FRACTION" \
  LR="$LR" \
  bash "$SCRIPT_DIR/run_may19_100ep_alternating_fvdb_spconv_local.sh"

  log_status "Training pair complete or already finished: r30 d${d}"
  log_status "Running all-scene metric + timing inference for r30 d${d}"

  TRAIN_ROOT="$TRAIN_ROOT" \
  RESULT_ROOT="$pair_result_root" \
  CANONICAL_ROOT="$CANONICAL_ROOT" \
  FVDB_REPO="$FVDB_INFER_REPO" \
  SPCONV_REPO="$SPCONV_REPO" \
  FVDB_ENV="$FVDB_ENV" \
  SPCONV_ENV="$SPCONV_ENV" \
  CONDA_SH="$CONDA_SH" \
  RADII="$RADIUS" \
  D_VALUES="$d" \
  BACKENDS="fvdb spconv" \
  SCENES="$SCENES" \
  EPOCHS="$EPOCHS" \
  BATCH="$BATCH" \
  DEPTH="$DEPTH" \
  Z_SIZE="$Z_SIZE" \
  N_FRAMES="$N_FRAMES" \
  RUN_METRICS="$RUN_METRICS" \
  RUN_TIMING="$RUN_TIMING" \
  bash "$SCRIPT_DIR/run_100ep_completed_final_all_inference.sh"

  summary_csv="$(newest_original_summary_csv "$pair_result_root")"
  if [ -z "$summary_csv" ] || [ ! -f "$summary_csv" ]; then
    echo "ERROR: could not find summary CSV for d${d} under $pair_result_root/summaries" >&2
    exit 1
  fi

  log_status "summary_csv=$summary_csv"
  log_status "Generating compact summary and SVG plots for d${d}"

  python "$SCRIPT_DIR/summarize_final_all_inference.py" --csv "$summary_csv"
  python "$SCRIPT_DIR/plot_final_all_inference.py" --csv "$summary_csv"

  nice_md="${summary_csv%.csv}_nice_summary.md"
  compact_csv="${summary_csv%.csv}_compact.csv"
  plots_dir="$(dirname "$summary_csv")/plots_$(basename "${summary_csv%.csv}")"

  log_status "Copying d${d} lightweight artifacts into repo"
  copy_pair_artifacts_to_repo "$d" "$summary_csv" "$nice_md" "$compact_csv" "$plots_dir" "$pair_result_root"

  log_status "Pushing d${d} artifacts"
  push_artifacts "$d"

  log_status "PAIR DONE: r30 d${d} 200ep"
done

log_status ""
log_status "DONE: all requested r30 d-pairs completed."
log_status "Training root: $TRAIN_ROOT"
log_status "Inference root: $FINAL_RESULT_ROOT"
log_status "Repo artifact root: $REPO_ROOT/$ARTIFACT_ROOT"
