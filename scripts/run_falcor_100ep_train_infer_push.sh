#!/usr/bin/env bash
set -eo pipefail

# Train the Falcor synthetic 100-epoch benchmark, run final all-scene inference,
# generate compact summaries/plots, and push only lightweight artifacts to GitHub.
#
# Typical use on Linux:
#   cd /var/tmp/${USER}_repos/Adapted_fvdb_BACKENDOPTIMIZATION
#   git pull origin main
#   bash scripts/run_falcor_100ep_train_infer_push.sh
#
# What gets pushed:
#   results/falcor_final_all_inference_100ep/
#     - original final inference CSV
#     - nice Markdown summary
#     - compact CSV
#     - SVG plots
#     - run config/status files
#
# What does NOT get pushed:
#   checkpoints, predicted PVV folders, lossless videos, or raw inference output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_ROOT="${TRAIN_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_FALCOR_SYNTHETIC_100EP}"
FINAL_RESULT_ROOT="${FINAL_RESULT_ROOT:-/var/tmp/${USER}_runs/falcor_final_all_inference_100ep}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/var/tmp/${USER}_runs/canonical_data/FINAL_GVPVV}"

FVDB_TRAIN_REPO="${FVDB_TRAIN_REPO:-/home/atighedl/Adapted_fvdb}"
FVDB_INFER_REPO="${FVDB_INFER_REPO:-$REPO_ROOT}"
SPCONV_REPO="${SPCONV_REPO:-/home/atighedl/neuralpvs}"

FVDB_ENV="${FVDB_ENV:-fvdb_rc}"
SPCONV_ENV="${SPCONV_ENV:-cuda128}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

SOURCE_ROOT="${SOURCE_ROOT:-/var/tmp/${USER}_runs}"
FALCOR_SOURCE_PREFIX="${FALCOR_SOURCE_PREFIX:-falcor_synth_train_1000}"
FALCOR_STAGE_ROOT="${FALCOR_STAGE_ROOT:-$TRAIN_ROOT/falcor_scene_generation}"
RUN_SPECS="${RUN_SPECS:-r30:16 r30:8 r30:32 r60:16 r60:8 r60:32 r90:16 r90:8 r90:32}"

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
THRESHOLD="${THRESHOLD:-3000}"
N_FRAMES="${N_FRAMES:-0}"
RUN_TIMING="${RUN_TIMING:-1}"

PUSH_TO_GITHUB="${PUSH_TO_GITHUB:-1}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
ARTIFACT_DIR="${ARTIFACT_DIR:-results/falcor_final_all_inference_100ep}"
STAMP="$(date +%Y%m%d_%H%M%S)"

STATUS_DIR="$TRAIN_ROOT/summaries"
STATUS_FILE="$STATUS_DIR/falcor_100ep_train_infer_push_${STAMP}_status.txt"

mkdir -p "$STATUS_DIR"

log_status() {
  echo "$*" | tee -a "$STATUS_FILE"
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
  find "$FINAL_RESULT_ROOT/summaries" \
    -maxdepth 1 \
    -type f \
    -name "final_all_inference_${EPOCHS}ep_*.csv" \
    ! -name "*_compact.csv" \
    -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-
}

copy_artifacts_to_repo() {
  local summary_csv="$1"
  local nice_md="$2"
  local compact_csv="$3"
  local plots_dir="$4"
  local repo_artifacts="$REPO_ROOT/$ARTIFACT_DIR"

  if [ -z "$summary_csv" ] || [ ! -f "$summary_csv" ]; then
    echo "ERROR: summary CSV missing, cannot copy artifacts." >&2
    exit 1
  fi

  rm -rf "$repo_artifacts"
  mkdir -p "$repo_artifacts"

  cp "$summary_csv" "$repo_artifacts/"
  [ -f "$nice_md" ] && cp "$nice_md" "$repo_artifacts/"
  [ -f "$compact_csv" ] && cp "$compact_csv" "$repo_artifacts/"
  [ -d "$plots_dir" ] && cp -R "$plots_dir" "$repo_artifacts/plots"

  cp "$STATUS_FILE" "$repo_artifacts/status.txt"

  cat > "$repo_artifacts/README.md" <<EOF
# Falcor Final All-Scene Inference 100ep

Generated: ${STAMP}

This folder contains lightweight result artifacts only:

- original final all-scene inference CSV
- compact CSV
- Markdown summary
- SVG plots
- run status/config

Large files are intentionally not committed: checkpoints, predicted PVV folders,
lossless videos, and raw inference output remain on the Linux machine under:

\`${FINAL_RESULT_ROOT}\`

Training root:

\`${TRAIN_ROOT}\`

Falcor source prefix:

\`${SOURCE_ROOT}/${FALCOR_SOURCE_PREFIX}_rXX\`
EOF
}

prepare_falcor_source_root() {
  rm -rf "$FALCOR_STAGE_ROOT"
  mkdir -p "$FALCOR_STAGE_ROOT"

  for radius in r30 r60 r90; do
    local src="$SOURCE_ROOT/${FALCOR_SOURCE_PREFIX}_${radius}"
    local dst="$FALCOR_STAGE_ROOT/$radius"

    require_path "$src/gv" "Falcor $radius gv folder"
    require_path "$src/pvv" "Falcor $radius pvv folder"

    # The shared May19 training helper expects files directly under gv/ and pvv/.
    # Falcor exports may have an extra nesting level, so flatten them with symlinks.
    mkdir -p "$dst/gv" "$dst/pvv"

    while IFS= read -r -d '' file; do
      ln -s "$file" "$dst/gv/$(basename "$file")"
    done < <(find -L "$src/gv" -type f -name '*_gv.bin.gz' -print0)

    while IFS= read -r -d '' file; do
      ln -s "$file" "$dst/pvv/$(basename "$file")"
    done < <(find -L "$src/pvv" -type f -name '*_pvv.bin.gz' -print0)

    local gv_count
    local pvv_count
    gv_count="$(find -L "$dst/gv" -type f -name '*_gv.bin.gz' | wc -l)"
    pvv_count="$(find -L "$dst/pvv" -type f -name '*_pvv.bin.gz' | wc -l)"

    log_status "Falcor $radius source: $src"
    log_status "Falcor $radius staged: $dst"
    log_status "Falcor $radius GV=$gv_count PVV=$pvv_count"

    if [ "$gv_count" -eq 0 ] || [ "$pvv_count" -eq 0 ]; then
      echo "ERROR: Falcor $radius source has no GV/PVV files." >&2
      exit 1
    fi
  done
}

push_artifacts() {
  if [ "$PUSH_TO_GITHUB" != "1" ]; then
    log_status "PUSH_TO_GITHUB=0, skipping git push."
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

  git add "$ARTIFACT_DIR"

  if git diff --cached --quiet -- "$ARTIFACT_DIR"; then
    log_status "No artifact changes to commit."
    return
  fi

  git commit -m "Add Falcor 100ep final inference artifacts ${STAMP}"

  if git push "$GIT_REMOTE" "$GIT_BRANCH"; then
    log_status "Uploaded artifacts to $GIT_REMOTE/$GIT_BRANCH."
    return
  fi

  log_status "Main push failed. Trying rebase + retry."
  if git pull --rebase --autostash "$GIT_REMOTE" "$GIT_BRANCH" && git push "$GIT_REMOTE" "$GIT_BRANCH"; then
    log_status "Uploaded artifacts to $GIT_REMOTE/$GIT_BRANCH after rebase."
    return
  fi

  log_status "Main push still failed. Trying fallback branch upload."
  local safe_artifact
  local fallback_branch
  safe_artifact="$(echo "$ARTIFACT_DIR" | tr '/ ' '__')"
  fallback_branch="auto-upload-${safe_artifact}-${STAMP}"

  if git push "$GIT_REMOTE" "HEAD:refs/heads/$fallback_branch"; then
    log_status "Uploaded artifacts to fallback branch: $GIT_REMOTE/$fallback_branch"
    return
  fi

  local bundle="$TRAIN_ROOT/summaries/git_upload_failed_${STAMP}.bundle"
  git bundle create "$bundle" HEAD
  log_status "ERROR: GitHub upload failed. Local git bundle written to: $bundle"
  exit 1
}

require_path "$CONDA_SH" "conda activation script"
require_path "$SOURCE_ROOT" "SOURCE_ROOT"
require_path "$CANONICAL_ROOT" "CANONICAL_ROOT"
require_path "$FVDB_TRAIN_REPO/train.py" "fVDB train.py"
require_path "$FVDB_INFER_REPO/infer.py" "optimized fVDB infer.py"
require_path "$SPCONV_REPO/train.py" "spconv train.py"
require_path "$SPCONV_REPO/infer.py" "spconv infer.py"

log_status "============================================================"
log_status "FALCOR 100EP TRAIN + FINAL INFER + PUSH"
log_status "============================================================"
log_status "TRAIN_ROOT=$TRAIN_ROOT"
log_status "FINAL_RESULT_ROOT=$FINAL_RESULT_ROOT"
log_status "SOURCE_ROOT=$SOURCE_ROOT"
log_status "CANONICAL_ROOT=$CANONICAL_ROOT"
log_status "FVDB_TRAIN_REPO=$FVDB_TRAIN_REPO"
log_status "FVDB_INFER_REPO=$FVDB_INFER_REPO"
log_status "SPCONV_REPO=$SPCONV_REPO"
log_status "FALCOR_SOURCE_PREFIX=$FALCOR_SOURCE_PREFIX"
log_status "FALCOR_STAGE_ROOT=$FALCOR_STAGE_ROOT"
log_status "RUN_SPECS=$RUN_SPECS"
log_status "EPOCHS=$EPOCHS BATCH=$BATCH DEPTH=$DEPTH THRESHOLD=$THRESHOLD"
log_status "N_FRAMES=$N_FRAMES RUN_TIMING=$RUN_TIMING"
log_status "ARTIFACT_DIR=$ARTIFACT_DIR"
log_status "PUSH_TO_GITHUB=$PUSH_TO_GITHUB remote=$GIT_REMOTE branch=$GIT_BRANCH"
log_status "status_file=$STATUS_FILE"

log_status ""
log_status "Preparing Falcor source staging root."
prepare_falcor_source_root

log_status ""
log_status "Step 1/4: train Falcor model configs."
RESULT_ROOT="$TRAIN_ROOT" \
SOURCE_ROOT="$FALCOR_STAGE_ROOT" \
FVDB_REPO="$FVDB_TRAIN_REPO" \
SPCONV_REPO="$SPCONV_REPO" \
FVDB_ENV="$FVDB_ENV" \
SPCONV_ENV="$SPCONV_ENV" \
CONDA_SH="$CONDA_SH" \
RUN_SPECS="$RUN_SPECS" \
EPOCHS="$EPOCHS" \
BATCH="$BATCH" \
DEPTH="$DEPTH" \
THRESHOLD="$THRESHOLD" \
bash "$SCRIPT_DIR/run_may19_100ep_alternating_fvdb_spconv_local.sh"

log_status ""
log_status "Step 2/4: run clean final all-scene inference for all completed configs."
TRAIN_ROOT="$TRAIN_ROOT" \
RESULT_ROOT="$FINAL_RESULT_ROOT" \
CANONICAL_ROOT="$CANONICAL_ROOT" \
FVDB_REPO="$FVDB_INFER_REPO" \
SPCONV_REPO="$SPCONV_REPO" \
FVDB_ENV="$FVDB_ENV" \
SPCONV_ENV="$SPCONV_ENV" \
CONDA_SH="$CONDA_SH" \
EPOCHS="$EPOCHS" \
BATCH="$BATCH" \
DEPTH="$DEPTH" \
N_FRAMES="$N_FRAMES" \
RUN_TIMING="$RUN_TIMING" \
bash "$SCRIPT_DIR/run_100ep_completed_final_all_inference.sh"

SUMMARY_CSV="$(newest_original_summary_csv)"
if [ -z "$SUMMARY_CSV" ] || [ ! -f "$SUMMARY_CSV" ]; then
  echo "ERROR: could not find final summary CSV under $FINAL_RESULT_ROOT/summaries" >&2
  exit 1
fi

log_status "summary_csv=$SUMMARY_CSV"

log_status ""
log_status "Step 3/4: generate nice summary and SVG plots."
python "$SCRIPT_DIR/summarize_final_all_inference.py" --csv "$SUMMARY_CSV"
python "$SCRIPT_DIR/plot_final_all_inference.py" --csv "$SUMMARY_CSV"

NICE_MD="${SUMMARY_CSV%.csv}_nice_summary.md"
COMPACT_CSV="${SUMMARY_CSV%.csv}_compact.csv"
PLOTS_DIR="$(dirname "$SUMMARY_CSV")/plots_$(basename "${SUMMARY_CSV%.csv}")"

log_status "nice_md=$NICE_MD"
log_status "compact_csv=$COMPACT_CSV"
log_status "plots_dir=$PLOTS_DIR"

log_status ""
log_status "Step 4/4: copy lightweight artifacts into repo and push."
copy_artifacts_to_repo "$SUMMARY_CSV" "$NICE_MD" "$COMPACT_CSV" "$PLOTS_DIR"
push_artifacts

log_status ""
log_status "DONE"
log_status "Artifacts copied to: $REPO_ROOT/$ARTIFACT_DIR"
log_status "Linux raw result root: $FINAL_RESULT_ROOT"
