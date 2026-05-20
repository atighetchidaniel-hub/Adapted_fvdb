#!/usr/bin/env bash
set -eo pipefail

# Fast 5-epoch synthetic smoke benchmark.
#
# First run:
#   r30 d16 fvdb -> spconv
#
# Then continue with the remaining radius/d combinations, alternating
# fVDB and spconv for each setting.
#
# Typical use:
#   cd /home/atighedl/Adapted_fvdb
#   git pull
#   bash scripts/run_may19_5ep_alternating_fvdb_spconv_local.sh
#
# Override local storage if needed:
#   RESULT_ROOT="/var/tmp/${USER}_runs/NEURALPVS_5EP" \
#   bash scripts/run_may19_5ep_alternating_fvdb_spconv_local.sh

SOURCE_ROOT="${SOURCE_ROOT:-/home/atighedl/May9_lunch_1000framescodexparamters/SceneGeneration}"
RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_5EP_ALTERNATING}"

FVDB_REPO="${FVDB_REPO:-/home/atighedl/Adapted_fvdb}"
SPCONV_REPO="${SPCONV_REPO:-/home/atighedl/neuralpvs}"

FVDB_ENV="${FVDB_ENV:-fvdb_rc}"
SPCONV_ENV="${SPCONV_ENV:-cuda128}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

# Ordered as requested: start with r30 d16, then the rest.
RUN_SPECS="${RUN_SPECS:-r30:16 r30:8 r30:32 r60:16 r60:8 r60:32 r90:16 r90:8 r90:32}"

THRESHOLD="${THRESHOLD:-3000}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
Z_SIZE="${Z_SIZE:-256}"
TEST_FRACTION="${TEST_FRACTION:-0.05}"
LR="${LR:-0.001}"

require_path() {
  local path="$1"
  local label="$2"
  if [ ! -e "$path" ]; then
    echo "ERROR: $label not found: $path" >&2
    exit 1
  fi
}

require_path "$SOURCE_ROOT" "SOURCE_ROOT"
require_path "$FVDB_REPO/train.py" "fVDB train.py"
require_path "$SPCONV_REPO/train.py" "spconv train.py"
require_path "$CONDA_SH" "conda activation script"

mkdir -p "$RESULT_ROOT/data/datasets" \
         "$RESULT_ROOT/fvdb_out" \
         "$RESULT_ROOT/spconv_out" \
         "$RESULT_ROOT/logs" \
         "$RESULT_ROOT/summaries"

MASTER_LOG="$RESULT_ROOT/logs/master_$(date +%Y%m%d_%H%M%S).log"

# shellcheck source=/dev/null
source "$CONDA_SH"

make_filtered_dataset() {
  local radius="$1"
  local src="$SOURCE_ROOT/$radius"
  local ds="synthetic_may19_${radius}_1000_t${THRESHOLD}"
  local dst="$RESULT_ROOT/data/datasets/$ds"

  require_path "$src/gv" "$radius gv folder"
  require_path "$src/pvv" "$radius pvv folder"

  if [ -d "$dst/gv" ] && [ -d "$dst/pvv" ]; then
    local existing_gv
    local existing_pvv
    existing_gv="$(find -L "$dst/gv" -type f -name '*_gv.bin.gz' 2>/dev/null | wc -l)"
    existing_pvv="$(find -L "$dst/pvv" -type f -name '*_pvv.bin.gz' 2>/dev/null | wc -l)"
    if [ "$existing_gv" -ge 10 ] && [ "$existing_pvv" -ge 10 ]; then
      echo "$ds"
      return
    fi
    echo "Existing filtered dataset $ds has GV=$existing_gv PVV=$existing_pvv; rebuilding it." >&2
  fi

  echo "Creating filtered dataset: $ds" >&2
  rm -rf "$dst"
  mkdir -p "$dst/gv" "$dst/pvv"

  python - <<PY
from pathlib import Path
import os
import sys

src = Path("$src")
dst = Path("$dst")
threshold = int("$THRESHOLD")

gv_dir = src / "gv"
pvv_dir = src / "pvv"

pairs = []
for gv in sorted(gv_dir.glob("*_gv.bin.gz"), key=lambda p: int(p.name.split("_")[0])):
    idx = int(gv.name.split("_")[0])
    pvv = pvv_dir / f"{idx}_pvv.bin.gz"
    if pvv.exists() and gv.stat().st_size > threshold and pvv.stat().st_size > threshold:
        pairs.append((idx, gv, pvv))

for _, gv, pvv in pairs:
    os.symlink(gv, dst / "gv" / gv.name)
    os.symlink(pvv, dst / "pvv" / pvv.name)

print(f"Filtered {src.name}: {len(pairs)} valid pairs", file=sys.stderr, flush=True)
if len(pairs) < 10:
    raise SystemExit(f"Too few valid pairs for {src}")
PY

  echo "$ds"
}

latest_matching_dir() {
  local out_dir="$1"
  local tag="$2"
  find "$out_dir" -maxdepth 1 -type d -name "*${tag}*" | sort | tail -1
}

has_finished() {
  local log="$1"
  [ -f "$log" ] && grep -q "Training finished" "$log"
}

run_train() {
  local backend="$1"
  local radius="$2"
  local d="$3"
  local dataset="$4"

  local repo
  local out_dir
  local env_name

  if [ "$backend" = "fvdb" ]; then
    repo="$FVDB_REPO"
    out_dir="$RESULT_ROOT/fvdb_out"
    env_name="$FVDB_ENV"
  else
    repo="$SPCONV_REPO"
    out_dir="$RESULT_ROOT/spconv_out"
    env_name="$SPCONV_ENV"
  fi

  local tag="synthetic_may19_${radius}_${backend}_d${d}_b${BATCH}_depth${DEPTH}_${EPOCHS}ep"
  local log="$RESULT_ROOT/logs/${tag}.log"

  {
    echo "============================================================"
    echo "START TRAIN"
    echo "backend=$backend"
    echo "radius=$radius"
    echo "d=$d"
    echo "dataset=$dataset"
    echo "epochs=$EPOCHS"
    echo "batch=$BATCH"
    echo "result_root=$RESULT_ROOT"
    echo "log=$log"
    echo "============================================================"
  } | tee -a "$MASTER_LOG"

  if has_finished "$log"; then
    echo "SKIP already finished: $tag" | tee -a "$MASTER_LOG"
    return
  fi

  conda activate "$env_name"
  cd "$repo"

  python train.py \
    --root "$RESULT_ROOT/data" \
    --dataset_name "$dataset" \
    --z_size "$Z_SIZE" \
    --test_fraction "$TEST_FRACTION" \
    --model OACNNsInterleaved \
    --backend "$backend" \
    --model_depth "$DEPTH" \
    --interleaver_r "$d" \
    --batchSz "$BATCH" \
    --nEpochs "$EPOCHS" \
    --save_all_freq 999999 \
    --lr "$LR" \
    --opt adam \
    --loss dice,no_guess \
    --loss_weights 0.99,0.01 \
    --dice_alpha 0.001 \
    --out_dir "$out_dir" \
    --tag "$tag" \
    2>&1 | tee "$log"

  local exp_dir
  exp_dir="$(latest_matching_dir "$out_dir" "$tag")"

  {
    echo "Completed: $tag"
    echo "Output: $exp_dir"
    df -hT "$RESULT_ROOT" | tail -1
  } | tee -a "$MASTER_LOG"
}

{
  echo "Result root: $RESULT_ROOT"
  echo "Source root: $SOURCE_ROOT"
  echo "fVDB repo:   $FVDB_REPO"
  echo "spconv repo: $SPCONV_REPO"
  echo "Run specs:   $RUN_SPECS"
  echo "Order: for each radius/d pair, run fvdb then spconv"
} | tee -a "$MASTER_LOG"

declare -A DATASET_CACHE

for spec in $RUN_SPECS; do
  radius="${spec%%:*}"
  d="${spec##*:}"

  if [ -z "${DATASET_CACHE[$radius]+x}" ]; then
    dataset="$(make_filtered_dataset "$radius")"
    DATASET_CACHE[$radius]="$dataset"

    {
      echo "Dataset for $radius: $dataset"
      echo "GV count:  $(find -L "$RESULT_ROOT/data/datasets/$dataset/gv" -type f -name '*_gv.bin.gz' | wc -l)"
      echo "PVV count: $(find -L "$RESULT_ROOT/data/datasets/$dataset/pvv" -type f -name '*_pvv.bin.gz' | wc -l)"
    } | tee -a "$MASTER_LOG"
  else
    dataset="${DATASET_CACHE[$radius]}"
  fi

  run_train "fvdb" "$radius" "$d" "$dataset"
  run_train "spconv" "$radius" "$d" "$dataset"
done

echo "ALL TRAINING DONE" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG"
