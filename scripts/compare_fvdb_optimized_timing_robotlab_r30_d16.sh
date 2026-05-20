#!/usr/bin/env bash
set -eo pipefail

# Compare current fVDB timing against the optimized fVDB lab path for one pair.
#
# This reuses the same trained OACNNsInterleaved checkpoint and only reruns
# --timing inference:
#   scene:   RobotLab
#   radius:  r30
#   d:       16
#   epochs:  10
#
# Typical use:
#   cd /home/atighedl/Adapted_fvdb
#   git pull
#   bash scripts/compare_fvdb_optimized_timing_robotlab_r30_d16.sh

RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_10EP_ALTERNATING}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/var/tmp/atighedl_runs/canonical_data/FINAL_GVPVV}"

BASE_REPO="${BASE_REPO:-/home/atighedl/Adapted_fvdb}"
OPT_REPO="${OPT_REPO:-/home/atighedl/Adapted_fvdb_learned_ops_lab}"

FVDB_ENV="${FVDB_ENV:-fvdb_rc}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

SCENE_KEY="${SCENE_KEY:-robotlab}"
RADIUS="${RADIUS:-r30}"
D="${D:-16}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
Z_SIZE="${Z_SIZE:-256}"
TIMING_N_FRAMES="${TIMING_N_FRAMES:-0}" # 0 means full scene density pass.
OPT_INFER_EXTRA_ARGS="${OPT_INFER_EXTRA_ARGS:-}"

DATA_ROOT="$RESULT_ROOT/data"
OUT_DIR="$RESULT_ROOT/fvdb_out"
SUMMARY_DIR="$RESULT_ROOT/summaries"
LOG_DIR="$RESULT_ROOT/logs"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="$SUMMARY_DIR/compare_fvdb_optimized_${SCENE_KEY}_${RADIUS}_d${D}_${EPOCHS}ep_${STAMP}.csv"
BASE_LOG="$LOG_DIR/timing_compare_${SCENE_KEY}_${RADIUS}_d${D}_${EPOCHS}ep_base_${STAMP}.log"
OPT_LOG="$LOG_DIR/timing_compare_${SCENE_KEY}_${RADIUS}_d${D}_${EPOCHS}ep_optimized_${STAMP}.log"

mkdir -p "$DATA_ROOT/datasets" "$SUMMARY_DIR" "$LOG_DIR"

require_path() {
  local path="$1"
  local label="$2"
  if [ ! -e "$path" ]; then
    echo "ERROR: $label not found: $path" >&2
    exit 1
  fi
}

discover_opt_repo() {
  if [ -f "$OPT_REPO/infer.py" ]; then
    echo "$OPT_REPO"
    return 0
  fi

  local candidate
  for candidate in \
    /home/atighedl/Adapted_fvdb_learned_ops_lab \
    /home/atighedl/Adapted_fvdb_optimization_lab \
    /home/atighedl/*learned*ops* \
    /home/atighedl/*optimization*fvdb* \
    /home/atighedl/*Adapted*fvdb*learned* \
    /home/atighedl/*Adapted*fvdb*optimization*; do
    if [ -f "$candidate/infer.py" ]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

safe_name() {
  python - "$1" <<'PY'
import re
import sys
text = sys.argv[1].lower()
text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
print(text or "scene")
PY
}

norm_name() {
  python - "$1" <<'PY'
import re
import sys
print(re.sub(r"[^a-z0-9]+", "", sys.argv[1].lower()))
PY
}

scene_radius_dir() {
  local scene_key="$1"
  local radius="$2"
  local norm_key
  norm_key="$(norm_name "$scene_key")"

  while IFS= read -r rdir; do
    if [ ! -d "$rdir/gv" ] || [ ! -d "$rdir/pvv" ]; then
      continue
    fi
    local rel
    local norm_rel
    rel="${rdir#"$CANONICAL_ROOT"/}"
    norm_rel="$(norm_name "$rel")"
    if [[ "$norm_rel" == *"$norm_key"* ]]; then
      echo "$rdir"
      return 0
    fi
  done < <(find "$CANONICAL_ROOT" -type d -name "$radius" | sort)

  return 1
}

latest_exp_dir() {
  local pattern="*synthetic_may19_${RADIUS}_fvdb_d${D}_b${BATCH}_depth${DEPTH}_${EPOCHS}ep*"
  local dir

  while IFS= read -r dir; do
    if find "$dir" -maxdepth 1 -type f -name '*.pth' | grep -q .; then
      echo "$dir"
      return 0
    fi
  done < <(find "$OUT_DIR" -maxdepth 1 -type d -name "$pattern" | sort -r)

  return 1
}

checkpoint_suffix() {
  local exp_dir="$1"
  local exp_name="$2"

  if [ -f "$exp_dir/${exp_name}_BEST.pth" ]; then
    echo ""
    return
  fi

  if [ -f "$exp_dir/${exp_name}_last_epoch.pth" ]; then
    echo "last_epoch"
    return
  fi

  local latest_epoch
  latest_epoch="$({ find "$exp_dir" -maxdepth 1 -type f -name "${exp_name}_*_epoch.pth" 2>/dev/null || true; } | sed -E 's/.*_([0-9]+)_epoch\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [ -n "$latest_epoch" ] && [ -f "$latest_epoch" ]; then
    basename "$latest_epoch" | sed -E "s/^${exp_name}_(.*)\.pth$/\1/"
    return
  fi

  echo "ERROR_NO_CHECKPOINT"
}

make_scene_dataset() {
  local scene_src="$1"
  local scene_id
  scene_id="$(safe_name "$SCENE_KEY")"
  local dataset="infer_${scene_id}_${RADIUS}_${EPOCHS}ep_timing_compare"
  local dst="$DATA_ROOT/datasets/$dataset"

  rm -rf "$dst/gv" "$dst/pvv"
  mkdir -p "$dst"
  ln -s "$scene_src/gv" "$dst/gv"
  ln -s "$scene_src/pvv" "$dst/pvv"

  local gv_count
  local pvv_count
  gv_count="$(find -L "$dst/gv" -type f -name '*_gv.bin.gz' | wc -l)"
  pvv_count="$(find -L "$dst/pvv" -type f -name '*_pvv.bin.gz' | wc -l)"

  echo "Dataset: $dataset"
  echo "Scene:   $SCENE_KEY"
  echo "Radius:  $RADIUS"
  echo "Source:  $scene_src"
  echo "GV:      $gv_count"
  echo "PVV:     $pvv_count"

  if [ "$gv_count" -eq 0 ] || [ "$pvv_count" -eq 0 ]; then
    echo "ERROR: scene has no GV/PVV files after symlink." >&2
    exit 1
  fi

  echo "$dataset"
}

run_timing() {
  local label="$1"
  local repo="$2"
  local log="$3"
  local exp_name="$4"
  local dataset="$5"
  local ckpt_suffix="$6"

  echo "============================================================"
  echo "START TIMING: $label"
  echo "repo=$repo"
  echo "exp=$exp_name"
  echo "dataset=$dataset"
  echo "ckpt_suffix=${ckpt_suffix:-BEST}"
  if [ "$label" = "optimized" ] && [ -n "$OPT_INFER_EXTRA_ARGS" ]; then
    echo "extra_args=$OPT_INFER_EXTRA_ARGS"
  fi
  echo "log=$log"
  echo "============================================================"

  conda activate "$FVDB_ENV"
  cd "$repo"

  args=(
    python infer.py
    --root "$DATA_ROOT"
    --out_dir "$OUT_DIR"
    --dataset_name "$dataset"
    --z_size "$Z_SIZE"
    --exp_name "$exp_name"
    --infer_tag "${SCENE_KEY}_${RADIUS}_fvdb_d${D}_${EPOCHS}ep_${label}_timing_compare"
    --timing
  )

  if [ -n "$ckpt_suffix" ]; then
    args+=(--ckpt_suffix "$ckpt_suffix")
  fi

  if [ "$TIMING_N_FRAMES" != "0" ]; then
    args+=(--n_frames "$TIMING_N_FRAMES")
  fi

  if [ "$label" = "optimized" ] && [ -n "$OPT_INFER_EXTRA_ARGS" ]; then
    read -r -a extra_args <<< "$OPT_INFER_EXTRA_ARGS"
    args+=("${extra_args[@]}")
  fi

  "${args[@]}" 2>&1 | tee "$log"
}

write_summary() {
  python - "$SUMMARY_CSV" "$BASE_LOG" "$OPT_LOG" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

summary_csv, base_log, opt_log = sys.argv[1:]
keys = [
    "infer_time_mean",
    "infer_time_pure_mean",
    "interleaver_time_mean",
    "sparse_core_time_mean",
    "deinterleaver_time_mean",
    "peak_mem",
    "density_mean",
]

def parse_log(path):
    text = Path(path).read_text(errors="replace")
    row = {"log": path}
    for key in keys:
        matches = re.findall(rf"(?:^|\s|\|){re.escape(key)}:\s*([0-9.eE+-]+)", text)
        row[key] = float(matches[-1]) if matches else math.nan
    return row

rows = [("baseline", parse_log(base_log)), ("optimized", parse_log(opt_log))]
baseline = rows[0][1]
optimized = rows[1][1]

with Path(summary_csv).open("w", newline="") as f:
    fieldnames = ["variant"] + keys + ["log"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for name, row in rows:
        writer.writerow({"variant": name, **row})

def pct_drop(old, new):
    try:
        return (old - new) / old * 100.0
    except Exception:
        return math.nan

print()
print("FVDB OPTIMIZED TIMING COMPARISON")
print("variant    infer_ms   pure_ms    inter_ms   core_ms    deinter_ms peak_mb")
print("---------  ---------  ---------  ---------  ---------  ---------- -------")
for name, row in rows:
    print(
        f"{name:<9}  "
        f"{row['infer_time_mean']:>9.3f}  "
        f"{row['infer_time_pure_mean']:>9.3f}  "
        f"{row['interleaver_time_mean']:>9.3f}  "
        f"{row['sparse_core_time_mean']:>9.3f}  "
        f"{row['deinterleaver_time_mean']:>10.3f} "
        f"{row['peak_mem']:>7.1f}"
    )

print()
print(f"Total infer time drop: {pct_drop(baseline['infer_time_mean'], optimized['infer_time_mean']):.2f}%")
print(f"Pure/core time drop:   {pct_drop(baseline['infer_time_pure_mean'], optimized['infer_time_pure_mean']):.2f}%")
print(f"CSV written to: {summary_csv}")
PY
}

require_path "$CONDA_SH" "conda activation script"
require_path "$BASE_REPO/infer.py" "baseline infer.py"
OPT_REPO="$(discover_opt_repo || true)"
if [ -z "$OPT_REPO" ]; then
  echo "ERROR: optimized lab infer.py not found." >&2
  echo "Set OPT_REPO=/path/to/Adapted_fvdb_learned_ops_lab and rerun." >&2
  echo "Quick search command:" >&2
  echo "find /home/atighedl -maxdepth 2 -name infer.py | grep -Ei 'learned|optim|fvdb'" >&2
  exit 1
fi
require_path "$OPT_REPO/infer.py" "optimized lab infer.py"
require_path "$CANONICAL_ROOT" "CANONICAL_ROOT"
require_path "$OUT_DIR" "fVDB output root"

EXP_DIR="${EXP_DIR:-$(latest_exp_dir || true)}"
if [ -z "$EXP_DIR" ] || [ ! -d "$EXP_DIR" ]; then
  echo "ERROR: no completed fVDB $RADIUS d$D ${EPOCHS}ep checkpoint found in $OUT_DIR" >&2
  exit 1
fi

EXP_NAME="$(basename "$EXP_DIR")"
CKPT_SUFFIX="$(checkpoint_suffix "$EXP_DIR" "$EXP_NAME")"
if [ "$CKPT_SUFFIX" = "ERROR_NO_CHECKPOINT" ]; then
  echo "ERROR: no checkpoint found in $EXP_DIR" >&2
  exit 1
fi

SCENE_SRC="$(scene_radius_dir "$SCENE_KEY" "$RADIUS" || true)"
if [ -z "$SCENE_SRC" ]; then
  echo "ERROR: could not find $SCENE_KEY $RADIUS under $CANONICAL_ROOT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH"

DATASET="$(make_scene_dataset "$SCENE_SRC" | tail -1)"

echo "Using experiment: $EXP_NAME"
echo "Using checkpoint suffix: ${CKPT_SUFFIX:-BEST}"
echo

run_timing "baseline" "$BASE_REPO" "$BASE_LOG" "$EXP_NAME" "$DATASET" "$CKPT_SUFFIX"
run_timing "optimized" "$OPT_REPO" "$OPT_LOG" "$EXP_NAME" "$DATASET" "$CKPT_SUFFIX"
write_summary
