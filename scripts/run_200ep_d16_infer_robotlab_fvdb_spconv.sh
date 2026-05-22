#!/usr/bin/env bash
set -eo pipefail

# Infer RobotLab/ROBOT_FINAL using completed d=16 200-epoch fVDB and spconv runs.
#
# Defaults:
#   result root: /var/tmp/$USER_runs/NEURALPVS_MAY19_SYNTHETIC_D16_200EP
#   scene:       RobotLab r30 from ROBOT_FINAL
#
# Typical use:
#   cd /var/tmp/${USER}_repos/Adapted_fvdb_BACKENDOPTIMIZATION
#   git pull origin main
#   bash scripts/run_200ep_d16_infer_robotlab_fvdb_spconv.sh
#
# Override examples:
#   RADIUS=r60 bash scripts/run_200ep_d16_infer_robotlab_fvdb_spconv.sh
#   N_FRAMES=20 bash scripts/run_200ep_d16_infer_robotlab_fvdb_spconv.sh
#   SCENE_SRC=/path/to/ROBOT_FINAL/robotlab/r30 bash scripts/run_200ep_d16_infer_robotlab_fvdb_spconv.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_D16_200EP}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/var/tmp/${USER}_runs/canonical_data/FINAL_GVPVV}"

FVDB_REPO="${FVDB_REPO:-$REPO_ROOT}"
SPCONV_REPO="${SPCONV_REPO:-/home/atighedl/neuralpvs}"
ADAPTED_REPO="${ADAPTED_REPO:-/home/atighedl/Adapted_fvdb}"

FVDB_ENV="${FVDB_ENV:-fvdb_rc}"
SPCONV_ENV="${SPCONV_ENV:-cuda128}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

RADIUS="${RADIUS:-r30}"
D="${D:-16}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
EPOCHS="${EPOCHS:-200}"
Z_SIZE="${Z_SIZE:-256}"
N_FRAMES="${N_FRAMES:-0}"      # 0 means full RobotLab scene.
RUN_TIMING="${RUN_TIMING:-0}"  # 1 adds --timing instead of normal metric inference.

DATA_ROOT="$RESULT_ROOT/data"
FVDB_OUT="$RESULT_ROOT/fvdb_out"
SPCONV_OUT="$RESULT_ROOT/spconv_out"
SUMMARY_DIR="$RESULT_ROOT/summaries"
LOG_DIR="$RESULT_ROOT/logs"

mkdir -p "$DATA_ROOT/datasets" "$SUMMARY_DIR" "$LOG_DIR"

require_path() {
  local path="$1"
  local label="$2"
  if [ ! -e "$path" ]; then
    echo "ERROR: $label not found: $path" >&2
    exit 1
  fi
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

first_existing_scene() {
  for candidate in \
    "$ADAPTED_REPO/ROBOT_FINAL/robotlab/$RADIUS" \
    "$ADAPTED_REPO/ROBOT_FINAL/RobotLab/$RADIUS" \
    "$ADAPTED_REPO/ROBOTLAB_FINAL/robotlab/$RADIUS" \
    "$CANONICAL_ROOT/ROBOT_FINAL/robotlab/$RADIUS" \
    "$CANONICAL_ROOT/ROBOT_FINAL/RobotLab/$RADIUS" \
    "$CANONICAL_ROOT/ROBOTLAB_FINAL/robotlab/$RADIUS" \
    "$FVDB_REPO/ROBOT_FINAL/robotlab/$RADIUS"; do
    if [ -d "$candidate/gv" ] && [ -d "$candidate/pvv" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

latest_exp_dir() {
  local out_dir="$1"
  local backend="$2"
  local pattern="*synthetic_may19_${RADIUS}_${backend}_d${D}_b${BATCH}_depth${DEPTH}_${EPOCHS}ep*"
  local dir

  while IFS= read -r dir; do
    if find "$dir" -maxdepth 1 -type f -name '*.pth' | grep -q .; then
      echo "$dir"
      return 0
    fi
  done < <(find "$out_dir" -maxdepth 1 -type d -name "$pattern" | sort -r)

  return 1
}

require_finished_training_log() {
  local backend="$1"
  local log="$LOG_DIR/synthetic_may19_${RADIUS}_${backend}_d${D}_b${BATCH}_depth${DEPTH}_${EPOCHS}ep.log"

  require_path "$log" "$backend training log"
  if ! grep -q "Training finished" "$log"; then
    local epoch
    epoch="$(grep -o "epoch: [0-9]*" "$log" | tail -1 | awk '{print $2}')"
    echo "ERROR: $backend $RADIUS d=$D ${EPOCHS}ep training is not finished yet; last epoch=${epoch:-unknown}" >&2
    echo "Log: $log" >&2
    exit 1
  fi
}

print_exp_candidates() {
  local out_dir="$1"
  local backend="$2"
  local pattern="*synthetic_may19_${RADIUS}_${backend}_d${D}_b${BATCH}_depth${DEPTH}_${EPOCHS}ep*"
  echo "Matching $backend experiment folders:" >&2
  find "$out_dir" -maxdepth 1 -type d -name "$pattern" | sort | tail -10 >&2 || true
  echo "Matching $backend checkpoint files:" >&2
  find "$out_dir" -path "*$pattern*" -type f -name '*.pth' | sort | tail -20 >&2 || true
}

make_scene_dataset() {
  local scene_src="$1"
  local scene_name="$2"
  local scene_id
  scene_id="$(safe_name "$scene_name")"
  local dataset="infer_${scene_id}_${RADIUS}_d${D}_${EPOCHS}ep"
  local dst="$DATA_ROOT/datasets/$dataset"

  rm -rf "$dst/gv" "$dst/pvv"
  mkdir -p "$dst"
  ln -s "$scene_src/gv" "$dst/gv"
  ln -s "$scene_src/pvv" "$dst/pvv"

  local gv_count
  local pvv_count
  gv_count="$(find -L "$dst/gv" -type f -name '*_gv.bin.gz' | wc -l)"
  pvv_count="$(find -L "$dst/pvv" -type f -name '*_pvv.bin.gz' | wc -l)"

  echo "Scene source: $scene_src" >&2
  echo "Dataset:      $dataset" >&2
  echo "GV count:     $gv_count" >&2
  echo "PVV count:    $pvv_count" >&2

  if [ "$gv_count" -eq 0 ] || [ "$pvv_count" -eq 0 ]; then
    echo "ERROR: scene has no GV/PVV files after symlink." >&2
    exit 1
  fi

  echo "$dataset"
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

run_infer() {
  local backend="$1"
  local repo="$2"
  local env_name="$3"
  local out_dir="$4"
  local exp_dir="$5"
  local dataset="$6"
  local scene_name="$7"

  local exp_name
  exp_name="$(basename "$exp_dir")"
  local tag="${scene_name}_${RADIUS}_d${D}_${backend}_${EPOCHS}ep"
  local log="$LOG_DIR/infer_${scene_name}_${RADIUS}_${backend}_d${D}_${EPOCHS}ep.log"
  local ckpt_suffix
  ckpt_suffix="$(checkpoint_suffix "$exp_dir" "$exp_name")"

  if [ "$ckpt_suffix" = "ERROR_NO_CHECKPOINT" ]; then
    echo "ERROR: no checkpoint found for $backend in $exp_dir" >&2
    echo "Expected one of: ${exp_name}_BEST.pth, ${exp_name}_last_epoch.pth, or ${exp_name}_*_epoch.pth" >&2
    exit 1
  fi

  echo "============================================================"
  echo "START INFER"
  echo "backend=$backend"
  echo "repo=$repo"
  echo "exp=$exp_name"
  echo "ckpt_suffix=${ckpt_suffix:-BEST}"
  echo "dataset=$dataset"
  echo "log=$log"
  echo "============================================================"

  conda activate "$env_name"
  cd "$repo"

  args=(
    python infer.py
    --root "$DATA_ROOT"
    --out_dir "$out_dir"
    --dataset_name "$dataset"
    --z_size "$Z_SIZE"
    --exp_name "$exp_name"
    --infer_tag "$tag"
  )

  if [ -n "$ckpt_suffix" ]; then
    args+=(--ckpt_suffix "$ckpt_suffix")
  fi

  if [ "$N_FRAMES" != "0" ]; then
    args+=(--n_frames "$N_FRAMES")
  fi

  if [ "$RUN_TIMING" = "1" ]; then
    args+=(--timing)
  fi

  "${args[@]}" 2>&1 | tee "$log"

  local latest_output
  latest_output="$(find "$out_dir" -maxdepth 1 -type d -name "*${dataset}*${tag}*" | sort | tail -1)"
  if [ -z "$latest_output" ]; then
    echo "ERROR: could not find inference output for $backend" >&2
    exit 1
  fi

  echo "$latest_output"
}

summarize_outputs() {
  local summary_csv="$1"
  shift

  python - "$summary_csv" "$@" <<'PY'
import csv
import statistics as stats
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
pairs = sys.argv[2:]
metrics = ["dice", "loss", "fp", "fn", "fp_rate", "fn_rate", "fp_ratio", "gv_ratio"]

def load_stats(output):
    path = Path(output) / "eval_stats.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    out = {"frames": len(rows)}
    for key in metrics:
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        out[f"{key}_mean"] = stats.mean(vals) if vals else float("nan")
        out[f"{key}_std"] = stats.pstdev(vals) if len(vals) > 1 else 0.0
    return out

records = []
for item in pairs:
    backend, output = item.split("=", 1)
    row = load_stats(output)
    row["backend"] = backend
    row["output_dir"] = output
    row["predicted_pvv_folder"] = str(Path(output) / "inference" / "0")
    records.append(row)

print()
print("ROBOTLAB D16 200EP INFERENCE SUMMARY")
print("backend  frames  dice      loss      fp_rate   fn_rate   fp_ratio  gv_ratio")
print("-------  ------  --------  --------  --------  --------  --------  --------")
for row in records:
    print(
        f"{row['backend']:<7}  {row['frames']:>6}  "
        f"{row['dice_mean']:.6f}  {row['loss_mean']:.6f}  "
        f"{row['fp_rate_mean']:.6f}  {row['fn_rate_mean']:.6f}  "
        f"{row['fp_ratio_mean']:.6f}  {row['gv_ratio_mean']:.6f}"
    )

fieldnames = ["backend", "frames"]
for key in metrics:
    fieldnames += [f"{key}_mean", f"{key}_std"]
fieldnames += ["output_dir", "predicted_pvv_folder"]

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)

print()
print("Predicted PVV folders:")
for row in records:
    print(f"{row['backend']}: {row['predicted_pvv_folder']}")

print()
print(f"CSV written to: {summary_csv}")
PY
}

require_path "$CONDA_SH" "conda activation script"
require_path "$FVDB_REPO/infer.py" "fVDB infer.py"
require_path "$SPCONV_REPO/infer.py" "spconv infer.py"
require_path "$FVDB_OUT" "fVDB output root"
require_path "$SPCONV_OUT" "spconv output root"
require_finished_training_log "fvdb"
require_finished_training_log "spconv"

SCENE_SRC="${SCENE_SRC:-$(first_existing_scene || true)}"
if [ -z "$SCENE_SRC" ]; then
  echo "ERROR: could not auto-find RobotLab/ROBOT_FINAL $RADIUS scene." >&2
  echo "Set SCENE_SRC=/path/to/ROBOT_FINAL/robotlab/$RADIUS and rerun." >&2
  exit 1
fi
require_path "$SCENE_SRC/gv" "scene gv folder"
require_path "$SCENE_SRC/pvv" "scene pvv folder"

SCENE_NAME="${SCENE_NAME:-robotlab_${RADIUS}}"
SCENE_NAME="$(safe_name "$SCENE_NAME")"

# shellcheck source=/dev/null
source "$CONDA_SH"

DATASET="$(make_scene_dataset "$SCENE_SRC" "$SCENE_NAME")"

FVDB_EXP_DIR="${FVDB_EXP_DIR:-$(latest_exp_dir "$FVDB_OUT" "fvdb" || true)}"
SPCONV_EXP_DIR="${SPCONV_EXP_DIR:-$(latest_exp_dir "$SPCONV_OUT" "spconv" || true)}"

if [ -z "$FVDB_EXP_DIR" ] || [ ! -d "$FVDB_EXP_DIR" ]; then
  echo "ERROR: could not find fVDB $RADIUS d$D ${EPOCHS}ep experiment with a checkpoint in $FVDB_OUT" >&2
  print_exp_candidates "$FVDB_OUT" "fvdb"
  exit 1
fi

if [ -z "$SPCONV_EXP_DIR" ] || [ ! -d "$SPCONV_EXP_DIR" ]; then
  echo "ERROR: could not find spconv $RADIUS d$D ${EPOCHS}ep experiment with a checkpoint in $SPCONV_OUT" >&2
  print_exp_candidates "$SPCONV_OUT" "spconv"
  exit 1
fi

echo "Using RobotLab scene:     $SCENE_SRC"
echo "Using fVDB experiment:   $FVDB_EXP_DIR"
echo "Using spconv experiment: $SPCONV_EXP_DIR"

FVDB_OUTPUT="$(run_infer "fvdb" "$FVDB_REPO" "$FVDB_ENV" "$FVDB_OUT" "$FVDB_EXP_DIR" "$DATASET" "$SCENE_NAME" | tail -1)"
SPCONV_OUTPUT="$(run_infer "spconv" "$SPCONV_REPO" "$SPCONV_ENV" "$SPCONV_OUT" "$SPCONV_EXP_DIR" "$DATASET" "$SCENE_NAME" | tail -1)"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="$SUMMARY_DIR/infer_${SCENE_NAME}_${RADIUS}_d${D}_${EPOCHS}ep_fvdb_vs_spconv_${STAMP}.csv"

summarize_outputs "$SUMMARY_CSV" "fvdb=$FVDB_OUTPUT" "spconv=$SPCONV_OUTPUT"
