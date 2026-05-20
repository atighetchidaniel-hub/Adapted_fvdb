#!/usr/bin/env bash
set -eo pipefail

# Infer one scene using the finished 5-epoch r30 d16 fVDB and spconv runs.
#
# Defaults:
#   result root: /var/tmp/$USER_runs/NEURALPVS_MAY19_SYNTHETIC_5EP_ALTERNATING
#   scene:       RobotLab r30, preferring canonical data
#
# Typical use:
#   cd /home/atighedl/Adapted_fvdb
#   git pull
#   bash scripts/run_5ep_d16_infer_scene_fvdb_spconv.sh
#
# Override scene if needed:
#   SCENE_SRC=/path/to/scene/r30 SCENE_NAME=sponza_r30 bash scripts/run_5ep_d16_infer_scene_fvdb_spconv.sh

RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_5EP_ALTERNATING}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/var/tmp/atighedl_runs/canonical_data/FINAL_GVPVV}"

FVDB_REPO="${FVDB_REPO:-/home/atighedl/Adapted_fvdb}"
SPCONV_REPO="${SPCONV_REPO:-/home/atighedl/neuralpvs}"

FVDB_ENV="${FVDB_ENV:-fvdb_rc}"
SPCONV_ENV="${SPCONV_ENV:-cuda128}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

RADIUS="${RADIUS:-r30}"
D="${D:-16}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
EPOCHS="${EPOCHS:-5}"
Z_SIZE="${Z_SIZE:-256}"
N_FRAMES="${N_FRAMES:-0}"      # 0 means full scene.
RUN_TIMING="${RUN_TIMING:-0}"  # 1 adds --timing.

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
    "$CANONICAL_ROOT/ROBOT_FINAL/robotlab/$RADIUS" \
    "$CANONICAL_ROOT/ROBOTLAB_FINAL/robotlab/$RADIUS" \
    "$CANONICAL_ROOT/ROBOT_FINAL/RobotLab/$RADIUS" \
    "$CANONICAL_ROOT/ROBOTLAB_FINAL/RobotLab/$RADIUS" \
    "$FVDB_REPO/ROBOT_FINAL/robotlab/$RADIUS" \
    "$FVDB_REPO/ROBOTLAB_FINAL/robotlab/$RADIUS"; do
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
  find "$out_dir" -maxdepth 1 -type d -name "$pattern" | sort | tail -1
}

make_scene_dataset() {
  local scene_src="$1"
  local scene_name="$2"
  local scene_id
  scene_id="$(safe_name "$scene_name")"
  local dataset="infer_${scene_id}_${RADIUS}_5ep_d${D}"
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
  local tag="${RADIUS}_d${D}_${EPOCHS}ep_${backend}_to_${scene_name}"
  local log="$LOG_DIR/infer_${scene_name}_${backend}_${RADIUS}_d${D}_${EPOCHS}ep.log"

  require_path "$exp_dir/${exp_name}_BEST.pth" "$backend BEST checkpoint"

  echo "============================================================"
  echo "START INFER"
  echo "backend=$backend"
  echo "exp=$exp_name"
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

summarize_pair() {
  local fvdb_output="$1"
  local spconv_output="$2"
  local summary_csv="$3"

  python - "$fvdb_output/eval_stats.csv" "$spconv_output/eval_stats.csv" "$summary_csv" "$fvdb_output" "$spconv_output" <<'PY'
import csv
import statistics as stats
import sys
from pathlib import Path

fvdb_stats, spconv_stats, summary_csv, fvdb_output, spconv_output = sys.argv[1:]
metrics = ["dice", "loss", "fp", "fn", "fp_rate", "fn_rate", "fp_ratio", "gv_ratio"]

def load(path):
    rows = list(csv.DictReader(Path(path).open(newline="")))
    out = {"samples": len(rows)}
    for key in metrics:
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        if vals:
            out[f"{key}_mean"] = stats.mean(vals)
            out[f"{key}_std"] = stats.pstdev(vals) if len(vals) > 1 else 0.0
        else:
            out[f"{key}_mean"] = ""
            out[f"{key}_std"] = ""
    return out

records = [
    ("fvdb", load(fvdb_stats), fvdb_output),
    ("spconv", load(spconv_stats), spconv_output),
]

print()
print("5-EPOCH D16 SCENE INFERENCE SUMMARY")
print("backend  samples  dice      fp_rate   fn_rate   fp_ratio  gv_ratio")
print("-------  -------  --------  --------  --------  --------  --------")
for backend, row, output in records:
    print(
        f"{backend:<7}  {row['samples']:>7}  "
        f"{row['dice_mean']:.6f}  {row['fp_rate_mean']:.6f}  "
        f"{row['fn_rate_mean']:.6f}  {row['fp_ratio_mean']:.6f}  "
        f"{row['gv_ratio_mean']:.6f}"
    )

fieldnames = ["backend", "samples"]
for key in metrics:
    fieldnames += [f"{key}_mean", f"{key}_std"]
fieldnames += ["output_dir", "predicted_pvv_folder"]

with Path(summary_csv).open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for backend, row, output in records:
        record = {"backend": backend, **row}
        record["output_dir"] = output
        record["predicted_pvv_folder"] = str(Path(output) / "inference" / "0")
        writer.writerow(record)

print()
print(f"CSV written to: {summary_csv}")
print()
print("Predicted PVV folders:")
print(f"fvdb:   {Path(fvdb_output) / 'inference' / '0'}")
print(f"spconv: {Path(spconv_output) / 'inference' / '0'}")
PY
}

require_path "$CONDA_SH" "conda activation script"
require_path "$FVDB_REPO/infer.py" "fVDB infer.py"
require_path "$SPCONV_REPO/infer.py" "spconv infer.py"
require_path "$FVDB_OUT" "fVDB 5ep output root"
require_path "$SPCONV_OUT" "spconv 5ep output root"

SCENE_SRC="${SCENE_SRC:-$(first_existing_scene || true)}"
if [ -z "$SCENE_SRC" ]; then
  echo "ERROR: could not auto-find a default RobotLab $RADIUS scene." >&2
  echo "Set SCENE_SRC=/path/to/scene/$RADIUS and rerun." >&2
  exit 1
fi
require_path "$SCENE_SRC/gv" "scene gv folder"
require_path "$SCENE_SRC/pvv" "scene pvv folder"

SCENE_NAME="${SCENE_NAME:-robotlab_${RADIUS}}"
SCENE_NAME="$(safe_name "$SCENE_NAME")"
DATASET="$(make_scene_dataset "$SCENE_SRC" "$SCENE_NAME")"

# shellcheck source=/dev/null
source "$CONDA_SH"

FVDB_EXP_DIR="${FVDB_EXP_DIR:-$(latest_exp_dir "$FVDB_OUT" "fvdb")}"
SPCONV_EXP_DIR="${SPCONV_EXP_DIR:-$(latest_exp_dir "$SPCONV_OUT" "spconv")}"

if [ -z "$FVDB_EXP_DIR" ] || [ ! -d "$FVDB_EXP_DIR" ]; then
  echo "ERROR: could not find fVDB $RADIUS d$D ${EPOCHS}ep experiment in $FVDB_OUT" >&2
  exit 1
fi

if [ -z "$SPCONV_EXP_DIR" ] || [ ! -d "$SPCONV_EXP_DIR" ]; then
  echo "ERROR: could not find spconv $RADIUS d$D ${EPOCHS}ep experiment in $SPCONV_OUT" >&2
  exit 1
fi

echo "Using fVDB experiment:   $FVDB_EXP_DIR"
echo "Using spconv experiment: $SPCONV_EXP_DIR"

FVDB_OUTPUT="$(run_infer "fvdb" "$FVDB_REPO" "$FVDB_ENV" "$FVDB_OUT" "$FVDB_EXP_DIR" "$DATASET" "$SCENE_NAME" | tail -1)"
SPCONV_OUTPUT="$(run_infer "spconv" "$SPCONV_REPO" "$SPCONV_ENV" "$SPCONV_OUT" "$SPCONV_EXP_DIR" "$DATASET" "$SCENE_NAME" | tail -1)"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="$SUMMARY_DIR/infer_${SCENE_NAME}_${RADIUS}_d${D}_${EPOCHS}ep_fvdb_vs_spconv_${STAMP}.csv"

summarize_pair "$FVDB_OUTPUT" "$SPCONV_OUTPUT" "$SUMMARY_CSV"
