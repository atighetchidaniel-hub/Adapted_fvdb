#!/usr/bin/env bash
set -eo pipefail

# Run the finished May19 synthetic fVDB r30 model on every canonical r30 scene.
#
# Defaults target the first completed 200-epoch run:
#   backend: fVDB
#   radius:  r30
#   d:       8
#   epochs:  200
#
# Typical use:
#   cd /home/atighedl/Adapted_fvdb
#   git pull
#   bash scripts/run_fvdb_r30_200ep_infer_all_canonical.sh
#
# Override when needed:
#   D=16 bash scripts/run_fvdb_r30_200ep_infer_all_canonical.sh
#   RESULT_ROOT=/path/to/results bash scripts/run_fvdb_r30_200ep_infer_all_canonical.sh

ADAPTED_REPO="${ADAPTED_REPO:-/home/atighedl/Adapted_fvdb}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/var/tmp/atighedl_runs/canonical_data/FINAL_GVPVV}"
RESULT_ROOT="${RESULT_ROOT:-/home/atighedl/NEURALPVS_MAY19_SYNTHETIC_200EP}"

FVDB_ENV="${FVDB_ENV:-fvdb_rc}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

RADIUS="${RADIUS:-r30}"
D="${D:-8}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
EPOCHS="${EPOCHS:-200}"
Z_SIZE="${Z_SIZE:-256}"
N_FRAMES="${N_FRAMES:-0}"        # 0 means full scene.
RUN_TIMING="${RUN_TIMING:-0}"    # 1 adds --timing.

DATA_ROOT="$RESULT_ROOT/data"
OUT_DIR="$RESULT_ROOT/fvdb_out"
SUMMARY_DIR="$RESULT_ROOT/summaries"
LOG_DIR="$RESULT_ROOT/logs"

mkdir -p "$DATA_ROOT/datasets" "$SUMMARY_DIR" "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
INFER_TAG="${INFER_TAG:-${RADIUS}_fvdb_d${D}_${EPOCHS}ep_all_canonical}"
SUMMARY_CSV="$SUMMARY_DIR/fvdb_${RADIUS}_d${D}_${EPOCHS}ep_all_canonical_inference_${STAMP}.csv"
SUMMARY_MD="$SUMMARY_DIR/fvdb_${RADIUS}_d${D}_${EPOCHS}ep_all_canonical_inference_${STAMP}.md"
MASTER_LOG="$LOG_DIR/fvdb_${RADIUS}_d${D}_${EPOCHS}ep_all_canonical_inference_${STAMP}.log"

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

latest_exp_dir() {
  local pattern="*synthetic_may19_${RADIUS}_fvdb_d${D}_b${BATCH}_depth${DEPTH}_${EPOCHS}ep*"
  find "$OUT_DIR" -maxdepth 1 -type d -name "$pattern" | sort | tail -1
}

discover_scene_radius_dirs() {
  find "$CANONICAL_ROOT" -type d -name "$RADIUS" | sort | while read -r rdir; do
    if [ -d "$rdir/gv" ] && [ -d "$rdir/pvv" ]; then
      echo "$rdir"
    fi
  done
}

make_dataset_symlink() {
  local scene_label="$1"
  local src_radius_dir="$2"
  local scene_id="$3"
  local dataset="infer_${scene_id}_${RADIUS}"
  local dst="$DATA_ROOT/datasets/$dataset"

  rm -rf "$dst/gv" "$dst/pvv"
  mkdir -p "$dst"
  ln -s "$src_radius_dir/gv" "$dst/gv"
  ln -s "$src_radius_dir/pvv" "$dst/pvv"

  local gv_count
  local pvv_count
  gv_count="$(find -L "$dst/gv" -type f -name '*_gv.bin.gz' | wc -l)"
  pvv_count="$(find -L "$dst/pvv" -type f -name '*_pvv.bin.gz' | wc -l)"

  {
    echo "Dataset: $dataset"
    echo "Scene:   $scene_label"
    echo "Source:  $src_radius_dir"
    echo "GV:      $gv_count"
    echo "PVV:     $pvv_count"
  } | tee -a "$MASTER_LOG" >&2

  if [ "$gv_count" -eq 0 ] || [ "$pvv_count" -eq 0 ]; then
    echo "ERROR: $scene_label has no GV/PVV files after symlink." >&2
    exit 1
  fi

  echo "$dataset"
}

append_summary_row() {
  local scene_label="$1"
  local scene_id="$2"
  local src_radius_dir="$3"
  local dataset="$4"
  local output_dir="$5"
  local pvv_folder="$6"
  local stats_file="$7"

  python - "$SUMMARY_CSV" "$scene_label" "$scene_id" "$src_radius_dir" "$dataset" "$output_dir" "$pvv_folder" "$stats_file" <<'PY'
import csv
import math
import statistics as stats
import sys
from pathlib import Path

summary_csv, scene_label, scene_id, source, dataset, output_dir, pvv_folder, stats_file = sys.argv[1:]
stats_path = Path(stats_file)
out_path = Path(summary_csv)

keys = ["dice", "loss", "fp", "fn", "fp_rate", "fn_rate", "fp_ratio", "gv_ratio"]

rows = []
with stats_path.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def values(key):
    vals = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except Exception:
            pass
    return vals

record = {
    "scene": scene_label,
    "scene_id": scene_id,
    "dataset": dataset,
    "frames": len(rows),
    "source": source,
    "output_dir": output_dir,
    "predicted_pvv_folder": pvv_folder,
}

for key in keys:
    vals = values(key)
    if vals:
        record[f"{key}_mean"] = stats.mean(vals)
        record[f"{key}_std"] = stats.pstdev(vals) if len(vals) > 1 else 0.0
        record[f"{key}_min"] = min(vals)
        record[f"{key}_max"] = max(vals)
    else:
        record[f"{key}_mean"] = math.nan
        record[f"{key}_std"] = math.nan
        record[f"{key}_min"] = math.nan
        record[f"{key}_max"] = math.nan

fieldnames = [
    "scene", "scene_id", "dataset", "frames",
    "dice_mean", "dice_std", "dice_min", "dice_max",
    "loss_mean", "fp_mean", "fn_mean", "fp_rate_mean", "fn_rate_mean",
    "fp_ratio_mean", "gv_ratio_mean",
    "source", "output_dir", "predicted_pvv_folder",
]

write_header = not out_path.exists()
with out_path.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        writer.writeheader()
    writer.writerow(record)
PY
}

write_markdown_summary() {
  python - "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])

rows = list(csv.DictReader(csv_path.open(newline="")))

def fmt(value, places=6):
    try:
        return f"{float(value):.{places}f}"
    except Exception:
        return str(value)

lines = []
lines.append("# fVDB r30 200ep Canonical Inference Summary")
lines.append("")
lines.append("| Scene | Frames | Dice | FP rate | FN rate | FP ratio | GV ratio |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| {row['scene']} | {row['frames']} | {fmt(row['dice_mean'])} | "
        f"{fmt(row['fp_rate_mean'])} | {fmt(row['fn_rate_mean'])} | "
        f"{fmt(row['fp_ratio_mean'])} | {fmt(row['gv_ratio_mean'])} |"
    )

lines.append("")
lines.append("## Predicted PVV Folders")
lines.append("")
for row in rows:
    lines.append(f"- **{row['scene']}**: `{row['predicted_pvv_folder']}`")

md_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

require_path "$ADAPTED_REPO/infer.py" "Adapted_fvdb infer.py"
require_path "$CANONICAL_ROOT" "CANONICAL_ROOT"
require_path "$OUT_DIR" "fVDB output directory"
require_path "$CONDA_SH" "conda activation script"

EXP_DIR="${EXP_DIR:-$(latest_exp_dir)}"
if [ -z "$EXP_DIR" ] || [ ! -d "$EXP_DIR" ]; then
  echo "ERROR: Could not auto-find finished fVDB experiment in $OUT_DIR" >&2
  echo "Pattern used: *synthetic_may19_${RADIUS}_fvdb_d${D}_b${BATCH}_depth${DEPTH}_${EPOCHS}ep*" >&2
  echo "Set EXP_DIR=/path/to/experiment and rerun." >&2
  exit 1
fi

EXP_NAME="$(basename "$EXP_DIR")"
require_path "$EXP_DIR/${EXP_NAME}_BEST.pth" "BEST checkpoint"

mapfile -t SCENE_RDIRS < <(discover_scene_radius_dirs)
if [ "${#SCENE_RDIRS[@]}" -eq 0 ]; then
  echo "ERROR: no canonical $RADIUS scene folders found under $CANONICAL_ROOT" >&2
  exit 1
fi

{
  echo "============================================================"
  echo "FVDB CANONICAL INFERENCE"
  echo "repo=$ADAPTED_REPO"
  echo "canonical_root=$CANONICAL_ROOT"
  echo "result_root=$RESULT_ROOT"
  echo "experiment=$EXP_NAME"
  echo "radius=$RADIUS"
  echo "d=$D"
  echo "scene_count=${#SCENE_RDIRS[@]}"
  echo "summary_csv=$SUMMARY_CSV"
  echo "summary_md=$SUMMARY_MD"
  echo "============================================================"
} | tee -a "$MASTER_LOG"

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$FVDB_ENV"
cd "$ADAPTED_REPO"

for rdir in "${SCENE_RDIRS[@]}"; do
  rel="${rdir#"$CANONICAL_ROOT"/}"
  scene_label="${rel%/"$RADIUS"}"
  scene_id="$(safe_name "$scene_label")"
  dataset="$(make_dataset_symlink "$scene_label" "$rdir" "$scene_id")"

  scene_tag="${INFER_TAG}_${scene_id}"
  infer_log="$LOG_DIR/infer_${scene_id}_${RADIUS}_fvdb_d${D}_${EPOCHS}ep_${STAMP}.log"

  {
    echo "============================================================"
    echo "START INFER"
    echo "scene=$scene_label"
    echo "dataset=$dataset"
    echo "tag=$scene_tag"
    echo "log=$infer_log"
    echo "============================================================"
  } | tee -a "$MASTER_LOG"

  args=(
    python infer.py
    --root "$DATA_ROOT"
    --out_dir "$OUT_DIR"
    --dataset_name "$dataset"
    --z_size "$Z_SIZE"
    --exp_name "$EXP_NAME"
    --infer_tag "$scene_tag"
  )

  if [ "$N_FRAMES" != "0" ]; then
    args+=(--n_frames "$N_FRAMES")
  fi

  if [ "$RUN_TIMING" = "1" ]; then
    args+=(--timing)
  fi

  "${args[@]}" 2>&1 | tee "$infer_log"

  latest_output="$(find "$OUT_DIR" -maxdepth 1 -type d -name "*${dataset}*${scene_tag}*" | sort | tail -1)"
  if [ -z "$latest_output" ]; then
    echo "ERROR: could not find inference output for $scene_label" >&2
    exit 1
  fi

  stats_file="$latest_output/eval_stats.csv"
  pvv_folder="$latest_output/inference/0"
  require_path "$stats_file" "eval_stats.csv for $scene_label"
  require_path "$pvv_folder" "predicted PVV folder for $scene_label"

  append_summary_row "$scene_label" "$scene_id" "$rdir" "$dataset" "$latest_output" "$pvv_folder" "$stats_file"

  {
    echo "Completed scene: $scene_label"
    echo "Output: $latest_output"
    echo "Predicted PVVs: $(find "$pvv_folder" -type f -name '*_predicted_pvv.bin.gz' | wc -l)"
  } | tee -a "$MASTER_LOG"
done

echo "============================================================"
echo "ALL CANONICAL INFERENCE DONE"
echo "============================================================"
write_markdown_summary | tee -a "$MASTER_LOG"

echo ""
echo "CSV summary: $SUMMARY_CSV"
echo "Markdown summary: $SUMMARY_MD"
echo "Master log: $MASTER_LOG"
