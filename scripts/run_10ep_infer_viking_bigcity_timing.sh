#!/usr/bin/env bash
set -eo pipefail

# Run completed May19 10-epoch synthetic runs on RobotLab and BigCity.
#
# Defaults:
#   radii:    r30 r60
#   d values: 16 8 32
#   backends: fvdb spconv
#   scenes:   robotlab bigcity
#
# For each completed training run this performs:
#   1. normal inference, saving eval_stats.csv and predicted PVVs
#   2. --timing inference, saving timing logs
#
# Typical use:
#   cd /home/atighedl/Adapted_fvdb
#   git pull
#   bash scripts/run_10ep_infer_viking_bigcity_timing.sh

RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_10EP_ALTERNATING}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/var/tmp/atighedl_runs/canonical_data/FINAL_GVPVV}"

FVDB_REPO="${FVDB_REPO:-/home/atighedl/Adapted_fvdb}"
SPCONV_REPO="${SPCONV_REPO:-/home/atighedl/neuralpvs}"

FVDB_ENV="${FVDB_ENV:-fvdb_rc}"
SPCONV_ENV="${SPCONV_ENV:-cuda128}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

RADII="${RADII:-r30 r60}"
D_VALUES="${D_VALUES:-16 8 32}"
BACKENDS="${BACKENDS:-fvdb spconv}"
SCENE_KEYS="${SCENE_KEYS:-robotlab bigcity}"

EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
Z_SIZE="${Z_SIZE:-256}"
N_FRAMES="${N_FRAMES:-0}"              # 0 means full scene for normal inference.
TIMING_N_FRAMES="${TIMING_N_FRAMES:-0}" # 0 means full scene for timing density pass.

DATA_ROOT="$RESULT_ROOT/data"
FVDB_OUT="$RESULT_ROOT/fvdb_out"
SPCONV_OUT="$RESULT_ROOT/spconv_out"
SUMMARY_DIR="$RESULT_ROOT/summaries"
LOG_DIR="$RESULT_ROOT/logs"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="$SUMMARY_DIR/infer_robotlab_bigcity_r30_r60_${EPOCHS}ep_timing_${STAMP}.csv"
SUMMARY_MD="$SUMMARY_DIR/infer_robotlab_bigcity_r30_r60_${EPOCHS}ep_timing_${STAMP}.md"
MASTER_LOG="$LOG_DIR/infer_robotlab_bigcity_r30_r60_${EPOCHS}ep_timing_${STAMP}.log"

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
  local backend="$1"
  local radius="$2"
  local d="$3"
  local out_dir
  local pattern
  local dir

  if [ "$backend" = "fvdb" ]; then
    out_dir="$FVDB_OUT"
  else
    out_dir="$SPCONV_OUT"
  fi

  pattern="*synthetic_may19_${radius}_${backend}_d${d}_b${BATCH}_depth${DEPTH}_${EPOCHS}ep*"

  while IFS= read -r dir; do
    if find "$dir" -maxdepth 1 -type f -name '*.pth' | grep -q .; then
      echo "$dir"
      return 0
    fi
  done < <(find "$out_dir" -maxdepth 1 -type d -name "$pattern" | sort -r)

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
  local scene_key="$1"
  local radius="$2"
  local src="$3"
  local scene_id
  scene_id="$(safe_name "$scene_key")"
  local dataset="infer_${scene_id}_${radius}_${EPOCHS}ep"
  local dst="$DATA_ROOT/datasets/$dataset"

  rm -rf "$dst/gv" "$dst/pvv"
  mkdir -p "$dst"
  ln -s "$src/gv" "$dst/gv"
  ln -s "$src/pvv" "$dst/pvv"

  local gv_count
  local pvv_count
  gv_count="$(find -L "$dst/gv" -type f -name '*_gv.bin.gz' | wc -l)"
  pvv_count="$(find -L "$dst/pvv" -type f -name '*_pvv.bin.gz' | wc -l)"

  {
    echo "Dataset: $dataset"
    echo "Scene:   $scene_key"
    echo "Radius:  $radius"
    echo "Source:  $src"
    echo "GV:      $gv_count"
    echo "PVV:     $pvv_count"
  } | tee -a "$MASTER_LOG" >&2

  if [ "$gv_count" -eq 0 ] || [ "$pvv_count" -eq 0 ]; then
    echo "ERROR: $scene_key $radius has no GV/PVV files after symlink." >&2
    exit 1
  fi

  echo "$dataset"
}

run_one_infer() {
  local backend="$1"
  local radius="$2"
  local d="$3"
  local scene_key="$4"
  local dataset="$5"
  local exp_dir="$6"
  local timing="$7"

  local repo
  local env_name
  local out_dir
  if [ "$backend" = "fvdb" ]; then
    repo="$FVDB_REPO"
    env_name="$FVDB_ENV"
    out_dir="$FVDB_OUT"
  else
    repo="$SPCONV_REPO"
    env_name="$SPCONV_ENV"
    out_dir="$SPCONV_OUT"
  fi

  local exp_name
  local ckpt_suffix
  local scene_id
  local tag
  local log
  exp_name="$(basename "$exp_dir")"
  ckpt_suffix="$(checkpoint_suffix "$exp_dir" "$exp_name")"
  scene_id="$(safe_name "$scene_key")"

  if [ "$ckpt_suffix" = "ERROR_NO_CHECKPOINT" ]; then
    echo "SKIP: no checkpoint found for $backend $radius d$d in $exp_dir" | tee -a "$MASTER_LOG"
    echo ""
    return
  fi

  if [ "$timing" = "1" ]; then
    tag="${scene_id}_${radius}_${backend}_d${d}_${EPOCHS}ep_timing"
    log="$LOG_DIR/infer_${scene_id}_${radius}_${backend}_d${d}_${EPOCHS}ep_timing.log"
  else
    tag="${scene_id}_${radius}_${backend}_d${d}_${EPOCHS}ep_full"
    log="$LOG_DIR/infer_${scene_id}_${radius}_${backend}_d${d}_${EPOCHS}ep_full.log"
  fi

  {
    echo "============================================================"
    echo "START INFER"
    echo "backend=$backend"
    echo "radius=$radius"
    echo "d=$d"
    echo "scene=$scene_key"
    echo "dataset=$dataset"
    echo "timing=$timing"
    echo "exp=$exp_name"
    echo "ckpt_suffix=${ckpt_suffix:-BEST}"
    echo "log=$log"
    echo "============================================================"
  } | tee -a "$MASTER_LOG"

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

  if [ "$timing" = "1" ]; then
    args+=(--timing)
    if [ "$TIMING_N_FRAMES" != "0" ]; then
      args+=(--n_frames "$TIMING_N_FRAMES")
    fi
  elif [ "$N_FRAMES" != "0" ]; then
    args+=(--n_frames "$N_FRAMES")
  fi

  "${args[@]}" 2>&1 | tee "$log"

  if [ "$timing" = "1" ]; then
    echo "$log"
    return
  fi

  local latest_output
  latest_output="$(find "$out_dir" -maxdepth 1 -type d -name "*${dataset}*${tag}*" | sort | tail -1)"
  if [ -z "$latest_output" ]; then
    echo "ERROR: could not find inference output for $backend $radius d$d $scene_key" >&2
    exit 1
  fi

  echo "$latest_output"
}

append_summary_row() {
  local scene="$1"
  local radius="$2"
  local d="$3"
  local backend="$4"
  local output_dir="$5"
  local timing_log="$6"

  python - "$SUMMARY_CSV" "$scene" "$radius" "$d" "$backend" "$output_dir" "$timing_log" <<'PY'
import csv
import math
import re
import statistics as stats
import sys
from pathlib import Path

summary_csv, scene, radius, d, backend, output_dir, timing_log = sys.argv[1:]
output = Path(output_dir)
stats_path = output / "eval_stats.csv"
eval_log_path = output / "eval_log.csv"
pred_folder = output / "inference" / "0"
timing_path = Path(timing_log)

metrics = ["dice", "loss", "fp", "fn", "fp_rate", "fn_rate", "fp_ratio", "gv_ratio"]
timing_keys = [
    "infer_time_mean", "infer_time_std", "infer_time_min", "infer_time_max",
    "infer_time_pure_mean", "interleaver_time_mean", "deinterleaver_time_mean",
    "shapecriptor_time_mean", "start_mem", "end_mem", "peak_mem",
    "density_mean", "density_std", "density_min", "density_max",
]

stats_rows = list(csv.DictReader(stats_path.open(newline="")))
eval_rows = list(csv.DictReader(eval_log_path.open(newline=""))) if eval_log_path.exists() else []
predicted_count = len(list(pred_folder.glob("*_predicted_pvv.bin.gz"))) if pred_folder.exists() else 0
frame_count = len(eval_rows) or predicted_count

record = {
    "scene": scene,
    "radius": radius,
    "d": d,
    "backend": backend,
    "frames": frame_count,
    "output_dir": str(output),
    "predicted_pvv_folder": str(pred_folder),
    "timing_log": str(timing_path),
}

stats_by_metric = {}
if stats_rows and {"Metric", "Mean"}.issubset(stats_rows[0].keys()):
    for row in stats_rows:
        metric = row.get("Metric", "")
        if metric:
            stats_by_metric[metric] = row

def stat_value(row, name):
    try:
        return float(row[name])
    except Exception:
        return math.nan

def vals_for(key):
    vals = []
    for row in eval_rows:
        try:
            vals.append(float(row[key]))
        except Exception:
            pass
    return vals

for key in metrics:
    if key in stats_by_metric:
        row = stats_by_metric[key]
        record[f"{key}_mean"] = stat_value(row, "Mean")
        record[f"{key}_std"] = stat_value(row, "Std")
        record[f"{key}_min"] = stat_value(row, "Min")
        record[f"{key}_max"] = stat_value(row, "Max")
    else:
        vals = vals_for(key)
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

timing_text = timing_path.read_text(errors="replace") if timing_path.exists() else ""
for key in timing_keys:
    matches = re.findall(rf"(?:^|\s|\|){re.escape(key)}:\s*([0-9.eE+-]+)", timing_text)
    record[key] = float(matches[-1]) if matches else math.nan

fieldnames = [
    "scene", "radius", "d", "backend", "frames",
    "dice_mean", "loss_mean", "fp_mean", "fn_mean",
    "fp_rate_mean", "fn_rate_mean", "fp_ratio_mean", "gv_ratio_mean",
    "infer_time_mean", "infer_time_pure_mean", "peak_mem", "density_mean",
    "output_dir", "predicted_pvv_folder", "timing_log",
]

summary_path = Path(summary_csv)
write_header = not summary_path.exists()
with summary_path.open("a", newline="") as f:
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
rows = list(csv.DictReader(csv_path.open(newline=""))) if csv_path.exists() else []

def fmt(value, places=6):
    try:
        return f"{float(value):.{places}f}"
    except Exception:
        return str(value)

lines = []
lines.append("# 10ep RobotLab/BigCity Inference Timing Summary")
lines.append("")
lines.append("| Scene | Radius | d | Backend | Frames | Dice | FP rate | FN rate | GV ratio | Infer ms | Pure ms | Peak MB |")
lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| {row['scene']} | {row['radius']} | {row['d']} | {row['backend']} | {row['frames']} | "
        f"{fmt(row['dice_mean'])} | {fmt(row['fp_rate_mean'])} | {fmt(row['fn_rate_mean'])} | "
        f"{fmt(row['gv_ratio_mean'])} | {fmt(row['infer_time_mean'], 3)} | "
        f"{fmt(row['infer_time_pure_mean'], 3)} | {fmt(row['peak_mem'], 1)} |"
    )

lines.append("")
lines.append("## Predicted PVV Folders")
lines.append("")
for row in rows:
    lines.append(f"- **{row['scene']} {row['radius']} d{row['d']} {row['backend']}**: `{row['predicted_pvv_folder']}`")

text = "\n".join(lines) + "\n"
md_path.write_text(text)
print(text)
PY
}

require_path "$CONDA_SH" "conda activation script"
require_path "$CANONICAL_ROOT" "CANONICAL_ROOT"
require_path "$FVDB_REPO/infer.py" "fVDB infer.py"
require_path "$SPCONV_REPO/infer.py" "spconv infer.py"
require_path "$FVDB_OUT" "fVDB output root"
require_path "$SPCONV_OUT" "spconv output root"

# shellcheck source=/dev/null
source "$CONDA_SH"

{
  echo "============================================================"
  echo "10EP VIKING/BIGCITY INFERENCE + TIMING"
  echo "result_root=$RESULT_ROOT"
  echo "canonical_root=$CANONICAL_ROOT"
  echo "radii=$RADII"
  echo "d_values=$D_VALUES"
  echo "backends=$BACKENDS"
  echo "scenes=$SCENE_KEYS"
  echo "summary_csv=$SUMMARY_CSV"
  echo "summary_md=$SUMMARY_MD"
  echo "============================================================"
} | tee -a "$MASTER_LOG"

for radius in $RADII; do
  for scene_key in $SCENE_KEYS; do
    scene_src="$(scene_radius_dir "$scene_key" "$radius" || true)"
    if [ -z "$scene_src" ]; then
      echo "SKIP: could not find $scene_key $radius under $CANONICAL_ROOT" | tee -a "$MASTER_LOG"
      continue
    fi

    dataset="$(make_scene_dataset "$scene_key" "$radius" "$scene_src")"

    for d in $D_VALUES; do
      for backend in $BACKENDS; do
        exp_dir="$(latest_exp_dir "$backend" "$radius" "$d" || true)"
        if [ -z "$exp_dir" ]; then
          echo "SKIP: no completed $backend $radius d$d ${EPOCHS}ep checkpoint found" | tee -a "$MASTER_LOG"
          continue
        fi

        normal_output="$(run_one_infer "$backend" "$radius" "$d" "$scene_key" "$dataset" "$exp_dir" "0" | tail -1)"
        if [ -z "$normal_output" ] || [ ! -f "$normal_output/eval_stats.csv" ]; then
          echo "SKIP: normal inference did not produce eval_stats for $backend $radius d$d $scene_key" | tee -a "$MASTER_LOG"
          continue
        fi

        timing_log="$(run_one_infer "$backend" "$radius" "$d" "$scene_key" "$dataset" "$exp_dir" "1" | tail -1)"
        append_summary_row "$scene_key" "$radius" "$d" "$backend" "$normal_output" "$timing_log"

        {
          echo "Completed: scene=$scene_key radius=$radius d=$d backend=$backend"
          echo "Output: $normal_output"
          echo "Timing log: $timing_log"
          echo "Predicted PVVs: $(find "$normal_output/inference/0" -type f -name '*_predicted_pvv.bin.gz' 2>/dev/null | wc -l)"
        } | tee -a "$MASTER_LOG"
      done
    done
  done
done

echo "============================================================"
echo "INFERENCE RUNS DONE"
echo "============================================================"

if [ -f "$SUMMARY_CSV" ]; then
  write_markdown_summary | tee -a "$MASTER_LOG"
  echo "CSV summary:      $SUMMARY_CSV"
  echo "Markdown summary: $SUMMARY_MD"
else
  echo "No completed inference rows were written."
fi

echo "Master log:       $MASTER_LOG"
