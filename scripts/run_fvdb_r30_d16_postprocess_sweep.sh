#!/usr/bin/env bash
set -eo pipefail

# Small inference-only sweep for fVDB r30 d16 checkpoints.
#
# Purpose:
#   Reuse the existing 100-epoch fVDB checkpoint and test conservative
#   inference post-processing settings. This does not retrain the network.
#
# Typical Linux use:
#   cd /var/tmp/${USER}_repos/Adapted_fvdb_BACKENDOPTIMIZATION
#   bash scripts/run_fvdb_r30_d16_postprocess_sweep.sh
#
# Useful overrides:
#   SCENES="viking sponza" bash scripts/run_fvdb_r30_d16_postprocess_sweep.sh
#   N_FRAMES=50 bash scripts/run_fvdb_r30_d16_postprocess_sweep.sh
#   MAX_POOL_SIZES="-1 3 5 7 11" CACHE_SIZES="0 1 3" bash scripts/run_fvdb_r30_d16_postprocess_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_ROOT="${TRAIN_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_100EP}"
RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/fvdb_r30_d16_postprocess_sweep}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/var/tmp/${USER}_runs/canonical_data/FINAL_GVPVV}"

FVDB_REPO="${FVDB_REPO:-$REPO_ROOT}"
FVDB_ENV="${FVDB_ENV:-fvdb_rc}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

RADIUS="${RADIUS:-r30}"
D="${D:-16}"
BACKEND="fvdb"
SCENES="${SCENES:-viking bigcity industrial robotlab sponza}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
Z_SIZE="${Z_SIZE:-256}"
N_FRAMES="${N_FRAMES:-0}"
LOSS="${LOSS:-dice}"

# -1 disables dilation; positive values apply max-pool dilation.
# Cache combines recent predictions by voxel-wise OR. Use cache only for ordered
# camera paths; for shuffled datasets, keep CACHE_SIZES=0.
MAX_POOL_SIZES="${MAX_POOL_SIZES:--1 3 5 7 11}"
CACHE_SIZES="${CACHE_SIZES:-0}"

DATA_ROOT="$RESULT_ROOT/data"
OUT_ROOT="$RESULT_ROOT/fvdb_out"
LOG_DIR="$RESULT_ROOT/logs"
SUMMARY_DIR="$RESULT_ROOT/summaries"

mkdir -p "$DATA_ROOT/datasets" "$OUT_ROOT" "$LOG_DIR" "$SUMMARY_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="$SUMMARY_DIR/fvdb_${RADIUS}_d${D}_${EPOCHS}ep_postprocess_sweep_${STAMP}.csv"
SUMMARY_MD="$SUMMARY_DIR/fvdb_${RADIUS}_d${D}_${EPOCHS}ep_postprocess_sweep_${STAMP}.md"
MASTER_LOG="$LOG_DIR/fvdb_${RADIUS}_d${D}_${EPOCHS}ep_postprocess_sweep_${STAMP}.log"

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
print(text or "item")
PY
}

contains_word() {
  local needle="$1"
  local haystack="$2"
  local item
  for item in $haystack; do
    if [ "$item" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

scene_name_from_radius_dir() {
  basename "$(dirname "$1")" | tr '[:upper:]' '[:lower:]'
}

discover_scene_radius_dirs() {
  find "$CANONICAL_ROOT" -type d -name "$RADIUS" | sort | while read -r rdir; do
    if [ -d "$rdir/gv" ] && [ -d "$rdir/pvv" ]; then
      local scene
      scene="$(scene_name_from_radius_dir "$rdir")"
      if contains_word "$scene" "$SCENES"; then
        echo "$rdir"
      fi
    fi
  done
}

training_log_path() {
  echo "$TRAIN_ROOT/logs/synthetic_may19_${RADIUS}_${BACKEND}_d${D}_b${BATCH}_depth${DEPTH}_${EPOCHS}ep.log"
}

training_is_finished() {
  local log
  log="$(training_log_path)"
  [ -f "$log" ] && grep -q "Training finished" "$log"
}

latest_exp_dir() {
  local out_dir="$TRAIN_ROOT/fvdb_out"
  local pattern="*synthetic_may19_${RADIUS}_${BACKEND}_d${D}_b${BATCH}_depth${DEPTH}_${EPOCHS}ep*"
  local dir

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

link_experiment_into_out_root() {
  local exp_dir="$1"
  local exp_name
  local dst
  exp_name="$(basename "$exp_dir")"
  dst="$OUT_ROOT/$exp_name"

  if [ -e "$dst" ] && [ ! -L "$dst" ]; then
    echo "ERROR: $dst already exists and is not a symlink. Refusing to overwrite it." >&2
    exit 1
  fi
  ln -sfn "$exp_dir" "$dst"
}

make_scene_dataset() {
  local scene_src="$1"
  local scene_name="$2"
  local dataset="postprocess_${scene_name}_${RADIUS}"
  local dst="$DATA_ROOT/datasets/$dataset"

  rm -rf "$dst/gv" "$dst/pvv"
  mkdir -p "$dst"
  ln -s "$scene_src/gv" "$dst/gv"
  ln -s "$scene_src/pvv" "$dst/pvv"

  local gv_count
  local pvv_count
  gv_count="$(find -L "$dst/gv" -type f -name '*_gv.bin.gz' | wc -l)"
  pvv_count="$(find -L "$dst/pvv" -type f -name '*_pvv.bin.gz' | wc -l)"

  if [ "$gv_count" -eq 0 ] || [ "$pvv_count" -eq 0 ]; then
    echo "ERROR: $scene_name $RADIUS has no GV/PVV files after symlink." >&2
    exit 1
  fi

  echo "$dataset"
}

append_summary_row() {
  local scene_name="$1"
  local dataset="$2"
  local exp_dir="$3"
  local output_dir="$4"
  local log_path="$5"
  local cache_size="$6"
  local max_pool_size="$7"

  python - "$SUMMARY_CSV" "$scene_name" "$RADIUS" "$D" "$BACKEND" "$EPOCHS" "$dataset" "$exp_dir" "$output_dir" "$log_path" "$cache_size" "$max_pool_size" <<'PY'
import csv
import math
import sys
from pathlib import Path

(
    summary_csv,
    scene,
    radius,
    d,
    backend,
    epochs,
    dataset,
    exp_dir,
    output_dir,
    log_path,
    cache_size,
    max_pool_size,
) = sys.argv[1:]

metric_keys = ["dice", "loss", "fp", "fn", "tp", "tn", "fp_rate", "fn_rate", "fp_ratio", "gv_ratio"]

def load_eval_stats(output_dir):
    path = Path(output_dir) / "eval_stats.csv"
    result = {"frames": 0}
    if not path.exists():
        for key in metric_keys:
            result[f"{key}_mean"] = math.nan
        return result

    rows = list(csv.DictReader(path.open(newline="")))
    if rows and {"Metric", "Mean"}.issubset(rows[0].keys()):
        eval_log = Path(output_dir) / "eval_log.csv"
        if eval_log.exists():
            result["frames"] = sum(1 for _ in csv.DictReader(eval_log.open(newline="")))
        for row in rows:
            metric = row.get("Metric", "")
            if metric:
                try:
                    result[f"{metric}_mean"] = float(row.get("Mean", "nan"))
                except Exception:
                    result[f"{metric}_mean"] = math.nan
        for key in metric_keys:
            result.setdefault(f"{key}_mean", math.nan)
        return result

    result["frames"] = len(rows)
    for key in metric_keys:
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        result[f"{key}_mean"] = sum(vals) / len(vals) if vals else math.nan
    return result

record = {
    "scene": scene,
    "radius": radius,
    "d": int(d),
    "backend": backend,
    "epochs": int(epochs),
    "dataset": dataset,
    "cache_size": int(cache_size),
    "max_pool_size": int(max_pool_size),
    "train_exp_dir": exp_dir,
    "output_dir": output_dir,
    "predicted_pvv_folder": str(Path(output_dir) / "inference" / "0"),
    "log": log_path,
}
record.update(load_eval_stats(output_dir))

fieldnames = [
    "scene", "radius", "d", "backend", "epochs", "dataset",
    "cache_size", "max_pool_size", "frames",
    "dice_mean", "loss_mean", "fp_mean", "fn_mean", "tp_mean", "tn_mean",
    "fp_rate_mean", "fn_rate_mean", "fp_ratio_mean", "gv_ratio_mean",
    "train_exp_dir", "output_dir", "predicted_pvv_folder", "log",
]

out = Path(summary_csv)
write_header = not out.exists()
with out.open("a", newline="") as f:
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

def f(row, key, places=6):
    try:
        return f"{float(row[key]):.{places}f}"
    except Exception:
        return "nan"

lines = [
    "# fVDB r30 d16 Postprocess Sweep",
    "",
    "| Scene | cache | max pool | Frames | Dice | FP rate | FN rate | FP ratio | GV ratio |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    lines.append(
        f"| {row['scene']} | {row['cache_size']} | {row['max_pool_size']} | {row['frames']} | "
        f"{f(row, 'dice_mean')} | {f(row, 'fp_rate_mean')} | {f(row, 'fn_rate_mean')} | "
        f"{f(row, 'fp_ratio_mean')} | {f(row, 'gv_ratio_mean')} |"
    )

lines.append("")
lines.append("## Best Rows By False Negative Rate")
lines.append("")
best = sorted(rows, key=lambda r: float(r.get("fn_rate_mean", "inf")))[:10]
for row in best:
    lines.append(
        f"- {row['scene']} cache={row['cache_size']} max_pool={row['max_pool_size']}: "
        f"FN={f(row, 'fn_rate_mean')}, FP={f(row, 'fp_rate_mean')}, Dice={f(row, 'dice_mean')}"
    )

md_path.write_text("\n".join(lines) + "\n")

print()
print("POSTPROCESS SWEEP SUMMARY")
print("scene       cache mp   frames dice     fp_rate  fn_rate  fp_ratio gv_ratio")
print("---------- ----- ---- ------ -------- -------- -------- -------- --------")
for row in rows:
    print(
        f"{row['scene']:<10} {row['cache_size']:>5} {row['max_pool_size']:>4} "
        f"{row['frames']:>6} {f(row, 'dice_mean')} {f(row, 'fp_rate_mean')} "
        f"{f(row, 'fn_rate_mean')} {f(row, 'fp_ratio_mean')} {f(row, 'gv_ratio_mean')}"
    )

print()
print(f"CSV: {csv_path}")
print(f"MD:  {md_path}")
PY
}

run_infer() {
  local scene_name="$1"
  local dataset="$2"
  local exp_dir="$3"
  local cache_size="$4"
  local max_pool_size="$5"

  local exp_name
  local ckpt_suffix
  local tag
  local log
  local latest_output

  exp_name="$(basename "$exp_dir")"
  ckpt_suffix="$(checkpoint_suffix "$exp_dir" "$exp_name")"
  tag="${scene_name}_${RADIUS}_${BACKEND}_d${D}_${EPOCHS}ep_cache${cache_size}_mp${max_pool_size}_postprocess"
  log="$LOG_DIR/infer_${tag}.log"

  if [ "$ckpt_suffix" = "ERROR_NO_CHECKPOINT" ]; then
    echo "ERROR: no checkpoint found in $exp_dir" >&2
    exit 1
  fi

  {
    echo "============================================================"
    echo "START POSTPROCESS INFER"
    echo "scene=$scene_name"
    echo "dataset=$dataset"
    echo "radius=$RADIUS"
    echo "d=$D"
    echo "cache_size=$cache_size"
    echo "max_pool_size=$max_pool_size"
    echo "exp=$exp_name"
    echo "ckpt_suffix=${ckpt_suffix:-BEST}"
    echo "log=$log"
    echo "============================================================"
  } | tee -a "$MASTER_LOG"

  conda activate "$FVDB_ENV"
  cd "$FVDB_REPO"

  local args=(
    python infer.py
    --root "$DATA_ROOT"
    --out_dir "$OUT_ROOT"
    --dataset_name "$dataset"
    --z_size "$Z_SIZE"
    --exp_name "$exp_name"
    --infer_tag "$tag"
    --loss "$LOSS"
    --cache_size "$cache_size"
    --max_pool_size "$max_pool_size"
  )

  if [ -n "$ckpt_suffix" ]; then
    args+=(--ckpt_suffix "$ckpt_suffix")
  fi

  if [ "$N_FRAMES" != "0" ]; then
    args+=(--n_frames "$N_FRAMES")
  fi

  "${args[@]}" 2>&1 | tee "$log"

  latest_output="$(find "$OUT_ROOT" -maxdepth 1 -type d -name "*${dataset}*${tag}*" | sort | tail -1)"
  if [ -z "$latest_output" ]; then
    echo "ERROR: could not find output for $tag" >&2
    exit 1
  fi

  append_summary_row "$scene_name" "$dataset" "$exp_dir" "$latest_output" "$log" "$cache_size" "$max_pool_size"
}

require_path "$CONDA_SH" "conda activation script"
require_path "$TRAIN_ROOT" "TRAIN_ROOT"
require_path "$CANONICAL_ROOT" "CANONICAL_ROOT"
require_path "$FVDB_REPO/infer.py" "fVDB infer.py"

if ! training_is_finished; then
  echo "ERROR: training log is missing or does not contain 'Training finished': $(training_log_path)" >&2
  exit 1
fi

EXP_DIR="$(latest_exp_dir)"
require_path "$EXP_DIR" "fVDB training experiment"
link_experiment_into_out_root "$EXP_DIR"

{
  echo "============================================================"
  echo "FVDB R30 D16 POSTPROCESS SWEEP"
  echo "This reuses the existing checkpoint; no training is performed."
  echo "TRAIN_ROOT=$TRAIN_ROOT"
  echo "RESULT_ROOT=$RESULT_ROOT"
  echo "CANONICAL_ROOT=$CANONICAL_ROOT"
  echo "FVDB_REPO=$FVDB_REPO"
  echo "FVDB_ENV=$FVDB_ENV"
  echo "EXP_DIR=$EXP_DIR"
  echo "SCENES=$SCENES"
  echo "MAX_POOL_SIZES=$MAX_POOL_SIZES"
  echo "CACHE_SIZES=$CACHE_SIZES"
  echo "N_FRAMES=$N_FRAMES"
  echo "SUMMARY_CSV=$SUMMARY_CSV"
  echo "============================================================"
} | tee -a "$MASTER_LOG"

# shellcheck source=/dev/null
source "$CONDA_SH"

mapfile -t scene_dirs < <(discover_scene_radius_dirs)
if [ "${#scene_dirs[@]}" -eq 0 ]; then
  echo "ERROR: no scene directories found under $CANONICAL_ROOT for $RADIUS and SCENES='$SCENES'" >&2
  exit 1
fi

for scene_src in "${scene_dirs[@]}"; do
  scene_name="$(safe_name "$(scene_name_from_radius_dir "$scene_src")")"
  dataset="$(make_scene_dataset "$scene_src" "$scene_name")"
  for cache_size in $CACHE_SIZES; do
    for max_pool_size in $MAX_POOL_SIZES; do
      run_infer "$scene_name" "$dataset" "$EXP_DIR" "$cache_size" "$max_pool_size"
    done
  done
done

write_markdown_summary

{
  echo
  echo "Done."
  echo "summary_csv=$SUMMARY_CSV"
  echo "summary_md=$SUMMARY_MD"
  echo "master_log=$MASTER_LOG"
} | tee -a "$MASTER_LOG"
