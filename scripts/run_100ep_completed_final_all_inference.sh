#!/usr/bin/env bash
set -eo pipefail

# Run final clean inference for all completed May19 synthetic 100-epoch models.
#
# What this does:
#   - uses the optimized fVDB repo for fVDB inference timing
#   - uses the spconv repo/env for spconv inference
#   - discovers only completed 100ep training runs
#   - skips missing/partial runs
#   - runs normal inference for metrics/predicted PVVs
#   - runs a second --timing inference pass for timing numbers
#   - saves everything under /var/tmp/$USER_runs/final_all_inference by default
#   - writes one combined CSV and Markdown summary
#
# Typical use on Linux:
#   cd /var/tmp/${USER}_repos/Adapted_fvdb_BACKENDOPTIMIZATION
#   git pull origin main
#   bash scripts/run_100ep_completed_final_all_inference.sh
#
# Useful overrides:
#   N_FRAMES=20 bash scripts/run_100ep_completed_final_all_inference.sh
#   SCENES="robotlab bigcity" bash scripts/run_100ep_completed_final_all_inference.sh
#   RADII="r30 r60" D_VALUES="8 16" bash scripts/run_100ep_completed_final_all_inference.sh
#   RUN_TIMING=0 bash scripts/run_100ep_completed_final_all_inference.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_ROOT="${TRAIN_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_100EP}"
RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/final_all_inference}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/var/tmp/${USER}_runs/canonical_data/FINAL_GVPVV}"

FVDB_REPO="${FVDB_REPO:-$REPO_ROOT}"
SPCONV_REPO="${SPCONV_REPO:-/home/atighedl/neuralpvs}"

FVDB_ENV="${FVDB_ENV:-fvdb_rc}"
SPCONV_ENV="${SPCONV_ENV:-cuda128}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

RADII="${RADII:-r30 r60 r90}"
D_VALUES="${D_VALUES:-8 16 32}"
BACKENDS="${BACKENDS:-fvdb spconv}"
SCENES="${SCENES:-}"          # Empty means every scene found for the radius.

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
Z_SIZE="${Z_SIZE:-256}"
N_FRAMES="${N_FRAMES:-0}"     # 0 means full scene.
RUN_METRICS="${RUN_METRICS:-1}"
RUN_TIMING="${RUN_TIMING:-1}"
LOSS="${LOSS:-dice}"
LOSS_WEIGHTS="${LOSS_WEIGHTS:-}"

DATA_ROOT="$RESULT_ROOT/data"
FVDB_OUT="$RESULT_ROOT/fvdb_out"
SPCONV_OUT="$RESULT_ROOT/spconv_out"
LOG_DIR="$RESULT_ROOT/logs"
SUMMARY_DIR="$RESULT_ROOT/summaries"

mkdir -p "$DATA_ROOT/datasets" "$FVDB_OUT" "$SPCONV_OUT" "$LOG_DIR" "$SUMMARY_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="$SUMMARY_DIR/final_all_inference_${EPOCHS}ep_${STAMP}.csv"
SUMMARY_MD="$SUMMARY_DIR/final_all_inference_${EPOCHS}ep_${STAMP}.md"
CONFIG_TXT="$SUMMARY_DIR/final_all_inference_${EPOCHS}ep_${STAMP}_config.txt"
MASTER_LOG="$LOG_DIR/final_all_inference_${EPOCHS}ep_${STAMP}.log"

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
  local radius="$1"
  find "$CANONICAL_ROOT" -type d -name "$radius" | sort | while read -r rdir; do
    if [ -d "$rdir/gv" ] && [ -d "$rdir/pvv" ]; then
      local scene
      scene="$(scene_name_from_radius_dir "$rdir")"
      if [ -z "$SCENES" ] || contains_word "$scene" "$SCENES"; then
        echo "$rdir"
      fi
    fi
  done
}

training_log_path() {
  local radius="$1"
  local d="$2"
  local backend="$3"
  echo "$TRAIN_ROOT/logs/synthetic_may19_${radius}_${backend}_d${d}_b${BATCH}_depth${DEPTH}_${EPOCHS}ep.log"
}

backend_train_out() {
  local backend="$1"
  if [ "$backend" = "fvdb" ]; then
    echo "$TRAIN_ROOT/fvdb_out"
  else
    echo "$TRAIN_ROOT/spconv_out"
  fi
}

backend_clean_out() {
  local backend="$1"
  if [ "$backend" = "fvdb" ]; then
    echo "$FVDB_OUT"
  else
    echo "$SPCONV_OUT"
  fi
}

backend_repo() {
  local backend="$1"
  if [ "$backend" = "fvdb" ]; then
    echo "$FVDB_REPO"
  else
    echo "$SPCONV_REPO"
  fi
}

backend_env() {
  local backend="$1"
  if [ "$backend" = "fvdb" ]; then
    echo "$FVDB_ENV"
  else
    echo "$SPCONV_ENV"
  fi
}

training_is_finished() {
  local radius="$1"
  local d="$2"
  local backend="$3"
  local log
  log="$(training_log_path "$radius" "$d" "$backend")"
  [ -f "$log" ] && grep -q "Training finished" "$log"
}

latest_exp_dir() {
  local radius="$1"
  local d="$2"
  local backend="$3"
  local out_dir
  local pattern
  local dir
  out_dir="$(backend_train_out "$backend")"
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

link_experiment_into_clean_out() {
  local backend="$1"
  local exp_dir="$2"
  local clean_out
  local exp_name
  local dst
  clean_out="$(backend_clean_out "$backend")"
  exp_name="$(basename "$exp_dir")"
  dst="$clean_out/$exp_name"

  mkdir -p "$clean_out"
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then
    echo "ERROR: $dst already exists and is not a symlink. Refusing to overwrite it." >&2
    exit 1
  fi
  ln -sfn "$exp_dir" "$dst"
}

make_scene_dataset() {
  local radius="$1"
  local scene_src="$2"
  local scene_name="$3"
  local dataset="final_infer_${scene_name}_${radius}"
  local dst="$DATA_ROOT/datasets/$dataset"

  rm -rf "$dst/gv" "$dst/pvv"
  mkdir -p "$dst"
  ln -s "$scene_src/gv" "$dst/gv"
  ln -s "$scene_src/pvv" "$dst/pvv"

  local gv_count
  local pvv_count
  gv_count="$(find -L "$dst/gv" -type f -name '*_gv.bin.gz' | wc -l)"
  pvv_count="$(find -L "$dst/pvv" -type f -name '*_pvv.bin.gz' | wc -l)"

  {
    echo "Dataset: $dataset"
    echo "Scene:   $scene_name"
    echo "Radius:  $radius"
    echo "Source:  $scene_src"
    echo "GV:      $gv_count"
    echo "PVV:     $pvv_count"
  } | tee -a "$MASTER_LOG" >&2

  if [ "$gv_count" -eq 0 ] || [ "$pvv_count" -eq 0 ]; then
    echo "ERROR: $scene_name $radius has no GV/PVV files after symlink." >&2
    exit 1
  fi

  echo "$dataset"
}

RUN_INFER_OUTPUT=""

run_infer() {
  local backend="$1"
  local radius="$2"
  local d="$3"
  local scene_name="$4"
  local dataset="$5"
  local exp_dir="$6"
  local mode="$7"

  local repo
  local env_name
  local out_dir
  local exp_name
  local ckpt_suffix
  local tag
  local log

  repo="$(backend_repo "$backend")"
  env_name="$(backend_env "$backend")"
  out_dir="$(backend_clean_out "$backend")"
  exp_name="$(basename "$exp_dir")"
  ckpt_suffix="$(checkpoint_suffix "$exp_dir" "$exp_name")"
  tag="final_${scene_name}_${radius}_${backend}_d${d}_${EPOCHS}ep_${mode}"
  log="$LOG_DIR/infer_${scene_name}_${radius}_${backend}_d${d}_${EPOCHS}ep_${mode}.log"

  if [ "$ckpt_suffix" = "ERROR_NO_CHECKPOINT" ]; then
    echo "ERROR: no checkpoint found for $backend in $exp_dir" >&2
    exit 1
  fi

  require_path "$repo/infer.py" "$backend infer.py"

  {
    echo "============================================================"
    echo "START INFER"
    echo "mode=$mode"
    echo "backend=$backend"
    echo "repo=$repo"
    echo "env=$env_name"
    echo "scene=$scene_name"
    echo "radius=$radius"
    echo "d=$d"
    echo "epochs=$EPOCHS"
    echo "dataset=$dataset"
    echo "exp=$exp_name"
    echo "ckpt_suffix=${ckpt_suffix:-BEST}"
    echo "out_dir=$out_dir"
    echo "log=$log"
    echo "============================================================"
  } | tee -a "$MASTER_LOG"

  conda activate "$env_name"
  cd "$repo"

  local args=(
    python infer.py
    --root "$DATA_ROOT"
    --out_dir "$out_dir"
    --dataset_name "$dataset"
    --z_size "$Z_SIZE"
    --exp_name "$exp_name"
    --infer_tag "$tag"
    --loss "$LOSS"
  )

  if [ -n "$LOSS_WEIGHTS" ]; then
    args+=(--loss_weights "$LOSS_WEIGHTS")
  fi

  if [ -n "$ckpt_suffix" ]; then
    args+=(--ckpt_suffix "$ckpt_suffix")
  fi

  if [ "$N_FRAMES" != "0" ]; then
    args+=(--n_frames "$N_FRAMES")
  fi

  if [ "$mode" = "timing" ]; then
    args+=(--timing)
  fi

  "${args[@]}" 2>&1 | tee "$log"

  local latest_output
  latest_output="$(find "$out_dir" -maxdepth 1 -type d -name "*${dataset}*${tag}*" | sort | tail -1)"
  if [ -z "$latest_output" ]; then
    echo "ERROR: could not find inference output for $backend $scene_name $radius d$d mode=$mode" >&2
    exit 1
  fi

  RUN_INFER_OUTPUT="$latest_output"
}

append_summary_row() {
  local scene_name="$1"
  local radius="$2"
  local d="$3"
  local backend="$4"
  local exp_dir="$5"
  local metrics_output="$6"
  local timing_output="$7"
  local metrics_log="$8"
  local timing_log="$9"

  python - "$SUMMARY_CSV" "$scene_name" "$radius" "$d" "$backend" "$EPOCHS" "$exp_dir" "$metrics_output" "$timing_output" "$metrics_log" "$timing_log" <<'PY'
import csv
import math
import re
import statistics as stats
import sys
from pathlib import Path

(
    summary_csv,
    scene,
    radius,
    d,
    backend,
    epochs,
    exp_dir,
    metrics_output,
    timing_output,
    metrics_log,
    timing_log,
) = sys.argv[1:]

metric_keys = ["dice", "loss", "fp", "fn", "fp_rate", "fn_rate", "fp_ratio", "gv_ratio"]
timing_keys = [
    "infer_time_mean",
    "infer_time_std",
    "infer_time_min",
    "infer_time_max",
    "infer_time_pure_mean",
    "interleaver_time_mean",
    "sparse_core_time_mean",
    "deinterleaver_time_mean",
    "peak_mem",
    "density_mean",
]

def load_eval_stats(output_dir):
    path = Path(output_dir) / "eval_stats.csv"
    result = {"frames": 0}
    if not path.exists():
        for key in metric_keys:
            result[f"{key}_mean"] = math.nan
            result[f"{key}_std"] = math.nan
        return result

    rows = list(csv.DictReader(path.open(newline="")))
    result["frames"] = len(rows)
    for key in metric_keys:
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        result[f"{key}_mean"] = stats.mean(vals) if vals else math.nan
        result[f"{key}_std"] = stats.pstdev(vals) if len(vals) > 1 else 0.0
    return result

def load_timing(log_path):
    result = {key: math.nan for key in timing_keys}
    path = Path(log_path)
    if not path.exists():
        return result
    text = path.read_text(errors="replace")
    for key in timing_keys:
        matches = re.findall(rf"(?:^|[| ]){re.escape(key)}:\s*([0-9.eE+-]+)", text)
        if matches:
            try:
                result[key] = float(matches[-1])
            except Exception:
                pass
    return result

record = {
    "scene": scene,
    "radius": radius,
    "d": int(d),
    "backend": backend,
    "epochs": int(epochs),
    "train_exp_dir": exp_dir,
    "metrics_output_dir": metrics_output,
    "timing_output_dir": timing_output,
    "predicted_pvv_folder": str(Path(metrics_output) / "inference" / "0") if metrics_output else "",
    "metrics_log": metrics_log,
    "timing_log": timing_log,
}
record.update(load_eval_stats(metrics_output))
record.update(load_timing(timing_log))

fieldnames = [
    "scene", "radius", "d", "backend", "epochs", "frames",
    "dice_mean", "loss_mean", "fp_mean", "fn_mean",
    "fp_rate_mean", "fn_rate_mean", "fp_ratio_mean", "gv_ratio_mean",
    "infer_time_mean", "infer_time_pure_mean",
    "interleaver_time_mean", "sparse_core_time_mean", "deinterleaver_time_mean",
    "peak_mem", "density_mean",
    "train_exp_dir", "metrics_output_dir", "timing_output_dir",
    "predicted_pvv_folder", "metrics_log", "timing_log",
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

def fmt(row, key, places=6):
    try:
        return f"{float(row[key]):.{places}f}"
    except Exception:
        return "nan"

lines = []
lines.append("# Final All-Scene Inference Summary")
lines.append("")
lines.append("| Scene | Radius | d | Backend | Frames | Dice | FP rate | FN rate | FP ratio | GV ratio | Infer ms | Pure ms | Peak MB |")
lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| {row['scene']} | {row['radius']} | {row['d']} | {row['backend']} | {row['frames']} | "
        f"{fmt(row, 'dice_mean')} | {fmt(row, 'fp_rate_mean')} | {fmt(row, 'fn_rate_mean')} | "
        f"{fmt(row, 'fp_ratio_mean')} | {fmt(row, 'gv_ratio_mean')} | "
        f"{fmt(row, 'infer_time_mean', 3)} | {fmt(row, 'infer_time_pure_mean', 3)} | {fmt(row, 'peak_mem', 1)} |"
    )

lines.append("")
lines.append("## Predicted PVV Folders")
lines.append("")
for row in rows:
    lines.append(f"- **{row['scene']} {row['radius']} d{row['d']} {row['backend']}**: `{row['predicted_pvv_folder']}`")

md_path.write_text("\n".join(lines) + "\n")

print()
print("FINAL ALL-SCENE INFERENCE TABLE")
print("scene       radius d   backend frames dice     fp_rate  fn_rate  fp_ratio gv_ratio infer_ms pure_ms")
print("---------- ------ --- ------- ------ -------- -------- -------- -------- -------- -------- --------")
for row in rows:
    print(
        f"{row['scene']:<10} {row['radius']:<6} {row['d']:>3} {row['backend']:<7} "
        f"{row['frames']:>6} {fmt(row, 'dice_mean')} {fmt(row, 'fp_rate_mean')} "
        f"{fmt(row, 'fn_rate_mean')} {fmt(row, 'fp_ratio_mean')} {fmt(row, 'gv_ratio_mean')} "
        f"{fmt(row, 'infer_time_mean', 3)} {fmt(row, 'infer_time_pure_mean', 3)}"
    )

print()
print(f"CSV: {csv_path}")
print(f"MD:  {md_path}")
PY
}

write_config() {
  {
    echo "TRAIN_ROOT=$TRAIN_ROOT"
    echo "RESULT_ROOT=$RESULT_ROOT"
    echo "CANONICAL_ROOT=$CANONICAL_ROOT"
    echo "FVDB_REPO=$FVDB_REPO"
    echo "SPCONV_REPO=$SPCONV_REPO"
    echo "FVDB_ENV=$FVDB_ENV"
    echo "SPCONV_ENV=$SPCONV_ENV"
    echo "RADII=$RADII"
    echo "D_VALUES=$D_VALUES"
    echo "BACKENDS=$BACKENDS"
    echo "SCENES=$SCENES"
    echo "EPOCHS=$EPOCHS"
    echo "BATCH=$BATCH"
    echo "DEPTH=$DEPTH"
    echo "Z_SIZE=$Z_SIZE"
    echo "N_FRAMES=$N_FRAMES"
    echo "RUN_METRICS=$RUN_METRICS"
    echo "RUN_TIMING=$RUN_TIMING"
    echo "LOSS=$LOSS"
    echo "LOSS_WEIGHTS=$LOSS_WEIGHTS"
    echo "SUMMARY_CSV=$SUMMARY_CSV"
    echo "SUMMARY_MD=$SUMMARY_MD"
  } | tee "$CONFIG_TXT"
}

require_path "$CONDA_SH" "conda activation script"
require_path "$TRAIN_ROOT" "TRAIN_ROOT"
require_path "$CANONICAL_ROOT" "CANONICAL_ROOT"
require_path "$FVDB_REPO/infer.py" "optimized fVDB infer.py"
require_path "$SPCONV_REPO/infer.py" "spconv infer.py"

write_config

{
  echo "============================================================"
  echo "FINAL ALL-SCENE INFERENCE"
  echo "Only completed training logs with checkpoints are run."
  echo "============================================================"
} | tee -a "$MASTER_LOG"

# shellcheck source=/dev/null
source "$CONDA_SH"

completed_count=0
skipped_count=0
inference_count=0

for radius in $RADII; do
  mapfile -t scene_dirs < <(discover_scene_radius_dirs "$radius")
  if [ "${#scene_dirs[@]}" -eq 0 ]; then
    echo "SKIP radius=$radius: no canonical scenes found" | tee -a "$MASTER_LOG"
    continue
  fi

  for d in $D_VALUES; do
    for backend in $BACKENDS; do
      if ! training_is_finished "$radius" "$d" "$backend"; then
        echo "SKIP training incomplete/missing: $radius d=$d $backend" | tee -a "$MASTER_LOG"
        skipped_count=$((skipped_count + 1))
        continue
      fi

      exp_dir="$(latest_exp_dir "$radius" "$d" "$backend" || true)"
      if [ -z "$exp_dir" ] || [ ! -d "$exp_dir" ]; then
        echo "SKIP no checkpoint folder: $radius d=$d $backend" | tee -a "$MASTER_LOG"
        skipped_count=$((skipped_count + 1))
        continue
      fi

      link_experiment_into_clean_out "$backend" "$exp_dir"
      completed_count=$((completed_count + 1))

      for scene_src in "${scene_dirs[@]}"; do
        scene_name="$(safe_name "$(scene_name_from_radius_dir "$scene_src")")"
        dataset="$(make_scene_dataset "$radius" "$scene_src" "$scene_name")"

        metrics_output=""
        timing_output=""
        metrics_log="$LOG_DIR/infer_${scene_name}_${radius}_${backend}_d${d}_${EPOCHS}ep_metrics.log"
        timing_log="$LOG_DIR/infer_${scene_name}_${radius}_${backend}_d${d}_${EPOCHS}ep_timing.log"

        if [ "$RUN_METRICS" = "1" ]; then
          run_infer "$backend" "$radius" "$d" "$scene_name" "$dataset" "$exp_dir" "metrics"
          metrics_output="$RUN_INFER_OUTPUT"
        fi

        if [ "$RUN_TIMING" = "1" ]; then
          run_infer "$backend" "$radius" "$d" "$scene_name" "$dataset" "$exp_dir" "timing"
          timing_output="$RUN_INFER_OUTPUT"
        fi

        append_summary_row "$scene_name" "$radius" "$d" "$backend" "$exp_dir" "$metrics_output" "$timing_output" "$metrics_log" "$timing_log"
        inference_count=$((inference_count + 1))
      done
    done
  done
done

write_markdown_summary

{
  echo
  echo "Done."
  echo "completed_model_configs=$completed_count"
  echo "skipped_model_configs=$skipped_count"
  echo "inference_rows=$inference_count"
  echo "summary_csv=$SUMMARY_CSV"
  echo "summary_md=$SUMMARY_MD"
  echo "master_log=$MASTER_LOG"
} | tee -a "$MASTER_LOG"
