#!/usr/bin/env bash
set -eo pipefail

# Compare normal inference metrics for the current fVDB repo against the
# dedicated fVDB optimization-lab repo on one checkpoint/scene pair.
#
# This verifies that the faster optimized inference path keeps dice/fp/fn style
# metrics effectively unchanged before using its timing in final comparisons.
#
# Typical use on Linux:
#   cd /home/atighedl/Adapted_fvdb
#   git pull
#   bash scripts/compare_fvdb_optimization_lab_metrics_robotlab_r30_d16.sh

RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_10EP_ALTERNATING}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/var/tmp/atighedl_runs/canonical_data/FINAL_GVPVV}"

BASE_REPO="${BASE_REPO:-/home/atighedl/Adapted_fvdb}"
OPT_REPO="${OPT_REPO:-}"

FVDB_ENV="${FVDB_ENV:-fvdb_rc}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

SCENE_KEY="${SCENE_KEY:-robotlab}"
RADIUS="${RADIUS:-r30}"
D="${D:-16}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
Z_SIZE="${Z_SIZE:-256}"
N_FRAMES="${N_FRAMES:-0}" # 0 means full scene.

OPT_INFER_EXTRA_ARGS="${OPT_INFER_EXTRA_ARGS:---fast_dense_deinterleave --fuse_linear_bn --fast_eval_fvdb_bn --fast_oa_scatter --fast_adaptive_mix --fast_cluster_ids --fast_native_scatter --fast_native_cluster_ids}"

DATA_ROOT="$RESULT_ROOT/data"
OUT_DIR="$RESULT_ROOT/fvdb_out"
SUMMARY_DIR="$RESULT_ROOT/summaries"
LOG_DIR="$RESULT_ROOT/logs"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="$SUMMARY_DIR/compare_fvdb_optimization_metrics_${SCENE_KEY}_${RADIUS}_d${D}_${EPOCHS}ep_${STAMP}.csv"
BASE_LOG="$LOG_DIR/metrics_compare_${SCENE_KEY}_${RADIUS}_d${D}_${EPOCHS}ep_base_${STAMP}.log"
OPT_LOG="$LOG_DIR/metrics_compare_${SCENE_KEY}_${RADIUS}_d${D}_${EPOCHS}ep_optimized_${STAMP}.log"

mkdir -p "$DATA_ROOT/datasets" "$SUMMARY_DIR" "$LOG_DIR"

require_path() {
  local path="$1"
  local label="$2"
  if [ ! -e "$path" ]; then
    echo "ERROR: $label not found: $path" >&2
    exit 1
  fi
}

find_opt_repo() {
  local candidate
  for candidate in \
    "$OPT_REPO" \
    "/var/tmp/${USER}_repos/Adapted_fvdb_optimization_lab" \
    "/home/atighedl/Adapted_fvdb_optimization_lab" \
    /home/atighedl/*optimization*lab* \
    /home/atighedl/*Adapted*fvdb*optimization*; do
    if [ -n "$candidate" ] && [ -f "$candidate/infer.py" ]; then
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
  local dataset="infer_${scene_id}_${RADIUS}_${EPOCHS}ep_metrics_compare"
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

run_inference() {
  local label="$1"
  local repo="$2"
  local log="$3"
  local exp_name="$4"
  local dataset="$5"
  local ckpt_suffix="$6"

  echo "============================================================"
  echo "START NORMAL INFERENCE: $label"
  echo "repo=$repo"
  echo "exp=$exp_name"
  echo "dataset=$dataset"
  echo "ckpt_suffix=${ckpt_suffix:-BEST}"
  if [ "$label" = "optimized" ]; then
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
    --infer_tag "${SCENE_KEY}_${RADIUS}_fvdb_d${D}_${EPOCHS}ep_${label}_metrics_compare"
  )

  if [ -n "$ckpt_suffix" ]; then
    args+=(--ckpt_suffix "$ckpt_suffix")
  fi

  if [ "$N_FRAMES" != "0" ]; then
    args+=(--n_frames "$N_FRAMES")
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
keys = ["dice", "loss", "fp", "fn", "fp_rate", "fn_rate", "fp_ratio", "gv_ratio"]

def stats_path_from_log(log_path):
    text = Path(log_path).read_text(errors="replace")
    matches = re.findall(r"Statistics written to (.+?eval_stats\.csv)", text)
    if not matches:
        raise SystemExit(f"Could not find eval_stats.csv path in {log_path}")
    path = Path(matches[-1].strip())
    if not path.exists():
        raise SystemExit(f"eval_stats.csv path from log does not exist: {path}")
    return path

def parse_eval_stats(path):
    rows = list(csv.reader(path.open(newline="")))
    if not rows:
        raise SystemExit(f"Empty eval_stats.csv: {path}")

    # Current repo format: Metric,Mean,Std,Min,Max
    if rows[0] and rows[0][0].strip().lower() == "metric":
        out = {}
        for row in rows[1:]:
            if len(row) >= 2:
                try:
                    out[row[0].strip()] = float(row[1])
                except ValueError:
                    pass
        return out

    # Fallback: one-row CSV with metric columns.
    header = [h.strip() for h in rows[0]]
    out = {}
    for row in rows[1:]:
        for h, v in zip(header, row):
            if h in keys:
                try:
                    out[h] = float(v)
                except ValueError:
                    pass
    return out

def pred_count(stats_path):
    pred_dir = stats_path.parent / "inference" / "0"
    return sum(1 for _ in pred_dir.glob("*_predicted_pvv.bin.gz")) if pred_dir.exists() else 0

base_stats = stats_path_from_log(base_log)
opt_stats = stats_path_from_log(opt_log)
base = parse_eval_stats(base_stats)
opt = parse_eval_stats(opt_stats)

with Path(summary_csv).open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "baseline", "optimized", "abs_diff", "rel_diff_pct"])
    for key in keys:
        b = base.get(key, math.nan)
        o = opt.get(key, math.nan)
        diff = o - b
        rel = (diff / b * 100.0) if b not in (0, math.nan) and not math.isnan(b) else math.nan
        writer.writerow([key, b, o, diff, rel])

print()
print("FVDB OPTIMIZATION NORMAL-INFERENCE METRIC CHECK")
print(f"baseline stats:  {base_stats}")
print(f"optimized stats: {opt_stats}")
print(f"baseline PVVs:   {pred_count(base_stats)}")
print(f"optimized PVVs:  {pred_count(opt_stats)}")
print()
print("metric    baseline      optimized     abs_diff      rel_diff_%")
print("--------  ------------  ------------  ------------  ----------")
for key in keys:
    b = base.get(key, math.nan)
    o = opt.get(key, math.nan)
    diff = o - b
    rel = (diff / b * 100.0) if b not in (0, math.nan) and not math.isnan(b) else math.nan
    print(f"{key:<8}  {b:>12.8f}  {o:>12.8f}  {diff:>12.8f}  {rel:>10.4f}")
print()
print(f"CSV written to: {summary_csv}")
PY
}

require_path "$CONDA_SH" "conda activation script"
require_path "$BASE_REPO/infer.py" "baseline infer.py"
OPT_REPO="$(find_opt_repo || true)"
if [ -z "$OPT_REPO" ]; then
  echo "ERROR: Adapted_fvdb_optimization_lab was not found." >&2
  echo "Run this once, then rerun:" >&2
  echo "  mkdir -p /var/tmp/\${USER}_repos" >&2
  echo "  cd /var/tmp/\${USER}_repos" >&2
  echo "  git clone https://github.com/atighetchidaniel-hub/Adapted_fvdb_optimization_lab.git" >&2
  echo "  cd /home/atighedl/Adapted_fvdb" >&2
  echo "  bash scripts/compare_fvdb_optimization_lab_metrics_robotlab_r30_d16.sh" >&2
  exit 1
fi
require_path "$OPT_REPO/infer.py" "optimization lab infer.py"
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
echo "Using optimization lab: $OPT_REPO"
echo

run_inference "baseline" "$BASE_REPO" "$BASE_LOG" "$EXP_NAME" "$DATASET" "$CKPT_SUFFIX"
run_inference "optimized" "$OPT_REPO" "$OPT_LOG" "$EXP_NAME" "$DATASET" "$CKPT_SUFFIX"
write_summary
