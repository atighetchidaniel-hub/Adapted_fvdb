#!/usr/bin/env bash
set -u

cd /home/atighedl/Adapted_fvdb || exit 1

ROOT="/home/atighedl/Adapted_fvdb/data_for_test"
DATASET_NAME="bigcity/bigcity/r30"
DATASET_REAL="$ROOT/$DATASET_NAME"
SWEEP_ID="$(date +%Y%m%d-%H%M%S)_bigcity_fvdb_oacnn_200ep_sweep"

mkdir -p logs/sweeps results/sweeps results/paper_metrics
LOG="logs/sweeps/${SWEEP_ID}.log"
SUMMARY="results/sweeps/${SWEEP_ID}_summary.csv"

exec > >(tee -a "$LOG") 2>&1

echo "SWEEP_ID=$SWEEP_ID"
echo "LOG=$LOG"
echo "SUMMARY=$SUMMARY"

mkdir -p "$ROOT/datasets/bigcity/bigcity"
ln -sfn "$DATASET_REAL" "$ROOT/datasets/$DATASET_NAME"

python - <<'PY'
from modules.dataset import PVSVoxelDataset
root = "/home/atighedl/Adapted_fvdb/data_for_test/bigcity/bigcity/r30"
ds = PVSVoxelDataset(root=root, mode="infer", z_size=256)
print("samples:", len(ds))
print("input:", ds[0]["input"].shape, ds[0]["input"].sum())
print("target:", ds[0]["target"].shape, ds[0]["target"].sum())
PY

python - "$SUMMARY" <<'PY'
import csv, sys
with open(sys.argv[1], "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "name", "d", "depth", "dice_alpha", "loss_weights", "lr",
        "weighted_dice", "fn_rate", "fp_rate", "fp_ratio", "gv_ratio",
        "paper_fnr", "paper_fpr", "pred_visible_over_gt", "eval_stats"
    ])
PY

append_summary () {
  local eval_stats="$1"
  local name="$2"
  local d="$3"
  local depth="$4"
  local alpha="$5"
  local weights="$6"
  local lr="$7"

  python - "$SUMMARY" "$eval_stats" "$name" "$d" "$depth" "$alpha" "$weights" "$lr" <<'PY'
import csv, sys
summary_csv, eval_stats, name, d, depth, alpha, weights, lr = sys.argv[1:]
vals = {}
with open(eval_stats, newline="") as f:
    for row in csv.DictReader(f):
        vals[row["Metric"]] = float(row["Mean"])

tp = vals["tp"]
fn = vals["fn"]
fp = vals["fp"]
gtp = tp + fn
paper_fnr = fn / gtp
paper_fpr = fp / gtp
pred_visible = (tp + fp) / gtp

with open(summary_csv, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        name, d, depth, alpha, weights, lr,
        vals.get("dice", ""),
        vals.get("fn_rate", ""),
        vals.get("fp_rate", ""),
        vals.get("fp_ratio", ""),
        vals.get("gv_ratio", ""),
        paper_fnr,
        paper_fpr,
        pred_visible,
        eval_stats,
    ])
PY
}

TESTS=(
  "d8_depth2_a0001|8|2|0.0001|0.99,0.01|0.001"
  "d16_depth2_a0001|16|2|0.0001|0.99,0.01|0.001"
  "d32_depth2_a0001|32|2|0.0001|0.99,0.01|0.001"

  "d8_depth3_a0001|8|3|0.0001|0.99,0.01|0.001"
  "d16_depth3_a0001|16|3|0.0001|0.99,0.01|0.001"
  "d32_depth3_a0001|32|3|0.0001|0.99,0.01|0.001"

  "d8_depth2_a001|8|2|0.001|0.99,0.01|0.001"
  "d16_depth2_a001|16|2|0.001|0.99,0.01|0.001"
  "d32_depth2_a001|32|2|0.001|0.99,0.01|0.001"
)

for TEST in "${TESTS[@]}"; do
  IFS='|' read -r NAME D DEPTH ALPHA WEIGHTS LR <<< "$TEST"
  TAG="${SWEEP_ID}_${NAME}"

  echo "============================================================"
  echo "START TRAIN $NAME"
  echo "d=$D depth=$DEPTH alpha=$ALPHA weights=$WEIGHTS lr=$LR"
  echo "============================================================"

  python train.py \
    --root "$ROOT" \
    --dataset_name "$DATASET_NAME" \
    --z_size 256 \
    --test_fraction 0.05 \
    --model OACNNsInterleaved \
    --backend fvdb \
    --model_depth "$DEPTH" \
    --interleaver_r "$D" \
    --loss dice,no_guess \
    --loss_weights "$WEIGHTS" \
    --dice_alpha "$ALPHA" \
    --batchSz 1 \
    --nEpochs 200 \
    --lr "$LR" \
    --opt adam \
    --save_all_freq 50 \
    --out_dir out \
    --tag "$TAG"

  TRAIN_STATUS=$?
  if [ "$TRAIN_STATUS" -ne 0 ]; then
    echo "TRAIN FAILED $NAME status=$TRAIN_STATUS"
    continue
  fi

  EXP_PATH=$(find "$ROOT/out" -type f -name 'training_arguments.json' \
    | grep "$TAG" \
    | grep -v "paperstyle_eval" \
    | sed 's#/training_arguments.json##' \
    | sort \
    | tail -n 1)

  if [ -z "$EXP_PATH" ]; then
    echo "Could not find EXP_PATH for $NAME"
    continue
  fi

  EXP_NAME=${EXP_PATH#"$ROOT/out/"}
  CKPT=$(find "$EXP_PATH" -maxdepth 1 -type f -name '*_BEST.pth' | head -n 1)

  if [ -z "$CKPT" ]; then
    echo "Could not find BEST checkpoint for $NAME"
    continue
  fi

  EXPECTED_CKPT="$ROOT/out/$EXP_NAME/${EXP_NAME}_BEST.pth"
  mkdir -p "$(dirname "$EXPECTED_CKPT")"
  ln -sfn "$(realpath "$CKPT")" "$EXPECTED_CKPT"

  INFER_TAG="${TAG}_paperstyle_eval"

  echo "------------------------------------------------------------"
  echo "START INFER $NAME"
  echo "------------------------------------------------------------"

  python infer.py \
    --root "$ROOT" \
    --out_dir out \
    --exp_name "$EXP_NAME" \
    --dataset_name "$DATASET_NAME" \
    --z_size 256 \
    --infer_tag "$INFER_TAG"

  INFER_STATUS=$?
  if [ "$INFER_STATUS" -ne 0 ]; then
    echo "INFER FAILED $NAME status=$INFER_STATUS"
    continue
  fi

  VIS_EXP=$(find "$ROOT/out" -type f -name 'eval_stats.csv' \
    | grep "$INFER_TAG" \
    | sed 's#/eval_stats.csv##' \
    | sort \
    | tail -n 1)

  if [ -z "$VIS_EXP" ]; then
    echo "Could not find VIS_EXP for $NAME"
    continue
  fi

  python scripts/summarize_paper_metrics.py "$VIS_EXP" \
    | tee "results/paper_metrics/${SWEEP_ID}_${NAME}_paper_metrics.txt"

  append_summary "$VIS_EXP/eval_stats.csv" "$NAME" "$D" "$DEPTH" "$ALPHA" "$WEIGHTS" "$LR"

  echo "Predicted PVV count:"
  find "$VIS_EXP/inference/0" -type f -name '*_predicted_pvv.bin.gz' | wc -l
done

echo "============================================================"
echo "SWEEP COMPLETE"
echo "SUMMARY=$SUMMARY"
echo "Best rows by fn_rate:"
python - "$SUMMARY" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
rows.sort(key=lambda r: float(r["fn_rate"]) if r["fn_rate"] else 999)
for r in rows[:20]:
    print(
        f'{r["fn_rate"]} fn_rate | {r["fp_rate"]} fp_rate | '
        f'{r["fp_ratio"]} fp_ratio | {r["gv_ratio"]} gv_ratio | '
        f'{r["name"]}'
    )
PY
