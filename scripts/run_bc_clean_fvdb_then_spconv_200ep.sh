#!/usr/bin/env bash
set -uo pipefail

ADAPTED="/home/atighedl/Adapted_fvdb"
NPVS="/home/atighedl/neuralpvs"
DATA_SRC="$ADAPTED/GV_PVV_BigCity/bigcity/r30"
DATASET="bc_clean"
RUN_ID=$(date +"%Y%m%d-%H%M%S")

SUMMARY="$ADAPTED/results/sweeps/${RUN_ID}_bc_clean_fvdb_spconv_d16_200ep_summary.csv"
MASTER_LOG="$ADAPTED/logs/${RUN_ID}_bc_clean_fvdb_spconv_d16_200ep.log"

mkdir -p "$ADAPTED/data_for_test/datasets" "$ADAPTED/logs" "$ADAPTED/results/sweeps"
mkdir -p "$NPVS/data_for_test/datasets" "$NPVS/logs"

echo "backend,tag,exp_path,infer_path,predicted_count,eval_stats,fn_rate,fp_rate,dice" > "$SUMMARY"

echo "============================================================" | tee -a "$MASTER_LOG"
echo "START bc_clean fVDB then spconv 200ep" | tee -a "$MASTER_LOG"
echo "RUN_ID=$RUN_ID" | tee -a "$MASTER_LOG"
echo "DATA_SRC=$DATA_SRC" | tee -a "$MASTER_LOG"
echo "SUMMARY=$SUMMARY" | tee -a "$MASTER_LOG"
echo "TIME=$(date)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

if [ ! -d "$DATA_SRC/gv" ] || [ ! -d "$DATA_SRC/pvv" ]; then
  echo "ERROR: missing dataset gv/pvv under $DATA_SRC" | tee -a "$MASTER_LOG"
  exit 1
fi

ln -sfn "$DATA_SRC" "$ADAPTED/data_for_test/datasets/$DATASET"
ln -sfn "$DATA_SRC" "$NPVS/data_for_test/datasets/$DATASET"

echo "GV count:  $(find "$DATA_SRC/gv" -maxdepth 1 -name '*_gv.bin.gz' | wc -l)" | tee -a "$MASTER_LOG"
echo "PVV count: $(find "$DATA_SRC/pvv" -maxdepth 1 -name '*_pvv.bin.gz' | wc -l)" | tee -a "$MASTER_LOG"

append_summary() {
  local backend="$1"
  local tag="$2"
  local exp_path="$3"
  local infer_path="$4"
  local summary="$5"

  local pred_count
  local eval_stats
  pred_count=$(find "$infer_path/inference/0" -name '*_predicted_pvv.bin.gz' 2>/dev/null | wc -l)
  eval_stats="$infer_path/eval_stats.csv"

  python - "$backend" "$tag" "$exp_path" "$infer_path" "$pred_count" "$eval_stats" "$summary" <<'PY'
import csv
import sys

backend, tag, exp_path, infer_path, pred_count, eval_stats, summary = sys.argv[1:]

vals = {}
with open(eval_stats) as f:
    for row in csv.DictReader(f):
        vals[row["Metric"]] = float(row["Mean"])

with open(summary, "a") as f:
    f.write(
        f"{backend},{tag},{exp_path},{infer_path},{pred_count},{eval_stats},"
        f"{vals.get('fn_rate', float('nan')):.8f},"
        f"{vals.get('fp_rate', float('nan')):.8f},"
        f"{vals.get('dice', float('nan')):.8f}\n"
    )
PY
}

echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "START FVDB 200ep" | tee -a "$MASTER_LOG"
echo "TIME=$(date)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

cd "$ADAPTED"
source /home/atighedl/miniforge3/etc/profile.d/conda.sh
conda activate fvdb_rc

FVDB_TAG="${RUN_ID}_bc_clean_fvdb_d16_200ep"

python train.py \
  --root "$ADAPTED/data_for_test" \
  --dataset_name "$DATASET" \
  --z_size 256 \
  --test_fraction 0.05 \
  --model OACNNsInterleaved \
  --backend fvdb \
  --model_depth 3 \
  --interleaver_r 16 \
  --loss dice,no_guess \
  --loss_weights 0.99,0.01 \
  --dice_alpha 0.001 \
  --batchSz 2 \
  --nEpochs 200 \
  --save_all_freq 50 \
  --lr 0.001 \
  --opt adam \
  --out_dir out \
  --tag "$FVDB_TAG" \
  2>&1 | tee "$ADAPTED/logs/${FVDB_TAG}.log"

FVDB_STATUS=${PIPESTATUS[0]}

if [ "$FVDB_STATUS" -ne 0 ]; then
  echo "FVDB TRAIN FAILED exit=$FVDB_STATUS" | tee -a "$MASTER_LOG"
else
  FVDB_EXP_PATH=$(find "$ADAPTED/data_for_test/out" -type f -name 'training_arguments.json' \
    | grep "$FVDB_TAG" \
    | sed 's#/training_arguments.json##' \
    | sort \
    | tail -n 1)

  FVDB_EXP_NAME=${FVDB_EXP_PATH#"$ADAPTED/data_for_test/out/"}

  echo "FVDB_EXP_PATH=$FVDB_EXP_PATH" | tee -a "$MASTER_LOG"
  echo "START FVDB INFER" | tee -a "$MASTER_LOG"

  python infer.py \
    --root data_for_test \
    --out_dir out \
    --exp_name "$FVDB_EXP_NAME" \
    --dataset_name "$DATASET" \
    --z_size 256 \
    --infer_tag "${FVDB_TAG}_infer" \
    2>&1 | tee "$ADAPTED/logs/${FVDB_TAG}_infer.log"

  FVDB_INFER_STATUS=${PIPESTATUS[0]}

  if [ "$FVDB_INFER_STATUS" -ne 0 ]; then
    echo "FVDB INFER FAILED exit=$FVDB_INFER_STATUS" | tee -a "$MASTER_LOG"
  else
    FVDB_INFER_PATH=$(find "$ADAPTED/data_for_test/out" -type d -name "*${FVDB_TAG}_infer*" | sort | tail -n 1)
    echo "FVDB_INFER_PATH=$FVDB_INFER_PATH" | tee -a "$MASTER_LOG"
    append_summary "fvdb" "$FVDB_TAG" "$FVDB_EXP_PATH" "$FVDB_INFER_PATH" "$SUMMARY"
  fi
fi

echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "START SPCONV 200ep" | tee -a "$MASTER_LOG"
echo "TIME=$(date)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

cd "$NPVS"
conda activate cuda128

SPCONV_TAG="${RUN_ID}_bc_clean_spconv_d16_200ep"

python train.py \
  --root "$NPVS/data_for_test" \
  --dataset_name "$DATASET" \
  --z_size 256 \
  --test_fraction 0.05 \
  --model OACNNsInterleaved \
  --backend spconv \
  --model_depth 3 \
  --interleaver_r 16 \
  --loss dice,no_guess \
  --loss_weights 0.99,0.01 \
  --dice_alpha 0.001 \
  --batchSz 2 \
  --nEpochs 200 \
  --save_all_freq 50 \
  --lr 0.001 \
  --opt adam \
  --out_dir out_spconv \
  --tag "$SPCONV_TAG" \
  2>&1 | tee "$NPVS/logs/${SPCONV_TAG}.log"

SPCONV_STATUS=${PIPESTATUS[0]}

if [ "$SPCONV_STATUS" -ne 0 ]; then
  echo "SPCONV TRAIN FAILED exit=$SPCONV_STATUS" | tee -a "$MASTER_LOG"
else
  SPCONV_EXP_PATH=$(find "$NPVS/data_for_test/out_spconv" -type f -name 'training_arguments.json' \
    | grep "$SPCONV_TAG" \
    | sed 's#/training_arguments.json##' \
    | sort \
    | tail -n 1)

  SPCONV_EXP_NAME=${SPCONV_EXP_PATH#"$NPVS/data_for_test/out_spconv/"}

  echo "SPCONV_EXP_PATH=$SPCONV_EXP_PATH" | tee -a "$MASTER_LOG"
  echo "START SPCONV INFER" | tee -a "$MASTER_LOG"

  python infer.py \
    --root data_for_test \
    --out_dir out_spconv \
    --exp_name "$SPCONV_EXP_NAME" \
    --dataset_name "$DATASET" \
    --z_size 256 \
    --infer_tag "${SPCONV_TAG}_infer" \
    2>&1 | tee "$NPVS/logs/${SPCONV_TAG}_infer.log"

  SPCONV_INFER_STATUS=${PIPESTATUS[0]}

  if [ "$SPCONV_INFER_STATUS" -ne 0 ]; then
    echo "SPCONV INFER FAILED exit=$SPCONV_INFER_STATUS" | tee -a "$MASTER_LOG"
  else
    SPCONV_INFER_PATH=$(find "$NPVS/data_for_test/out_spconv" -type d -name "*${SPCONV_TAG}_infer*" | sort | tail -n 1)
    echo "SPCONV_INFER_PATH=$SPCONV_INFER_PATH" | tee -a "$MASTER_LOG"

    ADAPTED_SPCONV_DST="$ADAPTED/data_for_test/out_spconv/$(basename "$SPCONV_INFER_PATH")"
    mkdir -p "$ADAPTED/data_for_test/out_spconv"

    rsync -av \
      --include '*/' \
      --include '*_predicted_pvv.bin.gz' \
      --include 'eval_stats.csv' \
      --include 'eval_log.csv' \
      --include 'training_arguments.json' \
      --exclude '*' \
      "$SPCONV_INFER_PATH/" "$ADAPTED_SPCONV_DST/" \
      2>&1 | tee -a "$MASTER_LOG"

    append_summary "spconv" "$SPCONV_TAG" "$SPCONV_EXP_PATH" "$ADAPTED_SPCONV_DST" "$SUMMARY"
  fi
fi

echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "BC CLEAN FVDB + SPCONV 200ep COMPLETE" | tee -a "$MASTER_LOG"
echo "SUMMARY=$SUMMARY" | tee -a "$MASTER_LOG"
echo "TIME=$(date)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
cat "$SUMMARY" | tee -a "$MASTER_LOG"
