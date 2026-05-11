#!/usr/bin/env bash
set -uo pipefail

cd /home/atighedl/Adapted_fvdb

RUN_ID=$(date +"%Y%m%d-%H%M%S")
RAW_BASE="/home/atighedl/Adapted_fvdb/gv_ppv_generations_eachscene"
SUMMARY="results/sweeps/${RUN_ID}_bigcity_industrial_Sponza_fvdb_d_sweep_200ep_summary.csv"
MASTER_LOG="logs/${RUN_ID}_bigcity_industrial_Sponza_fvdb_d_sweep_200ep.log"

SCENES=("bigcity" "industrial" "Sponza")
DS=(8 16 32)

mkdir -p data_for_test/datasets logs results/sweeps results/paper_metrics

echo "scene,d,tag,exp_path,infer_path,predicted_count,eval_stats,fn_rate,fp_rate,dice" > "$SUMMARY"

echo "START FVDB THREE-SCENE d-SWEEP" | tee -a "$MASTER_LOG"
echo "RUN_ID=$RUN_ID" | tee -a "$MASTER_LOG"
echo "SUMMARY=$SUMMARY" | tee -a "$MASTER_LOG"
echo "TIME=$(date)" | tee -a "$MASTER_LOG"

for SCENE in "${SCENES[@]}"; do
  RAW="${RAW_BASE}/${SCENE}/r30"
  DATASET="${SCENE}_r30"

  echo "============================================================" | tee -a "$MASTER_LOG"
  echo "SCENE=$SCENE" | tee -a "$MASTER_LOG"
  echo "RAW=$RAW" | tee -a "$MASTER_LOG"
  echo "DATASET=$DATASET" | tee -a "$MASTER_LOG"

  if [ ! -d "$RAW/gv" ] || [ ! -d "$RAW/pvv" ]; then
    echo "SKIP $SCENE: missing $RAW/gv or $RAW/pvv" | tee -a "$MASTER_LOG"
    continue
  fi

  ln -sfn "$RAW" "data_for_test/datasets/$DATASET"

  for D in "${DS[@]}"; do
    TAG="${RUN_ID}_${SCENE}_fvdb_oacnn_interleaved_d${D}_depth3_200ep"
    TRAIN_LOG="logs/${TAG}.log"

    echo "START FVDB TRAIN: scene=$SCENE d=$D" | tee -a "$MASTER_LOG"
    echo "TAG=$TAG" | tee -a "$MASTER_LOG"
    echo "TIME=$(date)" | tee -a "$MASTER_LOG"

    python train.py \
      --root /home/atighedl/Adapted_fvdb/data_for_test \
      --dataset_name "$DATASET" \
      --z_size 256 \
      --test_fraction 0.1 \
      --model OACNNsInterleaved \
      --backend fvdb \
      --model_depth 3 \
      --interleaver_r "$D" \
      --batchSz 3 \
      --nEpochs 200 \
      --save_all_freq 50 \
      --lr 0.001 \
      --opt adam \
      --loss dice,no_guess \
      --loss_weights 0.99,0.01 \
      --dice_alpha 0.001 \
      --out_dir out \
      --tag "$TAG" \
      2>&1 | tee "$TRAIN_LOG"

    TRAIN_STATUS=${PIPESTATUS[0]}

    if [ "$TRAIN_STATUS" -ne 0 ]; then
      echo "FVDB TRAIN FAILED: scene=$SCENE d=$D exit=$TRAIN_STATUS" | tee -a "$MASTER_LOG"
      continue
    fi

    EXP_PATH=$(find data_for_test/out -type f -name 'training_arguments.json' \
      | grep "$TAG" \
      | sed 's#/training_arguments.json##' \
      | sort \
      | tail -n 1)

    if [ -z "$EXP_PATH" ]; then
      echo "FAILED: could not find EXP_PATH for TAG=$TAG" | tee -a "$MASTER_LOG"
      continue
    fi

    EXP_NAME=${EXP_PATH#data_for_test/out/}

    echo "START FVDB INFER: scene=$SCENE d=$D" | tee -a "$MASTER_LOG"

    python infer.py \
      --root data_for_test \
      --out_dir out \
      --exp_name "$EXP_NAME" \
      --dataset_name "$DATASET" \
      --z_size 256 \
      --infer_tag "${TAG}_infer" \
      2>&1 | tee "logs/${TAG}_infer.log"

    INFER_STATUS=${PIPESTATUS[0]}

    if [ "$INFER_STATUS" -ne 0 ]; then
      echo "FVDB INFER FAILED: scene=$SCENE d=$D exit=$INFER_STATUS" | tee -a "$MASTER_LOG"
      continue
    fi

    INFER_PATH=$(find data_for_test/out -type d -name "*${TAG}_infer*" | sort | tail -n 1)
    PRED_COUNT=$(find "$INFER_PATH/inference/0" -name '*_predicted_pvv.bin.gz' 2>/dev/null | wc -l)
    EVAL_STATS="$INFER_PATH/eval_stats.csv"

    python scripts/summarize_paper_metrics.py "$INFER_PATH" \
      2>&1 | tee "results/paper_metrics/${TAG}_paper_metrics.txt"

    python - "$SCENE" "$D" "$TAG" "$EXP_PATH" "$INFER_PATH" "$PRED_COUNT" "$EVAL_STATS" "$SUMMARY" <<'PY'
import csv
import sys

scene, d, tag, exp_path, infer_path, pred_count, eval_stats, summary = sys.argv[1:]

vals = {}
with open(eval_stats) as f:
    for row in csv.DictReader(f):
        vals[row["Metric"]] = float(row["Mean"])

with open(summary, "a") as f:
    f.write(
        f"{scene},{d},{tag},{exp_path},{infer_path},{pred_count},{eval_stats},"
        f"{vals.get('fn_rate', float('nan')):.8f},"
        f"{vals.get('fp_rate', float('nan')):.8f},"
        f"{vals.get('dice', float('nan')):.8f}\n"
    )
PY

    echo "FINISHED FVDB: scene=$SCENE d=$D" | tee -a "$MASTER_LOG"
    echo "TIME=$(date)" | tee -a "$MASTER_LOG"
  done
done

echo "FVDB THREE-SCENE d-SWEEP COMPLETE" | tee -a "$MASTER_LOG"
echo "SUMMARY=$SUMMARY" | tee -a "$MASTER_LOG"
echo "TIME=$(date)" | tee -a "$MASTER_LOG"
cat "$SUMMARY" | tee -a "$MASTER_LOG"
