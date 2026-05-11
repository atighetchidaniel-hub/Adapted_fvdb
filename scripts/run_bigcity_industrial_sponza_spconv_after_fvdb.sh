#!/usr/bin/env bash
set -uo pipefail

ADAPTED="/home/atighedl/Adapted_fvdb"
NPVS="/home/atighedl/neuralpvs"
RAW_BASE="$ADAPTED/gv_ppv_generations_eachscene"

RUN_ID=$(date +"%Y%m%d-%H%M%S")
SUMMARY="$ADAPTED/results/sweeps/${RUN_ID}_bigcity_industrial_Sponza_spconv_d_sweep_200ep_summary.csv"
MASTER_LOG="$ADAPTED/logs/${RUN_ID}_bigcity_industrial_Sponza_spconv_d_sweep_200ep.log"

SCENES=("bigcity" "industrial" "Sponza")
DS=(8 16 32)

mkdir -p "$ADAPTED/logs" "$ADAPTED/results/sweeps"
mkdir -p "$NPVS/data_for_test/datasets" "$NPVS/logs"

echo "Waiting for current FvDB sweep to finish..." | tee -a "$MASTER_LOG"

while pgrep -f "run_bigcity_industrial_sponza_d_sweep_overnight.sh" >/dev/null; do
  echo "FvDB sweep still running at $(date). Checking again in 5 minutes..." | tee -a "$MASTER_LOG"
  sleep 300
done

echo "FvDB sweep finished. Starting spconv sweep." | tee -a "$MASTER_LOG"
echo "TIME=$(date)" | tee -a "$MASTER_LOG"

source /home/atighedl/miniforge3/etc/profile.d/conda.sh
conda activate cuda128

cd "$NPVS"

echo "scene,d,tag,exp_path,infer_path,predicted_count,eval_stats,fn_rate,fp_rate,dice" > "$SUMMARY"

for SCENE in "${SCENES[@]}"; do
  RAW="${RAW_BASE}/${SCENE}/r30"
  DATASET="${SCENE}_r30"

  echo "============================================================" | tee -a "$MASTER_LOG"
  echo "SCENE=$SCENE" | tee -a "$MASTER_LOG"
  echo "RAW=$RAW" | tee -a "$MASTER_LOG"

  if [ ! -d "$RAW/gv" ] || [ ! -d "$RAW/pvv" ]; then
    echo "SKIP $SCENE: missing $RAW/gv or $RAW/pvv" | tee -a "$MASTER_LOG"
    continue
  fi

  ln -sfn "$RAW" "$NPVS/data_for_test/datasets/$DATASET"

  for D in "${DS[@]}"; do
    TAG="${RUN_ID}_${SCENE}_spconv_oacnn_interleaved_d${D}_depth3_200ep"
    TRAIN_LOG="$NPVS/logs/${TAG}.log"

    echo "START SPCONV TRAIN: scene=$SCENE d=$D" | tee -a "$MASTER_LOG"
    echo "TAG=$TAG" | tee -a "$MASTER_LOG"
    echo "TIME=$(date)" | tee -a "$MASTER_LOG"

    python train.py \
      --root "$NPVS/data_for_test" \
      --dataset_name "$DATASET" \
      --z_size 256 \
      --test_fraction 0.1 \
      --model OACNNsInterleaved \
      --backend spconv \
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
      --out_dir out_spconv \
      --tag "$TAG" \
      2>&1 | tee "$TRAIN_LOG"

    TRAIN_STATUS=${PIPESTATUS[0]}

    if [ "$TRAIN_STATUS" -ne 0 ]; then
      echo "SPCONV TRAIN FAILED: scene=$SCENE d=$D exit=$TRAIN_STATUS" | tee -a "$MASTER_LOG"
      continue
    fi

    EXP_PATH=$(find "$NPVS/data_for_test/out_spconv" -type f -name 'training_arguments.json' \
      | grep "$TAG" \
      | sed 's#/training_arguments.json##' \
      | sort \
      | tail -n 1)

    if [ -z "$EXP_PATH" ]; then
      echo "FAILED: could not find EXP_PATH for TAG=$TAG" | tee -a "$MASTER_LOG"
      continue
    fi

    EXP_NAME=${EXP_PATH#"$NPVS/data_for_test/out_spconv/"}

    echo "START SPCONV INFER: scene=$SCENE d=$D" | tee -a "$MASTER_LOG"

    python infer.py \
      --root data_for_test \
      --out_dir out_spconv \
      --exp_name "$EXP_NAME" \
      --dataset_name "$DATASET" \
      --z_size 256 \
      --infer_tag "${TAG}_infer" \
      2>&1 | tee "$NPVS/logs/${TAG}_infer.log"

    INFER_STATUS=${PIPESTATUS[0]}

    if [ "$INFER_STATUS" -ne 0 ]; then
      echo "SPCONV INFER FAILED: scene=$SCENE d=$D exit=$INFER_STATUS" | tee -a "$MASTER_LOG"
      continue
    fi

    INFER_PATH=$(find "$NPVS/data_for_test/out_spconv" -type d -name "*${TAG}_infer*" | sort | tail -n 1)
    PRED_COUNT=$(find "$INFER_PATH/inference/0" -name '*_predicted_pvv.bin.gz' 2>/dev/null | wc -l)
    EVAL_STATS="$INFER_PATH/eval_stats.csv"

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

    echo "FINISHED SPCONV: scene=$SCENE d=$D" | tee -a "$MASTER_LOG"
    echo "TIME=$(date)" | tee -a "$MASTER_LOG"
  done
done

echo "SPCONV THREE-SCENE d-SWEEP COMPLETE" | tee -a "$MASTER_LOG"
echo "SUMMARY=$SUMMARY" | tee -a "$MASTER_LOG"
echo "TIME=$(date)" | tee -a "$MASTER_LOG"
cat "$SUMMARY" | tee -a "$MASTER_LOG"
