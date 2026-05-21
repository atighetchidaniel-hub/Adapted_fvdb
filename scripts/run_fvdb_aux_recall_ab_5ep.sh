#!/usr/bin/env bash
set -eo pipefail

# Quick fVDB-only A/B test for auxiliary recall supervision.
#
# It compares the same OACNNsInterleaved fVDB run with and without the
# auxiliary decoder recall loss. Defaults are intentionally small so this can
# be used as a directional smoke test before a 100-epoch run.

RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/FVDB_AUX_RECALL_AB_5EP}"
DATA_ROOT="${DATA_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_100EP/data}"
DSNAME="${DSNAME:-synthetic_may19_r30_1000_t3000}"

D="${D:-16}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-3}"
DEPTH="${DEPTH:-3}"
DEC_DEPTH="${DEC_DEPTH:-3}"
Z_SIZE="${Z_SIZE:-256}"
TEST_FRACTION="${TEST_FRACTION:-0.05}"
LR="${LR:-0.001}"

# Format: label:aux_recall_weight
AUX_SPECS="${AUX_SPECS:-aux0:0.0 aux005:0.05}"

mkdir -p "$RESULT_ROOT/fvdb_out" "$RESULT_ROOT/logs" "$RESULT_ROOT/summaries"

echo "Result root: $RESULT_ROOT"
echo "Dataset:     $DATA_ROOT/datasets/$DSNAME"
echo "D:           $D"
echo "Epochs:      $EPOCHS"
echo "Dec depth:   $DEC_DEPTH"
echo "A/B specs:   $AUX_SPECS"
echo

if [ ! -d "$DATA_ROOT/datasets/$DSNAME/gv" ] || [ ! -d "$DATA_ROOT/datasets/$DSNAME/pvv" ]; then
  echo "ERROR: dataset not found under $DATA_ROOT/datasets/$DSNAME" >&2
  exit 1
fi

for spec in $AUX_SPECS; do
  label="${spec%%:*}"
  aux_weight="${spec##*:}"
  tag="r30_fvdb_d${D}_decdepth${DEC_DEPTH}_${label}_${EPOCHS}ep"
  log="$RESULT_ROOT/logs/${tag}.log"

  echo "============================================================"
  echo "START $tag"
  echo "aux_recall_weight=$aux_weight"
  echo "============================================================"

  python train.py \
    --root "$DATA_ROOT" \
    --dataset_name "$DSNAME" \
    --z_size "$Z_SIZE" \
    --test_fraction "$TEST_FRACTION" \
    --model OACNNsInterleaved \
    --backend fvdb \
    --model_depth "$DEPTH" \
    --dec_depth "$DEC_DEPTH" \
    --aux_recall_weight "$aux_weight" \
    --aux_recall_alpha 0.001 \
    --interleaver_r "$D" \
    --batchSz "$BATCH" \
    --nEpochs "$EPOCHS" \
    --save_all_freq "$EPOCHS" \
    --lr "$LR" \
    --opt adam \
    --loss dice,no_guess \
    --loss_weights 0.99,0.01 \
    --dice_alpha 0.001 \
    --out_dir "$RESULT_ROOT/fvdb_out" \
    --tag "$tag" \
    2>&1 | tee "$log"

  echo "FINISHED $tag"
done

RESULT_ROOT="$RESULT_ROOT" python - <<'PY'
from pathlib import Path
import os
import re
import statistics as stats

result_root = Path(os.environ["RESULT_ROOT"])
keys = ["dice", "loss", "fp", "fn", "fp_rate", "fn_rate", "fp_ratio", "gv_ratio"]

def parse(line):
    row = {}
    for key in keys + ["epoch"]:
        m = re.search(rf"(?:^| )\|? ?{key}: ([0-9.eE+-]+)", line)
        if m:
            row[key] = float(m.group(1))
    return row

print()
print("=" * 80)
print("FVDB AUX RECALL A/B SUMMARY")
print("=" * 80)
print("variant,epoch,samples,dice_mean,loss_mean,fp_rate_mean,fn_rate_mean,fp_ratio_mean,gv_ratio_mean")

for log in sorted((result_root / "logs").glob("*.log")):
    rows = [parse(line) for line in log.read_text(errors="replace").splitlines() if "[eval]" in line]
    if not rows:
        continue
    epoch = int(max(row["epoch"] for row in rows if "epoch" in row))
    final = [row for row in rows if int(row.get("epoch", -1)) == epoch]
    def mean(key):
        vals = [row[key] for row in final if key in row]
        return stats.mean(vals) if vals else float("nan")
    print(
        f"{log.stem},"
        f"{epoch},"
        f"{len(final)},"
        f"{mean('dice'):.8f},"
        f"{mean('loss'):.8f},"
        f"{mean('fp_rate'):.8f},"
        f"{mean('fn_rate'):.8f},"
        f"{mean('fp_ratio'):.8f},"
        f"{mean('gv_ratio'):.8f}"
    )
PY
