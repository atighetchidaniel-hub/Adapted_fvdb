#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/view_fvdb_visual_result.sh [sample_id]

Opens the existing Open3D GV/PVV/diff visualization for the fVDB Viking
visual inference result. The sample id defaults to 0.

Environment overrides:
  DATASET_PATH   Dataset root containing gv/ and pvv/.
  EXP_PATH       Experiment output folder containing inference/0/*.bin.gz.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SAMPLE_ID="${1:-0}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/data_for_test/viking/r30}"
DEFAULT_EXP_PATH="${REPO_ROOT}/data_for_test/out/VNet_viking_r30_dice_20260413-121838_viking_visual_vnet-visual_infer_M6Ll37wx"
EXP_PATH="${EXP_PATH:-${DEFAULT_EXP_PATH}}"

if [[ ! -d "${EXP_PATH}" ]]; then
  latest_exp="$(find "${REPO_ROOT}/data_for_test/out" -maxdepth 1 -type d -name 'VNet_viking_r30_dice_*visual_infer*' | sort | tail -n 1 || true)"
  if [[ -n "${latest_exp}" ]]; then
    EXP_PATH="${latest_exp}"
  fi
fi

GV_FILE="${DATASET_PATH}/gv/${SAMPLE_ID}_gv.bin.gz"
PVV_FILE="${DATASET_PATH}/pvv/${SAMPLE_ID}_pvv.bin.gz"
PRED_FILE="${EXP_PATH}/inference/0/${SAMPLE_ID}_predicted_pvv.bin.gz"

if [[ ! -f "${GV_FILE}" ]]; then
  echo "Missing GV file: ${GV_FILE}" >&2
  exit 1
fi
if [[ ! -f "${PVV_FILE}" ]]; then
  echo "Missing PVV file: ${PVV_FILE}" >&2
  exit 1
fi
if [[ ! -f "${PRED_FILE}" ]]; then
  echo "Missing predicted PVV file: ${PRED_FILE}" >&2
  echo "Run infer.py first or set EXP_PATH to the correct inference experiment." >&2
  exit 1
fi

echo "Dataset: ${DATASET_PATH}"
echo "Experiment: ${EXP_PATH}"
echo "Sample id: ${SAMPLE_ID}"
echo

echo "Opening Open3D visualization. Close the Open3D window to return to the terminal."
python "${SCRIPT_DIR}/vis_geometry.py"   --dataset_path "${DATASET_PATH}"   --exp_path "${EXP_PATH}"   --id "${SAMPLE_ID}"
