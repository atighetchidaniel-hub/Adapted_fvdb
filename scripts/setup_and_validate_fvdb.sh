#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${1:-neuralpvs_fvdb}"
DATASET_ROOT="${2:-${REPO_ROOT}/data_for_test/viking/r30}"
Z_SIZE="${Z_SIZE:-256}"
SAVE_JSON="${SAVE_JSON:-results/fvdb_validation_summary.json}"

bash "${SCRIPT_DIR}/setup_conda_fvdb.sh" "${ENV_NAME}"

echo
echo "Running fvdb model validation in conda environment: ${ENV_NAME}"
conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/validate_fvdb_models.py" \
  --dataset-root "${DATASET_ROOT}" \
  --z-size "${Z_SIZE}" \
  --save-json "${SAVE_JSON}"
