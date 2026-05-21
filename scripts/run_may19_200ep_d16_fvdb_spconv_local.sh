#!/usr/bin/env bash
set -eo pipefail

# 200-epoch d=16 synthetic thesis benchmark.
#
# Runs exactly six trainings:
#   r30 d16 fvdb, r30 d16 spconv
#   r60 d16 fvdb, r60 d16 spconv
#   r90 d16 fvdb, r90 d16 spconv
#
# Typical use on Linux:
#   cd /var/tmp/${USER}_repos/Adapted_fvdb_BACKENDOPTIMIZATION
#   git pull origin main
#   bash scripts/run_may19_200ep_d16_fvdb_spconv_local.sh
#
# The script reuses the main synthetic runner and overrides only the run grid.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export SOURCE_ROOT="${SOURCE_ROOT:-/home/atighedl/May9_lunch_1000framescodexparamters/SceneGeneration}"
export RESULT_ROOT="${RESULT_ROOT:-/var/tmp/${USER}_runs/NEURALPVS_MAY19_SYNTHETIC_D16_200EP}"

export FVDB_REPO="${FVDB_REPO:-$REPO_ROOT}"
export SPCONV_REPO="${SPCONV_REPO:-/home/atighedl/neuralpvs}"

export RUN_SPECS="${RUN_SPECS:-r30:16 r60:16 r90:16}"
export EPOCHS="${EPOCHS:-200}"
export BATCH="${BATCH:-3}"
export DEPTH="${DEPTH:-3}"
export THRESHOLD="${THRESHOLD:-3000}"
export Z_SIZE="${Z_SIZE:-256}"
export TEST_FRACTION="${TEST_FRACTION:-0.05}"
export LR="${LR:-0.001}"

echo "Running d=16 200-epoch fVDB/spconv benchmark"
echo "Repo root:    $REPO_ROOT"
echo "Source root:  $SOURCE_ROOT"
echo "Result root:  $RESULT_ROOT"
echo "fVDB repo:    $FVDB_REPO"
echo "spconv repo:  $SPCONV_REPO"
echo "Run specs:    $RUN_SPECS"
echo

exec bash "$SCRIPT_DIR/run_may19_100ep_alternating_fvdb_spconv_local.sh"
