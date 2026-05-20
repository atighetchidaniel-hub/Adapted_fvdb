#!/usr/bin/env bash
set -eo pipefail

# Compare the current fVDB repo against the dedicated optimization-lab repo.
#
# Typical use on Linux:
#   cd /home/atighedl/Adapted_fvdb
#   git pull
#   bash scripts/compare_fvdb_optimization_lab_timing_robotlab_r30_d16.sh
#
# If the optimization lab is missing:
#   mkdir -p /var/tmp/${USER}_repos
#   cd /var/tmp/${USER}_repos
#   git clone https://github.com/atighetchidaniel-hub/Adapted_fvdb_optimization_lab.git
#   cd /home/atighedl/Adapted_fvdb
#   bash scripts/compare_fvdb_optimization_lab_timing_robotlab_r30_d16.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_opt_repo() {
  local candidate
  for candidate in \
    "${OPT_REPO:-}" \
    "/var/tmp/${USER}_repos/Adapted_fvdb_optimization_lab" \
    "/home/atighedl/Adapted_fvdb_optimization_lab" \
    "/home/atighedl/*optimization*lab*" \
    "/home/atighedl/*Adapted*fvdb*optimization*"; do
    if [ -n "$candidate" ] && [ -f "$candidate/infer.py" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

OPT_REPO="$(find_opt_repo || true)"
if [ -z "$OPT_REPO" ]; then
  echo "ERROR: Adapted_fvdb_optimization_lab was not found." >&2
  echo "Run this once, then rerun the script:" >&2
  echo "  mkdir -p /var/tmp/\${USER}_repos" >&2
  echo "  cd /var/tmp/\${USER}_repos" >&2
  echo "  git clone https://github.com/atighetchidaniel-hub/Adapted_fvdb_optimization_lab.git" >&2
  echo "  cd /home/atighedl/Adapted_fvdb" >&2
  echo "  bash scripts/compare_fvdb_optimization_lab_timing_robotlab_r30_d16.sh" >&2
  exit 1
fi

export OPT_REPO
export OPT_INFER_EXTRA_ARGS="${OPT_INFER_EXTRA_ARGS:---profile_stages --fast_dense_deinterleave --fuse_linear_bn --fast_eval_fvdb_bn --fast_oa_scatter --fast_adaptive_mix --fast_cluster_ids --fast_native_scatter --fast_native_cluster_ids}"

echo "Using optimization lab repo: $OPT_REPO"
echo "Using optimization flags: $OPT_INFER_EXTRA_ARGS"

bash "$SCRIPT_DIR/compare_fvdb_optimized_timing_robotlab_r30_d16.sh"
