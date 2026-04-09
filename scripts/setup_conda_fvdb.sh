#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-neuralpvs_fvdb}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
PYG_VERSION="${PYG_VERSION:-2.6.1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH."
  echo "Install Miniconda or Anaconda first, then rerun this script."
  exit 1
fi

echo "Creating or reusing conda environment: ${ENV_NAME}"
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
else
  echo "Conda environment '${ENV_NAME}' already exists; reusing it."
fi

echo "Upgrading pip"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

echo "Installing PyTorch ${TORCH_VERSION} + CUDA 12.8"
conda run -n "${ENV_NAME}" python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url https://download.pytorch.org/whl/cu128

echo "Installing fvdb"
conda run -n "${ENV_NAME}" python -m pip install fvdb-core

echo "Installing PyTorch Geometric core packages"
conda run -n "${ENV_NAME}" python -m pip install "torch-geometric==${PYG_VERSION}"
conda run -n "${ENV_NAME}" python -m pip install \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

echo "Installing repo packages needed for validation"
conda run -n "${ENV_NAME}" python -m pip install \
  numpy pandas tqdm tensorboard torchinfo

echo
echo "Environment setup complete."
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"
