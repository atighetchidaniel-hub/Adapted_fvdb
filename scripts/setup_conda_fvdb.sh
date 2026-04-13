#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-neuralpvs_fvdb}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
PYG_VERSION="${PYG_VERSION:-2.6.1}"
CUDA_VERSION="${CUDA_VERSION:-12.8}"

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

echo "Installing base packaging tools"
conda install -n "${ENV_NAME}" pip setuptools wheel -y
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel build

ENV_PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
ENV_NVCC="${ENV_PREFIX}/bin/nvcc"

if [ ! -x "${ENV_NVCC}" ]; then
  echo "No env-local nvcc found; installing CUDA toolkit ${CUDA_VERSION} into the conda environment"
  conda install -n "${ENV_NAME}" -c nvidia "cuda-toolkit=${CUDA_VERSION}" -y
fi

if [ ! -x "${ENV_NVCC}" ]; then
  echo "Expected nvcc at ${ENV_NVCC}, but it is still missing."
  echo "Please verify the CUDA toolkit installation in the conda environment."
  exit 1
fi

export CUDA_HOME="${ENV_PREFIX}"
export CUDA_PATH="${CUDA_HOME}"
export CUDACXX="${ENV_NVCC}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"
export CUDAToolkit_ROOT="${CUDA_HOME}"
export CMAKE_PREFIX_PATH="${CUDA_HOME}:${CMAKE_PREFIX_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA toolkit configured from conda environment"
echo "  CUDA_HOME=${CUDA_HOME}"
echo "  CUDACXX=${CUDACXX}"

echo "Installing PyTorch ${TORCH_VERSION} + CUDA 12.8"
conda run -n "${ENV_NAME}" python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url https://download.pytorch.org/whl/cu128

echo "Installing native build prerequisites for fvdb-core"
conda run -n "${ENV_NAME}" python -m pip install scikit-build-core cmake ninja pybind11

echo "Installing fvdb"
conda run -n "${ENV_NAME}" python -m pip install --no-build-isolation fvdb-core

echo "Installing PyTorch Geometric core packages"
conda run -n "${ENV_NAME}" python -m pip install "torch-geometric==${PYG_VERSION}"
conda run -n "${ENV_NAME}" python -m pip install \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

echo "Installing repo packages needed for validation and visualization"
conda run -n "${ENV_NAME}" python -m pip install \
  numpy pandas tqdm tensorboard torchinfo matplotlib open3d

echo
echo "Environment setup complete."
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"
