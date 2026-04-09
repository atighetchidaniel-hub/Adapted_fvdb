# Conda Setup and Validation Guide

This document describes the Python environment that was used to validate the `fvdb` migration in this repo.

It is written for a fresh clone and focuses on reproducing the current `fvdb` model tests, including the real-data Viking validation.

## Validated Stack

- Python `3.11`
- PyTorch `2.8.0` with CUDA `12.8`
- `fvdb` via the `fvdb-core` package
- PyTorch Geometric `2.6.1` plus the matching compiled extensions

Suggested conda environment name:

- `neuralpvs_fvdb`

## Fastest Reproducible Path

If you want one command for setup and one for validation, use the checked-in scripts:

```bash
cd Adapted_fvdb
bash scripts/setup_conda_fvdb.sh neuralpvs_fvdb
conda activate neuralpvs_fvdb
bash scripts/run_validation.sh --save-json results/fvdb_validation_summary.json
```

Or run both setup and validation in one shot:

```bash
cd Adapted_fvdb
bash scripts/setup_and_validate_fvdb.sh neuralpvs_fvdb
```

## 1. Create the Conda Environment

```bash
conda create -n neuralpvs_fvdb python=3.11 -y
conda activate neuralpvs_fvdb
python -m pip install --upgrade pip
```

## 2. Install Core Dependencies

Install PyTorch first:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Install `fvdb` after PyTorch:

```bash
python -m pip install fvdb-core
```

Install the PyTorch Geometric packages needed by the OA-CNN models:

```bash
python -m pip install torch-geometric==2.6.1
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Install the repo packages needed for training, logging, and the current validation scripts:

```bash
python -m pip install numpy pandas tqdm tensorboard torchinfo
```

## 3. Optional Packages

These were not required for the core `fvdb` validation sweeps, but they are useful for auxiliary scripts and analysis:

```bash
python -m pip install cupy-cuda12x matplotlib seaborn scipy open3d av opencv-python
```

Notes:

- `cupy-cuda12x` is only needed if you want to use the `--cupy` data-loading path.
- `open3d`, `matplotlib`, `av`, and `opencv-python` are mainly used by visualization and video-analysis scripts.

## 4. Quick Environment Smoke Test

From the repo root:

```bash
cd Adapted_fvdb
python - <<'PY'
import torch
import fvdb

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("fvdb import ok:", fvdb is not None)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

Expected result:

- `cuda available: True`
- `fvdb import ok: True`

## 5. Sparse Roundtrip Smoke Test

```bash
cd Adapted_fvdb
python - <<'PY'
import torch
from utils.tensor import to_sparse, to_dense

x = torch.zeros(1, 1, 8, 8, 8, device="cuda")
x[:, :, 2:6, 2:6, 2:6] = 1

s = to_sparse(x, "fvdb")
y = to_dense(s, x.shape)

print("type:", type(s).__name__)
print("shape:", tuple(y.shape))
print("equal:", torch.equal(x, y))
PY
```

Expected result:

- `type: FvdbTensor`
- `equal: True`

## 6. Real-Data Dataset Assumption

The real-data validation below assumes that the repo contains the Viking dataset at:

```text
data_for_test/viking/r30/
  gv/
  pvv/
```

If this folder is not present in the clone, either:

- generate it from the Unity NeuralPVS repo first, or
- adjust the dataset root in the commands below

The current validation assumes:

- `z_size = 256`

## 7. Viking Dataset Loader Check

```bash
cd Adapted_fvdb
python - <<'PY'
from modules.dataset import PVSVoxelDataset

dataset = PVSVoxelDataset(
    root="data_for_test/viking/r30",
    mode="infer",
    z_size=256,
)

sample = dataset[0]
print("num samples:", len(dataset))
print("input shape:", sample["input"].shape)
print("target shape:", sample["target"].shape)
print("input sum:", float(sample["input"].sum()))
print("target sum:", float(sample["target"].sum()))
PY
```

The validated Viking dataset used for the current progress report produced:

- `num samples: 818`
- `input shape: (1, 256, 256, 256)`
- `target shape: (1, 256, 256, 256)`

## 8. Real-Data Forward Sweep Across All `fvdb` Models

```bash
cd Adapted_fvdb
python - <<'PY'
import argparse
import torch

from modules.dataset import PVSVoxelDataset
from utils.init import init_model, init_loss
from utils.tensor import to_sparse, to_dense

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_root = "data_for_test/viking/r30"

dataset = PVSVoxelDataset(
    root=dataset_root,
    mode="infer",
    z_size=256,
)

sample = dataset[0]
x = torch.from_numpy(sample["input"]).unsqueeze(0).to(device)
target = torch.from_numpy(sample["target"]).unsqueeze(0).to(device)

args = argparse.Namespace(interleaver_r=2, dice_alpha=0.1)
criterion = init_loss("dice", "fvdb", 1, args)

models = [
    "VNet",
    "VNetInterleaved",
    "VNetLighter",
    "VNetLight",
    "OACNNs",
    "OACNNsInterleaved",
]

for name in models:
    if name.startswith("OACNNs"):
        model = init_model(name, "fvdb", 1, 1, model_depth=2, args=args).to(device).eval()
        model_input = x
    else:
        model = init_model(name, "fvdb", 1, 1, args=args).to(device).eval()
        model_input = to_sparse(x, "fvdb")

    with torch.no_grad():
        y = model(model_input)
        y_dense = to_dense(y, x.shape)
        loss, metrics = criterion(y, target, {})

    print(
        f"{name}: ok | type={type(y).__name__} | shape={tuple(y_dense.shape)} | "
        f"finite={bool(torch.isfinite(y_dense).all())} | loss={float(loss):.6f} | dice={metrics['dice']}"
    )
PY
```

This is the main real-data forward validation sweep used in the current status update.

## 9. Real-Data Train-Step Sweep Across All `fvdb` Models

```bash
cd Adapted_fvdb
python - <<'PY'
import argparse
import torch

from modules.dataset import PVSVoxelDataset
from utils.init import init_model, init_loss, init_optimizer
from utils.tensor import to_sparse

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_root = "data_for_test/viking/r30"

dataset = PVSVoxelDataset(
    root=dataset_root,
    mode="infer",
    z_size=256,
)

sample = dataset[0]
x = torch.from_numpy(sample["input"]).unsqueeze(0).to(device)
target = torch.from_numpy(sample["target"]).unsqueeze(0).to(device)

args = argparse.Namespace(interleaver_r=2, dice_alpha=0.1)

models = [
    "VNet",
    "VNetInterleaved",
    "VNetLighter",
    "VNetLight",
    "OACNNs",
    "OACNNsInterleaved",
]

for name in models:
    if name.startswith("OACNNs"):
        model = init_model(name, "fvdb", 1, 1, model_depth=2, args=args).to(device).train()
        model_input = x
    else:
        model = init_model(name, "fvdb", 1, 1, args=args).to(device).train()
        model_input = to_sparse(x, "fvdb")

    criterion = init_loss("dice", "fvdb", 1, args)
    optimizer, _ = init_optimizer("adam", model, lr=1e-4, no_scheduler=True)

    optimizer.zero_grad(set_to_none=True)
    y = model(model_input)
    loss, metrics = criterion(y, target, {})
    loss.backward()

    grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )

    optimizer.step()

    print(
        f"{name}: train_step_ok | loss={float(loss.detach()):.6f} | "
        f"dice={metrics['dice']} | grad_ok={grad_ok}"
    )
PY
```

This is the main real-data training-breadth validation sweep used in the current status update.

## 10. Optional `train.py` Smoke Run

If you also want to reproduce a one-epoch `train.py` run on the Viking dataset, first expose it under the dataset layout expected by the training CLI:

```bash
cd Adapted_fvdb
mkdir -p data_for_test/datasets
ln -s ../viking/r30 data_for_test/datasets/viking_r30
```

Then run:

```bash
cd Adapted_fvdb
python train.py \
  --root data_for_test \
  --dataset_name viking_r30 \
  --z_size 256 \
  --test_fraction 0.1 \
  --model VNet \
  --backend fvdb \
  --batchSz 1 \
  --nEpochs 1 \
  --lr 0.0001 \
  --opt adam \
  --out_dir out \
  --tag viking_realdata_smoke
```

## 11. Related Repo Documents

- `FVDB_MIGRATION_SUMMARY.md`
- `STATUS_UPDATE_2026-04-09.md`

These two files summarize:

- what was changed in the `spconv` to `fvdb` migration
- what has already been validated on synthetic and real Unity-generated data

## 12. References

- PyTorch install guide: <https://pytorch.org/get-started/locally/>
- fVDB docs: <https://openvdb.github.io/fvdb-core/>
- PyTorch Geometric install guide: <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>
