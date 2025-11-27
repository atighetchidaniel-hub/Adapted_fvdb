# NeuralPVS: Learned Estimation of Potentially Visible Sets

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2509.24677)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://windingwind.github.io/neuralpvs/)

<img width="2754" height="730" alt="image" src="https://github.com/user-attachments/assets/5ac2c585-05e6-433f-ace2-75b3193f69b3" />

Official training code for **NeuralPVS**, a deep learning approach for real-time visibility computation presented at **SIGGRAPH Asia 2025**.

## Overview

NeuralPVS is the first deep-learning approach for visibility computation that efficiently determines from-region visibility in large scenes, running at ~100 Hz with less than 1% missing geometry. The network operates on a voxelized (froxelized) representation of the scene and combines sparse convolution with 3D volume-preserving interleaving for data compression.

## Installation

```bash
# Clone the repository
git clone https://github.com/windingwind/neuralpvs.git
cd neuralpvs

# Set up the Python virtual environment
python scripts/setup_venv.py
```

**Requirements:**

- Python 3.12
- PyTorch 2.7 with CUDA 12.8
- spconv (for sparse convolution backend)

The specified spconv whl is for RTX 5090 or higher, as the spconv releases on PyPI do not support the latest GPU architectures. Adjust accordingly for other GPUs.

The setup is tested on Oracle Linux 9.6 and Windows 10/11. If it does not work for you, please manually install the PyTorch and build spconv from source.

## Dataset and Output Structure

The dataset should be organized as follows:

```text
<root_dir>/
├── datasets/
│   └── <dataset_name>/
│       ├── gv/           # Geometry voxel grids (input)
│       │   ├── 0000_gv.bin.gz
│       │   ├── 0001_gv.bin.gz
│       │   └── ...
│       └── pvv/          # Potentially visible voxels (ground truth)
│           ├── 0000_pvv.bin.gz
│           ├── 0001_pvv.bin.gz
│           └── ...
└── <out_dir>/
    └── <experiment_name>/
        ├── training_arguments.json
        ├── <experiment_name>_BEST.pth
        ├── <experiment_name>_last_epoch.pth
        └── ...
```

Each `.bin.gz` file contains a bit-packed voxel grid. Training checkpoints and logs are saved to `<root_dir>/<out_dir>/<experiment_name>/`, where `<experiment_name>` is auto-generated from model, dataset, loss, timestamp, and tag.

## Training

```bash
python train.py \
    --root <root_dir> \
    --dataset_name <dataset_name> \
    --z_size 256 \
    --test_fraction 0.05 \
    --model OACNNsInterleaved \
    --backend spconv \
    --model_depth 3 \
    --loss dice,no_guess \
    --loss_weights 0.99,0.01 \
    --dice_alpha 0.001 \
    --batchSz 2 \
    --nEpochs 100 \
    --lr 0.001 \
    --opt adam \
    --out_dir <out_dir> \
    --tag <experiment_tag>
```

**Key arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | Root directory for data and outputs | `./data` |
| `--out_dir` | Output directory under `<root>/` | `out` |
| `--dataset_name` | Name of dataset under `<root>/datasets/` | `tiny2` |
| `--z_size` | Voxel grid Z dimension | required |
| `--test_fraction` | Validation split ratio | required |
| `--model` | Model architecture (`VNet`, `VNetInterleaved`, `OACNNsInterleaved`) | `VNet` |
| `--backend` | Convolution backend (`torchnn`, `spconv`) | `torchnn` |
| `--model_depth` | Network depth | `3` |
| `--interleaver_r` | Interleaving factor for interleaved models | `2` |
| `--loss` | Loss function(s), comma-separated (`dice`, `no_guess`, `focal`) | `dice` |
| `--loss_weights` | Weights for multiple losses, comma-separated | - |
| `--dice_alpha` | Alpha parameter for weighted Dice loss | `0.1` |
| `--batchSz` | Batch size | `2` |
| `--nEpochs` | Number of training epochs | `100` |
| `--lr` | Learning rate | `1e-2` |
| `--opt` | Optimizer (`adam`, `sgd`, `adamw`, etc.) | `adam` |
| `--no_scheduler` | Disable learning rate scheduler | `False` |
| `--amp` | Enable automatic mixed precision | `False` |
| `--resume` | Path to checkpoint for resuming training | - |

## Inference

```bash
python infer.py \
    --root <root_dir> \
    --out_dir <out_dir> \
    --exp_name <experiment_name> \
    --dataset_name <dataset_name>
```

**Inference arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | Root directory for data and outputs | `.` |
| `--out_dir` | Output directory under `<root>/` | `out` |
| `--exp_name` | Experiment name (folder under `<out_dir>/`) | required |
| `--dataset_name` | Override dataset for inference | from training |
| `--ckpt_suffix` | Checkpoint suffix (e.g., `59_epoch`) | `BEST` |
| `--z_size` | Override voxel grid Z dimension | from training |
| `--cache_size` | Temporal smoothing cache size | `0` |
| `--max_pool_size` | Max pooling dilation kernel size | `-1` |
| `--timing` | Run timing benchmark | `False` |

## Batch Evaluation

Run evaluation across multiple experiments and datasets:

```bash
python scripts/run_eval.py \
    --root <root_dir> \
    --exp_path <out_dir> \
    --save_path <save_dir> \
    --fov 30 \
    --size 256 \
    --z_size 256 \
    --datasets viking,robotlab,bigcity \
    --keyword <filter_keyword> \
    --max_concurrent 4
```

**Key arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--exp_path` | Directory containing experiment folders | required |
| `--save_path` | Directory to save evaluation results | required |
| `--fov` | Field of view (used in dataset prefix) | `60` |
| `--size` | Voxel grid XY size | `256` |
| `--z_size` | Voxel grid Z size | `256` |
| `--datasets` | Comma-separated list of dataset suffixes | all default |
| `--keyword` | Filter experiments by keyword | `""` |
| `--cache_size` | Temporal smoothing cache size | from training |
| `--max_pool_size` | Max pooling dilation kernel size | from training |
| `--ckpt_suffix` | Checkpoint suffix to evaluate | `BEST` |
| `--use_all_ckpt` | Evaluate all checkpoints in each experiment | `False` |
| `--max_concurrent` | Number of parallel evaluations | `1` |

## Model Architectures

- **VNet**: Baseline V-Net [^1]
- **VNetInterleaved**: V-Net with 3D interleaving layers
- **OACNNsInterleaved**: OA-CNN [^2] with 3D interleaving layers (recommended)

## Citation

```bibtex
@misc{wang2025neuralpvs,
  title={NeuralPVS: Learned Estimation of Potentially Visible Sets},
  author={Xiangyu Wang and Thomas Köhler and Jun Lin Qiu and Shohei Mori and Markus Steinberger and Dieter Schmalstieg},
  year={2025},
  eprint={2509.24677},
  archivePrefix={arXiv},
  primaryClass={cs.GR}
}
```

[^1]: Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. 2016. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. In International Conference on 3D Vision. arXiv, 565–571. doi:10.1109/3DV.2016.79

[^2]: Bohao Peng, Xiaoyang Wu, Li Jiang, Yukang Chen, Hengshuang Zhao, Zhuotao Tian, and Jiaya Jia. 2024. OA-CNNs: Omni-Adaptive Sparse CNNs for 3D Semantic Segmentation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Seattle, WA, USA, 21305–21315. doi:10.1109/CVPR52733.2024.02013
