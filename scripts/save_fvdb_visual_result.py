#!/usr/bin/env python3
"""Save a stable PNG summary of the fVDB Viking visual inference result.

This uses the same files as the Open3D visualizer, but writes a normal image
instead of opening an interactive GUI window. Each panel is a max projection over
one axis of the 3D voxel grid.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXP_PATH = (
    REPO_ROOT
    / "data_for_test"
    / "out"
    / "VNet_viking_r30_dice_20260413-121838_viking_visual_vnet-visual_infer_M6Ll37wx"
)


def read_binary_file(file_path: Path, size: int = 256, depth: int = 256) -> np.ndarray:
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    data = data.byteswap().view(np.uint8)
    return np.unpackbits(data).reshape((size, size, depth))


def projection(volume: np.ndarray, axis: int) -> np.ndarray:
    return volume.max(axis=axis)


def error_projection(gt: np.ndarray, pred: np.ndarray, axis: int) -> np.ndarray:
    fp = (pred > 0) & (gt == 0)
    fn = (pred == 0) & (gt > 0)
    tp = (pred > 0) & (gt > 0)

    # 0 background, 1 true positive, 2 false positive, 3 false negative.
    err = np.zeros(gt.shape, dtype=np.uint8)
    err[tp] = 1
    err[fp] = 2
    err[fn] = 3
    return err.max(axis=axis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a PNG comparing GV, Unity PVV, fVDB predicted PVV, and error."
    )
    parser.add_argument("--id", type=int, default=0, help="Sample id to visualize.")
    parser.add_argument(
        "--dataset-path",
        default=str(REPO_ROOT / "data_for_test" / "viking" / "r30"),
        help="Dataset root containing gv/ and pvv/.",
    )
    parser.add_argument(
        "--exp-path",
        default=str(DEFAULT_EXP_PATH),
        help="Inference experiment folder containing inference/0/*.bin.gz.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to results/visuals/fvdb_viking_sample_<id>.png.",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="Axis used for max projection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    exp_path = Path(args.exp_path).expanduser().resolve()
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else REPO_ROOT / "results" / "visuals" / f"fvdb_viking_sample_{args.id}.png"
    )

    gv_path = dataset_path / "gv" / f"{args.id}_gv.bin.gz"
    gt_path = dataset_path / "pvv" / f"{args.id}_pvv.bin.gz"
    pred_path = exp_path / "inference" / "0" / f"{args.id}_predicted_pvv.bin.gz"

    for path in (gv_path, gt_path, pred_path):
        if not path.exists():
            raise FileNotFoundError(path)

    gv = read_binary_file(gv_path)
    gt = read_binary_file(gt_path)
    pred = read_binary_file(pred_path)

    gt = np.where(gv > 0, gt > 0, False)
    pred = np.where(gv > 0, pred > 0, False)
    gv = gv > 0

    panels = [
        ("GV geometry", projection(gv, args.axis), "gray", None),
        ("Unity ground-truth PVV", projection(gt, args.axis), "viridis", None),
        ("fVDB predicted PVV", projection(pred, args.axis), "plasma", None),
        (
            "Error overlay\nTP=green FP=red FN=blue",
            error_projection(gt, pred, args.axis),
            ListedColormap(["black", "limegreen", "red", "royalblue"]),
            (0, 3),
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    for ax, (title, image, cmap, clim) in zip(axes, panels):
        ax.imshow(image.T, origin="lower", cmap=cmap, interpolation="nearest")
        if clim is not None:
            ax.images[-1].set_clim(*clim)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"fVDB Viking visual result | sample {args.id} | projection axis {args.axis}",
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved visual result to {out_path}")


if __name__ == "__main__":
    main()
