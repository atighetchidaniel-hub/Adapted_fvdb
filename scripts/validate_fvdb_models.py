#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.dataset import PVSVoxelDataset  # noqa: E402
from utils.init import init_loss, init_model, init_optimizer  # noqa: E402
from utils.tensor import to_dense, to_sparse  # noqa: E402


MODEL_NAMES = [
    "VNet",
    "VNetInterleaved",
    "VNetLighter",
    "VNetLight",
    "OACNNs",
    "OACNNsInterleaved",
]


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _dataset_sample(dataset_root: Path, z_size: int):
    dataset = PVSVoxelDataset(root=str(dataset_root), mode="infer", z_size=z_size)
    sample = dataset[0]
    return dataset, sample


def _torchify_sample(sample: dict, device: str):
    x = torch.from_numpy(sample["input"]).unsqueeze(0).to(device)
    target = torch.from_numpy(sample["target"]).unsqueeze(0).to(device)
    return x, target


def _build_model(name: str, args, device: str, train: bool):
    kwargs = {"args": args}
    if name.startswith("OACNNs"):
        kwargs["model_depth"] = 2
    model = init_model(name, "fvdb", 1, 1, **kwargs).to(device)
    return model.train() if train else model.eval()


def run_loader_check(dataset_root: Path, z_size: int) -> dict:
    dataset, sample = _dataset_sample(dataset_root, z_size)
    result = {
        "num_samples": len(dataset),
        "input_shape": tuple(sample["input"].shape),
        "target_shape": tuple(sample["target"].shape),
        "input_sum": float(sample["input"].sum()),
        "target_sum": float(sample["target"].sum()),
    }

    if result["num_samples"] <= 0:
        raise RuntimeError("Dataset contains no samples.")
    if result["input_sum"] <= 0 or result["target_sum"] <= 0:
        raise RuntimeError("Dataset sample sums must be positive.")

    return result


def run_forward_sweep(dataset_root: Path, z_size: int, device: str) -> list[dict]:
    dataset, sample = _dataset_sample(dataset_root, z_size)
    del dataset

    x, target = _torchify_sample(sample, device)
    args = argparse.Namespace(interleaver_r=2, dice_alpha=0.1)
    criterion = init_loss("dice", "fvdb", 1, args)

    results = []
    for name in MODEL_NAMES:
        model = _build_model(name, args, device, train=False)
        model_input = x if name.startswith("OACNNs") else to_sparse(x, "fvdb")

        with torch.no_grad():
            y = model(model_input)
            y_dense = to_dense(y, x.shape)
            loss, metrics = criterion(y, target, {})

        finite = bool(torch.isfinite(y_dense).all())
        shape = tuple(y_dense.shape)
        dice = np.asarray(metrics["dice"]).tolist()

        if not finite:
            raise RuntimeError(f"{name} produced non-finite output in forward sweep.")
        if shape != tuple(x.shape):
            raise RuntimeError(f"{name} returned shape {shape}, expected {tuple(x.shape)}.")

        results.append(
            {
                "model": name,
                "output_type": type(y).__name__,
                "output_shape": shape,
                "finite": finite,
                "loss": float(loss),
                "dice": dice,
            }
        )

    return results


def run_train_step_sweep(dataset_root: Path, z_size: int, device: str) -> list[dict]:
    dataset, sample = _dataset_sample(dataset_root, z_size)
    del dataset

    x, target = _torchify_sample(sample, device)
    args = argparse.Namespace(interleaver_r=2, dice_alpha=0.1)

    results = []
    for name in MODEL_NAMES:
        model = _build_model(name, args, device, train=True)
        model_input = x if name.startswith("OACNNs") else to_sparse(x, "fvdb")

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
        if not grad_ok:
            raise RuntimeError(f"{name} produced missing or non-finite gradients.")

        optimizer.step()

        results.append(
            {
                "model": name,
                "loss": float(loss.detach()),
                "dice": np.asarray(metrics["dice"]).tolist(),
                "grad_ok": grad_ok,
            }
        )

    return results


def print_loader_result(result: dict):
    print("Loader check:")
    print(f"  num samples: {result['num_samples']}")
    print(f"  input shape: {result['input_shape']}")
    print(f"  target shape: {result['target_shape']}")
    print(f"  input sum: {result['input_sum']}")
    print(f"  target sum: {result['target_sum']}")


def print_forward_results(results: list[dict]):
    print("\nForward sweep:")
    for result in results:
        print(
            f"  {result['model']}: ok | type={result['output_type']} | "
            f"shape={result['output_shape']} | finite={result['finite']} | "
            f"loss={result['loss']:.6f} | dice={result['dice']}"
        )


def print_train_step_results(results: list[dict]):
    print("\nTrain-step sweep:")
    for result in results:
        print(
            f"  {result['model']}: train_step_ok | loss={result['loss']:.6f} | "
            f"dice={result['dice']} | grad_ok={result['grad_ok']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate all fvdb VNet and OA-CNN model variants on a real or synthetic dataset."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "data_for_test" / "viking" / "r30"),
        help="Path to the dataset directory containing gv/ and pvv/.",
    )
    parser.add_argument(
        "--z-size",
        type=int,
        default=256,
        help="Voxel grid z dimension used to decode the .bin.gz files.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to write the validation summary as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not (dataset_root / "gv").exists() or not (dataset_root / "pvv").exists():
        raise FileNotFoundError(
            f"Expected gv/ and pvv/ under dataset root: {dataset_root}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")

    summary = {
        "dataset_root": _repo_relative(dataset_root),
        "z_size": args.z_size,
        "device": args.device,
        "loader": run_loader_check(dataset_root, args.z_size),
        "forward": run_forward_sweep(dataset_root, args.z_size, args.device),
        "train_step": run_train_step_sweep(dataset_root, args.z_size, args.device),
    }

    print(f"Dataset root: {summary['dataset_root']}")
    print_loader_result(summary["loader"])
    print_forward_results(summary["forward"])
    print_train_step_results(summary["train_step"])
    print("\nVALIDATION PASSED")

    if args.save_json:
        save_path = Path(args.save_json).expanduser()
        if not save_path.is_absolute():
            save_path = REPO_ROOT / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary JSON to {_repo_relative(save_path.resolve())}")


if __name__ == "__main__":
    main()
