#!/usr/bin/env python3
import argparse
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses.dice import SparseWeightedDiceLoss
from utils.init import init_cuda, init_model, init_optimizer
from utils.tensor import to_dense, to_sparse


MODEL_NAMES = (
    "VNet",
    "VNetInterleaved",
    "VNetLighter",
    "VNetLight",
    "OACNNs",
    "OACNNsInterleaved",
)


def make_volume(size: int, density: float, device: str):
    x = (torch.rand(1, 1, size, size, size, device=device) < density).float()
    if not torch.any(x > 0):
        center = size // 2
        x[:, :, center - 1:center + 1, center - 1:center + 1, center - 1:center + 1] = 1.0
    target = x * (torch.rand_like(x) < 0.5).float()
    if not torch.any(target > 0):
        target = x.clone()
    return x, target


def model_input_for(name: str, x: torch.Tensor):
    if name.startswith("OACNNs"):
        # The optimized OACNNsInterleaved path intentionally accepts dense
        # input so interleaving happens before the smaller fVDB grid is built.
        return x
    return to_sparse(x, "fvdb")


def call_model(model, model_input):
    params = inspect.signature(model.forward).parameters
    if len(params) >= 2:
        return model(model_input, {})
    return model(model_input)


def run_one(name: str, args, x: torch.Tensor, target: torch.Tensor, train_step: bool):
    model_depth = args.oacnn_depth if name.startswith("OACNNs") else 3
    model_args = SimpleNamespace(interleaver_r=args.interleaver_r)
    model = init_model(
        name,
        "fvdb",
        in_channels=1,
        classes=1,
        model_depth=model_depth,
        args=model_args,
    ).to(args.device)
    model.train(mode=train_step)

    criterion = SparseWeightedDiceLoss(classes=1, alpha=args.dice_alpha)
    optimizer = None
    if train_step:
        optimizer, _ = init_optimizer("adam", model, lr=1e-4, no_scheduler=True)
        optimizer.zero_grad(set_to_none=True)

    model_input = model_input_for(name, x)
    output = call_model(model, model_input)
    dense = to_dense(output, target.shape)
    loss, metrics = criterion(output, target, {})

    if not torch.isfinite(dense).all():
        raise RuntimeError(f"{name} produced non-finite dense output")
    if dense.shape != target.shape:
        raise RuntimeError(f"{name} output shape {tuple(dense.shape)} != target shape {tuple(target.shape)}")
    if not torch.isfinite(loss):
        raise RuntimeError(f"{name} produced non-finite loss")

    grad_ok = None
    if train_step:
        loss.backward()
        grad_ok = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
        if not grad_ok:
            raise RuntimeError(f"{name} produced missing or non-finite gradients")
        optimizer.step()

    dice = metrics["dice"]
    if hasattr(dice, "tolist"):
        dice = dice.tolist()

    return {
        "model": name,
        "params": sum(p.numel() for p in model.parameters()),
        "output_type": type(output).__name__,
        "output_shape": tuple(dense.shape),
        "loss": float(loss.detach().cpu()),
        "dice": dice,
        "grad_ok": grad_ok,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test every fVDB model class.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--interleaver-r", type=int, default=2)
    parser.add_argument("--oacnn-depth", type=int, default=2)
    parser.add_argument("--dice-alpha", type=float, default=0.001)
    parser.add_argument("--no-train-step", action="store_true")
    parser.add_argument("--models", nargs="*", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")

    init_cuda(args.device.startswith("cuda"), cupy=False, seed=0, inference=False)
    torch.manual_seed(0)

    x, target = make_volume(args.size, args.density, args.device)
    print("FVDB all-model smoke test")
    print(f"device={args.device} size={args.size} density={args.density} train_step={not args.no_train_step}")
    print(f"input active={int(x.sum().item())} target active={int(target.sum().item())}")
    print("")

    for name in args.models:
        result = run_one(name, args, x, target, train_step=not args.no_train_step)
        print(
            f"{result['model']}: ok | params={result['params']} | "
            f"type={result['output_type']} | shape={result['output_shape']} | "
            f"loss={result['loss']:.6f} | dice={result['dice']} | grad_ok={result['grad_ok']}"
        )

    print("")
    print("ALL FVDB MODEL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
