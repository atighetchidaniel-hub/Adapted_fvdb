#!/usr/bin/env python3
"""Run one NeuralPVS fVDB prediction for Unity's generated GV file.

This script intentionally lives in the Unity project copy, but imports the
adapted NeuralPVS Python repo through --neuralpvs-root. It keeps the renderer
repo small and makes the bridge explicit at the process boundary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one PVV file from one Unity-generated GV file."
    )
    parser.add_argument("--neuralpvs-root", required=True)
    parser.add_argument("--gv", required=True, help="Input <id>_gv.bin.gz file.")
    parser.add_argument("--out", required=True, help="Output <id>_predicted_pvv.bin.gz file.")
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint path.")
    parser.add_argument("--model", default="OACNNsInterleaved")
    parser.add_argument("--backend", default="fvdb")
    parser.add_argument("--classes", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--model-depth", type=int, default=2)
    parser.add_argument("--interleaver-r", type=int, default=2)
    parser.add_argument("--z-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-pool-size", type=int, default=-1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def wait_for_stable_file(path: Path, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    last_size = -1
    stable_count = 0

    while time.time() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 0 and size == last_size:
                stable_count += 1
                if stable_count >= 2:
                    return
            else:
                stable_count = 0
            last_size = size
        time.sleep(0.25)

    raise TimeoutError(f"GV file was not stable before timeout: {path}")


def main() -> int:
    args = parse_args()

    neuralpvs_root = Path(args.neuralpvs_root).expanduser().resolve()
    gv_path = Path(args.gv).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    if not neuralpvs_root.exists():
        raise FileNotFoundError(f"NeuralPVS root not found: {neuralpvs_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(neuralpvs_root))

    import torch
    import torch.nn.functional as F

    from modules.dataset import load_volume
    from utils.init import init_model
    from utils.tensor import to_dense, to_sparse
    from utils.train import restore_checkpoint, save_volume

    wait_for_stable_file(gv_path)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_args = argparse.Namespace(
        interleaver_r=args.interleaver_r,
        dice_alpha=0.1,
    )

    model = init_model(
        args.model,
        args.backend,
        args.in_channels,
        args.classes,
        args.model_depth,
        model_args,
    ).to(device)
    restore_checkpoint(model, str(checkpoint_path))
    model.eval()

    gv = load_volume(str(gv_path), amp=False, z_size=args.z_size, cupy=False)
    x = torch.from_numpy(gv).unsqueeze(0).to(device)
    model_input = x if args.model.startswith("OACNNs") else to_sparse(x, args.backend)

    with torch.no_grad():
        y = model(model_input)
        y_dense = to_dense(y, x.shape).sigmoid()

        if args.max_pool_size and args.max_pool_size > 0:
            kernel = args.max_pool_size
            padding = kernel // 2
            y_dense = F.max_pool3d(y_dense, kernel_size=kernel, stride=1, padding=padding)

        pred = (y_dense > args.threshold).int()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_volume(pred.cpu(), str(out_path))

    active_voxels = int(pred.sum().item())
    print(f"wrote={out_path}")
    print(f"active_voxels={active_voxels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
