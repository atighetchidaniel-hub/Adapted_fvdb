#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.dataset import PVSVoxelDataset
from utils.init import init_cuda, init_model
from utils.tensor import FvdbTensor, to_dense, to_sparse
from utils.train import restore_checkpoint


def load_training_args(exp_dir: Path):
    args_path = exp_dir / "training_arguments.json"
    if not args_path.exists():
        return {}
    with args_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_checkpoint(exp_dir: Path):
    best = sorted(exp_dir.glob("*_BEST.pth"))
    if best:
        return best[-1]
    last = sorted(exp_dir.glob("*_last_epoch.pth"))
    if last:
        return last[-1]
    any_ckpt = sorted(exp_dir.glob("*.pth"))
    if any_ckpt:
        return any_ckpt[-1]
    raise FileNotFoundError(f"No .pth checkpoint found in {exp_dir}")


def sample_to_tensors(sample: dict, device: str):
    input_tensor = torch.from_numpy(sample["input"]).unsqueeze(0).to(device)
    target = torch.from_numpy(sample["target"]).unsqueeze(0).to(device)
    return input_tensor, target


def model_input_for(model_name: str, x: torch.Tensor, backend: str):
    if backend == "fvdb" and model_name.startswith("OACNNs"):
        return x
    return to_sparse(x, backend)


def call_model(model, model_input, data_dict):
    try:
        return model(model_input, data_dict)
    except TypeError:
        return model(model_input)


def output_topology_mask(output, target_shape):
    if isinstance(output, torch.Tensor):
        return torch.ones(target_shape, device=output.device, dtype=torch.bool)
    if not isinstance(output, FvdbTensor):
        raise TypeError(f"Unsupported output type: {type(output)}")

    channels = target_shape[1]
    active = output.data.jdata.shape[0]
    ones = torch.ones((active, channels), device=output.data.jdata.device, dtype=torch.float32)
    jagged_ones = output.grid.jagged_like(ones.contiguous())
    dense = output.grid.inject_to_dense_cmajor(
        jagged_ones,
        min_coord=[0, 0, 0],
        grid_size=list(target_shape[2:]),
    )
    if dense.shape != target_shape:
        raise ValueError(f"Topology dense shape {tuple(dense.shape)} != target shape {tuple(target_shape)}")
    return dense > 0


def ratio(num: torch.Tensor, den: torch.Tensor):
    den_f = den.float().clamp_min(1.0)
    return (num.float() / den_f).detach().cpu().item()


def diagnose_one(model, model_name, backend, sample, device, threshold):
    x, target = sample_to_tensors(sample, device)
    data_dict = {"input": x, "target": target, "extras": {"gv": "input", "pvv": "target"}}
    model_input = model_input_for(model_name, x, backend)

    output = call_model(model, model_input, data_dict)
    topology = output_topology_mask(output, target.shape)
    dense_logits = to_dense(output, target.shape).float()

    gv = x > 0
    pvv = target > 0
    probs = torch.sigmoid(dense_logits)
    pred = torch.where(gv, probs > threshold, torch.zeros_like(gv, dtype=torch.bool))

    pvv_active = pvv.sum()
    gv_active = gv.sum()
    topology_active = topology.sum()
    pvv_covered = (pvv & topology).sum()
    gv_covered = (gv & topology).sum()
    hard_topology_fn = (pvv & ~topology).sum()

    fp = (pred & ~pvv).sum()
    fn = (~pred & pvv).sum()
    tn = (~pred & ~pvv).sum()
    tp = (pred & pvv).sum()

    return {
        "pvv_active": int(pvv_active.detach().cpu().item()),
        "gv_active": int(gv_active.detach().cpu().item()),
        "topology_active": int(topology_active.detach().cpu().item()),
        "pvv_covered": int(pvv_covered.detach().cpu().item()),
        "gv_covered": int(gv_covered.detach().cpu().item()),
        "hard_topology_fn": int(hard_topology_fn.detach().cpu().item()),
        "hard_topology_fn_rate": ratio(hard_topology_fn, pvv_active),
        "pvv_topology_recall": ratio(pvv_covered, pvv_active),
        "gv_topology_recall": ratio(gv_covered, gv_active),
        "fp": int(fp.detach().cpu().item()),
        "fn": int(fn.detach().cpu().item()),
        "tn": int(tn.detach().cpu().item()),
        "tp": int(tp.detach().cpu().item()),
        "fp_rate": ratio(fp, fp + tn),
        "fn_rate": ratio(fn, fn + tp),
        "fp_ratio": ratio(fp, gv_active),
        "gv_ratio": ratio(pred.sum(), gv_active),
    }


def mean(rows, key):
    if not rows:
        return float("nan")
    return sum(float(row[key]) for row in rows) / len(rows)


def print_summary(rows):
    keys = [
        "pvv_active",
        "gv_active",
        "topology_active",
        "hard_topology_fn",
        "hard_topology_fn_rate",
        "pvv_topology_recall",
        "gv_topology_recall",
        "fp_rate",
        "fn_rate",
        "fp_ratio",
        "gv_ratio",
    ]
    print("")
    print("FVDB TOPOLOGY FN SUMMARY")
    print(f"frames: {len(rows)}")
    for key in keys:
        print(f"{key:22s}: {mean(rows, key):.8f}")

    hard_rate = mean(rows, "hard_topology_fn_rate")
    print("")
    if hard_rate > 0:
        print("DIAGNOSIS: output topology misses PVV voxels. This is a backend-level hard FN floor.")
    else:
        print("DIAGNOSIS: output topology covers all PVV voxels in sampled frames. FN is probably logits/training, not topology.")


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose fVDB output topology false-negative floor.")
    parser.add_argument("--exp-dir", type=Path, default=None, help="Experiment folder containing training_arguments.json and checkpoint.")
    parser.add_argument("--ckpt", type=Path, default=None, help="Checkpoint path. Overrides --exp-dir checkpoint discovery.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset folder containing gv/ and pvv/.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default="fvdb")
    parser.add_argument("--model-depth", type=int, default=None)
    parser.add_argument("--interleaver-r", type=int, default=None)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--classes", type=int, default=None)
    parser.add_argument("--z-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    if args.backend != "fvdb":
        raise ValueError("This diagnostic is intended for backend=fvdb.")

    train_args = load_training_args(args.exp_dir) if args.exp_dir else {}
    ckpt = args.ckpt or (find_checkpoint(args.exp_dir) if args.exp_dir else None)
    if ckpt is None:
        raise ValueError("Provide --ckpt or --exp-dir.")

    model_name = args.model or train_args.get("model", "OACNNsInterleaved")
    model_depth = args.model_depth or int(train_args.get("model_depth", 3))
    interleaver_r = args.interleaver_r or int(train_args.get("interleaver_r", 16))
    in_channels = args.in_channels or int(train_args.get("inChannels", 1))
    classes = args.classes or int(train_args.get("classes", 1))
    z_size = args.z_size or int(train_args.get("z_size", 256))

    init_cuda(args.device.startswith("cuda"), cupy=False, seed=0, inference=True)

    model_args = SimpleNamespace(interleaver_r=interleaver_r)
    model = init_model(
        model_name,
        "fvdb",
        in_channels=in_channels,
        classes=classes,
        model_depth=model_depth,
        args=model_args,
    )
    restore_checkpoint(model, str(ckpt))
    model.to(args.device).eval()

    dataset = PVSVoxelDataset(root=str(args.dataset_root), mode="infer", z_size=z_size)
    frame_count = min(args.frames, len(dataset)) if args.frames is not None and args.frames > 0 else len(dataset)

    print("FVDB topology false-negative diagnostic")
    print(f"model:        {model_name}")
    print(f"checkpoint:   {ckpt}")
    print(f"dataset:      {args.dataset_root}")
    print(f"frames:       {frame_count}/{len(dataset)}")
    print(f"threshold:    {args.threshold}")
    print(f"interleaver:  {interleaver_r}")
    print("")

    rows = []
    with torch.no_grad():
        for i in range(frame_count):
            row = diagnose_one(model, model_name, "fvdb", dataset[i], args.device, args.threshold)
            row["frame"] = i
            rows.append(row)
            print(
                f"frame {i:04d}: "
                f"hard_fn={row['hard_topology_fn']} "
                f"hard_fn_rate={row['hard_topology_fn_rate']:.8f} "
                f"topo_recall={row['pvv_topology_recall']:.8f} "
                f"fn_rate={row['fn_rate']:.8f} "
                f"fp_rate={row['fp_rate']:.8f}"
            )

    print_summary(rows)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["frame"] + [key for key in rows[0].keys() if key != "frame"]
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to: {args.out_csv}")


if __name__ == "__main__":
    main()
