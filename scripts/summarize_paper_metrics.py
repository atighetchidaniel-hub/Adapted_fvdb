#!/usr/bin/env python3
"""Summarize NeuralPVS paper-style voxel metrics from eval_stats.csv.

The training/inference runner writes `eval_stats.csv` with mean TP, FP, FN, and
TN counts. This helper converts those counts into the paper-style voxel metrics:

  FNR = FN / GTP
  FPR = FP / GTP
  GTP = TP + FN, the number of ground-truth visible voxels/froxels

It also prints the repo's existing metrics so it is clear which values are using
paper-style definitions and which are internal helper ratios.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_METRICS = ("tp", "fp", "fn", "tn")
OPTIONAL_METRICS = ("dice", "fn_rate", "fp_rate", "fp_ratio", "gv_ratio", "loss")


def resolve_eval_stats_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        path = path / "eval_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"eval_stats.csv not found: {path}")
    return path


def load_means(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "Metric" not in reader.fieldnames or "Mean" not in reader.fieldnames:
            raise ValueError(f"Expected columns Metric and Mean in {path}")
        for row in reader:
            metric = row["Metric"]
            try:
                values[metric] = float(row["Mean"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Could not parse Mean for metric {metric!r} in {path}") from exc
    missing = [metric for metric in REQUIRED_METRICS if metric not in values]
    if missing:
        raise ValueError(f"Missing required metrics in {path}: {', '.join(missing)}")
    return values


def fmt(value: float) -> str:
    return f"{value:.6f}"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.3f}%"


def print_summary(path: Path) -> None:
    values = load_means(path)
    tp = values["tp"]
    fp = values["fp"]
    fn = values["fn"]
    tn = values["tn"]

    gtp = tp + fn
    if gtp <= 0:
        raise ValueError(f"GTP = TP + FN must be positive in {path}")

    paper_fnr = fn / gtp
    paper_fpr = fp / gtp
    standard_dice = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) > 0 else 0.0
    predicted_to_gt = (tp + fp) / gtp

    print(f"eval_stats: {path}")
    print("Mean voxel counts:")
    print(f"  TP:  {fmt(tp)}")
    print(f"  FN:  {fmt(fn)}")
    print(f"  FP:  {fmt(fp)}")
    print(f"  TN:  {fmt(tn)}")
    print(f"  GTP = TP + FN: {fmt(gtp)}")
    print()
    print("Paper-style voxel metrics:")
    print(f"  FNR = FN / GTP: {fmt(paper_fnr)} ({fmt_pct(paper_fnr)})")
    print(f"  FPR = FP / GTP: {fmt(paper_fpr)} ({fmt_pct(paper_fpr)})")
    print(f"  Predicted-visible / GTP: {fmt(predicted_to_gt)}")
    print()
    print("Additional repo metrics:")
    print(f"  Standard Dice from thresholded counts: {fmt(standard_dice)}")
    for metric in OPTIONAL_METRICS:
        if metric in values:
            label = metric
            if metric == "dice":
                label = "weighted Dice reported by runner"
            elif metric == "fp_rate":
                label = "repo fp_rate = FP / (FP + TN)"
            elif metric == "fn_rate":
                label = "repo fn_rate = FN / (FN + TP)"
            elif metric == "fp_ratio":
                label = "repo fp_ratio = FP / GV"
            print(f"  {label}: {fmt(values[metric])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print NeuralPVS paper-style FNR/FPR from one or more eval_stats.csv files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Path(s) to eval_stats.csv or to inference experiment folders containing eval_stats.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for i, path in enumerate(args.paths):
        if i:
            print("\n" + "=" * 72 + "\n")
        print_summary(resolve_eval_stats_path(path))


if __name__ == "__main__":
    main()
