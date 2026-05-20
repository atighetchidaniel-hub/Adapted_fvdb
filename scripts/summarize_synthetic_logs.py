#!/usr/bin/env python3
"""Summarize NeuralPVS train/eval logs into a compact table.

Examples
--------
python scripts/summarize_synthetic_logs.py \
  /var/tmp/$USER_runs/NEURALPVS_MAY19_SYNTHETIC_5EP_ALTERNATING/logs \
  '*r30*d16*.log'

The script reads matching log files, takes eval rows from the final epoch in
each file, and reports mean metrics.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import statistics as stats
from pathlib import Path


METRICS = [
    "dice",
    "loss",
    "fp",
    "fn",
    "fp_rate",
    "fn_rate",
    "fp_ratio",
    "gv_ratio",
]


def parse_metric(line: str, key: str) -> float | None:
    match = re.search(rf"(?:^|\s|\|){re.escape(key)}:\s*([0-9.eE+-]+)", line)
    if not match:
        return None
    return float(match.group(1))


def infer_setting(path: Path) -> dict[str, str]:
    name = path.name
    record = {
        "backend": "",
        "radius": "",
        "d": "",
        "epochs": "",
        "file": str(path),
    }

    backend_match = re.search(r"_(fvdb|spconv)_", name)
    radius_match = re.search(r"_(r[0-9]+)_", name)
    d_match = re.search(r"_d([0-9]+)_", name)
    epoch_match = re.search(r"_([0-9]+)ep(?:\.|_)", name)

    if backend_match:
        record["backend"] = backend_match.group(1)
    if radius_match:
        record["radius"] = radius_match.group(1)
    if d_match:
        record["d"] = d_match.group(1)
    if epoch_match:
        record["epochs"] = epoch_match.group(1)

    return record


def summarize_log(path: Path) -> dict[str, object] | None:
    lines = path.read_text(errors="replace").splitlines()
    eval_lines = [line.strip() for line in lines if "[eval]" in line]
    if not eval_lines:
        return None

    epochs: list[int] = []
    for line in eval_lines:
        match = re.search(r"epoch:\s*([0-9]+)", line)
        if match:
            epochs.append(int(match.group(1)))

    if not epochs:
        return None

    final_epoch = max(epochs)
    final_lines = [line for line in eval_lines if f"epoch: {final_epoch}" in line]

    record: dict[str, object] = infer_setting(path)
    record["final_epoch"] = final_epoch
    record["samples"] = len(final_lines)

    for key in METRICS:
        vals = [parse_metric(line, key) for line in final_lines]
        vals = [v for v in vals if v is not None]
        if vals:
            record[f"{key}_mean"] = stats.mean(vals)
            record[f"{key}_std"] = stats.pstdev(vals) if len(vals) > 1 else 0.0
            record[f"{key}_min"] = min(vals)
            record[f"{key}_max"] = max(vals)
        else:
            record[f"{key}_mean"] = ""
            record[f"{key}_std"] = ""
            record[f"{key}_min"] = ""
            record[f"{key}_max"] = ""

    return record


def fmt(value: object, digits: int = 6) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: list[dict[str, object]]) -> None:
    headers = [
        "backend",
        "radius",
        "d",
        "epoch",
        "samples",
        "dice",
        "loss",
        "fp_rate",
        "fn_rate",
        "fp_ratio",
        "gv_ratio",
    ]

    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row.get("backend", ""),
                row.get("radius", ""),
                row.get("d", ""),
                row.get("final_epoch", ""),
                row.get("samples", ""),
                fmt(row.get("dice_mean", "")),
                fmt(row.get("loss_mean", "")),
                fmt(row.get("fp_rate_mean", "")),
                fmt(row.get("fn_rate_mean", "")),
                fmt(row.get("fp_ratio_mean", "")),
                fmt(row.get("gv_ratio_mean", "")),
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(parts: list[object]) -> str:
        return "  ".join(str(part).ljust(widths[i]) for i, part in enumerate(parts))

    print(line(headers))
    print(line(["-" * w for w in widths]))
    for row in table_rows:
        print(line(row))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Directory containing .log files")
    parser.add_argument(
        "pattern",
        nargs="?",
        default="*.log",
        help="Glob pattern inside log_dir, for example '*r30*d16*.log'",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path. Defaults to log_dir/summary_<pattern>.csv",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    files = [Path(p) for p in sorted(glob.glob(str(log_dir / args.pattern)))]

    rows = []
    for path in files:
        row = summarize_log(path)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"No completed eval rows found in {log_dir / args.pattern}")
        return 1

    rows.sort(key=lambda r: (str(r.get("radius", "")), int(r.get("d") or 0), str(r.get("backend", ""))))

    print_table(rows)

    csv_path = Path(args.csv) if args.csv else log_dir / "summary.csv"
    fieldnames = [
        "backend",
        "radius",
        "d",
        "epochs",
        "final_epoch",
        "samples",
    ]
    for key in METRICS:
        fieldnames.extend([f"{key}_mean", f"{key}_std", f"{key}_min", f"{key}_max"])
    fieldnames.append("file")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"CSV written to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
