#!/usr/bin/env python3
"""Summarize completed d=16 200-epoch fVDB/spconv training logs.

Default result root:
  /var/tmp/$USER_runs/NEURALPVS_MAY19_SYNTHETIC_D16_200EP
"""

from __future__ import annotations

import argparse
import csv
import os
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

EXPECTED = [
    ("r30", "fvdb"),
    ("r30", "spconv"),
    ("r60", "fvdb"),
    ("r60", "spconv"),
    ("r90", "fvdb"),
    ("r90", "spconv"),
]


def parse_metric(line: str, key: str) -> float | None:
    match = re.search(rf"(?:^|\s|\|){re.escape(key)}:\s*([0-9.eE+-]+)", line)
    return float(match.group(1)) if match else None


def infer_setting(path: Path) -> dict[str, str]:
    name = path.name
    radius = re.search(r"_(r[0-9]+)_", name)
    backend = re.search(r"_(fvdb|spconv)_", name)
    d_value = re.search(r"_d([0-9]+)_", name)
    epochs = re.search(r"_([0-9]+)ep(?:\.|_)", name)
    return {
        "radius": radius.group(1) if radius else "",
        "backend": backend.group(1) if backend else "",
        "d": d_value.group(1) if d_value else "",
        "epochs": epochs.group(1) if epochs else "",
        "log": str(path),
    }


def summarize_log(path: Path) -> dict[str, object] | None:
    text = path.read_text(errors="replace")
    if "Training finished" not in text:
        return None

    lines = text.splitlines()
    eval_lines = [line.strip() for line in lines if "[eval]" in line]
    if not eval_lines:
        return None

    epochs = []
    for line in eval_lines:
        match = re.search(r"epoch:\s*([0-9]+)", line)
        if match:
            epochs.append(int(match.group(1)))
    if not epochs:
        return None

    final_epoch = max(epochs)
    final_lines = [line for line in eval_lines if f"epoch: {final_epoch}" in line]

    batch_times = []
    for line in lines:
        match = re.search(r"Average batch time:\s*([0-9.eE+-]+)", line)
        if match:
            batch_times.append(float(match.group(1)))

    row: dict[str, object] = infer_setting(path)
    row["final_epoch"] = final_epoch
    row["samples"] = len(final_lines)
    row["batch_time_entries"] = len(batch_times)
    row["batch_time_first"] = batch_times[0] if batch_times else ""
    row["batch_time_last"] = batch_times[-1] if batch_times else ""
    row["batch_time_mean"] = stats.mean(batch_times) if batch_times else ""
    row["batch_time_median"] = stats.median(batch_times) if batch_times else ""

    for key in METRICS:
        values = [parse_metric(line, key) for line in final_lines]
        values = [value for value in values if value is not None]
        row[f"{key}_mean"] = stats.mean(values) if values else ""
        row[f"{key}_std"] = stats.pstdev(values) if len(values) > 1 else (0.0 if values else "")
        row[f"{key}_min"] = min(values) if values else ""
        row[f"{key}_max"] = max(values) if values else ""

    return row


def fmt(value: object, digits: int = 6) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: list[dict[str, object]]) -> None:
    headers = [
        "radius",
        "backend",
        "epoch",
        "samples",
        "dice",
        "fp_rate",
        "fn_rate",
        "fp_ratio",
        "gv_ratio",
        "batch_med_s",
        "batch_last_s",
    ]
    table = []
    for row in rows:
        table.append(
            [
                row.get("radius", ""),
                row.get("backend", ""),
                row.get("final_epoch", ""),
                row.get("samples", ""),
                fmt(row.get("dice_mean", "")),
                fmt(row.get("fp_rate_mean", "")),
                fmt(row.get("fn_rate_mean", "")),
                fmt(row.get("fp_ratio_mean", "")),
                fmt(row.get("gv_ratio_mean", "")),
                fmt(row.get("batch_time_median", "")),
                fmt(row.get("batch_time_last", "")),
            ]
        )

    widths = [len(header) for header in headers]
    for cells in table:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(str(cell)))

    def line(cells: list[object]) -> str:
        return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells))

    print(line(headers))
    print(line(["-" * width for width in widths]))
    for cells in table:
        print(line(cells))


def main() -> int:
    user = os.environ.get("USER", "atighedl")
    default_root = Path("/var/tmp") / f"{user}_runs" / "NEURALPVS_MAY19_SYNTHETIC_D16_200EP"

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=default_root)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    log_dir = args.result_root / "logs"
    summary_dir = args.result_root / "summaries"
    csv_path = args.csv or summary_dir / "d16_200ep_finished_training_summary.csv"

    print(f"Result root: {args.result_root}")
    print(f"Log dir:     {log_dir}")
    print()

    print("STATUS")
    rows = []
    for radius, backend in EXPECTED:
        log = log_dir / f"synthetic_may19_{radius}_{backend}_d16_b3_depth3_200ep.log"
        if not log.exists():
            print(f"MISSING     {radius} d=16 {backend}")
            continue
        if "Training finished" not in log.read_text(errors="replace"):
            epoch_match = None
            for epoch_match in re.finditer(r"epoch:\s*([0-9]+)", log.read_text(errors="replace")):
                pass
            epoch = epoch_match.group(1) if epoch_match else "unknown"
            print(f"PARTIAL     {radius} d=16 {backend} epoch={epoch}")
            continue
        print(f"DONE        {radius} d=16 {backend}")
        row = summarize_log(log)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nNo completed logs found.")
        return 1

    rows.sort(key=lambda row: (str(row.get("radius", "")), str(row.get("backend", ""))))

    print()
    print("COMPLETED TRAINING SUMMARY")
    print_table(rows)

    print()
    print("CSV SUMMARY TO PASTE")
    csv_headers = [
        "radius",
        "backend",
        "epoch",
        "samples",
        "dice_mean",
        "loss_mean",
        "fp_rate_mean",
        "fn_rate_mean",
        "fp_ratio_mean",
        "gv_ratio_mean",
        "batch_time_median",
        "batch_time_last",
    ]
    print(",".join(csv_headers))
    for row in rows:
        print(
            ",".join(
                [
                    str(row.get("radius", "")),
                    str(row.get("backend", "")),
                    str(row.get("final_epoch", "")),
                    str(row.get("samples", "")),
                    fmt(row.get("dice_mean", ""), 8),
                    fmt(row.get("loss_mean", ""), 8),
                    fmt(row.get("fp_rate_mean", ""), 8),
                    fmt(row.get("fn_rate_mean", ""), 8),
                    fmt(row.get("fp_ratio_mean", ""), 8),
                    fmt(row.get("gv_ratio_mean", ""), 8),
                    fmt(row.get("batch_time_median", ""), 8),
                    fmt(row.get("batch_time_last", ""), 8),
                ]
            )
        )

    summary_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "radius",
        "backend",
        "d",
        "epochs",
        "final_epoch",
        "samples",
        "batch_time_entries",
        "batch_time_first",
        "batch_time_last",
        "batch_time_mean",
        "batch_time_median",
    ]
    for key in METRICS:
        fieldnames.extend([f"{key}_mean", f"{key}_std", f"{key}_min", f"{key}_max"])
    fieldnames.append("log")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"CSV written to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
