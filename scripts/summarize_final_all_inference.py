#!/usr/bin/env python3
"""Summarize final all-scene inference CSVs into thesis-friendly tables."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path("/var/tmp") / f"{Path.home().name}_runs" / "final_all_inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final all-scene NeuralPVS inference results.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="final_all_inference root. Default: /var/tmp/$USER_runs/final_all_inference",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Specific final_all_inference CSV. Default: newest CSV under root/summaries.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional Markdown output path. Default: root/summaries/<csv-stem>_nice_summary.md",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional compact CSV output path. Default: root/summaries/<csv-stem>_compact.csv",
    )
    return parser.parse_args()


def newest_summary_csv(root: Path) -> Path:
    candidates = sorted((root / "summaries").glob("final_all_inference_*ep_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No final_all_inference CSV found under {root / 'summaries'}")
    return candidates[-1]


def as_float(row: dict[str, str], key: str) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0")))
    except Exception:
        return 0


def fmt(value: float, places: int = 6) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.{places}f}"


def fmt_ms(value: float) -> str:
    return fmt(value, 3)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def compact_record(row: dict[str, str]) -> dict[str, str]:
    return {
        "scene": row["scene"],
        "radius": row["radius"],
        "d": str(as_int(row, "d")),
        "backend": row["backend"],
        "frames": str(as_int(row, "frames")),
        "dice": fmt(as_float(row, "dice_mean")),
        "fp_rate": fmt(as_float(row, "fp_rate_mean")),
        "fn_rate": fmt(as_float(row, "fn_rate_mean")),
        "fp_ratio": fmt(as_float(row, "fp_ratio_mean")),
        "gv_ratio": fmt(as_float(row, "gv_ratio_mean")),
        "infer_ms": fmt_ms(as_float(row, "infer_time_mean")),
        "pure_ms": fmt_ms(as_float(row, "infer_time_pure_mean")),
        "peak_mb": fmt(as_float(row, "peak_mem"), 1),
        "predicted_pvv_folder": row.get("predicted_pvv_folder", ""),
    }


def sort_key(row: dict[str, str]) -> tuple[str, int, int, str]:
    backend_order = {"fvdb": 0, "spconv": 1}
    return (row["scene"], int(row["radius"].removeprefix("r")), int(row["d"]), str(backend_order.get(row["backend"], 9)))


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def make_main_table(compact: list[dict[str, str]]) -> list[str]:
    table_rows = []
    for row in compact:
        table_rows.append(
            [
                row["scene"],
                row["radius"],
                f"d{row['d']}",
                row["backend"],
                row["frames"],
                row["dice"],
                row["fp_rate"],
                row["fn_rate"],
                row["fp_ratio"],
                row["gv_ratio"],
                row["infer_ms"],
                row["pure_ms"],
                row["peak_mb"],
            ]
        )
    return markdown_table(
        [
            "Scene",
            "Radius",
            "d",
            "Backend",
            "Frames",
            "Dice",
            "FP rate",
            "FN rate",
            "FP ratio",
            "GV ratio",
            "Infer ms",
            "Pure ms",
            "Peak MB",
        ],
        table_rows,
    )


def make_pair_table(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["scene"], row["radius"], as_int(row, "d"))
        grouped[key][row["backend"]] = row

    pair_rows = []
    for (scene, radius, d), backends in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1].removeprefix("r")), item[0][2])):
        if "fvdb" not in backends or "spconv" not in backends:
            continue
        fv = backends["fvdb"]
        sp = backends["spconv"]

        fv_fn = as_float(fv, "fn_rate_mean")
        sp_fn = as_float(sp, "fn_rate_mean")
        fv_fp = as_float(fv, "fp_rate_mean")
        sp_fp = as_float(sp, "fp_rate_mean")
        fv_dice = as_float(fv, "dice_mean")
        sp_dice = as_float(sp, "dice_mean")
        fv_time = as_float(fv, "infer_time_mean")
        sp_time = as_float(sp, "infer_time_mean")
        time_drop = 100.0 * (sp_time - fv_time) / sp_time if sp_time and not math.isnan(sp_time) else math.nan

        pair_rows.append(
            [
                scene,
                radius,
                f"d{d}",
                fmt(fv_dice - sp_dice),
                fmt(fv_fp - sp_fp),
                fmt(fv_fn - sp_fn),
                fmt(fv_time - sp_time, 3),
                fmt(time_drop, 2),
                "fVDB" if fv_fn < sp_fn else "spconv",
                "fVDB" if fv_time < sp_time else "spconv",
            ]
        )

    if not pair_rows:
        return ["No fVDB/spconv pairs found."]

    return markdown_table(
        [
            "Scene",
            "Radius",
            "d",
            "Dice Δ fVDB-spconv",
            "FP rate Δ",
            "FN rate Δ",
            "Infer ms Δ",
            "fVDB speedup %",
            "Lower FN",
            "Faster",
        ],
        pair_rows,
    )


def make_averages_table(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["radius"], as_int(row, "d"), row["backend"])].append(row)

    avg_rows = []
    for (radius, d, backend), items in sorted(grouped.items(), key=lambda item: (int(item[0][0].removeprefix("r")), item[0][1], item[0][2])):
        avg_rows.append(
            [
                radius,
                f"d{d}",
                backend,
                str(len(items)),
                fmt(sum(as_float(r, "dice_mean") for r in items) / len(items)),
                fmt(sum(as_float(r, "fp_rate_mean") for r in items) / len(items)),
                fmt(sum(as_float(r, "fn_rate_mean") for r in items) / len(items)),
                fmt_ms(sum(as_float(r, "infer_time_mean") for r in items) / len(items)),
                fmt_ms(sum(as_float(r, "infer_time_pure_mean") for r in items) / len(items)),
            ]
        )

    return markdown_table(
        ["Radius", "d", "Backend", "Scenes", "Dice", "FP rate", "FN rate", "Infer ms", "Pure ms"],
        avg_rows,
    )


def write_compact_csv(path: Path, compact: list[dict[str, str]]) -> None:
    fieldnames = [
        "scene",
        "radius",
        "d",
        "backend",
        "frames",
        "dice",
        "fp_rate",
        "fn_rate",
        "fp_ratio",
        "gv_ratio",
        "infer_ms",
        "pure_ms",
        "peak_mb",
        "predicted_pvv_folder",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compact)


def main() -> None:
    args = parse_args()
    csv_path = args.csv or newest_summary_csv(args.root)
    out_md = args.out or csv_path.with_name(csv_path.stem + "_nice_summary.md")
    out_csv = args.csv_out or csv_path.with_name(csv_path.stem + "_compact.csv")

    rows = sorted(load_rows(csv_path), key=sort_key)
    compact = [compact_record(row) for row in rows]
    write_compact_csv(out_csv, compact)

    lines = []
    lines.append("# Final All-Scene Inference Nice Summary")
    lines.append("")
    lines.append(f"Source CSV: `{csv_path}`")
    lines.append("")
    lines.append(f"Rows: **{len(rows)}**")
    lines.append("")
    lines.append("## Main Table")
    lines.append("")
    lines.extend(make_main_table(compact))
    lines.append("")
    lines.append("## fVDB vs spconv Deltas")
    lines.append("")
    lines.append("Negative deltas mean fVDB is lower. Positive speedup means fVDB is faster.")
    lines.append("")
    lines.extend(make_pair_table(rows))
    lines.append("")
    lines.append("## Averages By Radius/d/Backend")
    lines.append("")
    lines.extend(make_averages_table(rows))
    lines.append("")

    out_md.write_text("\n".join(lines))

    print("\n".join(lines))
    print()
    print(f"Wrote Markdown: {out_md}")
    print(f"Wrote compact CSV: {out_csv}")


if __name__ == "__main__":
    main()
