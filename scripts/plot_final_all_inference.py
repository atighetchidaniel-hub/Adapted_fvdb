#!/usr/bin/env python3
"""Create dependency-free SVG plots for final all-scene inference results."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from summarize_final_all_inference import as_float, as_int, enrich_rows, load_rows, newest_summary_csv


DEFAULT_ROOT = Path("/var/tmp") / f"{Path.home().name}_runs" / "final_all_inference"
BACKEND_COLORS = {"fvdb": "#2563EB", "spconv": "#F59E0B"}
GOOD = "#10B981"
BAD = "#EF4444"
GRID = "#E5E7EB"
TEXT = "#111827"
MUTED = "#6B7280"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write SVG plots for final all-scene NeuralPVS inference results.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="final_all_inference root. Default: /var/tmp/$USER_runs/final_all_inference",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Specific original final_all_inference CSV.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: root/summaries/plots_<csv-stem>",
    )
    return parser.parse_args()


def clean_float(value: float) -> float | None:
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    return value


def mean(values: list[float]) -> float:
    vals = [v for v in values if clean_float(v) is not None]
    return sum(vals) / len(vals) if vals else math.nan


def safe_num(value: float | None, default: float = 0.0) -> float:
    return value if value is not None and not math.isnan(value) and not math.isinf(value) else default


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x: float, y: float, content: object, size: int = 12, anchor: str = "start", weight: str = "400", fill: str = TEXT, rotate: float | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-family="Arial, sans-serif" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{transform}>{esc(content)}</text>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", opacity: float = 1.0) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(0, w):.2f}" height="{max(0, h):.2f}" fill="{fill}" stroke="{stroke}" opacity="{opacity:.3f}" />'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = GRID, width: float = 1.0) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width:.2f}" />'


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = "white", width: float = 1.0) -> str:
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.2f}" />'


def save_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "white"),
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def grouped_by_config(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["radius"], as_int(row, "d"), row["backend"])].append(row)
    return grouped


def config_order(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    configs = {(row["radius"], as_int(row, "d")) for row in rows}
    return sorted(configs, key=lambda item: (int(item[0].removeprefix("r")), item[1]))


def metric_mean(items: list[dict[str, str]], key: str) -> float:
    return mean([as_float(row, key) for row in items])


def paired_rows(rows: list[dict[str, str]]) -> list[tuple[tuple[str, str, int], dict[str, str], dict[str, str]]]:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["scene"], row["radius"], as_int(row, "d"))][row["backend"]] = row
    pairs = []
    for key, backends in grouped.items():
        if "fvdb" in backends and "spconv" in backends:
            pairs.append((key, backends["fvdb"], backends["spconv"]))
    return sorted(pairs, key=lambda item: (item[0][0], int(item[0][1].removeprefix("r")), item[0][2]))


def axis_ticks(max_val: float, count: int = 4) -> list[float]:
    if max_val <= 0 or math.isnan(max_val):
        return [0, 1]
    raw = max_val / count
    exponent = math.floor(math.log10(raw)) if raw > 0 else 0
    base = raw / (10**exponent)
    if base <= 1:
        step = 1
    elif base <= 2:
        step = 2
    elif base <= 5:
        step = 5
    else:
        step = 10
    step *= 10**exponent
    top = math.ceil(max_val / step) * step
    return [i * step for i in range(int(round(top / step)) + 1)]


def fmt_tick(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def panel_grouped_bars(body: list[str], x0: float, y0: float, w: float, h: float, title: str, labels: list[str], values_by_backend: dict[str, list[float]], y_label: str) -> None:
    body.append(text(x0, y0 - 12, title, 15, weight="700"))
    body.append(text(x0, y0 + h + 50, y_label, 11, fill=MUTED))
    flat_values = [v for values in values_by_backend.values() for v in values if clean_float(v) is not None]
    max_val = max(flat_values) if flat_values else 1.0
    ticks = axis_ticks(max_val * 1.08)
    top = ticks[-1] if ticks else max_val

    for tick in ticks:
        yy = y0 + h - (tick / top) * h if top else y0 + h
        body.append(line(x0, yy, x0 + w, yy, GRID, 0.8))
        body.append(text(x0 - 8, yy + 4, fmt_tick(tick), 10, anchor="end", fill=MUTED))

    group_w = w / max(1, len(labels))
    bar_w = min(22, group_w * 0.28)
    for i, label in enumerate(labels):
        gx = x0 + i * group_w + group_w / 2
        for j, backend in enumerate(["fvdb", "spconv"]):
            value = clean_float(values_by_backend.get(backend, [math.nan] * len(labels))[i])
            if value is None:
                continue
            bh = (value / top) * h if top else 0
            bx = gx + (j - 0.5) * bar_w - bar_w / 2
            body.append(rect(bx, y0 + h - bh, bar_w, bh, BACKEND_COLORS[backend], opacity=0.92))
        body.append(text(gx, y0 + h + 18, label, 10, anchor="middle", fill=MUTED))

    legend_x = x0 + w - 130
    for j, backend in enumerate(["fvdb", "spconv"]):
        body.append(rect(legend_x + j * 65, y0 - 25, 12, 12, BACKEND_COLORS[backend]))
        body.append(text(legend_x + 17 + j * 65, y0 - 14, backend, 11, fill=MUTED))


def plot_average_metrics(rows: list[dict[str, str]], out_dir: Path) -> Path:
    grouped = grouped_by_config(rows)
    configs = config_order(rows)
    labels = [f"{r} d{d}" for r, d in configs]
    metrics = [
        ("dice_mean", "Average Dice", "higher is better"),
        ("fn_rate_mean", "Average FN rate", "lower is better"),
        ("fp_rate_mean", "Average FP rate", "lower is better"),
        ("infer_time_mean", "Average inference time (ms)", "lower is better"),
    ]
    width, height = 1320, 900
    body = [text(40, 42, "Final Inference Averages Across Scenes", 22, weight="700")]
    positions = [(95, 100), (735, 100), (95, 505), (735, 505)]
    for (key, title, y_label), (x0, y0) in zip(metrics, positions):
        values_by_backend = {}
        for backend in ["fvdb", "spconv"]:
            values_by_backend[backend] = [metric_mean(grouped.get((radius, d, backend), []), key) for radius, d in configs]
        panel_grouped_bars(body, x0, y0, 500, 270, title, labels, values_by_backend, y_label)
    path = out_dir / "final_average_metrics_by_config.svg"
    save_svg(path, width, height, body)
    return path


def panel_delta_bars(body: list[str], x0: float, y0: float, w: float, h: float, title: str, labels: list[str], values: list[float], lower_is_good: bool = True) -> None:
    body.append(text(x0, y0 - 14, title, 15, weight="700"))
    vals = [v for v in values if clean_float(v) is not None]
    if not vals:
        vals = [0.0]
    max_abs = max(abs(v) for v in vals) * 1.15
    if max_abs == 0:
        max_abs = 1.0
    zero_y = y0 + h / 2
    body.append(line(x0, zero_y, x0 + w, zero_y, TEXT, 0.8))
    for tick in [-max_abs, -max_abs / 2, 0, max_abs / 2, max_abs]:
        yy = zero_y - (tick / max_abs) * (h / 2)
        body.append(line(x0, yy, x0 + w, yy, GRID, 0.7))
        body.append(text(x0 - 8, yy + 4, fmt_tick(tick), 10, anchor="end", fill=MUTED))
    group_w = w / max(1, len(labels))
    bar_w = max(4, min(14, group_w * 0.65))
    for i, value in enumerate(values):
        if clean_float(value) is None:
            continue
        gx = x0 + i * group_w + group_w / 2
        y = zero_y - (value / max_abs) * (h / 2)
        top_y = min(y, zero_y)
        bh = abs(y - zero_y)
        good = value < 0 if lower_is_good else value > 0
        body.append(rect(gx - bar_w / 2, top_y, bar_w, bh, GOOD if good else BAD, opacity=0.9))
    step = max(1, math.ceil(len(labels) / 12))
    for i, label in enumerate(labels):
        if i % step == 0:
            gx = x0 + i * group_w + group_w / 2
            body.append(text(gx - 4, y0 + h + 18, label, 9, anchor="end", fill=MUTED, rotate=-60))


def plot_pair_deltas(rows: list[dict[str, str]], out_dir: Path) -> Path:
    pairs = paired_rows(rows)
    labels = [f"{scene} {radius} d{d}" for (scene, radius, d), _fv, _sp in pairs]
    fn_delta = [as_float(fv, "fn_rate_mean") - as_float(sp, "fn_rate_mean") for _key, fv, sp in pairs]
    fp_delta = [as_float(fv, "fp_rate_mean") - as_float(sp, "fp_rate_mean") for _key, fv, sp in pairs]
    time_delta = [as_float(fv, "infer_time_mean") - as_float(sp, "infer_time_mean") for _key, fv, sp in pairs]
    width = max(1400, 60 * len(labels))
    height = 980
    body = [text(40, 42, "Paired Backend Deltas", 22, weight="700"), text(40, 66, "Delta = fVDB - spconv. Green favors fVDB; red favors spconv.", 12, fill=MUTED)]
    panel_delta_bars(body, 100, 120, width - 160, 210, "FN rate delta", labels, fn_delta)
    panel_delta_bars(body, 100, 430, width - 160, 210, "FP rate delta", labels, fp_delta)
    panel_delta_bars(body, 100, 740, width - 160, 150, "Inference time delta (ms)", labels, time_delta)
    path = out_dir / "final_fvdb_spconv_pair_deltas.svg"
    save_svg(path, width, height, body)
    return path


def plot_speed_vs_fn(rows: list[dict[str, str]], out_dir: Path) -> Path:
    width, height = 980, 650
    x0, y0, w, h = 90, 85, 800, 465
    body = [text(40, 42, "Speed vs False-Negative Rate", 22, weight="700")]
    xs = [as_float(row, "infer_time_mean") for row in rows]
    ys = [as_float(row, "fn_rate_mean") for row in rows]
    x_max = max([x for x in xs if clean_float(x) is not None] or [1]) * 1.1
    y_max = max([y for y in ys if clean_float(y) is not None] or [1]) * 1.15
    for tick in axis_ticks(x_max):
        xx = x0 + (tick / x_max) * w
        body.append(line(xx, y0, xx, y0 + h, GRID, 0.7))
        body.append(text(xx, y0 + h + 20, fmt_tick(tick), 10, anchor="middle", fill=MUTED))
    for tick in axis_ticks(y_max):
        yy = y0 + h - (tick / y_max) * h
        body.append(line(x0, yy, x0 + w, yy, GRID, 0.7))
        body.append(text(x0 - 8, yy + 4, fmt_tick(tick), 10, anchor="end", fill=MUTED))
    body.append(line(x0, y0 + h, x0 + w, y0 + h, TEXT, 0.9))
    body.append(line(x0, y0, x0, y0 + h, TEXT, 0.9))
    body.append(text(x0 + w / 2, y0 + h + 52, "Inference time (ms)", 12, anchor="middle"))
    body.append(text(24, y0 + h / 2, "FN rate", 12, anchor="middle", rotate=-90))
    marker_radius = {8: 5, 16: 7, 32: 9}
    for row in rows:
        xv = clean_float(as_float(row, "infer_time_mean"))
        yv = clean_float(as_float(row, "fn_rate_mean"))
        if xv is None or yv is None:
            continue
        xx = x0 + (xv / x_max) * w
        yy = y0 + h - (yv / y_max) * h
        backend = row["backend"]
        d = as_int(row, "d")
        body.append(circle(xx, yy, marker_radius.get(d, 6), BACKEND_COLORS[backend], width=1.2))
    body.append(rect(705, 92, 160, 84, "white", GRID, 0.95))
    for i, backend in enumerate(["fvdb", "spconv"]):
        body.append(circle(724, 118 + i * 24, 6, BACKEND_COLORS[backend]))
        body.append(text(740, 123 + i * 24, backend, 11, fill=MUTED))
    body.append(text(724, 170, "circle size: d8/d16/d32", 10, fill=MUTED))
    path = out_dir / "final_speed_vs_fn_rate.svg"
    save_svg(path, width, height, body)
    return path


def plot_fn_by_scene(rows: list[dict[str, str]], out_dir: Path) -> Path:
    sorted_rows = sorted(rows, key=lambda r: (r["scene"], int(r["radius"].removeprefix("r")), as_int(r, "d"), r["backend"]))
    labels = [f"{r['scene']} {r['radius']} d{as_int(r, 'd')} {r['backend']}" for r in sorted_rows]
    values = [as_float(r, "fn_rate_mean") for r in sorted_rows]
    width = 1120
    row_h = 22
    height = max(500, 110 + row_h * len(labels))
    x0, y0, w = 300, 75, 720
    max_val = max([v for v in values if clean_float(v) is not None] or [1]) * 1.12
    body = [text(40, 42, "False-Negative Rate By Scene and Configuration", 22, weight="700")]
    for tick in axis_ticks(max_val):
        xx = x0 + (tick / max_val) * w
        body.append(line(xx, y0, xx, y0 + row_h * len(labels), GRID, 0.7))
        body.append(text(xx, y0 - 8, fmt_tick(tick), 10, anchor="middle", fill=MUTED))
    for i, (label, value) in enumerate(zip(labels, values)):
        yy = y0 + i * row_h
        backend = "fvdb" if label.endswith("fvdb") else "spconv"
        body.append(text(x0 - 10, yy + 15, label, 10, anchor="end", fill=MUTED))
        if clean_float(value) is None:
            continue
        bw = (value / max_val) * w
        body.append(rect(x0, yy + 4, bw, 13, BACKEND_COLORS[backend], opacity=0.85))
        body.append(text(x0 + bw + 5, yy + 15, f"{value:.3f}", 9, fill=MUTED))
    path = out_dir / "final_fn_rate_by_scene_config.svg"
    save_svg(path, width, height, body)
    return path


def write_average_csv(rows: list[dict[str, str]], out_dir: Path) -> Path:
    grouped = grouped_by_config(rows)
    path = out_dir / "final_average_metrics_by_config.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["radius", "d", "backend", "scenes", "dice_mean", "fp_rate_mean", "fn_rate_mean", "infer_time_mean", "pure_time_mean"])
        for radius, d in config_order(rows):
            for backend in ["fvdb", "spconv"]:
                items = grouped.get((radius, d, backend), [])
                if not items:
                    continue
                writer.writerow([
                    radius,
                    d,
                    backend,
                    len(items),
                    f"{metric_mean(items, 'dice_mean'):.8f}",
                    f"{metric_mean(items, 'fp_rate_mean'):.8f}",
                    f"{metric_mean(items, 'fn_rate_mean'):.8f}",
                    f"{metric_mean(items, 'infer_time_mean'):.3f}",
                    f"{metric_mean(items, 'infer_time_pure_mean'):.3f}",
                ])
    return path


def main() -> None:
    args = parse_args()
    csv_path = args.csv or newest_summary_csv(args.root)
    raw_rows = load_rows(csv_path)
    if raw_rows and "dice_mean" not in raw_rows[0]:
        raise SystemExit(f"{csv_path} looks like a generated compact CSV. Use the original final_all_inference CSV.")
    for row in raw_rows:
        if "metrics_output_dir" not in row and row.get("output_dir"):
            row["metrics_output_dir"] = row["output_dir"]
    rows = enrich_rows(raw_rows)
    out_dir = args.out_dir or (args.root / "summaries" / f"plots_{csv_path.stem}")

    written = [
        plot_average_metrics(rows, out_dir),
        plot_pair_deltas(rows, out_dir),
        plot_speed_vs_fn(rows, out_dir),
        plot_fn_by_scene(rows, out_dir),
        write_average_csv(rows, out_dir),
    ]

    print("Wrote graph files:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
