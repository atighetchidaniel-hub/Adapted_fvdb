#!/usr/bin/env python3
"""Create side-by-side fVDB/spconv PVS render error figures.

The script compares two predicted render videos against one ground-truth render
video. It computes per-frame pixel error rate (PER) and a lightweight SSIM
value, selects a mixture of best and worst PER frames, and writes a paper-style
figure with GT, fVDB error overlay, and spconv error overlay rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float | None
    nb_frames: int | None


@dataclass
class Metric:
    per_percent: float
    ssim: float
    error_pixels: int
    total_pixels: int


@dataclass
class FrameRecord:
    frame: int
    fvdb: Metric
    spconv: Metric

    @property
    def combined_per(self) -> float:
        return 0.5 * (self.fvdb.per_percent + self.spconv.per_percent)

    @property
    def max_per(self) -> float:
        return max(self.fvdb.per_percent, self.spconv.per_percent)


@dataclass
class SelectedRecord:
    label: str
    record: FrameRecord
    gt: np.ndarray
    fvdb_overlay: np.ndarray
    spconv_overlay: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fVDB and spconv rendered PVS videos against GT and draw red error overlays."
    )
    parser.add_argument("--gt-video", type=Path, required=True)
    parser.add_argument("--fvdb-video", type=Path, required=True)
    parser.add_argument("--spconv-video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="RGB difference threshold in 0..255 units. A pixel is wrong if max channel diff exceeds this.",
    )
    parser.add_argument("--best-count", type=int, default=3, help="Number of lowest-PER frames to include.")
    parser.add_argument("--worst-count", type=int, default=3, help="Number of highest-PER frames to include.")
    parser.add_argument(
        "--frame-ids",
        type=str,
        default="",
        help="Comma-separated frame ids. Overrides automatic best/worst selection.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("combined", "max", "fvdb", "spconv"),
        default="combined",
        help="Metric used to select automatic best/worst frames.",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=30,
        help="Minimum frame gap for automatic selection to avoid near-duplicate frames.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Only evaluate every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional max compared frames; 0 means full video.")
    parser.add_argument("--thumb-width", type=int, default=330, help="Thumbnail width in the final figure.")
    parser.add_argument("--bbox", action="store_true", help="Draw one red bbox around all error pixels.")
    parser.add_argument("--title", default="", help="Optional figure title.")
    parser.add_argument("--caption", default="", help="Optional caption under the figure.")
    parser.add_argument("--ffmpeg", default="", help="Optional path to ffmpeg.")
    parser.add_argument("--ffprobe", default="", help="Optional path to ffprobe.")
    return parser.parse_args()


def require_tool(name: str, explicit: str = "") -> str:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"{name} not found at {explicit}")
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} was not found on PATH")
    return found


def run_json(cmd: list[str]) -> dict:
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(result.stdout)


def probe_video(video: Path, ffprobe: str) -> VideoInfo:
    data = run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,duration",
            "-of",
            "json",
            str(video),
        ]
    )
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {video}")
    stream = streams[0]
    nb_frames_raw = stream.get("nb_frames")
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration=float(stream["duration"]) if stream.get("duration") else None,
        nb_frames=int(nb_frames_raw) if nb_frames_raw and str(nb_frames_raw).isdigit() else None,
    )


def start_raw_reader(video: Path, width: int, height: int, ffmpeg: str) -> subprocess.Popen[bytes]:
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_frame(proc: subprocess.Popen[bytes], width: int, height: int) -> np.ndarray | None:
    assert proc.stdout is not None
    frame_bytes = width * height * 3
    data = proc.stdout.read(frame_bytes)
    if len(data) == 0:
        return None
    if len(data) != frame_bytes:
        return None
    return np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))


def cleanup_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.kill()
    try:
        proc.communicate(timeout=2)
    except Exception:
        pass


def grayscale(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    return 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]


def global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    x = grayscale(a)
    y = grayscale(b)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = float(x.mean())
    mu_y = float(y.mean())
    var_x = float(((x - mu_x) ** 2).mean())
    var_y = float(((y - mu_y) ** 2).mean())
    cov_xy = float(((x - mu_x) * (y - mu_y)).mean())
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return numerator / denominator if denominator else 1.0


def metric_for(gt: np.ndarray, pred: np.ndarray, threshold: float) -> tuple[Metric, np.ndarray]:
    diff = np.max(np.abs(pred.astype(np.int16) - gt.astype(np.int16)), axis=2)
    mask = diff > threshold
    error_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    metric = Metric(
        per_percent=100.0 * error_pixels / total_pixels,
        ssim=global_ssim(pred, gt),
        error_pixels=error_pixels,
        total_pixels=total_pixels,
    )
    return metric, mask


def overlay_errors(pred: np.ndarray, mask: np.ndarray, draw_bbox: bool) -> np.ndarray:
    overlay = np.array(pred, copy=True)
    overlay[mask] = np.array([255, 0, 0], dtype=np.uint8)
    if draw_bbox and int(mask.sum()):
        ys, xs = np.nonzero(mask)
        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            outline=(255, 0, 0),
            width=max(2, min(pred.shape[0], pred.shape[1]) // 180),
        )
        overlay = np.array(image)
    return overlay


def score(record: FrameRecord, metric_name: str) -> float:
    if metric_name == "combined":
        return record.combined_per
    if metric_name == "max":
        return record.max_per
    if metric_name == "fvdb":
        return record.fvdb.per_percent
    if metric_name == "spconv":
        return record.spconv.per_percent
    raise ValueError(metric_name)


def parse_frame_ids(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def far_enough(frame: int, selected: Iterable[int], min_gap: int) -> bool:
    return all(abs(frame - other) >= min_gap for other in selected)


def pick_with_gap(
    candidates: list[FrameRecord],
    count: int,
    selected: list[int],
    min_gap: int,
) -> list[FrameRecord]:
    picked: list[FrameRecord] = []
    for candidate in candidates:
        if candidate.frame in selected:
            continue
        if far_enough(candidate.frame, selected, min_gap):
            picked.append(candidate)
            selected.append(candidate.frame)
        if len(picked) == count:
            return picked

    # Relax the gap only if there are not enough separated frames.
    for candidate in candidates:
        if candidate.frame in selected:
            continue
        picked.append(candidate)
        selected.append(candidate.frame)
        if len(picked) == count:
            break
    return picked


def select_records(
    records: list[FrameRecord],
    frame_ids: list[int],
    selection_metric: str,
    best_count: int,
    worst_count: int,
    min_gap: int,
) -> list[tuple[str, FrameRecord]]:
    if frame_ids:
        by_frame = {record.frame: record for record in records}
        missing = [frame_id for frame_id in frame_ids if frame_id not in by_frame]
        if missing:
            raise RuntimeError(f"Requested frame ids were not available: {missing}")
        return [(f"manual {i + 1}", by_frame[frame_id]) for i, frame_id in enumerate(frame_ids)]

    by_low = sorted(records, key=lambda record: score(record, selection_metric))
    by_high = sorted(records, key=lambda record: score(record, selection_metric), reverse=True)

    selected_frames: list[int] = []
    best = pick_with_gap(by_low, best_count, selected_frames, min_gap)
    worst = pick_with_gap(by_high, worst_count, selected_frames, min_gap)

    labeled: list[tuple[str, FrameRecord]] = []
    labeled.extend((f"best {i + 1}", record) for i, record in enumerate(best))
    labeled.extend((f"worst {i + 1}", record) for i, record in enumerate(worst))
    return labeled


def compare_metrics(
    gt_video: Path,
    fvdb_video: Path,
    spconv_video: Path,
    width: int,
    height: int,
    ffmpeg: str,
    threshold: float,
    stride: int,
    max_frames: int,
) -> list[FrameRecord]:
    gt_proc = start_raw_reader(gt_video, width, height, ffmpeg)
    fvdb_proc = start_raw_reader(fvdb_video, width, height, ffmpeg)
    spconv_proc = start_raw_reader(spconv_video, width, height, ffmpeg)
    records: list[FrameRecord] = []
    frame_id = 0
    try:
        while True:
            gt = read_frame(gt_proc, width, height)
            fvdb = read_frame(fvdb_proc, width, height)
            spconv = read_frame(spconv_proc, width, height)
            if gt is None or fvdb is None or spconv is None:
                break

            if frame_id % stride == 0:
                fvdb_metric, _ = metric_for(gt, fvdb, threshold)
                spconv_metric, _ = metric_for(gt, spconv, threshold)
                records.append(FrameRecord(frame=frame_id, fvdb=fvdb_metric, spconv=spconv_metric))
                if len(records) % 100 == 0:
                    print(f"Compared {len(records)} frames...", flush=True)

            frame_id += 1
            if max_frames and frame_id >= max_frames:
                break
    finally:
        cleanup_process(gt_proc)
        cleanup_process(fvdb_proc)
        cleanup_process(spconv_proc)

    if not records:
        raise RuntimeError("No frames were compared")
    return records


def extract_selected_frames(
    selected: list[tuple[str, FrameRecord]],
    gt_video: Path,
    fvdb_video: Path,
    spconv_video: Path,
    width: int,
    height: int,
    ffmpeg: str,
    threshold: float,
    draw_bbox: bool,
) -> list[SelectedRecord]:
    needed = {record.frame for _label, record in selected}
    label_by_frame = {record.frame: label for label, record in selected}
    record_by_frame = {record.frame: record for _label, record in selected}

    gt_proc = start_raw_reader(gt_video, width, height, ffmpeg)
    fvdb_proc = start_raw_reader(fvdb_video, width, height, ffmpeg)
    spconv_proc = start_raw_reader(spconv_video, width, height, ffmpeg)
    extracted: dict[int, SelectedRecord] = {}
    frame_id = 0
    try:
        while needed - extracted.keys():
            gt = read_frame(gt_proc, width, height)
            fvdb = read_frame(fvdb_proc, width, height)
            spconv = read_frame(spconv_proc, width, height)
            if gt is None or fvdb is None or spconv is None:
                break

            if frame_id in needed:
                _fvdb_metric, fvdb_mask = metric_for(gt, fvdb, threshold)
                _spconv_metric, spconv_mask = metric_for(gt, spconv, threshold)
                extracted[frame_id] = SelectedRecord(
                    label=label_by_frame[frame_id],
                    record=record_by_frame[frame_id],
                    gt=np.array(gt, copy=True),
                    fvdb_overlay=overlay_errors(fvdb, fvdb_mask, draw_bbox),
                    spconv_overlay=overlay_errors(spconv, spconv_mask, draw_bbox),
                )
            frame_id += 1
    finally:
        cleanup_process(gt_proc)
        cleanup_process(fvdb_proc)
        cleanup_process(spconv_proc)

    missing = sorted(needed - extracted.keys())
    if missing:
        raise RuntimeError(f"Could not extract selected frames: {missing}")
    return [extracted[record.frame] for _label, record in selected]


def write_all_csv(path: Path, records: list[FrameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "fvdb_per_percent",
                "fvdb_ssim",
                "fvdb_error_pixels",
                "spconv_per_percent",
                "spconv_ssim",
                "spconv_error_pixels",
                "per_delta_fvdb_minus_spconv",
                "ssim_delta_fvdb_minus_spconv",
                "combined_per_percent",
                "max_per_percent",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.frame,
                    f"{record.fvdb.per_percent:.8f}",
                    f"{record.fvdb.ssim:.8f}",
                    record.fvdb.error_pixels,
                    f"{record.spconv.per_percent:.8f}",
                    f"{record.spconv.ssim:.8f}",
                    record.spconv.error_pixels,
                    f"{record.fvdb.per_percent - record.spconv.per_percent:.8f}",
                    f"{record.fvdb.ssim - record.spconv.ssim:.8f}",
                    f"{record.combined_per:.8f}",
                    f"{record.max_per:.8f}",
                ]
            )


def write_selected_csv(path: Path, selected: list[SelectedRecord]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "frame",
                "fvdb_per_percent",
                "fvdb_ssim",
                "spconv_per_percent",
                "spconv_ssim",
                "per_delta_fvdb_minus_spconv",
                "combined_per_percent",
            ]
        )
        for item in selected:
            record = item.record
            writer.writerow(
                [
                    item.label,
                    record.frame,
                    f"{record.fvdb.per_percent:.8f}",
                    f"{record.fvdb.ssim:.8f}",
                    f"{record.spconv.per_percent:.8f}",
                    f"{record.spconv.ssim:.8f}",
                    f"{record.fvdb.per_percent - record.spconv.per_percent:.8f}",
                    f"{record.combined_per:.8f}",
                ]
            )


def resize_for_figure(frame: np.ndarray, thumb_width: int) -> np.ndarray:
    image = Image.fromarray(frame)
    ratio = thumb_width / image.width
    thumb_height = max(1, round(image.height * ratio))
    return np.array(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS))


def write_frame_pngs(selected: list[SelectedRecord], out: Path, thumb_width: int) -> Path:
    frame_dir = out.with_suffix("").parent / (out.with_suffix("").name + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        frame = item.record.frame
        Image.fromarray(resize_for_figure(item.gt, thumb_width)).save(frame_dir / f"frame_{frame:06d}_gt.png")
        Image.fromarray(resize_for_figure(item.fvdb_overlay, thumb_width)).save(
            frame_dir / f"frame_{frame:06d}_fvdb_overlay.png"
        )
        Image.fromarray(resize_for_figure(item.spconv_overlay, thumb_width)).save(
            frame_dir / f"frame_{frame:06d}_spconv_overlay.png"
        )
    return frame_dir


def make_figure(selected: list[SelectedRecord], out: Path, title: str, caption: str, thumb_width: int) -> list[Path]:
    cols = len(selected)
    rows = 3
    fig_width = max(10.0, cols * 1.95)
    fig_height = 6.2 + (0.4 if title else 0.0) + (0.5 if caption else 0.0)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "text.color": "#000000",
        }
    )

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    row_names = ("GT", "fVDB", "spconv")

    for col, item in enumerate(selected):
        images = (item.gt, item.fvdb_overlay, item.spconv_overlay)
        for row, image in enumerate(images):
            ax = axes[row, col]
            ax.imshow(resize_for_figure(image, thumb_width))
            ax.set_axis_off()
            if col == 0:
                ax.text(
                    -0.06,
                    0.5,
                    row_names[row],
                    ha="right",
                    va="center",
                    rotation=90,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                )

        axes[0, col].set_title(f"{item.label}\nframe {item.record.frame}", fontsize=10)
        axes[1, col].text(
            0.5,
            -0.06,
            f"PER={item.record.fvdb.per_percent:.4g}%\nSSIM={item.record.fvdb.ssim:.4f}",
            ha="center",
            va="top",
            transform=axes[1, col].transAxes,
            fontsize=8.5,
        )
        axes[2, col].text(
            0.5,
            -0.06,
            f"PER={item.record.spconv.per_percent:.4g}%\nSSIM={item.record.spconv.ssim:.4f}",
            ha="center",
            va="top",
            transform=axes[2, col].transAxes,
            fontsize=8.5,
        )

    if title:
        fig.suptitle(title, y=0.99, fontsize=14)
    if caption:
        fig.text(0.01, 0.015, caption, ha="left", va="bottom", fontsize=10, wrap=True)

    fig.tight_layout(rect=(0.02, 0.08 if caption else 0.04, 1, 0.95 if title else 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    svg = out.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def mean(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    return float(arr.mean())


def main() -> None:
    args = parse_args()
    ffmpeg = require_tool("ffmpeg", args.ffmpeg)
    ffprobe = require_tool("ffprobe", args.ffprobe)

    infos = {
        "gt": probe_video(args.gt_video, ffprobe),
        "fvdb": probe_video(args.fvdb_video, ffprobe),
        "spconv": probe_video(args.spconv_video, ffprobe),
    }
    sizes = {(info.width, info.height) for info in infos.values()}
    if len(sizes) != 1:
        raise RuntimeError(
            "Video dimensions do not match: "
            + ", ".join(f"{name}={info.width}x{info.height}" for name, info in infos.items())
        )

    width, height = infos["gt"].width, infos["gt"].height
    records = compare_metrics(
        gt_video=args.gt_video,
        fvdb_video=args.fvdb_video,
        spconv_video=args.spconv_video,
        width=width,
        height=height,
        ffmpeg=ffmpeg,
        threshold=args.threshold,
        stride=args.stride,
        max_frames=args.max_frames,
    )

    selected_specs = select_records(
        records,
        parse_frame_ids(args.frame_ids),
        args.selection_metric,
        args.best_count,
        args.worst_count,
        args.min_gap,
    )
    selected = extract_selected_frames(
        selected_specs,
        args.gt_video,
        args.fvdb_video,
        args.spconv_video,
        width,
        height,
        ffmpeg,
        args.threshold,
        args.bbox,
    )

    stem = args.out.with_suffix("")
    all_csv = stem.parent / f"{stem.name}_metrics.csv"
    selected_csv = stem.parent / f"{stem.name}_selected.csv"
    write_all_csv(all_csv, records)
    write_selected_csv(selected_csv, selected)
    frame_dir = write_frame_pngs(selected, args.out, args.thumb_width)
    written = make_figure(selected, args.out, args.title, args.caption, args.thumb_width)

    print()
    print("PVS backend render-error comparison")
    print(f"frames compared : {len(records)}")
    print(f"resolution      : {width}x{height}")
    print(f"threshold       : {args.threshold}")
    print()
    print("backend  PER_mean_%  PER_max_%  SSIM_mean  SSIM_min")
    print("-------  ----------  ---------  ---------  --------")
    for backend in ("fvdb", "spconv"):
        per_values = [getattr(record, backend).per_percent for record in records]
        ssim_values = [getattr(record, backend).ssim for record in records]
        print(
            f"{backend:<7}  "
            f"{mean(per_values):10.6f}  "
            f"{max(per_values):9.6f}  "
            f"{mean(ssim_values):9.6f}  "
            f"{min(ssim_values):8.6f}"
        )

    print()
    print("Selected frames:")
    print("label    frame  fvdb_PER_%  spconv_PER_%  fvdb_SSIM  spconv_SSIM")
    print("-------  -----  ----------  ------------  ---------  -----------")
    for item in selected:
        record = item.record
        print(
            f"{item.label:<7}  "
            f"{record.frame:5d}  "
            f"{record.fvdb.per_percent:10.6f}  "
            f"{record.spconv.per_percent:12.6f}  "
            f"{record.fvdb.ssim:9.6f}  "
            f"{record.spconv.ssim:11.6f}"
        )

    print()
    print("Wrote:")
    for path in written:
        print(f"  {path}")
    print(f"  {all_csv}")
    print(f"  {selected_csv}")
    print(f"  {frame_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
