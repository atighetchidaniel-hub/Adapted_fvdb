#!/usr/bin/env python3
"""Create paper-style PVS render error figures from paired videos.

The script compares a predicted render video against a ground-truth render
video frame by frame. It computes a thresholded pixel error rate (PER), a
lightweight per-frame SSIM value, stores per-frame metrics, and writes selected
frames with error pixels marked in red.

Example:
    python scripts/make_pvs_error_figure.py \
      --gt-video "T:/FINAL_GVPVV/INDUSTRIAL_FINAL/industrial/r30/predicted_unity_pvv/00_color/_rendering.mkv" \
      --pred-video "T:/fvdb_out/.../inference/0/00_color/_rendering.mkv" \
      --out "T:/render_metrics/industrial_r30_d16_fvdb_error_figure.png" \
      --threshold 10 \
      --top-k 6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
class FrameMetric:
    frame: int
    per_percent: float
    ssim: float
    error_pixels: int
    total_pixels: int


@dataclass
class SelectedFrame:
    metric: FrameMetric
    overlay: np.ndarray
    pred: np.ndarray
    gt: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute rendered PVS pixel error rate and create a red-overlay figure."
    )
    parser.add_argument("--gt-video", type=Path, required=True, help="Ground-truth/reference render video.")
    parser.add_argument("--pred-video", type=Path, required=True, help="Predicted/model render video.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path. SVG is written too.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="RGB difference threshold in 0..255 units. A pixel is wrong if max channel diff exceeds this.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of high-PER frames to show when --frame-ids is not supplied.",
    )
    parser.add_argument(
        "--frame-ids",
        type=str,
        default="",
        help="Comma-separated frame ids to show instead of automatically selecting top-PER frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Only evaluate every Nth frame. Keep this at 1 for final figures.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional maximum number of compared frames. 0 means full video.",
    )
    parser.add_argument(
        "--overlay-on",
        choices=("pred", "gt"),
        default="pred",
        help="Draw red error pixels over the predicted frame or ground-truth frame.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="",
        help="Optional caption text printed under the figure.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure title.",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=360,
        help="Width of each saved/displayed thumbnail in pixels.",
    )
    parser.add_argument(
        "--bbox",
        action="store_true",
        help="Draw a red bounding box around all error pixels in selected frames.",
    )
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


def grayscale(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    return 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]


def global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Small dependency-free SSIM approximation over the full grayscale image."""
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


def compute_metric_and_overlay(
    gt: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    overlay_on: str,
    draw_bbox: bool,
) -> tuple[FrameMetric, np.ndarray]:
    diff = np.max(np.abs(pred.astype(np.int16) - gt.astype(np.int16)), axis=2)
    error_mask = diff > threshold
    error_pixels = int(error_mask.sum())
    total_pixels = int(error_mask.size)
    per_percent = 100.0 * error_pixels / total_pixels
    ssim = global_ssim(pred, gt)

    base = pred if overlay_on == "pred" else gt
    overlay = np.array(base, copy=True)
    overlay[error_mask] = np.array([255, 0, 0], dtype=np.uint8)

    if draw_bbox and error_pixels:
        ys, xs = np.nonzero(error_mask)
        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            outline=(255, 0, 0),
            width=max(2, min(gt.shape[0], gt.shape[1]) // 180),
        )
        overlay = np.array(image)

    metric = FrameMetric(
        frame=-1,
        per_percent=per_percent,
        ssim=ssim,
        error_pixels=error_pixels,
        total_pixels=total_pixels,
    )
    return metric, overlay


def parse_frame_ids(value: str) -> list[int]:
    if not value.strip():
        return []
    ids = []
    for part in value.split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    return ids


def resize_for_figure(frame: np.ndarray, thumb_width: int) -> np.ndarray:
    image = Image.fromarray(frame)
    ratio = thumb_width / image.width
    thumb_height = max(1, round(image.height * ratio))
    return np.array(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS))


def write_frame_pngs(selected: list[SelectedFrame], out: Path, thumb_width: int) -> Path:
    frame_dir = out.with_suffix("").parent / (out.with_suffix("").name + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for selected_frame in selected:
        frame = selected_frame.metric.frame
        overlay = resize_for_figure(selected_frame.overlay, thumb_width)
        pred = resize_for_figure(selected_frame.pred, thumb_width)
        gt = resize_for_figure(selected_frame.gt, thumb_width)
        Image.fromarray(overlay).save(frame_dir / f"frame_{frame:06d}_overlay.png")
        Image.fromarray(pred).save(frame_dir / f"frame_{frame:06d}_pred.png")
        Image.fromarray(gt).save(frame_dir / f"frame_{frame:06d}_gt.png")
    return frame_dir


def write_csv(path: Path, rows: Iterable[FrameMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "per_percent", "ssim", "error_pixels", "total_pixels"])
        for row in rows:
            writer.writerow(
                [
                    row.frame,
                    f"{row.per_percent:.8f}",
                    f"{row.ssim:.8f}",
                    row.error_pixels,
                    row.total_pixels,
                ]
            )


def select_frames(
    metrics: list[FrameMetric],
    stored_frames: dict[int, SelectedFrame],
    manual_ids: list[int],
    top_k: int,
) -> list[SelectedFrame]:
    if manual_ids:
        missing = [frame_id for frame_id in manual_ids if frame_id not in stored_frames]
        if missing:
            raise RuntimeError(f"Requested frame ids were not available: {missing}")
        return [stored_frames[frame_id] for frame_id in manual_ids]

    selected_metrics = sorted(metrics, key=lambda item: (item.per_percent, -item.ssim), reverse=True)[:top_k]
    return [stored_frames[item.frame] for item in selected_metrics]


def make_figure(selected: list[SelectedFrame], out: Path, title: str, caption: str, thumb_width: int) -> list[Path]:
    if not selected:
        raise RuntimeError("No selected frames to plot")

    thumbs = [resize_for_figure(item.overlay, thumb_width) for item in selected]
    cols = len(thumbs)
    fig_width = max(8.0, cols * 2.0)
    fig_height = 2.35 + (0.35 if title else 0.0) + (0.45 if caption else 0.0)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "text.color": "#000000",
        }
    )

    fig, axes = plt.subplots(1, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_list = axes[0].tolist()
    for ax, item, thumb in zip(axes_list, selected, thumbs):
        ax.imshow(thumb)
        ax.set_axis_off()
        ax.text(
            0.5,
            -0.08,
            f"frame={item.metric.frame}\nPER={item.metric.per_percent:.4g}%, SSIM={item.metric.ssim:.4f}",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )

    if title:
        fig.suptitle(title, y=0.98, fontsize=13)
    if caption:
        fig.text(0.01, 0.02, caption, ha="left", va="bottom", fontsize=10, wrap=True)

    fig.tight_layout(rect=(0, 0.12 if caption else 0.08, 1, 0.92 if title else 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    svg = out.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def cleanup_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.kill()
    try:
        proc.communicate(timeout=2)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    ffmpeg = require_tool("ffmpeg", args.ffmpeg)
    ffprobe = require_tool("ffprobe", args.ffprobe)

    gt_info = probe_video(args.gt_video, ffprobe)
    pred_info = probe_video(args.pred_video, ffprobe)
    if (gt_info.width, gt_info.height) != (pred_info.width, pred_info.height):
        raise RuntimeError(
            "Video dimensions do not match: "
            f"GT={gt_info.width}x{gt_info.height}, Pred={pred_info.width}x{pred_info.height}"
        )

    width, height = gt_info.width, gt_info.height
    manual_ids = parse_frame_ids(args.frame_ids)
    manual_set = set(manual_ids)
    keep_all_frames = not manual_ids

    gt_proc = start_raw_reader(args.gt_video, width, height, ffmpeg)
    pred_proc = start_raw_reader(args.pred_video, width, height, ffmpeg)

    metrics: list[FrameMetric] = []
    stored_frames: dict[int, SelectedFrame] = {}
    frame_id = 0

    try:
        while True:
            gt = read_frame(gt_proc, width, height)
            pred = read_frame(pred_proc, width, height)
            if gt is None or pred is None:
                break

            if frame_id % args.stride == 0:
                metric, overlay = compute_metric_and_overlay(
                    gt=gt,
                    pred=pred,
                    threshold=args.threshold,
                    overlay_on=args.overlay_on,
                    draw_bbox=args.bbox,
                )
                metric.frame = frame_id
                metrics.append(metric)

                if keep_all_frames or frame_id in manual_set:
                    stored_frames[frame_id] = SelectedFrame(
                        metric=metric,
                        overlay=np.array(overlay, copy=True),
                        pred=np.array(pred, copy=True),
                        gt=np.array(gt, copy=True),
                    )

                if len(metrics) % 100 == 0:
                    print(f"Compared {len(metrics)} frames...", flush=True)

            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break
    finally:
        cleanup_process(gt_proc)
        cleanup_process(pred_proc)

    if not metrics:
        raise RuntimeError("No frames were compared")

    selected = select_frames(metrics, stored_frames, manual_ids, args.top_k)

    metrics_csv = args.out.with_suffix("").parent / (args.out.with_suffix("").name + "_metrics.csv")
    selected_csv = args.out.with_suffix("").parent / (args.out.with_suffix("").name + "_selected.csv")
    write_csv(metrics_csv, metrics)
    write_csv(selected_csv, [item.metric for item in selected])
    frame_dir = write_frame_pngs(selected, args.out, args.thumb_width)
    written = make_figure(selected, args.out, args.title, args.caption, args.thumb_width)

    per_values = np.array([row.per_percent for row in metrics], dtype=float)
    ssim_values = np.array([row.ssim for row in metrics], dtype=float)

    print()
    print("PVS render pixel-error summary")
    print(f"frames compared : {len(metrics)}")
    print(f"resolution      : {width}x{height}")
    print(f"threshold       : {args.threshold}")
    print(f"PER mean        : {per_values.mean():.8f}%")
    print(f"PER max         : {per_values.max():.8f}%")
    print(f"SSIM mean       : {ssim_values.mean():.8f}")
    print(f"SSIM min        : {ssim_values.min():.8f}")
    print()
    print("Selected frames:")
    for item in selected:
        print(
            f"  frame {item.metric.frame:6d} | "
            f"PER={item.metric.per_percent:.8f}% | "
            f"SSIM={item.metric.ssim:.8f} | "
            f"error_pixels={item.metric.error_pixels}"
        )
    print()
    print("Wrote:")
    for path in written:
        print(f"  {path}")
    print(f"  {metrics_csv}")
    print(f"  {selected_csv}")
    print(f"  {frame_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
