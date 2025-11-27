"""Video quality metrics evaluation and difference visualization tool.

This module provides functionality to:
1. Calculate video quality metrics (SSIM, PSNR, VMAF, FLIP)
2. Generate visual difference videos between reference and distorted videos
"""

import setup_paths  # noqa: F401

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import av
import numpy as np
import torchvision.transforms.functional as tf
from flip import LDRFLIPLoss
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_FPS = 30.0
DEFAULT_THRESHOLD = 0.05
DEFAULT_CODEC = "libx264"
DEFAULT_PIX_FMT = "yuv420p"
DIFF_COLOR = np.array([255, 0, 0], dtype=np.uint8)

WINDOWS_FFMPEG_PATHS = [
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
]


def find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg executable using PATH scan, common locations, and WinGet fallback.

    Returns:
        The absolute path to ffmpeg executable or None if not found.
    """
    ffmpeg_exe = os.environ.get("FFMPEG_PATH")
    if ffmpeg_exe and os.path.isfile(ffmpeg_exe):
        return ffmpeg_exe

    ffmpeg_exe = _scan_path_for_ffmpeg()
    if ffmpeg_exe:
        return ffmpeg_exe

    if os.name == "nt":
        ffmpeg_exe = _check_common_windows_paths()
        if ffmpeg_exe:
            return ffmpeg_exe

        ffmpeg_exe = _check_winget_paths()
        if ffmpeg_exe:
            return ffmpeg_exe

    return None


def _scan_path_for_ffmpeg() -> Optional[str]:
    """Scan PATH environment variable for ffmpeg executable."""
    path_env = os.environ.get("PATH", "")
    executable_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

    for path in path_env.split(os.pathsep):
        if not path:
            continue
        candidate = os.path.join(path, executable_name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _check_common_windows_paths() -> Optional[str]:
    """Check common Windows installation paths for ffmpeg."""
    for path in WINDOWS_FFMPEG_PATHS:
        if os.path.isfile(path):
            return path
    return None


def _check_winget_paths() -> Optional[str]:
    """Check WinGet package locations for ffmpeg."""
    local_appdata = os.environ.get("LOCALAPPDATA") or os.path.join(
        os.path.expanduser("~"), "AppData", "Local"
    )
    winget_path = os.path.join(local_appdata, "Microsoft", "WinGet", "Packages")

    if not os.path.isdir(winget_path):
        return None

    try:
        dirs = [d for d in os.listdir(winget_path) if d.startswith("Gyan.FFmpeg")]
    except Exception:
        return None

    for d in dirs:
        base = os.path.join(winget_path, d)
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower() == "ffmpeg.exe":
                    return os.path.join(root, f)
    return None


def get_ffmpeg_executable() -> str:
    """Get FFmpeg executable path."""
    ffmpeg_exe = find_ffmpeg()
    if ffmpeg_exe is None:
        raise FileNotFoundError(
            "ffmpeg executable not found. Install ffmpeg or add it to your PATH, "
            "or set the FFMPEG_PATH environment variable to the ffmpeg executable."
        )
    return ffmpeg_exe


def extract_metrics(
    reference_path: Union[str, Path], distorted_path: Union[str, Path]
) -> Dict[str, Optional[float]]:
    """Extract SSIM, PSNR, and VMAF metrics using FFmpeg.

    Args:
        reference_path: Path to reference video
        distorted_path: Path to distorted video

    Returns:
        Dictionary containing metric values (may contain None for failed metrics)
    """
    ffmpeg_exe = get_ffmpeg_executable()
    ffmpeg_command = _build_ffmpeg_command(ffmpeg_exe, reference_path, distorted_path)

    logger.info("Running FFmpeg for SSIM/PSNR/VMAF...")
    process = subprocess.Popen(
        ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    _, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{stderr}")

    return _parse_ffmpeg_output(stderr)


def _build_ffmpeg_command(
    ffmpeg_exe: str, reference_path: Union[str, Path], distorted_path: Union[str, Path]
) -> list:
    """Build FFmpeg command for metrics extraction."""
    return [
        ffmpeg_exe,
        "-i",
        str(distorted_path),
        "-i",
        str(reference_path),
        "-lavfi",
        (
            "[0:v]setpts=PTS-STARTPTS[dist];"
            "[1:v]setpts=PTS-STARTPTS[ref];"
            "[dist]split=3[dist1][dist2][dist3];"
            "[ref]split=3[ref1][ref2][ref3];"
            "[dist1][ref1]libvmaf=log_fmt=json;"
            "[dist2][ref2]psnr;"
            "[dist3][ref3]ssim"
        ),
        "-f",
        "null",
        "-",
    ]


def _parse_ffmpeg_output(stderr: str) -> Dict[str, Optional[float]]:
    """Parse FFmpeg stderr output to extract metric values."""
    ssim_match = re.search(r"All:(\d+\.\d+)", stderr)
    psnr_match = re.search(r"average:\s*(\d+\.\d+)", stderr)
    vmaf_match = re.search(r"VMAF score:\s*(\d+\.\d+)", stderr)

    return {
        "ssim_all": float(ssim_match.group(1)) if ssim_match else None,
        "psnr_avg": float(psnr_match.group(1)) if psnr_match else None,
        "vmaf": float(vmaf_match.group(1)) if vmaf_match else None,
    }


def compute_flip(
    reference_path: Union[str, Path], distorted_path: Union[str, Path]
) -> float:
    """Compute per-frame FLIP metric using PyAV.

    Args:
        reference_path: Path to reference video
        distorted_path: Path to distorted video

    Returns:
        Average FLIP score across all frames
    """
    logger.info("Evaluating per-frame FLIP (PyAV)...")

    with av.open(str(reference_path)) as container_ref, av.open(
        str(distorted_path)
    ) as container_dist:

        flip = LDRFLIPLoss()
        stream_ref = container_ref.streams.video[0]
        stream_dist = container_dist.streams.video[0]

        frames_ref = container_ref.decode(stream_ref)
        frames_dist = container_dist.decode(stream_dist)

        flip_scores = []
        for ref_frame, dist_frame in tqdm(
            zip(frames_ref, frames_dist), desc="FLIP frames", unit="frame"
        ):
            flip_score = _compute_frame_flip(ref_frame, dist_frame, flip)
            flip_scores.append(flip_score)

    if not flip_scores:
        raise RuntimeError("No frames found or decoded.")

    return float(np.mean(flip_scores))


def _compute_frame_flip(ref_frame, dist_frame, flip) -> float:
    """Compute FLIP score for a single frame pair."""
    ref_rgb = ref_frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
    dist_rgb = dist_frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0

    ref_rgb = tf.to_tensor(ref_rgb).unsqueeze(0).cuda()
    dist_rgb = tf.to_tensor(dist_rgb).unsqueeze(0).cuda()

    if ref_rgb.shape != dist_rgb.shape:
        raise ValueError(f"Frame shape mismatch: {ref_rgb.shape} vs {dist_rgb.shape}")

    return flip(dist_rgb, ref_rgb).mean().item()


def run_video_metrics(
    reference_path: Union[str, Path], distorted_path: Union[str, Path]
) -> Dict[str, Optional[float]]:
    """Run comprehensive video quality metrics evaluation.

    Args:
        reference_path: Path to reference video
        distorted_path: Path to distorted video

    Returns:
        Dictionary containing all computed metrics
    """
    ref = Path(reference_path).resolve()
    dist = Path(distorted_path).resolve()

    if not ref.exists() or not dist.exists():
        raise FileNotFoundError("Reference or distorted video file not found.")

    results_path = dist.parent / f"{dist.stem}_results.json"

    metrics = extract_metrics(ref, dist)
    flip_avg = compute_flip(ref, dist)
    metrics["flip_avg"] = flip_avg

    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Saved metrics to {results_path}")
    return metrics


def save_diff_video(
    reference_path: Union[str, Path],
    distorted_path: Union[str, Path],
    *,
    diff_output: Union[bool, str, Path, None] = True,
    side_by_side_output: Union[bool, str, Path, None] = False,
    threshold: float = DEFAULT_THRESHOLD,
    codec: str = DEFAULT_CODEC,
) -> None:
    """Create videos showing per-pixel differences between distorted and reference frames.

    Differences larger than `threshold` (0..1) are marked in red on top of the distorted frame.

    Args:
        reference_path: Path to reference video
        distorted_path: Path to distorted video
        diff_output: Controls diff-only video output:
            - True: auto-generate path (default)
            - False/None: skip diff video
            - str/Path: use specific output path
        side_by_side_output: Controls side-by-side video output:
            - True: auto-generate path
            - False/None: skip side-by-side video (default)
            - str/Path: use specific output path
        threshold: Per-channel absolute difference threshold normalized by 255 (0..1)
        codec: Video codec for encoding
    """
    ref = Path(reference_path).resolve()
    dist = Path(distorted_path).resolve()

    if not ref.exists() or not dist.exists():
        raise FileNotFoundError("Reference or distorted video file not found.")

    # Determine what to save and setup output paths
    save_diff_only = bool(diff_output)
    save_side_by_side = bool(side_by_side_output)

    # Setup diff output path
    if diff_output is True:
        diff_out = dist.parent / f"{dist.stem}_diff.mp4"
    elif isinstance(diff_output, (str, Path)):
        diff_out = Path(diff_output).resolve()
    else:  # False or None
        diff_out = None

    # Setup side-by-side output path
    if side_by_side_output is True:
        sbs_out = dist.parent / f"{dist.stem}_sbs.mp4"
    elif isinstance(side_by_side_output, (str, Path)):
        sbs_out = Path(side_by_side_output).resolve()
    else:  # False or None
        sbs_out = None

    # Validate that at least one output is requested
    if not save_diff_only and not save_side_by_side:
        raise ValueError(
            "At least one output type must be enabled (diff_output or side_by_side_output)"
        )

    _process_videos(
        ref,
        dist,
        diff_out,
        sbs_out,
        threshold,
        codec,
        save_diff_only,
        save_side_by_side,
    )


def _process_videos(
    ref: Path,
    dist: Path,
    diff_out: Optional[Path],
    sbs_out: Optional[Path],
    threshold: float,
    codec: str,
    save_diff_only: bool,
    save_side_by_side: bool,
) -> None:
    """Process videos and generate diff outputs."""
    with av.open(str(ref)) as container_ref, av.open(str(dist)) as container_dist:
        stream_ref = container_ref.streams.video[0]
        stream_dist = container_dist.streams.video[0]

        fps = _get_fps(stream_dist)
        width = stream_dist.codec_context.width
        height = stream_dist.codec_context.height

        diff_container, diff_stream, sbs_container, sbs_stream = (
            _setup_output_containers(
                diff_out,
                sbs_out,
                fps,
                width,
                height,
                codec,
                save_diff_only,
                save_side_by_side,
            )
        )

        frames_written = _process_frames(
            container_ref,
            container_dist,
            stream_ref,
            stream_dist,
            diff_container,
            diff_stream,
            sbs_container,
            sbs_stream,
            threshold,
            save_diff_only,
            save_side_by_side,
            fps,
        )

        _cleanup_containers(
            diff_container,
            diff_stream,
            sbs_container,
            sbs_stream,
            save_diff_only,
            save_side_by_side,
        )

    _log_results(frames_written, diff_out, sbs_out, save_diff_only, save_side_by_side)


def _get_fps(stream) -> float:
    """Extract FPS from video stream with fallback."""
    try:
        return (
            float(stream.average_rate)
            if stream.average_rate
            else float(stream.base_rate)
        )
    except Exception:
        return DEFAULT_FPS


def _setup_output_containers(
    diff_out: Optional[Path],
    sbs_out: Optional[Path],
    fps: float,
    width: int,
    height: int,
    codec: str,
    save_diff_only: bool,
    save_side_by_side: bool,
) -> Tuple[
    Optional[Any],  # av.Container
    Optional[Any],  # av.Stream
    Optional[Any],  # av.Container
    Optional[Any],  # av.Stream
]:
    """Setup output containers and streams."""
    diff_container = diff_stream = sbs_container = sbs_stream = None

    if save_diff_only and diff_out is not None:
        diff_container = av.open(str(diff_out), mode="w")
        diff_stream = diff_container.add_stream(codec, rate=int(round(fps)))
        diff_stream.width = width
        diff_stream.height = height
        diff_stream.pix_fmt = DEFAULT_PIX_FMT

    if save_side_by_side and sbs_out is not None:
        sbs_container = av.open(str(sbs_out), mode="w")
        sbs_stream = sbs_container.add_stream(codec, rate=int(round(fps)))
        sbs_stream.width = width * 3  # Triple width for side-by-side
        sbs_stream.height = height
        sbs_stream.pix_fmt = DEFAULT_PIX_FMT

    return diff_container, diff_stream, sbs_container, sbs_stream


def _process_frames(
    container_ref,
    container_dist,
    stream_ref,
    stream_dist,
    diff_container,
    diff_stream,
    sbs_container,
    sbs_stream,
    threshold: float,
    save_diff_only: bool,
    save_side_by_side: bool,
    fps: float,
) -> Dict[str, int]:
    """Process video frames and generate diff outputs."""
    frames_ref = container_ref.decode(stream_ref)
    frames_dist = container_dist.decode(stream_dist)

    frames_written = {"diff": 0, "sbs": 0}

    logger.info(f"Writing diff(s) (threshold={threshold}, fps={fps:.2f})...")

    for ref_frame, dist_frame in tqdm(
        zip(frames_ref, frames_dist), desc="diff frames", unit="frame"
    ):
        ref_rgb = ref_frame.to_ndarray(format="rgb24")
        dist_rgb = dist_frame.to_ndarray(format="rgb24")

        if ref_rgb.shape != dist_rgb.shape:
            raise ValueError(
                f"Frame shape mismatch: {ref_rgb.shape} vs {dist_rgb.shape}"
            )

        diff_frame = _create_diff_frame(ref_rgb, dist_rgb, threshold)

        if save_diff_only and diff_stream is not None:
            _encode_frame(diff_frame, diff_container, diff_stream)
            frames_written["diff"] += 1

        if save_side_by_side and sbs_stream is not None:
            sbs_rgb = np.concatenate([ref_rgb, dist_rgb, diff_frame], axis=1)
            _encode_frame(sbs_rgb, sbs_container, sbs_stream)
            frames_written["sbs"] += 1

    return frames_written


def _create_diff_frame(
    ref_rgb: np.ndarray, dist_rgb: np.ndarray, threshold: float
) -> np.ndarray:
    """Create difference frame highlighting pixels above threshold."""
    diff = np.max(np.abs(dist_rgb.astype(np.int16) - ref_rgb.astype(np.int16)), axis=2)
    diff_norm = diff.astype(np.float32) / 255.0

    mask = diff_norm > threshold

    diff_frame = dist_rgb.copy()
    diff_frame[mask] = DIFF_COLOR

    return diff_frame


def _encode_frame(frame_rgb: np.ndarray, container, stream) -> None:
    """Encode a single frame to the output container."""
    vframe = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
    yuv_frame = vframe.reformat(format=DEFAULT_PIX_FMT)
    for packet in stream.encode(yuv_frame):
        container.mux(packet)


def _cleanup_containers(
    diff_container,
    diff_stream,
    sbs_container,
    sbs_stream,
    save_diff_only: bool,
    save_side_by_side: bool,
) -> None:
    """Flush encoders and close containers."""
    if save_diff_only and diff_stream is not None:
        for packet in diff_stream.encode():
            diff_container.mux(packet)
        diff_container.close()

    if save_side_by_side and sbs_stream is not None:
        for packet in sbs_stream.encode():
            sbs_container.mux(packet)
        sbs_container.close()


def _log_results(
    frames_written: Dict[str, int],
    diff_out: Optional[Path],
    sbs_out: Optional[Path],
    save_diff_only: bool,
    save_side_by_side: bool,
) -> None:
    """Log processing results."""
    if save_diff_only and diff_out is not None:
        logger.info(
            f"Saved diff-only video with {frames_written['diff']} frames to {diff_out}"
        )
    if save_side_by_side and sbs_out is not None:
        logger.info(
            f"Saved side-by-side video with {frames_written['sbs']} frames to {sbs_out}"
        )


def main() -> None:
    """Main CLI entry point."""
    if len(sys.argv) != 3:
        print("Usage: python run_video_metrics.py <distorted_video> <reference_video>")
        print("  distorted_video: Path to the distorted/test video")
        print("  reference_video: Path to the reference/ground truth video")
        print("\nThis script can:")
        print("  1. Calculate video quality metrics (SSIM, PSNR, VMAF, FLIP)")
        print("  2. Generate visual difference videos highlighting changes")
        sys.exit(1)

    dist_video = sys.argv[1]
    ref_video = sys.argv[2]

    try:
        logger.info("Processing videos:")
        logger.info(f"  Reference: {ref_video}")
        logger.info(f"  Distorted: {dist_video}")

        # Uncomment to run full metrics evaluation
        logger.info("Running comprehensive metrics evaluation...")
        metrics = run_video_metrics(ref_video, dist_video)
        logger.info(f"Metrics: {metrics}")

        # Generate difference video
        logger.info("Generating difference video...")
        save_diff_video(
            ref_video,
            dist_video,
            threshold=DEFAULT_THRESHOLD,
            diff_output=True,
            side_by_side_output=True,
        )
        logger.info("Processing completed successfully!")

    except Exception as e:
        logger.error(f"Error processing videos: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
