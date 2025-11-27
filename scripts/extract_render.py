import os
import glob
import zipfile
import json
import numpy as np
import pandas as pd
import cv2
import argparse


def process_render_metrics(json_path: str, output_dir: str = None):
    """
    Reads a JSON file of render metrics, writes per-frame metrics to render_frames.csv,
    re-computes mean/max/min/std across frames, and writes those to render_stats.csv
    in the same folder as the JSON (or in output_dir if provided).
    """
    if output_dir is None:
        output_dir = os.path.dirname(json_path)
    # 1. Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # 2. Flatten per-frame metrics
    records = []
    for frame_id, metrics in data.get("metrics_per_view", {}).items():
        rec = {"frame": frame_id}
        for k, v in metrics.items():
            if isinstance(v, str) and v.lower() == "infinity":
                rec[k] = np.inf
            else:
                rec[k] = float(v)
        records.append(rec)

    df = pd.DataFrame.from_records(records).set_index("frame")

    # 3. Write per-frame CSV
    frames_csv = os.path.join(output_dir, "render_frames.csv")
    df.to_csv(frames_csv, index=True)

    # 4. Re-compute stats
    stats = pd.DataFrame(
        {"mean": df.mean(), "max": df.max(), "min": df.min(), "std": df.std()}
    )

    # 5. Write stats CSV
    stats_csv = os.path.join(output_dir, "render_stats.csv")
    stats.to_csv(stats_csv, index=True)

    print(f"Wrote:\n  {frames_csv}\n  {stats_csv}")


def process_all_zips(zips_folder: str):
    """
    For each .zip in zips_folder:
      1. Unzip into a subfolder named after the zip (without .zip)
      2. Find the JSON and MP4 inside
      3. Call process_render_metrics() on the JSON
      4. Extract every frame from the MP4 into a 'frames' subfolder
    """
    zip_paths = glob.glob(os.path.join(zips_folder, "*.zip"))

    for zip_path in zip_paths:
        base_name = os.path.splitext(os.path.basename(zip_path))[0]
        extract_dir = os.path.join(zips_folder, base_name)
        os.makedirs(extract_dir, exist_ok=True)

        # 1. Unzip
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        # 2. Locate JSON and MP4
        json_files = glob.glob(os.path.join(extract_dir, "*.json"))
        mp4_files = glob.glob(os.path.join(extract_dir, "*.mp4"))

        if not json_files:
            print(f"[{base_name}] No JSON found—skipping.")
            continue
        if not mp4_files:
            print(f"[{base_name}] No MP4 found—skipping.")
            continue

        json_path = json_files[0]
        mp4_path = mp4_files[0]

        # 3. Generate CSVs
        process_render_metrics(json_path, output_dir=extract_dir)

        # 4. Extract frames
        frames_dir = os.path.join(extract_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(mp4_path)
        frame_idx = 0
        success, frame = cap.read()
        while success:
            frame_filename = os.path.join(frames_dir, f"{frame_idx}.png")
            cv2.imwrite(frame_filename, frame)
            frame_idx += 1
            success, frame = cap.read()
        cap.release()

        print(f"[{base_name}] Extracted {frame_idx} frames to {frames_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process render metric zips: extract, analyze, and save frames."
    )
    parser.add_argument(
        "zips_folder",
        type=str,
        help="Path to the folder containing .zip files to process.",
    )
    args = parser.parse_args()
    process_all_zips(args.zips_folder)
