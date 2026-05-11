# RobotLab 10-Epoch Unity Video Testing

## Purpose
This folder collects the fixed RobotLab Unity-rendered video comparisons for the 10-epoch fVDB and spconv runs.

## Key Result

| Backend | SSIM ↑ | PSNR ↑ | VMAF ↑ | FLIP ↓ |
|---|---:|---:|---:|---:|
| fVDB | 0.998503 | 48.523265 | 96.666082 | 0.001803 |
| spconv | 0.998370 | 48.136690 | 96.489759 | 0.001916 |

## Interpretation
The fVDB backend preserves visual quality compared with the original spconv backend. In this fixed RobotLab Unity-video comparison, fVDB is slightly better on all reported metrics.

## Folder Guide

- `00_summary_metrics/`: JSON metric outputs.
- `01_straight_render_videos/`: ground-truth Unity render video.
- `02_even_lossless_videos/`: fVDB and spconv rendered videos after even-dimension/lossless fixing.
- `03_diff_videos/`: thresholded visual difference videos.
- `04_side_by_side_videos/`: side-by-side comparison videos.
- `05_logs/`: metric-generation logs.
