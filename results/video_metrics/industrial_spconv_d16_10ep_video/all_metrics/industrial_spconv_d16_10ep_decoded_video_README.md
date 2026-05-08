# Industrial Clean spconv Results

## Experiment Setup

| Field | Value |
|---|---|
| Scene | Industrial clean |
| Backend | spconv |
| Model | OACNNsInterleaved |
| Interleaver d | 16 |
| Epochs | 10 |
| Compared frames | 600 |

## Voxel-Level Metrics

These are computed directly from predicted PVV voxels against ground-truth PVV voxels.

| Metric | Value |
|---|---:|
| False negative rate | 0.00048654 (0.0487%) |
| False positive rate | 0.00314675 (0.3147%) |
| Dice | 0.99229836 |

## Decoded Video-Frame Metrics

These compare decoded frames from the normalized GT and prediction videos.

| Metric | Value |
|---|---:|
| Decoded-frame SSIM mean | 0.95984581 |
| Decoded-frame SSIM min | 0.95216101 |
| Decoded-frame SSIM max | 0.98822385 |
| MAE mean | 0.00929640 |
| RMSE mean | 0.01842808 |

## Repo Video Metrics

These are computed using the repository video-metric pipeline.

| Metric | Value |
|---|---:|
| Video SSIM | 0.97852800 |
| PSNR | 37.943336 |
| VMAF | 93.446777 |
| FLIP | 0.05059883 |

## Interpretation

This run used Unity's Video export mode. Therefore, the frame-level metrics above are decoded-video-frame metrics rather than raw PNG metrics.

The repo video metrics are the main values for comparison with the repository pipeline. The decoded-frame metrics are a sanity check after video normalization/decoding.

## Source Artifacts

| Artifact | Path |
|---|---|
| Prediction video | `results/video_metrics/industrial_spconv_d16_10ep_video/industrial_spconv_d16_10ep_normalized.mp4` |
| GT video | `results/video_metrics/industrial_spconv_d16_10ep_video/industrial_gt_pvv_video_normalized.mp4` |
| Voxel eval stats | `data_for_test/out_spconv/OACNNsInterleaved_industrial_clean_dice_20260508-170334_industrial_clean_spconv_d16_10ep-industrial_clean_spconv_d16_10ep_infer_F5Rc9ZAo/eval_stats.csv` |
| Repo video metrics JSON | `results/video_metrics/industrial_spconv_d16_10ep_video/industrial_spconv_d16_10ep_normalized_results.json` |
