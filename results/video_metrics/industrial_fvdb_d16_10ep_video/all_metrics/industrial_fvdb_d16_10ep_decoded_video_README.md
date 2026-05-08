# Industrial Clean fvdb Results

## Experiment Setup

| Field | Value |
|---|---|
| Scene | Industrial clean |
| Backend | fvdb |
| Model | OACNNsInterleaved |
| Interleaver d | 16 |
| Epochs | 10 |
| Compared frames | 600 |

## Voxel-Level Metrics

These are computed directly from predicted PVV voxels against ground-truth PVV voxels.

| Metric | Value |
|---|---:|
| False negative rate | 0.00061132 (0.0611%) |
| False positive rate | 0.00297566 (0.2976%) |
| Dice | 0.99269353 |

## Decoded Video-Frame Metrics

These compare decoded frames from the normalized GT and prediction videos.

| Metric | Value |
|---|---:|
| Decoded-frame SSIM mean | 0.95923107 |
| Decoded-frame SSIM min | 0.94932103 |
| Decoded-frame SSIM max | 0.98788112 |
| MAE mean | 0.00939307 |
| RMSE mean | 0.01924759 |

## Repo Video Metrics

These are computed using the repository video-metric pipeline.

| Metric | Value |
|---|---:|
| Video SSIM | 0.97806800 |
| PSNR | 37.535405 |
| VMAF | 93.059299 |
| FLIP | 0.05094023 |

## Interpretation

This run used Unity's Video export mode. Therefore, the frame-level metrics above are decoded-video-frame metrics rather than raw PNG metrics.

The repo video metrics are the main values for comparison with the repository pipeline. The decoded-frame metrics are a sanity check after video normalization/decoding.

## Source Artifacts

| Artifact | Path |
|---|---|
| Prediction video | `results/video_metrics/industrial_fvdb_d16_10ep_video/industrial_fvdb_d16_10ep_normalized.mp4` |
| GT video | `results/video_metrics/industrial_fvdb_d16_10ep_video/industrial_gt_pvv_video_normalized.mp4` |
| Voxel eval stats | `data_for_test/out/OACNNsInterleaved_industrial_clean_dice_20260508-165150_industrial_clean_fvdb_d16_10ep-industrial_clean_fvdb_d16_10ep_infer_EtgImGP3/eval_stats.csv` |
| Repo video metrics JSON | `results/video_metrics/industrial_fvdb_d16_10ep_video/industrial_fvdb_d16_10ep_normalized_results.json` |
