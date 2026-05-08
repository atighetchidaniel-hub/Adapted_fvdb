# BigCity Clean fVDB Results

## Experiment Setup

| Field | Value |
|---|---|
| Scene | BigCity clean |
| Backend | fVDB |
| Model | OACNNsInterleaved |
| Interleaver d | 16 |
| Epochs | 200 |
| Rendered frames | 3598 |

## Voxel-Level Metrics

These are computed directly from predicted PVV voxels against ground-truth PVV voxels.

| Metric | Value |
|---|---:|
| False negative rate | 0.00125306 (0.1253%) |
| False positive rate | 0.00439525 (0.4395%) |
| Dice | 0.99168952 |

## Raw Unity Frame Metrics

These compare the rendered PNG image sequence before video compression.

| Metric | Value |
|---|---:|
| Raw SSIM mean | 0.99895062 |
| Raw SSIM min | 0.97514182 |
| Raw SSIM max | 1.00000000 |
| MAE mean | 0.00012761 |
| RMSE mean | 0.00410213 |

## Repo Video Metrics

These are computed using the repository video-metric pipeline after video encoding.

| Metric | Value |
|---|---:|
| Video SSIM | 0.97983900 |
| PSNR | 42.537567 |
| VMAF | 96.001006 |
| FLIP | 0.03952160 |


