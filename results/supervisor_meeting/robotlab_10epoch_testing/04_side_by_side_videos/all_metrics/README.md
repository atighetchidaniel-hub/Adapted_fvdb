# RobotLab d16 10 Epoch: All Metrics

This folder collects the relevant metric files for the side-by-side RobotLab comparison.

## Contents

- `robotlab_d16_10ep_key_metrics.csv`: compact table with fVDB vs spconv key metrics.
- `robotlab_d16_10ep_all_metrics_summary.json`: full structured summary.
- `fvdb/`: copied fVDB eval/video/config files.
- `spconv/`: copied spconv eval/video/config files.

## Metric Types

- Voxel metrics: dice, loss, fn_rate, fp_rate, fp_ratio, gv_ratio.
- Direct Unity video metrics: SSIM, PSNR, VMAF, FLIP from Unity `_rendering.mkv`.
- Even lossless video metrics: SSIM, PSNR, VMAF, FLIP after even-dimension lossless normalization for diff/side-by-side video generation.
