# RobotLab d16 10 Epoch: fVDB vs spconv Summary

This summary compares the important voxel and video metrics for the RobotLab d16 10-epoch trial.

Lower is better for loss, false-negative rate, false-positive rate, false-positive ratio, GV ratio, and FLIP.
Higher is better for Dice, SSIM, PSNR, and VMAF.

## Voxel Metrics

| Metric | fVDB | spconv | fVDB - spconv | Better |
|---|---:|---:|---:|---|
| Dice | 0.986030 | 0.986073 | -0.000043 | spconv |
| Loss | 0.013970 | 0.013927 | 0.000043 | spconv |
| False negative rate | 0.000512 | 0.000877 | -0.000365 | fVDB |
| False positive rate | 0.009786 | 0.009329 | 0.000457 | spconv |
| False positive ratio | 0.540296 | 0.515596 | 0.024700 | spconv |
| GV ratio | 0.917510 | 0.892671 | 0.024839 | spconv |

## Direct Unity Video Metrics

| Metric | fVDB | spconv | fVDB - spconv | Better |
|---|---:|---:|---:|---|
| SSIM | n/a | n/a | n/a | n/a |
| PSNR | n/a | n/a | n/a | n/a |
| VMAF | n/a | n/a | n/a | n/a |
| FLIP | n/a | n/a | n/a | n/a |

## Even Lossless Video Metrics

| Metric | fVDB | spconv | fVDB - spconv | Better |
|---|---:|---:|---:|---|
| SSIM | 0.998503 | 0.998370 | 0.000133 | fVDB |
| PSNR | 48.523265 | 48.136690 | 0.386575 | fVDB |
| VMAF | 96.666082 | 96.489759 | 0.176323 | fVDB |
| FLIP | 0.001803 | 0.001916 | -0.000113 | fVDB |

## Short Interpretation

- Direct Unity-video SSIM winner: **n/a**.
- False-negative-rate winner: **fVDB**.
- False-positive-rate winner: **spconv**.
- Direct Unity-video FLIP winner: **n/a**.

The differences are small, so this should be interpreted as a close backend comparison rather than a large quality gap.
