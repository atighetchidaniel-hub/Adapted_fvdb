# Logs

## How These Logs Were Created
These logs were produced while running the fixed RobotLab 10-epoch video metric pipeline. They include the metric computation and diff/side-by-side video generation.

The logs confirm that the diff videos were generated over 3600 frames and that processing completed successfully.

## Final Reported Metrics

| Backend | SSIM ↑ | PSNR ↑ | VMAF ↑ | FLIP ↓ |
|---|---:|---:|---:|---:|
| fVDB | 0.998503 | 48.523265 | 96.666082 | 0.001803 |
| spconv | 0.998370 | 48.136690 | 96.489759 | 0.001916 |

## Interpretation
The logs support the final result: fVDB preserves visual quality and is slightly better than spconv in this fixed RobotLab video comparison.
