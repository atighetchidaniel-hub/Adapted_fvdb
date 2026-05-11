# Difference Videos

## How These Videos Were Created
These videos were generated from the fixed even-dimension lossless videos. Each predicted rendering was compared frame-by-frame against the ground-truth Unity render.

The diff videos highlight pixels whose difference exceeds the configured threshold. In this run, the diff generation used threshold = 0.05 over 3600 frames.

These videos are for visual inspection, not the source of the numeric metrics.

## Source Metrics

| Backend | SSIM ↑ | PSNR ↑ | VMAF ↑ | FLIP ↓ |
|---|---:|---:|---:|---:|
| fVDB diff source | 0.998503 | 48.523265 | 96.666082 | 0.001803 |
| spconv diff source | 0.998370 | 48.136690 | 96.489759 | 0.001916 |

## Interpretation
Both diff videos should show very small visible error. The numeric metrics indicate that fVDB is slightly closer to the ground-truth render in this run.
