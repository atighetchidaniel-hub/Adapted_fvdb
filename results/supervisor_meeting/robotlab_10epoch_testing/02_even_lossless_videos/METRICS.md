# Even Lossless Videos

## How These Videos Were Created
These are the corrected videos used for the final metric computation. They come from Unity `RenderPVV` mode: Unity loads predicted PVV files, binds them to the renderer, and records the visible result.

A correction step was needed because the original rendered video dimensions were not safely divisible by 2. Some video metrics/codecs require even dimensions, so the videos were converted to an even-dimension lossless version before computing SSIM, PSNR, VMAF, and FLIP.

## Final Metrics

| Backend | SSIM ↑ | PSNR ↑ | VMAF ↑ | FLIP ↓ |
|---|---:|---:|---:|---:|
| fVDB | 0.998503 | 48.523265 | 96.666082 | 0.001803 |
| spconv | 0.998370 | 48.136690 | 96.489759 | 0.001916 |

## Difference fVDB - spconv

| Metric | Difference |
|---|---:|
| SSIM | +0.000133 |
| PSNR | +0.386575 dB |
| VMAF | +0.176323 |
| FLIP | -0.000112690 |

For FLIP, lower is better, so the negative difference favors fVDB.
