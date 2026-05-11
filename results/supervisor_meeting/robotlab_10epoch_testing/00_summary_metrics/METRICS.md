# Metrics Summary

## How These Results Were Created
These metrics come from the fixed RobotLab 10-epoch Unity-video comparison. The videos were produced from Unity rendering in `RenderPVV` mode, where Unity loads predicted PVV files and renders the scene using the PVS-aware shader.

The original Unity-rendered videos needed a fixed/even-dimension lossless version before metric computation, because video codecs and FFmpeg filters can fail or behave inconsistently when frame dimensions are not divisible by 2. Metrics were computed on the corrected even-dimension lossless videos.

## Key Result

| Backend | SSIM ↑ | PSNR ↑ | VMAF ↑ | FLIP ↓ |
|---|---:|---:|---:|---:|
| fVDB | 0.998503 | 48.523265 | 96.666082 | 0.001803 |
| spconv | 0.998370 | 48.136690 | 96.489759 | 0.001916 |

## Short Interpretation
The fVDB backend preserves visual quality compared with the original spconv backend. In this RobotLab fixed Unity-video test, fVDB is slightly better on all reported metrics.
