# Straight Render Videos

## How This Video Was Created
This folder contains the ground-truth Unity render video. It was created directly from Unity rendering and acts as the reference video for the metric comparisons.

This is not the neural prediction itself. It is the baseline/reference rendering used to compare the fVDB and spconv predicted-PVV renderings.

## Role In The Comparison
The predicted fVDB and spconv videos are compared against this ground-truth Unity render.

| Backend Compared Against GT | SSIM ↑ | PSNR ↑ | VMAF ↑ | FLIP ↓ |
|---|---:|---:|---:|---:|
| fVDB vs GT | 0.998503 | 48.523265 | 96.666082 | 0.001803 |
| spconv vs GT | 0.998370 | 48.136690 | 96.489759 | 0.001916 |
