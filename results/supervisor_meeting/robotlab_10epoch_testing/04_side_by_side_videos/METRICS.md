# Side-By-Side Videos

## How These Videos Were Created
These side-by-side videos were generated after the videos were fixed to even dimensions and encoded losslessly. They place the ground-truth Unity render next to the predicted-PVV render so visual differences can be inspected directly.

The predicted videos come from Unity `RenderPVV` mode, meaning Unity rendered the scene using PVVs predicted by either fVDB or spconv.

## Source Metrics

| Backend | SSIM ↑ | PSNR ↑ | VMAF ↑ | FLIP ↓ |
|---|---:|---:|---:|---:|
| fVDB side-by-side source | 0.998503 | 48.523265 | 96.666082 | 0.001803 |
| spconv side-by-side source | 0.998370 | 48.136690 | 96.489759 | 0.001916 |

## Interpretation
Use these videos for qualitative discussion with the supervisor. The numerical result says that both methods are visually very close to ground truth, with fVDB slightly ahead in this RobotLab test.
