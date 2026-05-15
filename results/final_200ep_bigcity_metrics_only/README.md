# BigCity Final 200 Epoch Metrics

Metrics-only export for BigCity final 200-epoch OA-CNN Interleaved runs.

This folder intentionally excludes videos, checkpoints, and predicted PVV binary outputs.

Source run root: `/var/tmp/atighedl_runs/final_all_scenes_oacnn_interleaved_200ep/20260513-180853`

Rows included: `18 / 18 BigCity runs`


## Summary

| Radius | d | Backend | Pred | Dice | fp_rate | fn_rate | gv_ratio | Infer time mean | Peak MB | Params |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r30 | 8 | fvdb | 97 | 0.994762 | 0.003218 | 0.001022 | 0.296752 | 38.762 | 256.3 | 9168192 |
| r30 | 8 | spconv | 97 | 0.993004 | 0.003473 | 0.000867 | 0.306336 | 17.037 | 316.6 | 9548096 |
| r30 | 16 | fvdb | 97 | 0.988137 | 0.004953 | 0.001301 | 0.361645 | 41.344 | 360.5 | 15708992 |
| r30 | 16 | spconv | 97 | 0.987327 | 0.005085 | 0.001365 | 0.366524 | 17.662 | 386.4 | 16088896 |
| r30 | 32 | fvdb | 97 | 0.982058 | 0.008607 | 0.007032 | 0.497707 | 43.536 | 705.2 | 68035392 |
| r30 | 32 | spconv | 97 | 0.989948 | 0.005090 | 0.001499 | 0.367249 | 23.574 | 666.9 | 68415296 |
| r60 | 8 | fvdb | 55 | 0.992888 | 0.007465 | 0.002128 | 0.489010 | 38.475 | 254.8 | 9168192 |
| r60 | 8 | spconv | 55 | 0.992935 | 0.005547 | 0.002121 | 0.417728 | 17.307 | 314.6 | 9548096 |
| r60 | 16 | fvdb | 55 | 0.987178 | 0.008947 | 0.002397 | 0.544699 | 41.216 | 358.8 | 15708992 |
| r60 | 16 | spconv | 55 | 0.986923 | 0.007964 | 0.002948 | 0.507767 | 17.706 | 384.8 | 16088896 |
| r60 | 32 | fvdb | 55 | 0.978388 | 0.015953 | 0.002746 | 0.807196 | 44.293 | 714.9 | 68035392 |
| r60 | 32 | spconv | 55 | 0.982839 | 0.012103 | 0.002363 | 0.663277 | 23.253 | 672.9 | 68415296 |
| r90 | 8 | fvdb | 41 | 0.991616 | 0.014608 | 0.000274 | 0.819391 | 38.902 | 254.5 | 9168192 |
| r90 | 8 | spconv | 41 | 0.993468 | 0.008374 | 0.001441 | 0.585574 | 17.275 | 313.9 | 9548096 |
| r90 | 16 | fvdb | 41 | 0.988257 | 0.011992 | 0.001491 | 0.721439 | 40.865 | 351.7 | 15708992 |
| r90 | 16 | spconv | 41 | 0.988549 | 0.010076 | 0.001470 | 0.649335 | 17.553 | 380.5 | 16088896 |
| r90 | 32 | fvdb | 41 | 0.979912 | 0.016445 | 0.002798 | 0.886553 | 43.599 | 711.3 | 68035392 |
| r90 | 32 | spconv | 41 | 0.983037 | 0.013731 | 0.001954 | 0.786431 | 23.143 | 670.9 | 68415296 |

## Contents

- `bigcity_200ep_summary.csv`: all BigCity rows from the full sweep summary.

- `bigcity_preflight.csv`: GV/PVV/predicted/render availability checks for BigCity.

- `runs/*`: per-run eval stats, timing logs, training logs, and training arguments.

- `video_metrics/*`: video metric JSON/CSV/log files if already generated, excluding videos.
