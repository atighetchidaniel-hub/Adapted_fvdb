# BigCity Clean fVDB d16 200ep: All Metrics

This folder collects the important metric and trace files for the BigCity clean fVDB d16 200-epoch run.

## Files

| File | Contents |
|---|---|
| `bigcity_clean_fvdb_d16_200ep_summary_metrics.csv` | One-row summary with voxel metrics, raw frame metrics, and repo video metrics. |
| `bigcity_clean_fvdb_d16_200ep_summary_metrics.json` | Same summary in JSON format. |
| `bigcity_clean_fvdb_d16_200ep_eval_stats.csv` | Full inference/evaluation stats, including dice, fp, fn, tp, tn, fp_rate, fn_rate, fp_ratio, gv_ratio, and loss. |
| `bigcity_clean_fvdb_d16_200ep_train_log.csv` | Training log over epochs/steps. |
| `bigcity_clean_fvdb_d16_200ep_eval_log.csv` | Evaluation log during training. |
| `bigcity_clean_fvdb_d16_200ep_training_arguments.json` | Training configuration and hyperparameters. |
| `bigcity_clean_fvdb_d16_200ep_repo_video_metrics.json` | Raw output from the repository video metric script: SSIM, PSNR, VMAF, and FLIP. |

## Notes

The summary metrics are best for thesis tables.

The eval stats CSV is best for detailed analysis, because it includes the repository voxel-level metrics such as `fp_ratio` and `gv_ratio`.

The repo video metrics are computed from normalized encoded videos using the repository metric script.
