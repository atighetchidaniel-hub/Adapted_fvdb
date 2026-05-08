# Industrial Clean fvdb d16 10ep: All Metrics

This folder collects the important metric, configuration, and trace files for the Industrial clean fvdb d16 10-epoch run.

## Included Files

| File | Contents |
|---|---|
| `industrial_fvdb_d16_10ep_eval_stats.csv` | Full inference/evaluation stats, including dice, fp, fn, tp, tn, fp_rate, fn_rate, fp_ratio, gv_ratio, and loss where available. |
| `industrial_fvdb_d16_10ep_repo_video_metrics.json` | Raw output from the repository video metric script: SSIM, PSNR, VMAF, and FLIP. |
| `industrial_fvdb_d16_10ep_decoded_video_metrics.json/csv` | Decoded-frame comparison summary from the normalized videos, if generated. |
| `industrial_fvdb_d16_10ep_raw_png_metrics.json/csv` | Raw PNG frame comparison summary, if generated after clean image-sequence rendering. |
| `industrial_fvdb_d16_10ep_train_log.csv` | Training log, if available. |
| `industrial_fvdb_d16_10ep_eval_log.csv` | Evaluation log during training, if available. |
| `industrial_fvdb_d16_10ep_training_arguments.json` | Training configuration and hyperparameters, if available. |
| `industrial_fvdb_d16_10ep_run_video_metrics.log` | Log from the repository video metrics run. |

## Source Paths

| Artifact | Path |
|---|---|
| Video metrics folder | `results/video_metrics/industrial_fvdb_d16_10ep_video` |
| Inference folder | `data_for_test/out/OACNNsInterleaved_industrial_clean_dice_20260508-165150_industrial_clean_fvdb_d16_10ep-industrial_clean_fvdb_d16_10ep_infer_EtgImGP3` |
| Training folder | `data_for_test/out/OACNNsInterleaved_industrial_clean_dice,no_guess_20260508-164212_industrial_clean_fvdb_d16_10ep_dhQ8xhWZ` |

## Notes

The repo video metrics are the main video-quality metrics for comparison with the repository pipeline.

The decoded-frame metrics are useful sanity checks for video exports.

The raw PNG metrics should only be used once the GT, fVDB, and spconv `00_color` folders are confirmed to have no missing frames.
