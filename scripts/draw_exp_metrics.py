
#!/usr/bin/env python
import os
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import argparse


def main(log_root=None, metrics=None):
    records = []
    for exp in sorted(os.listdir(log_root)):
        exp_dir = os.path.join(log_root, exp)
        if not os.path.isdir(exp_dir):
            continue

        tb_dir = os.path.join(exp_dir, "tensorboard")
        if not os.path.isdir(tb_dir):
            continue

        # read all scalar summaries under tensorboard/
        sr = SummaryReader(tb_dir)
        df = sr.scalars  # columns: [wall_time, step, tag, value]

        if df.empty:
            print(f"Warning: {tb_dir} is empty")
            continue

        # compute mean for each metric
        means = {"exp": exp}
        for m in metrics:
            vals = df.loc[df.tag == m, "value"]
            means[m] = vals.mean() if not vals.empty else float("nan")
        if any(pd.isna(means[m]) for m in metrics):
            print(f"Warning: {tb_dir} contains NaN values")
            continue
        records.append(means)

    df_means = pd.DataFrame.from_records(records).set_index("exp")

    # Sort by the first metric
    df_means = df_means.sort_values(metrics[0])

    # ── 3. plot ───────────────────────────────────────────────────────────────────
    ax = df_means.plot.bar(figsize=(10, 6))
    ax.set_ylabel("Mean value")
    ax.set_title("Per-experiment mean of val_loss and val_accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Save to CSV
    df_means.to_csv("data/metrics_{}.csv".format(os.path.basename(log_root)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw experiment metrics")
    parser.add_argument(
        "--log_root", type=str, default=".", help="Root directory of the logs")
    parser.add_argument(
        "--metrics", type=str, nargs="+", default=["infer/fn_rate", "infer/fp_ratio"],
        help="List of metrics to plot (default: val_loss, val_accuracy)"
    )
    args = parser.parse_args()
    log_root = args.log_root
    metrics = args.metrics

    print("Log root:", log_root)
    print("Metrics:", metrics)
    main(log_root, metrics)
