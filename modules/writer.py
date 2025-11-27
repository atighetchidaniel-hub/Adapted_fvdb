import csv
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir="logs", train_csv="train_log.csv", eval_csv="eval_log.csv"):
        """
        Initializes the logger with separate logs for training and evaluation.

        Args:
            log_dir (str): Directory where logs (TensorBoard, CSV) are stored.
            train_csv (str): CSV filename for training logs.
            eval_csv (str): CSV filename for evaluation logs.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard Writer
        self.writer = SummaryWriter(
            log_dir=os.path.join(log_dir, "tensorboard"))

        # File paths for CSV logs
        self.train_csv = os.path.join(log_dir, train_csv)
        self.eval_csv = os.path.join(log_dir, eval_csv)

        # Flags to track if CSV headers are written
        self.train_csv_initialized = False
        self.eval_csv_initialized = False

    def log_metrics(self, metrics: dict, step: int, mode: str = "train", log_to_stdout: bool = True, log_to_csv: bool = True):
        """
        Logs metrics to stdout, TensorBoard, and CSV.

        Args:
            metrics (dict): Dictionary where keys are metric names and values are numeric, torch.Tensor, or np.array.
            step (int): The step (or epoch) for logging.
            mode (str): Either 'train' or 'eval' to specify where to log the metrics.
            log_to_stdout (bool): Whether to print the metrics to stdout.
        """
        assert mode in [
            "train", "eval", "infer"], "Mode must be either 'train' or 'eval'"

        processed_metrics = {}

        # Convert tensors and numpy arrays to Python scalars
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                value = value.item() if value.size == 1 else value
            elif isinstance(value, list):
                value = np.array(value).flatten() if len(value) > 0 else value[0]
            processed_metrics[key] = value

        # Print to stdout
        if log_to_stdout:
            log_str = f"Step {step} [{mode}] - " + \
                " | ".join(f"{k}: {v}" for k, v in processed_metrics.items())
            print(log_str)

        # Write to TensorBoard
        for key, value in processed_metrics.items():
            tag = f"{mode}/{key}"
            if isinstance(value, (int, float)):
                self.writer.add_scalar(tag, value, step)
            elif isinstance(value, np.ndarray):
                # histogram – only if there’s actual data
                if value.size > 0:
                    self.writer.add_histogram(tag, value, step)
                else:
                    # skip empty arrays
                    continue
        # Write to the appropriate CSV file
        if log_to_csv:
            csv_file = self.train_csv if mode == "train" else self.eval_csv
            csv_initialized = self.train_csv_initialized if mode == "train" else self.eval_csv_initialized
            self._write_csv(csv_file, processed_metrics, step, csv_initialized)

        # Update initialization flag
        if mode == "train":
            self.train_csv_initialized = True
        else:
            self.eval_csv_initialized = True

    def _write_csv(self, filename: str, metrics: dict, step: int, initialized: bool):
        """Writes metrics to the specified CSV file."""
        write_header = not initialized

        with open(filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                # Write headers
                writer.writerow(["Step"] + list(metrics.keys()))
            writer.writerow([step] + list(metrics.values()))

    def compute_stats(self):
        """Computes statistics from the CSV files."""
        if not os.path.exists(self.eval_csv):
            print(f"Evaluation CSV file {self.eval_csv} does not exist.")
            return
        # Read the CSV file with pandas
        
        df = pd.read_csv(self.eval_csv)
        # For each column besides 'Step', compute mean, std, min, and max
        stats = {}
        for column in df.columns[1:]:
            stats[column] = {
                "mean": df[column].mean(),
                "std": df[column].std(),
                "min": df[column].min(),
                "max": df[column].max()
            }
        # Print the statistics
        for key, value in stats.items():
            print(f"Statistics for {key}:")
            print(f"  Mean: {value['mean']}")
            print(f"  Std: {value['std']}")
            print(f"  Min: {value['min']}")
            print(f"  Max: {value['max']}")
            print()
        
        # Write the statistics to a CSV file
        stats_filename = os.path.join(self.log_dir, "eval_stats.csv")
        with open(stats_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Mean", "Std", "Min", "Max"])
            for key, value in stats.items():
                writer.writerow([key, value["mean"], value["std"],
                                 value["min"], value["max"]])
                
        # Write to tensorboard
        for key, value in stats.items():
            self.writer.add_scalar(f"eval/{key}/mean", value["mean"], 0)
            self.writer.add_scalar(f"eval/{key}/std", value["std"], 0)
            self.writer.add_scalar(f"eval/{key}/min", value["min"], 0)
            self.writer.add_scalar(f"eval/{key}/max", value["max"], 0)

        print(f"Statistics written to {stats_filename}")

    def close(self):
        """Closes the TensorBoard writer."""
        self.writer.close()


# Example Usage
if __name__ == "__main__":
    logger = TrainingLogger(log_dir="training_logs")

    for step in range(1, 5):
        train_metrics = {
            "loss": torch.tensor(0.05 * step),
            "accuracy": np.array(0.85 + step * 0.01)
        }
        eval_metrics = {
            "loss": torch.tensor(0.03 * step),
            "accuracy": np.array(0.87 + step * 0.01)
        }

        logger.log_metrics(train_metrics, step, mode="train")
        logger.log_metrics(eval_metrics, step, mode="eval")

    logger.close()
