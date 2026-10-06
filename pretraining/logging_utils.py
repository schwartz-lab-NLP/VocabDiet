import json
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch
from dataclasses import asdict


class LossLogger:
    """Handles logging of training and validation losses to JSON files."""

    def __init__(self, log_dir: str, master_process: bool = True):
        self.log_dir = Path(log_dir)
        self.master_process = master_process

        if self.master_process:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.loss_logs = {
                "train": [],
                "val": [],
                "detailed": [],  # For additional loss components
            }

    def log_train_loss(
        self,
        step: int,
        loss: float,
        lr: float,
        additional_losses: Optional[Dict[str, float]] = None,
    ):
        """Log training loss with timestamp."""
        if not self.master_process:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "train_loss": float(loss),
            "learning_rate": float(lr),
            "training_time_ms": time.perf_counter() * 1000,
        }

        if additional_losses:
            entry.update({k: float(v) for k, v in additional_losses.items()})

        self.loss_logs["train"].append(entry)
        self._save_log("train")

    def log_val_loss(
        self,
        step: int,
        val_loss: float,
        training_time_ms: float,
        additional_losses: Optional[Dict[str, float]] = None,
    ):
        """Log validation loss with timestamp."""
        if not self.master_process:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "val_loss": float(val_loss),
            "training_time_ms": float(training_time_ms),
            "step_avg_ms": float(training_time_ms / max(step, 1)),
        }

        if additional_losses:
            entry.update({k: float(v) for k, v in additional_losses.items()})

        self.loss_logs["val"].append(entry)
        self._save_log("val")

    def _save_log(self, log_type: str):
        """Save specific log type to file."""
        if not self.master_process:
            return

        filepath = self.log_dir / f"{log_type}_losses.json"
        with open(filepath, "w") as f:
            json.dump(self.loss_logs[log_type], f, indent=2)


class RunTracker:
    """Tracks all training runs in a global CSV file."""

    def __init__(self, csv_path: str = "runs_log.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(
        self, hyperparams: Any, script_type: str, run_id: str, output_dir: str, log_dir: str = None
    ):
        """Log a new training run with all hyperparameters."""

        # Convert hyperparams dataclass to dict
        if hasattr(hyperparams, "__dict__"):
            params_dict = (
                asdict(hyperparams)
                if hasattr(hyperparams, "__dataclass_fields__")
                else vars(hyperparams)
            )
        else:
            params_dict = dict(hyperparams)

        # Create run entry
        run_entry = {
            "timestamp": datetime.now().isoformat(),
            "script_type": script_type,
            "run_id": str(run_id),
            "output_dir": str(output_dir),
            "log_dir": str(log_dir) if log_dir else None,
            **params_dict,
        }

        # Write to CSV
        file_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=run_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(run_entry)


class LogitsStatsCollector:
    """Collects and saves logits distribution statistics."""

    def __init__(self, stats_dir: str, master_process: bool = True, num_histogram_bins: int = 50):
        self.stats_dir = Path(stats_dir)
        self.master_process = master_process
        self.num_bins = num_histogram_bins

        if self.master_process:
            self.stats_dir.mkdir(parents=True, exist_ok=True)

    def collect_stats(self, logits_dict: Dict[str, torch.Tensor], step: int) -> Dict[str, Any]:
        """
        Collect statistics for logits tensors.

        Args:
            logits_dict: Dictionary mapping logits names to tensors
            step: Current training step

        Returns:
            Dictionary of statistics
        """
        if not self.master_process:
            return {}

        stats = {"step": step, "timestamp": datetime.now().isoformat()}

        for name, logits in logits_dict.items():
            if logits is None:
                continue

            # Convert to numpy for statistics
            logits_np = logits.detach().float().cpu().numpy().flatten()

            # Basic statistics
            tensor_stats = {
                f"{name}_min": float(np.min(logits_np)),
                f"{name}_max": float(np.max(logits_np)),
                f"{name}_mean": float(np.mean(logits_np)),
                f"{name}_std": float(np.std(logits_np)),
                f"{name}_shape": list(logits.shape),
            }

            # Histogram
            hist_counts, hist_edges = np.histogram(logits_np, bins=self.num_bins)
            tensor_stats[f"{name}_histogram"] = {
                "counts": hist_counts.tolist(),
                "edges": hist_edges.tolist(),
            }

            stats.update(tensor_stats)

        # Save to file
        self._save_stats(stats, step)
        return stats

    def _save_stats(self, stats: Dict[str, Any], step: int):
        """Save statistics to JSON file."""
        if not self.master_process:
            return

        filepath = self.stats_dir / f"logits_stats_step_{step}.json"
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)

    def get_wandb_logs(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract wandb-compatible logs from stats (without histograms).

        Args:
            stats: Statistics dictionary from collect_stats

        Returns:
            Dictionary suitable for wandb logging
        """
        wandb_logs = {}
        for key, value in stats.items():
            if key in ["step", "timestamp"]:
                continue
            if "_histogram" in key:
                continue  # Skip histograms for wandb (too large)
            if isinstance(value, (int, float)):
                wandb_logs[f"logits_stats/{key}"] = value

        return wandb_logs
