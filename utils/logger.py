"""
Logging utilities for Human Detection Model training.

This module provides logging functionality including TensorBoard integration,
console logging, and CSV logging.
"""

import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import json


class TrainingLogger:
    """
    Logger for training metrics and statistics.

    Supports TensorBoard logging, console output, and CSV logging.

    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        use_tensorboard: Whether to use TensorBoard for logging

    Example:
        >>> logger = TrainingLogger('logs', 'experiment_1')
        >>> logger.log_step(0, {'loss': 0.5, 'lr': 0.001})
        >>> logger.log_epoch(0, {'train_loss': 0.5, 'val_loss': 0.4, 'mAP': 0.3})
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = 'experiment',
        use_tensorboard: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard

        # Create log directory
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer if available
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.experiment_dir))
                print(f"TensorBoard logging enabled. Run: tensorboard --logdir={self.log_dir}")
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False

        # CSV file for metrics
        self.csv_path = self.experiment_dir / 'metrics.csv'
        self.csv_file = None
        self.csv_writer = None

        # Initialize CSV file
        self._init_csv()

        # Metrics history
        self.metrics_history = {
            'step': [],
            'epoch': []
        }

    def _init_csv(self):
        """Initialize CSV file for logging."""
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'step',
            'epoch',
            'timestamp',
            'loss',
            'loc_loss',
            'conf_loss',
            'learning_rate',
            'train_loss',
            'val_loss',
            'mAP',
            'precision',
            'recall'
        ])
        self.csv_file.flush()

    def log_step(
        self,
        step: int,
        metrics: Dict[str, float],
        epoch: Optional[int] = None
    ):
        """
        Log metrics for a single training step.

        Args:
            step: Training step number
            metrics: Dictionary of metric names to values
            epoch: Optional epoch number

        Example:
            >>> logger.log_step(100, {'loss': 0.5, 'lr': 0.001})
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Update history
        self.metrics_history['step'].append(step)
        if epoch is not None:
            self.metrics_history['epoch'].append(epoch)

        # Log to TensorBoard
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f'step/{name}', value, step)

        # Log to CSV
        self._write_csv_row(
            step=step,
            epoch=epoch if epoch is not None else '',
            timestamp=timestamp,
            **metrics
        )

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Log metrics for a complete epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric names to values

        Example:
            >>> logger.log_epoch(1, {
            ...     'train_loss': 0.5,
            ...     'val_loss': 0.4,
            ...     'mAP': 0.3
            ... })
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Log to TensorBoard
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f'epoch/{name}', value, epoch)

        # Log to console
        self._print_epoch_summary(epoch, metrics)

        # Log to CSV
        self._write_csv_row(
            step=epoch * 10000,  # Use large step numbers for epochs
            epoch=epoch,
            timestamp=timestamp,
            **metrics
        )

        # Save epoch metrics to JSON
        self._save_epoch_metrics(epoch, metrics)

    def _write_csv_row(
        self,
        step: int,
        epoch: int,
        timestamp: str,
        **metrics
    ):
        """Write a row to the CSV file."""
        row = [
            step,
            epoch,
            timestamp,
            metrics.get('loss', ''),
            metrics.get('loc_loss', ''),
            metrics.get('conf_loss', ''),
            metrics.get('learning_rate', ''),
            metrics.get('train_loss', ''),
            metrics.get('val_loss', ''),
            metrics.get('mAP', ''),
            metrics.get('precision', ''),
            metrics.get('recall', '')
        ]

        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def _print_epoch_summary(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Print epoch summary to console."""
        print(f"\nEpoch {epoch} Summary:")
        print("-" * 50)
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")
        print("-" * 50)

    def _save_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Save epoch metrics to JSON file."""
        metrics_file = self.experiment_dir / 'epoch_metrics.jsonl'

        with open(metrics_file, 'a') as f:
            record = {'epoch': epoch, **metrics}
            f.write(json.dumps(record) + '\n')

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            hparams: Dictionary of hyperparameters

        Example:
            >>> logger.log_hyperparameters({
            ...     'batch_size': 16,
            ...     'learning_rate': 0.001,
            ...     'epochs': 100
            ... })
        """
        # Log to TensorBoard
        if self.writer is not None:
            # Flatten nested dictionaries
            flat_hparams = {}
            for key, value in hparams.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_hparams[f"{key}/{subkey}"] = subvalue
                else:
                    flat_hparams[key] = value

            self.writer.add_hparams(flat_hparams, {})

        # Save to JSON file
        hparams_file = self.experiment_dir / 'hparams.json'
        with open(hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2)

        print(f"Hyperparameters saved to {hparams_file}")

    def close(self):
        """Close the logger and flush all buffers."""
        if self.csv_file is not None:
            self.csv_file.close()

        if self.writer is not None:
            self.writer.close()

        print(f"Logs saved to {self.experiment_dir}")


def setup_logging(
    log_dir: str,
    experiment_name: str,
    use_tensorboard: bool = True
) -> TrainingLogger:
    """
    Convenience function to setup training logger.

    Args:
        log_dir: Directory for logs
        experiment_name: Name of the experiment
        use_tensorboard: Whether to use TensorBoard

    Returns:
        TrainingLogger instance

    Example:
        >>> logger = setup_logging('logs', 'experiment_1')
    """
    logger = TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard
    )
    return logger
