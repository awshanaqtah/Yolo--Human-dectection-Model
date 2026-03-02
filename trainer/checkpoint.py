"""
Checkpoint management for Human Detection Model.

This module handles saving and loading model checkpoints during training.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """
    Manage model checkpoints during training.

    Args:
        checkpoint_dir: Directory to save checkpoints
        save_best_only: Whether to save only the best model
        save_freq: Save checkpoint every N epochs
        metric_name: Metric to monitor for best model (e.g., 'mAP')
        mode: 'min' or 'max' - whether lower or higher is better

    Example:
        >>> checkpoint_mgr = CheckpointManager(
        ...     'checkpoints',
        ...     save_best_only=True,
        ...     metric_name='mAP',
        ...     mode='max'
        ... )
        >>> checkpoint_mgr.save(
        ...     epoch=10,
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     metrics={'mAP': 0.5, 'loss': 0.3}
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_best_only: bool = True,
        save_freq: int = 10,
        metric_name: str = 'mAP',
        mode: str = 'max'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.metric_name = metric_name
        self.mode = mode

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track best metric value
        self.best_metric = float('-inf') if mode == 'max' else float('inf')

        print(f"Checkpoint manager initialized: {self.checkpoint_dir}")

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        metrics: Dict[str, float] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Save a checkpoint.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler to save
            metrics: Dictionary of metrics
            **kwargs: Additional items to save

        Returns:
            Path to saved checkpoint if saved, None otherwise

        Example:
            >>> checkpoint_mgr.save(
            ...     epoch=10,
            ...     model=model,
            ...     optimizer=optimizer,
            ...     metrics={'mAP': 0.5, 'loss': 0.3}
            ... )
        """
        # Check if we should save
        if self.save_best_only and metrics is not None:
            current_metric = metrics.get(self.metric_name, 0)

            # Check if this is the best model
            is_best = False
            if self.mode == 'max' and current_metric > self.best_metric:
                is_best = True
                self.best_metric = current_metric
            elif self.mode == 'min' and current_metric < self.best_metric:
                is_best = True
                self.best_metric = current_metric

            if not is_best:
                return None

            # Save best model
            checkpoint_name = 'best_model.pth'
        else:
            # Save periodic checkpoint
            if epoch % self.save_freq != 0:
                return None

            checkpoint_name = f'checkpoint_epoch_{epoch}.pth'

        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics if metrics is not None else {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **kwargs
        }

        # Save scheduler state if available
        if scheduler is not None:
            if hasattr(scheduler, 'state_dict'):
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            elif hasattr(scheduler, 'after_scheduler') and scheduler.after_scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.after_scheduler.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")
        if metrics is not None and self.metric_name in metrics:
            print(f"  {self.metric_name}: {metrics[self.metric_name]:.4f}")

        return str(checkpoint_path)

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Any = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint to

        Returns:
            Checkpoint dictionary

        Example:
            >>> checkpoint = checkpoint_mgr.load(
            ...     'checkpoints/best_model.pth',
            ...     model,
            ...     optimizer=optimizer
            ... )
            >>> start_epoch = checkpoint['epoch'] + 1
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif hasattr(scheduler, 'after_scheduler') and scheduler.after_scheduler:
                scheduler.after_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint and checkpoint['metrics']:
            print(f"  Metrics: {checkpoint['metrics']}")

        return checkpoint

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Any = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load the best checkpoint.

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint to

        Returns:
            Checkpoint dictionary

        Example:
            >>> checkpoint = checkpoint_mgr.load_best(model, optimizer)
        """
        best_checkpoint_path = self.checkpoint_dir / 'best_model.pth'

        if not best_checkpoint_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint_path}")

        return self.load(
            str(best_checkpoint_path),
            model,
            optimizer,
            scheduler,
            device
        )

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the most recent checkpoint.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist

        Example:
            >>> latest = checkpoint_mgr.get_latest_checkpoint()
        """
        checkpoints = list(self.checkpoint_dir.glob('*.pth'))

        if not checkpoints:
            return None

        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Prefer best_model.pth if it exists
        best_checkpoint = self.checkpoint_dir / 'best_model.pth'
        if best_checkpoint in checkpoints:
            return str(best_checkpoint)

        return str(checkpoints[0])

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Remove old checkpoints, keeping only the most recent N.

        Args:
            keep_last_n: Number of recent checkpoints to keep

        Example:
            >>> checkpoint_mgr.cleanup_old_checkpoints(keep_last_n=3)
        """
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))

        # Don't remove best_model.pth
        if len(checkpoints) <= keep_last_n:
            return

        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old checkpoints
        for checkpoint in checkpoints[keep_last_n:]:
            checkpoint.unlink()
            print(f"Removed old checkpoint: {checkpoint}")
