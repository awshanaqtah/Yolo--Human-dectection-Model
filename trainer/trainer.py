"""
Training loop for Human Detection Model.

This module implements the complete training loop with validation,
checkpointing, and metric tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import time

from ..models import SSD
from ..models.loss import MultiBoxLoss
from ..utils.metrics import calculate_map, calculate_precision_recall
from ..utils.common import AverageMeter, clip_gradients
from ..utils.logger import TrainingLogger
from .checkpoint import CheckpointManager
from .optimizer import create_scheduler, WarmupScheduler


class Trainer:
    """
    Trainer for Human Detection Model.

    Args:
        model: SSD model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        config: Configuration dictionary

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ...     device='cuda'
        ... )
        >>> trainer.train(num_epochs=100)
    """

    def __init__(
        self,
        model: SSD,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: MultiBoxLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        config: Dict = None,
        logger: Optional[TrainingLogger] = None,
        checkpoint_mgr: Optional[CheckpointManager] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        self.logger = logger
        self.checkpoint_mgr = checkpoint_mgr

        # Move model to device
        self.model.to(device)

        # Move loss function priors to device
        if hasattr(criterion, 'priors'):
            criterion.priors = criterion.priors.to(device)

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Best metric tracking
        self.best_metric = 0.0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics

        Example:
            >>> train_metrics = trainer.train_epoch()
            >>> print(f"Train loss: {train_metrics['loss']:.4f}")
        """
        self.model.train()

        # Metrics
        losses = AverageMeter()
        loc_losses = AverageMeter()
        conf_losses = AverageMeter()

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            # Forward pass
            class_pred, loc_pred = self.model(images)

            # Calculate loss
            loss, loss_dict = self.criterion(
                (loc_pred, class_pred),
                targets,
                self.model.priors.to(self.device)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip') is not None:
                clip_gradients(self.model, max_norm=self.config['grad_clip'])

            # Optimizer step
            self.optimizer.step()

            # Update metrics
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            loc_losses.update(loss_dict['loc_loss'], batch_size)
            conf_losses.update(loss_dict['conf_loss'], batch_size)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'loc': f'{loc_losses.avg:.4f}',
                'conf': f'{conf_losses.avg:.4f}'
            })

            # Log to TensorBoard
            if self.logger is not None and batch_idx % self.config.get('log_freq', 50) == 0:
                self.logger.log_step(
                    step=self.global_step,
                    metrics={
                        'loss': loss.item(),
                        'loc_loss': loss_dict['loc_loss'],
                        'conf_loss': loss_dict['conf_loss'],
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    },
                    epoch=self.current_epoch
                )

            self.global_step += 1

        return {
            'loss': losses.avg,
            'loc_loss': loc_losses.avg,
            'conf_loss': conf_losses.avg
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics

        Example:
            >>> val_metrics = trainer.validate()
            >>> print(f"mAP: {val_metrics['mAP']:.4f}")
        """
        self.model.eval()

        # Metrics
        losses = AverageMeter()

        # Collect predictions for mAP calculation
        all_predictions = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc='Validation')

        for images, targets in pbar:
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            # Forward pass
            class_pred, loc_pred = self.model(images)

            # Calculate loss
            loss, _ = self.criterion(
                (loc_pred, class_pred),
                targets,
                self.model.priors.to(self.device)
            )

            # Update loss metric
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)

            # Get detections for mAP calculation
            detections = self.model.detect(
                images,
                conf_threshold=self.config.get('conf_threshold', 0.5),
                nms_threshold=self.config.get('nms_threshold', 0.45),
                top_k=200,
                keep_top_k=100
            )

            all_predictions.extend(detections)
            all_targets.extend(targets)

        # Calculate mAP
        mAP, _ = calculate_map(
            predictions=all_predictions,
            targets=all_targets,
            iou_threshold=0.5,
            num_classes=self.model.num_classes
        )

        # Calculate precision and recall
        precision, recall = calculate_precision_recall(
            predictions=all_predictions,
            targets=all_targets,
            iou_threshold=0.5
        )

        return {
            'loss': losses.avg,
            'mAP': mAP,
            'precision': precision,
            'recall': recall
        }

    def train(self, num_epochs: int, start_epoch: int = 0):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming)

        Example:
            >>> trainer.train(num_epochs=100)
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, WarmupScheduler):
                    self.scheduler.step()
                else:
                    self.scheduler.step()

            # Log epoch results
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"mAP: {val_metrics['mAP']:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f}")
            print(f"Recall: {val_metrics['recall']:.4f}")

            # Log to TensorBoard
            if self.logger is not None:
                self.logger.log_epoch(
                    epoch=epoch,
                    metrics={
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'mAP': val_metrics['mAP'],
                        'precision': val_metrics['precision'],
                        'recall': val_metrics['recall']
                    }
                )

            # Save checkpoint
            if self.checkpoint_mgr is not None:
                self.checkpoint_mgr.save(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=val_metrics
                )

        print("\nTraining completed!")

        if self.logger is not None:
            self.logger.close()


def create_trainer(
    model: SSD,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda',
    config: Dict = None,
    log_dir: str = 'trainer/logs',
    checkpoint_dir: str = 'trainer/checkpoints'
) -> Trainer:
    """
    Convenience function to create a trainer.

    Args:
        model: SSD model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        config: Configuration dictionary
        log_dir: Directory for logs
        checkpoint_dir: Directory for checkpoints

    Returns:
        Trainer instance

    Example:
        >>> trainer = create_trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     config=config
        ... )
    """
    # Create loss function
    criterion = model.get_loss_criterion()

    # Create logger
    logger = TrainingLogger(
        log_dir=log_dir,
        experiment_name=f'training_{int(time.time())}',
        use_tensorboard=config.get('use_tensorboard', True) if config else True
    )

    # Log hyperparameters
    if config is not None:
        logger.log_hyperparameters(config)

    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        save_best_only=config.get('save_best_only', True) if config else True,
        save_freq=config.get('checkpoint_freq', 10) if config else 10,
        metric_name='mAP',
        mode='max'
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config or {},
        logger=logger,
        checkpoint_mgr=checkpoint_mgr
    )

    return trainer
