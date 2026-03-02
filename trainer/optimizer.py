"""
Optimizer and learning rate scheduler setup for training.

This module provides utilities for creating optimizers and learning rate schedulers
for training the Human Detection Model.
"""

import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, MultiStepLR
from typing import Dict, Any, Optional


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = 'sgd',
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    different_lr: bool = False
) -> torch.optim.Optimizer:
    """
    Create optimizer for model training.

    Args:
        model: PyTorch model
        optimizer_name: Type of optimizer ('sgd', 'adam', 'adamw')
        learning_rate: Learning rate
        momentum: Momentum for SGD
        weight_decay: Weight decay (L2 regularization)
        different_lr: Whether to use different learning rates for backbone and head

    Returns:
        Optimizer instance

    Example:
        >>> optimizer = create_optimizer(
        ...     model,
        ...     optimizer_name='sgd',
        ...     learning_rate=1e-3
        ... )
    """
    # Get parameter groups
    if different_lr:
        # Different learning rates for backbone and head
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name or 'features' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': learning_rate}
        ]
    else:
        param_groups = model.parameters()

    # Create optimizer
    if optimizer_name.lower() == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adam':
        optimizer = Adam(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adamw':
        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    print(f"Created {optimizer_name.upper()} optimizer with lr={learning_rate}")
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'multistep',
    steps: Optional[int] = None,
    milestones: list = None,
    gamma: float = 0.1,
    warmup_epochs: int = 0
) -> Optional[Any]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_name: Type of scheduler ('step', 'multistep', 'cosine', 'warmup')
        steps: Step size for StepLR
        milestones: List of epochs for MultiStepLR
        gamma: Multiplicative factor of learning rate decay
        warmup_epochs: Number of warmup epochs

    Returns:
        Scheduler instance

    Example:
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     scheduler_name='multistep',
        ...     milestones=[80, 100],
        ...     gamma=0.1
        ... )
    """
    if scheduler_name.lower() == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=steps if steps is not None else 30,
            gamma=gamma
        )
    elif scheduler_name.lower() == 'multistep':
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones if milestones is not None else [60, 80],
            gamma=gamma
        )
    elif scheduler_name.lower() == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=100,  # Maximum number of iterations
            eta_min=1e-6
        )
    elif scheduler_name.lower() == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    if scheduler is not None:
        print(f"Created {scheduler_name.upper()} scheduler")

    return scheduler


class WarmupScheduler:
    """
    Learning rate scheduler with warmup.

    Gradually increases learning rate during warmup period,
    then switches to the main scheduler.

    Args:
        optimizer: Optimizer instance
        warmup_epochs: Number of warmup epochs
        warmup_lr: Initial learning rate
        after_scheduler: Scheduler to use after warmup

    Example:
        >>> main_scheduler = MultiStepLR(optimizer, milestones=[60, 80])
        >>> scheduler = WarmupScheduler(optimizer, 5, 1e-5, main_scheduler)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_lr: float,
        after_scheduler: Any = None
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * lr_scale

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.after_scheduler is not None:
            # Use main scheduler
            self.after_scheduler.step()

        self.current_epoch += 1

    def state_dict(self):
        """Get scheduler state."""
        return {
            'current_epoch': self.current_epoch,
            'after_scheduler_state': self.after_scheduler.state_dict() if self.after_scheduler else None
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_epoch = state_dict['current_epoch']
        if self.after_scheduler and state_dict.get('after_scheduler_state'):
            self.after_scheduler.load_state_dict(state_dict['after_scheduler_state'])
