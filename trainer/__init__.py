"""
Trainer module for Human Detection Model.
Contains training loop, optimizer, and checkpoint management.
"""

from .trainer import Trainer, create_trainer
from .optimizer import create_optimizer, create_scheduler, WarmupScheduler
from .checkpoint import CheckpointManager

__all__ = [
    'Trainer',
    'create_trainer',
    'create_optimizer',
    'create_scheduler',
    'WarmupScheduler',
    'CheckpointManager'
]
