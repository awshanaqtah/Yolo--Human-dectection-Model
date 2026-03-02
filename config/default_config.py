"""
Default Configuration for Human Detection Model.

This class contains all hyperparameters and settings for training,
evaluation, and inference of the SSD-based human detection model.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Config:
    """
    Configuration class for Human Detection Model.

    Attributes:
        # Model Configuration
        input_size: Input image size (height, width)
        num_classes: Number of classes (including background)
        backbone: Backbone network name ('mobilenet_v2' or 'resnet50')

        # Anchor/Prior Box Configuration
        min_sizes: Minimum sizes for prior boxes at each feature map level
        max_sizes: Maximum sizes for prior boxes at each feature map level
        aspect_ratios: Aspect ratios for prior boxes
        feature_maps: Feature map sizes for each detection layer

        # Training Configuration
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        momentum: Momentum for SGD optimizer
        weight_decay: Weight decay (L2 regularization)
        lr_decay_steps: Epochs at which to decay learning rate
        lr_decay_gamma: Multiplicative factor of learning rate decay
        warmup_epochs: Number of warmup epochs

        # Data Configuration
        train_data_path: Path to training data
        val_data_path: Path to validation data
        test_data_path: Path to test data
        annotation_format: Format of annotations ('coco', 'voc', or 'yolo')
        num_workers: Number of data loading workers

        # Augmentation Configuration
        train_transforms: Whether to apply data augmentation during training
        horizontal_flip_prob: Probability of horizontal flip
        color_jitter: Color jitter parameters (brightness, contrast, saturation, hue)

        # Checkpoint Configuration
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Save checkpoint every N epochs
        resume_from: Path to checkpoint to resume from
        save_best_only: Whether to save only the best model

        # Logging Configuration
        log_dir: Directory to save logs
        log_freq: Log training metrics every N iterations
        use_tensorboard: Whether to use TensorBoard for logging

        # Inference Configuration
        conf_threshold: Confidence threshold for detections
        nms_threshold: IoU threshold for Non-Maximum Suppression
        top_k: Maximum number of detections per image
        variances: Variances for decoding bounding boxes

        # System Configuration
        device: Device to use ('cuda', 'mps', or 'cpu')
        seed: Random seed for reproducibility
        pin_memory: Whether to pin memory for faster GPU transfer
    """

    # ========== Model Configuration ==========
    input_size: Tuple[int, int] = (300, 300)
    num_classes: int = 2  # Background + Person
    backbone: str = 'mobilenet_v2'

    # ========== Prior Box Configuration ==========
    # SSD300 prior box configuration
    min_sizes: List[int] = field(default_factory=lambda: [30, 60, 111, 162, 213, 264])
    max_sizes: List[int] = field(default_factory=lambda: [60, 111, 162, 213, 264, 315])
    aspect_ratios: List[List[int]] = field(default_factory=lambda: [[2], [2, 3], [2, 3], [2, 3], [2], [2]])
    feature_maps: List[int] = field(default_factory=lambda: [38, 19, 10, 5, 3, 1])
    variances: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.1, 0.2])

    # ========== Training Configuration ==========
    batch_size: int = 16
    num_epochs: int = 120
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_decay_steps: List[int] = field(default_factory=lambda: [80, 100])
    lr_decay_gamma: float = 0.1
    warmup_epochs: int = 5

    # ========== Data Configuration ==========
    train_data_path: str = 'datasets/raw/images/train'
    val_data_path: str = 'datasets/raw/images/val'
    test_data_path: str = 'datasets/raw/images/test'
    train_annotation_path: str = 'datasets/raw/annotations/train.json'
    val_annotation_path: str = 'datasets/raw/annotations/val.json'
    test_annotation_path: str = 'datasets/raw/annotations/test.json'
    annotation_format: str = 'coco'  # 'coco', 'voc', or 'yolo'
    num_workers: int = 4

    # ========== Augmentation Configuration ==========
    train_transforms: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.1)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)    # ImageNet

    # ========== Checkpoint Configuration ==========
    checkpoint_dir: str = 'trainer/checkpoints'
    checkpoint_freq: int = 10
    resume_from: Optional[str] = None
    save_best_only: bool = True

    # ========== Logging Configuration ==========
    log_dir: str = 'trainer/logs'
    log_freq: int = 50
    use_tensorboard: bool = True

    # ========== Inference Configuration ==========
    conf_threshold: float = 0.5
    nms_threshold: float = 0.45
    top_k: int = 200
    keep_top_k: int = 100

    # ========== System Configuration ==========
    device: str = 'cuda'  # Will be set automatically if 'cuda'
    seed: int = 42
    pin_memory: bool = True

    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Set device automatically if needed
        if self.device == 'cuda':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Validate annotation format
        assert self.annotation_format in ['coco', 'voc', 'yolo'], \
            f"Invalid annotation format: {self.annotation_format}"

        # Validate backbone
        assert self.backbone in ['mobilenet_v2', 'resnet50'], \
            f"Invalid backbone: {self.backbone}"

    def save(self, path: str):
        """Save configuration to YAML file."""
        import yaml
        os.makedirs(os.path.dirname(path), exist_ok=True)

        config_dict = {
            'model': {
                'input_size': self.input_size,
                'num_classes': self.num_classes,
                'backbone': self.backbone,
            },
            'training': {
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay,
                'lr_decay_steps': self.lr_decay_steps,
                'lr_decay_gamma': self.lr_decay_gamma,
                'warmup_epochs': self.warmup_epochs,
            },
            'data': {
                'train_data_path': self.train_data_path,
                'val_data_path': self.val_data_path,
                'test_data_path': self.test_data_path,
                'train_annotation_path': self.train_annotation_path,
                'val_annotation_path': self.val_annotation_path,
                'test_annotation_path': self.test_annotation_path,
                'annotation_format': self.annotation_format,
                'num_workers': self.num_workers,
            },
            'augmentation': {
                'train_transforms': self.train_transforms,
                'horizontal_flip_prob': self.horizontal_flip_prob,
                'color_jitter': self.color_jitter,
            },
            'inference': {
                'conf_threshold': self.conf_threshold,
                'nms_threshold': self.nms_threshold,
                'top_k': self.top_k,
            },
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        import yaml

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Extract nested config
        kwargs = {}
        for section in config_dict.values():
            kwargs.update(section)

        return cls(**kwargs)

    def __str__(self) -> str:
        """Return string representation of configuration."""
        return f"Config(input_size={self.input_size}, num_classes={self.num_classes}, " \
               f"backbone={self.backbone}, batch_size={self.batch_size}, " \
               f"learning_rate={self.learning_rate})"
