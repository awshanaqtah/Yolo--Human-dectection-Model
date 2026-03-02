"""
Training script for Human Detection Model.

This script trains a custom SSD model for human detection.
"""

import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from models import create_ssd
from data import (
    HumanDetectionDataset,
    create_train_val_dataloaders,
    get_train_transform,
    get_val_transform
)
from trainer import create_trainer, create_optimizer
from utils import set_seed, get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Human Detection Model')

    # Data arguments
    parser.add_argument('--train-data', type=str, default='datasets/raw/images/train',
                        help='Path to training images')
    parser.add_argument('--train-ann', type=str, default='datasets/raw/annotations/train.json',
                        help='Path to training annotations')
    parser.add_argument('--val-data', type=str, default='datasets/raw/images/val',
                        help='Path to validation images')
    parser.add_argument('--val-ann', type=str, default='datasets/raw/annotations/val.json',
                        help='Path to validation annotations')
    parser.add_argument('--annotation-format', type=str, default='coco',
                        choices=['coco', 'voc', 'yolo'],
                        help='Annotation format')

    # Model arguments
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'resnet50'],
                        help='Backbone network')
    parser.add_argument('--input-size', type=int, nargs=2, default=[300, 300],
                        help='Input image size (height width)')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=120,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Warmup epochs')

    # System arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-dir', type=str, default='trainer/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='trainer/logs',
                        help='Directory to save logs')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("Human Detection Model Training")
    print("=" * 60)

    # Set random seed
    set_seed(args.seed)

    # Setup device
    if args.device is None:
        device = get_device()
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create configuration
    config = Config()
    config.train_data_path = args.train_data
    config.train_annotation_path = args.train_ann
    config.val_data_path = args.val_data
    config.val_annotation_path = args.val_ann
    config.annotation_format = args.annotation_format
    config.backbone = args.backbone
    config.input_size = tuple(args.input_size)
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.momentum = args.momentum
    config.weight_decay = args.weight_decay
    config.warmup_epochs = args.warmup
    config.num_workers = args.num_workers
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    config.device = device

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = HumanDetectionDataset(
        image_dir=config.train_data_path,
        annotation_path=config.train_annotation_path,
        annotation_format=config.annotation_format,
        transform=get_train_transform(
            input_size=config.input_size,
            horizontal_flip_prob=0.5,
            color_jitter=(0.2, 0.2, 0.2, 0.1)
        )
    )

    val_dataset = HumanDetectionDataset(
        image_dir=config.val_data_path,
        annotation_path=config.val_annotation_path,
        annotation_format=config.annotation_format,
        transform=get_val_transform(input_size=config.input_size)
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")

    # Create dataloaders
    train_loader, val_loader = create_train_val_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # Create model
    print("\nCreating model...")
    model = create_ssd(
        num_classes=config.num_classes,
        input_size=config.input_size,
        backbone=config.backbone,
        pretrained=True
    )

    # Print model info
    from utils import count_parameters, get_model_size
    num_params = count_parameters(model)
    model_size = get_model_size(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model_size:.2f} MB")

    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_name='sgd',
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Create trainer
    print("\nCreating trainer...")
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=vars(config),
        log_dir=config.log_dir,
        checkpoint_dir=config.checkpoint_dir
    )

    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=config.num_epochs, start_epoch=start_epoch)

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
