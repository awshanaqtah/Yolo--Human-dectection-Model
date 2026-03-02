"""
Evaluation script for Human Detection Model.

This script evaluates a trained model on a test dataset.
"""

import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import SSD, create_ssd
from data import HumanDetectionDataset, create_dataloader, get_val_transform
from utils.metrics import calculate_map, calculate_precision_recall
from utils import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Human Detection Model')

    # Data arguments
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test images')
    parser.add_argument('--test-ann', type=str, required=True,
                        help='Path to test annotations')
    parser.add_argument('--annotation-format', type=str, default='coco',
                        choices=['coco', 'voc', 'yolo'],
                        help='Annotation format')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'resnet50'],
                        help='Backbone network')
    parser.add_argument('--input-size', type=int, nargs=2, default=[300, 300],
                        help='Input image size (height width)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes')

    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.45,
                        help='NMS threshold')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # System arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device, conf_threshold, nms_threshold, num_classes):
    """Evaluate model on dataset."""
    model.eval()

    all_predictions = []
    all_targets = []

    print("Running evaluation...")

    for images, targets in dataloader:
        images = images.to(device)

        # Get predictions
        detections = model.detect(
            images,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=200,
            keep_top_k=100
        )

        # Move targets to CPU
        targets = [{k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        all_predictions.extend(detections)
        all_targets.extend(targets)

    # Calculate metrics
    print("\nCalculating metrics...")

    mAP, map_details = calculate_map(
        predictions=all_predictions,
        targets=all_targets,
        iou_threshold=0.5,
        num_classes=num_classes
    )

    precision, recall = calculate_precision_recall(
        predictions=all_predictions,
        targets=all_targets,
        iou_threshold=0.5
    )

    return {
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'details': map_details
    }


def main():
    """Main evaluation function."""
    args = parse_args()

    print("=" * 60)
    print("Human Detection Model Evaluation")
    print("=" * 60)

    # Setup device
    if args.device is None:
        device = get_device()
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = create_ssd(
        num_classes=checkpoint.get('num_classes', args.num_classes),
        input_size=tuple(checkpoint.get('input_size', args.input_size)),
        backbone=checkpoint.get('backbone', args.backbone),
        pretrained=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Model loaded successfully")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'metrics' in checkpoint:
        print(f"  Training metrics: {checkpoint['metrics']}")

    # Create dataset
    print("\nLoading test dataset...")
    test_dataset = HumanDetectionDataset(
        image_dir=args.test_data,
        annotation_path=args.test_ann,
        annotation_format=args.annotation_format,
        transform=get_val_transform(input_size=tuple(args.input_size))
    )

    print(f"Test dataset: {len(test_dataset)} images")

    # Create dataloader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Evaluate
    results = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        num_classes=model.num_classes
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"mAP@0.5: {results['mAP']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'] + 1e-7):.4f}")

    if 'details' in results and 'per_class' in results['details']:
        print("\nPer-class results:")
        for class_name, class_metrics in results['details']['per_class'].items():
            print(f"  {class_name}: AP={class_metrics['ap']:.4f}, "
                  f"GT={class_metrics['num_gt']}, Pred={class_metrics['num_pred']}")

    print("=" * 60)


if __name__ == '__main__':
    main()
