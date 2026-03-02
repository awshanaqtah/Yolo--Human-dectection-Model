"""
Evaluation metrics for Human Detection Model.

This module provides functions to calculate detection metrics including
mAP (mean Average Precision), IoU (Intersection over Union),
precision, and recall.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def compute_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor
) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in [x1, y1, x2, y2] format
        boxes2: [M, 4] boxes in [x1, y1, x2, y2] format

    Returns:
        IoU matrix [N, M]

    Example:
        >>> iou = compute_iou(pred_boxes, gt_boxes)
        >>> print(iou.shape)  # [N, M]
    """
    # Expand dimensions for broadcasting
    boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]

    # Compute intersection
    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Compute union
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = boxes1_area + boxes2_area - inter_area

    # Compute IoU
    iou = inter_area / (union_area + 1e-7)

    return iou


def compute_ap(
    rec: np.ndarray,
    prec: np.ndarray,
    use_11_point: bool = False
) -> float:
    """
    Compute Average Precision (AP) from precision-recall curve.

    Args:
        rec: Recall values
        prec: Precision values
        use_11_point: If True, use 11-point interpolation (VOC style).
                     If False, use all-point interpolation (COCO style).

    Returns:
        Average Precision value

    Example:
        >>> ap = compute_ap(recall, precision)
    """
    # Insert sentinel values at boundaries
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Compute precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    if use_11_point:
        # 11-point interpolation (VOC metric)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(mrec >= t) == 0:
                p = 0
            else:
                p = np.max(mpre[mrec >= t])
            ap += p / 11
    else:
        # All-point interpolation (COCO metric)
        # Find points where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def calculate_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 2
) -> Tuple[float, Dict]:
    """
    Calculate mean Average Precision (mAP) for object detection.

    Args:
        predictions: List of predictions, each containing:
            - 'boxes': [N, 4] predicted boxes
            - 'scores': [N] confidence scores
            - 'labels': [N] predicted labels
        targets: List of ground truth targets, each containing:
            - 'boxes': [M, 4] ground truth boxes
            - 'labels': [M] ground truth labels
        iou_threshold: IoU threshold for considering a detection as correct
        num_classes: Number of classes (including background)

    Returns:
        Tuple of (mAP, metrics_dict)

    Example:
        >>> mAP, metrics = calculate_map(predictions, targets)
        >>> print(f"mAP@0.5: {mAP:.4f}")
    """
    # Initialize data structures
    aps = []
    class_metrics = {}

    # Process each class (skip background class 0)
    for class_idx in range(1, num_classes):
        # Collect predictions and ground truth for this class
        class_preds = []
        class_gts = []

        # Count total ground truth objects for this class
        num_gt_per_image = []

        for pred, target in zip(predictions, targets):
            # Get predictions for this class
            class_mask = pred['labels'] == class_idx
            class_pred_boxes = pred['boxes'][class_mask].cpu().numpy()
            class_pred_scores = pred['scores'][class_mask].cpu().numpy()

            # Get ground truth for this class
            gt_mask = target['labels'] == class_idx
            gt_boxes = target['boxes'][gt_mask].cpu().numpy()
            num_gt = gt_mask.sum().item()

            # Store
            class_preds.append({
                'boxes': class_pred_boxes,
                'scores': class_pred_scores
            })
            class_gts.append(gt_boxes)
            num_gt_per_image.append(num_gt)

        # Skip if no ground truth for this class
        if sum(num_gt_per_image) == 0:
            continue

        # Sort all predictions by score (descending)
        all_scores = []
        all_matched = []

        for img_idx, (pred, gt_boxes) in enumerate(zip(class_preds, class_gts)):
            boxes = pred['boxes']
            scores = pred['scores']

            # Track which GT boxes have been matched
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)

            # Sort predictions by score
            sorted_indices = np.argsort(scores)[::-1]

            for pred_idx in sorted_indices:
                pred_box = boxes[pred_idx]
                pred_score = scores[pred_idx]

                # Find best matching GT box
                if len(gt_boxes) > 0:
                    gt_boxes_tensor = torch.tensor(gt_boxes)
                    pred_box_tensor = torch.tensor(pred_box).unsqueeze(0)
                    ious = compute_iou(pred_box_tensor, gt_boxes_tensor).numpy()[0]

                    max_iou_idx = np.argmax(ious)
                    max_iou = ious[max_iou_idx]

                    # Check if match
                    if max_iou >= iou_threshold:
                        if not gt_matched[max_iou_idx]:
                            # True positive
                            gt_matched[max_iou_idx] = True
                            all_matched.append(True)
                        else:
                            # Duplicate detection (false positive)
                            all_matched.append(False)
                    else:
                        # False positive
                        all_matched.append(False)
                else:
                    # False positive (no GT boxes)
                    all_matched.append(False)

                all_scores.append(pred_score)

        # Compute precision-recall curve
        if len(all_scores) > 0:
            all_scores = np.array(all_scores)
            all_matched = np.array(all_matched, dtype=bool)

            # Sort by score
            sorted_indices = np.argsort(all_scores)[::-1]
            all_matched = all_matched[sorted_indices]
            all_scores = all_scores[sorted_indices]

            # Compute cumulative true positives and false positives
            tp = np.cumsum(all_matched)
            fp = np.cumsum(~all_matched)

            # Compute precision and recall
            num_gt_total = sum(num_gt_per_image)
            recalls = tp / (num_gt_total + 1e-7)
            precisions = tp / (tp + fp + 1e-7)

            # Compute AP
            ap = compute_ap(recalls, precisions)
            aps.append(ap)

            class_metrics[f'class_{class_idx}'] = {
                'ap': ap,
                'num_gt': num_gt_total,
                'num_pred': len(all_scores)
            }
        else:
            # No predictions for this class
            class_metrics[f'class_{class_idx}'] = {
                'ap': 0.0,
                'num_gt': sum(num_gt_per_image),
                'num_pred': 0
            }

    # Calculate mAP
    if len(aps) > 0:
        mAP = np.mean(aps)
    else:
        mAP = 0.0

    metrics_dict = {
        'mAP': mAP,
        'per_class': class_metrics
    }

    return mAP, metrics_dict


def calculate_precision_recall(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Calculate overall precision and recall.

    Args:
        predictions: List of predictions
        targets: List of ground truth targets
        iou_threshold: IoU threshold for correct detection

    Returns:
        Tuple of (precision, recall)

    Example:
        >>> precision, recall = calculate_precision_recall(preds, targets)
        >>> print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    """
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        gt_boxes = target['boxes']
        pred_scores = pred['scores']

        if len(gt_boxes) == 0:
            # No ground truth, all predictions are false positives
            total_fp += len(pred_boxes)
            continue

        if len(pred_boxes) == 0:
            # No predictions, all ground truth is missed
            total_gt += len(gt_boxes)
            continue

        total_gt += len(gt_boxes)

        # Match predictions to ground truth
        ious = compute_iou(pred_boxes, gt_boxes)

        # Find best match for each prediction
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
        tp = 0

        # Sort by score (higher score first)
        sorted_indices = torch.argsort(pred_scores, descending=True)

        for pred_idx in sorted_indices:
            pred_ious = ious[pred_idx]
            max_iou_idx = torch.argmax(pred_ious)
            max_iou = pred_ious[max_iou_idx]

            if max_iou >= iou_threshold and not gt_matched[max_iou_idx]:
                tp += 1
                gt_matched[max_iou_idx] = True

        total_tp += tp
        total_fp += len(pred_boxes) - tp

    # Calculate precision and recall
    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_gt + 1e-7)

    return precision, recall
