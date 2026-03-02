"""
MultiBox Loss Function for SSD.

This module implements the MultiBox loss which combines:
- Localization loss (Smooth L1 for matched boxes)
- Confidence loss (Cross-entropy for classification)
- Matching strategy (IoU-based matching)
- Hard negative mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .prior_box import convert_boxes_to_cxcywh, convert_boxes_to_xyxy


class MultiBoxLoss(nn.Module):
    """
    MultiBox Loss for SSD training.

    Combines localization loss and confidence loss with
    matching strategy and hard negative mining.

    Args:
        num_classes: Number of classes (including background)
        overlap_threshold: IoU threshold for matching
        neg_pos_ratio: Ratio of negative to positive samples for hard negative mining
        variance: Variances for encoding/decoding boxes

    Example:
        >>> criterion = MultiBoxLoss(num_classes=2)
        >>> loss = criterion(preds, targets, priors)
    """

    def __init__(
        self,
        num_classes: int,
        overlap_threshold: float = 0.5,
        neg_pos_ratio: int = 3,
        variance: list = [0.1, 0.2]
    ):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.variance = variance

        # Smooth L1 loss for localization
        self.localization_loss = nn.SmoothL1Loss(reduction='sum')

    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        targets: list,
        priors: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate MultiBox loss.

        Args:
            predictions: Tuple of (loc_pred, conf_pred)
                - loc_pred: [batch_size, num_priors, 4] predicted offsets
                - conf_pred: [batch_size, num_priors, num_classes] class scores
            targets: List of target dictionaries, each containing:
                - 'boxes': [num_objects, 4] ground truth boxes in [x1, y1, x2, y2]
                - 'labels': [num_objects] ground truth labels
            priors: [num_priors, 4] prior boxes in [cx, cy, w, h] format

        Returns:
            Tuple of (total_loss, loss_dict)
            - total_loss: Combined loss
            - loss_dict: Dictionary with individual loss components

        Example:
            >>> loss, loss_dict = criterion(predictions, targets, priors)
            >>> print(f"Total loss: {loss.item():.4f}")
        """
        loc_pred, conf_pred = predictions

        batch_size = loc_pred.size(0)
        num_priors = priors.size(0)

        # Match priors to ground truth boxes
        matched_targets = self.match_priors(targets, priors)

        # Create ground truth tensors
        loc_targets = torch.zeros_like(loc_pred)  # [batch, num_priors, 4]
        conf_targets = torch.zeros_like(conf_pred)  # [batch, num_priors, num_classes]

        # Process each image in the batch
        for idx in range(batch_size):
            if len(targets[idx]['boxes']) == 0:
                # No ground truth boxes in this image
                continue

            gt_boxes = targets[idx]['boxes']  # [num_objects, 4]
            gt_labels = targets[idx]['labels']  # [num_objects]

            # Matched information for this image
            matched_gt_boxes = matched_targets[idx]['boxes']  # [num_priors, 4]
            matched_gt_labels = matched_targets[idx]['labels']  # [num_priors]
            matched_indices = matched_targets[idx]['indices']  # [num_priors]

            # Encode ground truth boxes relative to priors
            # Convert gt_boxes to [cx, cy, w, h] format
            gt_boxes_cxcywh = convert_boxes_to_cxcywh(gt_boxes)
            encoded_boxes = self.encode_boxes(
                gt_boxes_cxcywh[matched_indices],
                priors,
                self.variance
            )

            # Assign to targets (only for matched priors)
            pos_mask = matched_indices > -1
            loc_targets[idx, pos_mask] = encoded_boxes[pos_mask]

            # Assign class labels
            # Background class is 0, so we add 1 to labels
            conf_targets[idx, pos_mask] = self.one_hot_encoding(
                matched_gt_labels[pos_mask] - 1,  # Convert to 0-indexed (excluding background)
                self.num_classes
            )

        # Calculate localization loss (only for matched priors)
        pos_mask = conf_targets.sum(dim=2) > 0  # [batch, num_priors]
        pos_mask_idx = pos_mask.unsqueeze(2).expand_as(loc_pred)

        num_pos = pos_mask_idx.sum()

        if num_pos > 0:
            loc_loss = self.localization_loss(
                loc_pred[pos_mask_idx].view(-1, 4),
                loc_targets[pos_mask_idx].view(-1, 4)
            )
        else:
            loc_loss = torch.tensor(0.0, device=loc_pred.device)

        # Calculate confidence loss
        conf_loss = F.cross_entropy(
            conf_pred.view(-1, self.num_classes),
            conf_targets.argmax(dim=2).view(-1),
            reduction='none'
        )
        conf_loss = conf_loss.view(batch_size, num_priors)

        # Hard negative mining
        # Positive examples are already matched
        # We select hard negatives (high confidence loss) to maintain neg_pos_ratio

        if num_pos > 0:
            # Get negative loss values
            neg_loss = conf_loss.clone()
            neg_loss[pos_mask] = 0  # Zero out positive positions

            # Sort and select top k negatives
            _, neg_idx = neg_loss.sort(dim=1, descending=True)
            _, rank = neg_idx.sort(dim=1)

            num_pos_per_sample = pos_mask.sum(dim=1)
            num_neg = torch.clamp(num_pos_per_sample * self.neg_pos_ratio, max=num_priors - 1)

            neg_mask = rank < num_neg.unsqueeze(1)

            # Combine positive and negative masks
            conf_mask = pos_mask | neg_mask
        else:
            # If no positives, use top-k negatives
            num_neg = self.neg_pos_ratio * min(100, num_priors)
            _, neg_idx = conf_loss.sort(dim=1, descending=True)
            _, rank = neg_idx.sort(dim=1)
            conf_mask = rank < num_neg

        # Calculate final confidence loss
        conf_loss = conf_loss[conf_mask].sum()

        # Total loss
        num_conf = conf_mask.sum()
        if num_conf > 0:
            total_loss = loc_loss / num_pos + conf_loss / num_conf
        else:
            total_loss = loc_loss / max(1, num_pos)

        # Loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'loc_loss': (loc_loss / max(1, num_pos)).item(),
            'conf_loss': (conf_loss / max(1, num_conf)).item(),
            'num_pos': num_pos.item()
        }

        return total_loss, loss_dict

    def match_priors(self, targets: list, priors: torch.Tensor) -> list:
        """
        Match prior boxes to ground truth boxes using IoU.

        For each ground truth box:
        1. Find prior with highest IoU (match regardless of threshold)
        2. Match priors with IoU > threshold

        Args:
            targets: List of target dictionaries
            priors: [num_priors, 4] prior boxes in [cx, cy, w, h] format

        Returns:
            List of matched targets for each image
        """
        batch_size = len(targets)
        matched_targets = []

        # Convert priors to [x1, y1, x2, y2] format
        priors_xyxy = convert_boxes_to_xyxy(priors)

        for idx in range(batch_size):
            if len(targets[idx]['boxes']) == 0:
                matched_targets.append({
                    'boxes': torch.zeros(len(priors), 4, device=priors.device),
                    'labels': torch.zeros(len(priors), dtype=torch.long, device=priors.device),
                    'indices': torch.full((len(priors),), -1, dtype=torch.long, device=priors.device)
                })
                continue

            gt_boxes = targets[idx]['boxes']  # [num_objects, 4] in [x1, y1, x2, y2] format
            gt_labels = targets[idx]['labels']

            num_gt = len(gt_boxes)
            num_priors = len(priors)

            # Compute IoU between all priors and all GT boxes
            # iou: [num_priors, num_gt]
            iou = self.compute_iou(priors_xyxy, gt_boxes)

            # Initialize match arrays
            matched_gt_idx = torch.full((num_priors,), -1, dtype=torch.long, device=priors.device)
            matched_gt_boxes = torch.zeros(num_priors, 4, device=priors.device)
            matched_gt_labels = torch.zeros(num_priors, dtype=torch.long, device=priors.device)

            # Match each GT to the prior with highest IoU
            max_iou_for_each_gt, prior_idx_for_each_gt = iou.max(dim=0)
            for gt_idx in range(num_gt):
                prior_idx = prior_idx_for_each_gt[gt_idx]
                matched_gt_idx[prior_idx] = gt_idx
                matched_gt_boxes[prior_idx] = gt_boxes[gt_idx]
                matched_gt_labels[prior_idx] = gt_labels[gt_idx]

            # Match priors to GT if IoU > threshold
            max_iou_for_each_prior, gt_idx_for_each_prior = iou.max(dim=1)
            below_threshold = (max_iou_for_each_prior < self.threshold)
            below_threshold_idx = torch.where(below_threshold)[0]

            # Set low IoU matches to background (-1)
            matched_gt_idx[below_threshold_idx] = -1
            matched_gt_boxes[below_threshold_idx] = 0
            matched_gt_labels[below_threshold_idx] = 0

            # For high IoU matches, assign GT
            above_threshold = (max_iou_for_each_prior >= self.threshold)
            for prior_idx in torch.where(above_threshold)[0]:
                gt_idx = gt_idx_for_each_prior[prior_idx]
                if matched_gt_idx[prior_idx] == -1:  # Not already matched
                    matched_gt_idx[prior_idx] = gt_idx
                    matched_gt_boxes[prior_idx] = gt_boxes[gt_idx]
                    matched_gt_labels[prior_idx] = gt_labels[gt_idx]

            matched_targets.append({
                'boxes': matched_gt_boxes,
                'labels': matched_gt_labels,
                'indices': matched_gt_idx
            })

        return matched_targets

    def encode_boxes(
        self,
        boxes: torch.Tensor,
        priors: torch.Tensor,
        variances: list
    ) -> torch.Tensor:
        """
        Encode ground truth boxes relative to priors.

        Args:
            boxes: Ground truth boxes [N, 4] in [cx, cy, w, h] format
            priors: Prior boxes [N, 4] in [cx, cy, w, h] format
            variances: Variances for encoding

        Returns:
            Encoded offsets [N, 4]
        """
        # Encode
        g_cxcy = (boxes[:, :2] - priors[:, :2]) / (variances[0] * priors[:, 2:])
        g_wh = torch.log(boxes[:, 2:] / priors[:, 2:]) / variances[1]

        return torch.cat([g_cxcy, g_wh], dim=1)

    def compute_iou(
        self,
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

    def one_hot_encoding(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Convert labels to one-hot encoding.

        Args:
            labels: [N] labels (0-indexed, not including background)
            num_classes: Number of classes (including background)

        Returns:
            One-hot encoded labels [N, num_classes]
        """
        batch_size = labels.size(0)
        one_hot = torch.zeros(batch_size, num_classes, device=labels.device)

        # Labels are 0-indexed for foreground classes
        # Background is class 0, so we add 1
        one_hot[torch.arange(batch_size), labels + 1] = 1

        return one_hot
