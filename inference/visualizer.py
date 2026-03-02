"""
Visualization utilities for Human Detection Model.

This module provides functions to draw bounding boxes,
labels, and confidence scores on images.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional


def get_color_palette(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
    """
    Get a color palette for different classes.

    Args:
        num_classes: Number of classes

    Returns:
        Dictionary mapping class IDs to RGB colors

    Example:
        >>> colors = get_color_palette(2)
        >>> print(colors[1])  # RGB tuple for class 1
    """
    # Common colors for visualization
    colors = {
        0: (255, 0, 0),      # Red (background)
        1: (0, 255, 0),      # Green (person)
        2: (255, 255, 0),    # Yellow
        3: (255, 0, 255),    # Magenta
        4: (0, 255, 255),    # Cyan
        5: (255, 128, 0),    # Orange
        6: (128, 0, 255),    # Purple
        7: (255, 0, 128),    # Pink
        8: (0, 128, 255),    # Light blue
        9: (128, 255, 0),    # Lime
    }

    # Generate additional colors if needed
    for i in range(len(colors), num_classes):
        # Generate random distinct colors
        color = (
            (i * 37) % 256,
            (i * 73) % 256,
            (i * 151) % 256
        )
        colors[i] = color

    return colors


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: List[int],
    color: Tuple[int, int, int],
    linewidth: int = 3
):
    """
    Draw a bounding box on an image.

    Args:
        draw: PIL ImageDraw object
        box: Bounding box [x1, y1, x2, y2]
        color: RGB color tuple
        linewidth: Line width

    Example:
        >>> draw_box(draw, [10, 20, 100, 200], (0, 255, 0))
    """
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=linewidth)


def draw_label(
    draw: ImageDraw.ImageDraw,
    box: List[int],
    label: str,
    score: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    font: Optional[ImageFont.ImageFont] = None
):
    """
    Draw a label with optional score on an image.

    Args:
        draw: PIL ImageDraw object
        box: Bounding box [x1, y1, x2, y2]
        label: Label text
        score: Optional confidence score
        color: RGB color tuple
        font: PIL ImageFont object

    Example:
        >>> draw_label(draw, [10, 20, 100, 200], 'person', 0.95)
    """
    x1, y1, x2, y2 = box

    # Prepare text
    if score is not None:
        text = f"{label}: {score:.2f}"
    else:
        text = label

    # Load default font if not provided
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()

    # Get text size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Draw background rectangle
    draw.rectangle(
        [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
        fill=color
    )

    # Draw text
    draw.text((x1 + 2, y1 - text_height - 2), text, fill=(0, 0, 0), font=font)


def draw_detections(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: Dict[int, str] = None,
    show_scores: bool = True,
    score_threshold: Optional[float] = None,
    linewidth: int = 3
) -> Image.Image:
    """
    Draw detection results on an image.

    Args:
        image: PIL Image
        boxes: [N, 4] array of bounding boxes in [x1, y1, x2, y2] format
        scores: [N] array of confidence scores
        labels: [N] array of class labels
        class_names: Dictionary mapping class IDs to names
        show_scores: Whether to show confidence scores
        score_threshold: Minimum score to display (None = show all)
        linewidth: Bounding box line width

    Returns:
        PIL Image with drawn detections

    Example:
        >>> annotated = draw_detections(
        ...     image=image,
        ...     boxes=boxes,
        ...     scores=scores,
        ...     labels=labels,
        ...     class_names={1: 'person'}
        ... )
        >>> annotated.save('output.jpg')
    """
    # Make a copy of the image
    image = image.copy()

    # Create drawing context
    draw = ImageDraw.Draw(image)

    # Get color palette
    unique_labels = np.unique(labels)
    colors = get_color_palette(max(unique_labels) + 1)

    # Default class names
    if class_names is None:
        class_names = {1: 'person'}

    # Filter by score threshold
    if score_threshold is not None:
        mask = scores >= score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

    # Draw each detection
    for box, score, label in zip(boxes, scores, labels):
        box = box.astype(int)
        color = colors.get(int(label), (0, 255, 0))

        # Draw bounding box
        draw_box(draw, box, color, linewidth)

        # Draw label
        if show_scores:
            label_text = class_names.get(int(label), f'class_{label}')
            draw_label(
                draw,
                box,
                label_text,
                score,
                color=color
            )

    return image


def create_comparison_grid(
    images: List[Image.Image],
    titles: List[str] = None,
    rows: int = 1,
    cols: int = None
) -> Image.Image:
    """
    Create a grid comparison of multiple images.

    Args:
        images: List of PIL Images
        titles: Optional list of titles for each image
        rows: Number of rows in the grid
        cols: Number of columns (auto-calculated if None)

    Returns:
        PIL Image with grid layout

    Example:
        >>> grid = create_comparison_grid(
        ...     [img1, img2, img3, img4],
        ...     titles=['Original', 'Detected', 'GT', 'Overlay'],
        ...     rows=2
        ... )
    """
    if cols is None:
        cols = (len(images) + rows - 1) // rows

    # Get image size
    img_width, img_height = images[0].size

    # Create grid image
    grid_width = cols * img_width
    grid_height = rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')

    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        grid_image.paste(img, (x, y))

        # Add title if provided
        if titles is not None and idx < len(titles):
            draw = ImageDraw.Draw(grid_image)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            # Draw title background
            text_bbox = draw.textbbox((0, 0), titles[idx], font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            draw.rectangle(
                [x, y, x + text_width + 10, y + text_height + 10],
                fill=(0, 0, 0)
            )

            # Draw title
            draw.text((x + 5, y + 5), titles[idx], fill=(255, 255, 255), font=font)

    return grid_image


def visualize_dataset_sample(
    image: Image.Image,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray = None,
    pred_scores: np.ndarray = None,
    save_path: Optional[str] = None
) -> Image.Image:
    """
    Visualize a dataset sample with ground truth and predictions.

    Args:
        image: PIL Image
        gt_boxes: Ground truth boxes [N, 4]
        pred_boxes: Optional predicted boxes [M, 4]
        pred_scores: Optional prediction scores [M]
        save_path: Optional path to save visualization

    Returns:
        PIL Image with visualizations

    Example:
        >>> vis = visualize_dataset_sample(
        ...     image=image,
        ...     gt_boxes=gt_boxes,
        ...     pred_boxes=pred_boxes,
        ...     pred_scores=scores,
        ...     save_path='sample.jpg'
        ... )
    """
    # Create comparison
    images_to_compare = []
    titles = []

    # Original with ground truth
    gt_image = image.copy()
    draw_gt = ImageDraw.Draw(gt_image)
    for box in gt_boxes:
        box = box.astype(int)
        draw_gt.rectangle(box.tolist(), outline=(0, 255, 0), width=3)
    images_to_compare.append(gt_image)
    titles.append("Ground Truth")

    # Predictions if provided
    if pred_boxes is not None and len(pred_boxes) > 0:
        pred_image = image.copy()
        draw_pred = ImageDraw.Draw(pred_image)
        for i, box in enumerate(pred_boxes):
            box = box.astype(int)
            score = pred_scores[i] if pred_scores is not None else None
            draw_label(draw_pred, box.tolist(), 'pred', score, color=(255, 0, 0))
            draw_box(draw_pred, box.tolist(), (255, 0, 0))
        images_to_compare.append(pred_image)
        titles.append("Predictions")

    # Create grid
    grid = create_comparison_grid(images_to_compare, titles=titles, rows=1)

    # Save if path provided
    if save_path is not None:
        grid.save(save_path)
        print(f"Saved visualization to {save_path}")

    return grid
