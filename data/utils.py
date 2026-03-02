"""
Annotation parsing utilities for Human Detection Model.

This module provides functions to parse annotations from different formats:
- COCO: JSON format used by COCO dataset
- Pascal VOC: XML format used by Pascal VOC dataset
- YOLO: TXT format with normalized coordinates
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Union
import numpy as np


def parse_coco_annotations(annotation_path: str) -> Dict[str, List]:
    """
    Parse COCO format annotations from JSON file.

    COCO format:
    {
        "images": [{"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}, ...],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}, ...],
        "categories": [{"id": 1, "name": "person"}, ...]
    }

    Args:
        annotation_path: Path to COCO annotations JSON file

    Returns:
        Dictionary with image_id as key and list of annotations as value:
        {
            "image1.jpg": {
                "boxes": [[x1, y1, x2, y2], ...],  # in absolute pixels
                "labels": [1, 1, ...],  # class labels
                "image_id": 1
            },
            ...
        }

    Example:
        >>> annotations = parse_coco_annotations("annotations/train.json")
        >>> for img_name, ann in annotations.items():
        ...     boxes = ann['boxes']
        ...     labels = ann['labels']
    """
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # Build mapping from image_id to file name
    image_id_to_name = {}
    image_id_to_size = {}
    for image in coco_data['images']:
        image_id_to_name[image['id']] = image['file_name']
        image_id_to_size[image['id']] = (image['width'], image['height'])

    # Build mapping from category_id to category_id (1-indexed, 0 is background)
    # For human detection, we typically have category 1 = person
    category_mapping = {cat['id']: cat['id'] for cat in coco_data['categories']}

    # Parse annotations
    annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        file_name = image_id_to_name.get(image_id)

        if file_name is None:
            continue

        # COCO bbox format: [x, y, width, height]
        # Convert to [x1, y1, x2, y2] format
        x, y, w, h = ann['bbox']
        x1, y1 = x, y
        x2, y2 = x + w, y + h

        # Get label (ensure it's valid)
        label = category_mapping.get(ann['category_id'], 1)

        # Initialize or append to annotations
        if file_name not in annotations:
            annotations[file_name] = {
                'boxes': [],
                'labels': [],
                'image_id': image_id
            }

        annotations[file_name]['boxes'].append([x1, y1, x2, y2])
        annotations[file_name]['labels'].append(label)

    # Convert boxes to numpy arrays for easier processing
    for file_name in annotations:
        annotations[file_name]['boxes'] = np.array(annotations[file_name]['boxes'], dtype=np.float32)
        annotations[file_name]['labels'] = np.array(annotations[file_name]['labels'], dtype=np.int64)

    return annotations


def parse_voc_annotations(annotations_dir: str) -> Dict[str, Dict]:
    """
    Parse Pascal VOC format annotations from XML files.

    VOC format: One XML file per image with annotation data.

    Args:
        annotations_dir: Directory containing XML annotation files

    Returns:
        Dictionary with image filename as key and annotations as value:
        {
            "image1.jpg": {
                "boxes": [[x1, y1, x2, y2], ...],
                "labels": [1, ...],
                "difficult": [0, ...]
            },
            ...
        }

    Example:
        >>> annotations = parse_voc_annotations("annotations/train/")
        >>> boxes = annotations["image1.jpg"]['boxes']
    """
    annotations = {}
    annotation_files = Path(annotations_dir).glob("*.xml")

    for ann_file in annotation_files:
        tree = ET.parse(ann_file)
        root = tree.getroot()

        # Get image filename
        filename = root.find('filename').text
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            filename += '.jpg'  # Default extension

        # Initialize annotation
        annotation = {
            'boxes': [],
            'labels': [],
            'difficult': []
        }

        # Parse all objects
        for obj in root.findall('object'):
            # Get class label
            class_name = obj.find('name').text
            # For human detection, map 'person' to class 1
            label = 1 if class_name.lower() == 'person' else 0

            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Get difficult flag
            difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0

            annotation['boxes'].append([xmin, ymin, xmax, ymax])
            annotation['labels'].append(label)
            annotation['difficult'].append(difficult)

        # Convert to numpy arrays
        if annotation['boxes']:  # Only add if there are annotations
            annotation['boxes'] = np.array(annotation['boxes'], dtype=np.float32)
            annotation['labels'] = np.array(annotation['labels'], dtype=np.int64)
            annotation['difficult'] = np.array(annotation['difficult'], dtype=np.int64)
            annotations[filename] = annotation

    return annotations


def parse_yolo_annotations(annotations_dir: str, image_dir: str) -> Dict[str, Dict]:
    """
    Parse YOLO format annotations from TXT files.

    YOLO format: One TXT file per image with normalized coordinates.
    Each line: class_id center_x center_y width height (all normalized 0-1)

    Args:
        annotations_dir: Directory containing TXT annotation files
        image_dir: Directory containing corresponding images

    Returns:
        Dictionary with image filename as key and annotations as value

    Example:
        >>> annotations = parse_yolo_annotations("annotations/train/", "images/train/")
    """
    annotations = {}
    annotation_files = Path(annotations_dir).glob("*.txt")

    for ann_file in annotation_files:
        # Get corresponding image file
        image_name = ann_file.stem + '.jpg'  # Default to jpg
        image_path = Path(image_dir) / image_name

        if not image_path.exists():
            # Try other extensions
            for ext in ['.jpg', '.jpeg', '.png']:
                image_path = Path(image_dir) / (ann_file.stem + ext)
                if image_path.exists():
                    image_name = image_path.name
                    break

        # Get image size for denormalization
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except:
            # If image doesn't exist, skip
            continue

        # Parse annotations
        boxes = []
        labels = []

        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert from normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]
                    x1 = (center_x - width / 2) * img_width
                    y1 = (center_y - height / 2) * img_height
                    x2 = (center_x + width / 2) * img_width
                    y2 = (center_y + height / 2) * img_height

                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)  # YOLO is 0-indexed, we want 1-indexed

        if boxes:  # Only add if there are annotations
            annotations[image_name] = {
                'boxes': np.array(boxes, dtype=np.float32),
                'labels': np.array(labels, dtype=np.int64)
            }

    return annotations


def convert_bbox_format(
    boxes: np.ndarray,
    from_format: str,
    to_format: str,
    image_width: int = None,
    image_height: int = None
) -> np.ndarray:
    """
    Convert bounding boxes between different formats.

    Supported formats:
    - 'xyxy': [x1, y1, x2, y2] (top-left and bottom-right corners)
    - 'xywh': [x, y, width, height] (top-left corner and size)
    - 'cxcywh': [center_x, center_y, width, height] (center and size)

    Args:
        boxes: Array of bounding boxes
        from_format: Source format
        to_format: Target format
        image_width: Image width (needed for normalization)
        image_height: Image height (needed for normalization)

    Returns:
        Converted bounding boxes

    Example:
        >>> boxes = np.array([[10, 10, 50, 50]])
        >>> converted = convert_bbox_format(boxes, 'xyxy', 'xywh')
        >>> # Output: [[10, 10, 40, 40]]
    """
    if boxes.size == 0:
        return boxes

    boxes = boxes.copy()

    # Convert to xyxy first
    if from_format == 'xywh':
        boxes[:, 2] += boxes[:, 0]  # x2 = x1 + w
        boxes[:, 3] += boxes[:, 1]  # y2 = y1 + h
    elif from_format == 'cxcywh':
        boxes[:, 0] -= boxes[:, 2] / 2  # x1 = cx - w/2
        boxes[:, 1] -= boxes[:, 3] / 2  # y1 = cy - h/2
        boxes[:, 2] += boxes[:, 0]      # x2 = cx + w/2
        boxes[:, 3] += boxes[:, 1]      # y2 = cy + h/2

    # Convert from xyxy to target format
    if to_format == 'xywh':
        boxes[:, 2] -= boxes[:, 0]  # w = x2 - x1
        boxes[:, 3] -= boxes[:, 1]  # h = y2 - y1
    elif to_format == 'cxcywh':
        boxes[:, 2] -= boxes[:, 0]  # w = x2 - x1
        boxes[:, 3] -= boxes[:, 1]  # h = y2 - y1
        boxes[:, 0] += boxes[:, 2] / 2  # cx = x1 + w/2
        boxes[:, 1] += boxes[:, 3] / 2  # cy = y1 + h/2

    return boxes


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes: Array of boxes in [x1, y1, x2, y2] format
        width: Image width
        height: Image height

    Returns:
        Clipped boxes
    """
    boxes = boxes.copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width)   # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)   # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height)  # y2
    return boxes
