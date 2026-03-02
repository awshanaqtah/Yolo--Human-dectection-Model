# Human Detection Model

A custom SSD (Single Shot MultiBox Detector) implementation for human detection using PyTorch. This project includes a complete pipeline for training, evaluation, and inference with support for multiple annotation formats.

## Features

- **Custom SSD Architecture**: Built from scratch with MobileNetV2 backbone
- **Multi-format Support**: COCO, Pascal VOC, and YOLO annotation formats
- **Complete Training Pipeline**: Data augmentation, checkpointing, TensorBoard logging
- **Evaluation Metrics**: mAP, precision, recall calculations
- **Inference Tools**: Image and video detection with visualization
- **Modular Design**: Easy to extend and customize

## Architecture

The model uses SSD300 with:
- **Input**: 300x300 RGB images
- **Backbone**: MobileNetV2 (pre-trained on ImageNet)
- **Feature Maps**: 6 scales (38x38, 19x19, 10x10, 5x5, 3x3, 1x1)
- **Prior Boxes**: 8,732 default boxes
- **Loss**: MultiBox loss with hard negative mining

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/human-detection-model.git
cd human-detection-model

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Project Structure

```
human-detection-model/
├── config/              # Configuration files
├── data/                # Data loading pipeline
├── models/              # Model architecture
├── trainer/             # Training infrastructure
├── inference/           # Inference & visualization
├── utils/               # Utility functions
├── scripts/             # Executable scripts
├── notebooks/           # Jupyter notebooks
└── datasets/            # Your dataset
```

## Dataset Preparation

### Expected Structure

```
datasets/
├── raw/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
```

### Supported Formats

1. **COCO Format** (Recommended)
   - JSON files with 'images', 'annotations', and 'categories' keys
   - Bounding boxes in [x, y, width, height] format

2. **Pascal VOC Format**
   - One XML file per image
   - Bounding boxes in [xmin, ymin, xmax, ymax] format

3. **YOLO Format**
   - One TXT file per image
   - Normalized coordinates with class_id center_x center_y width height

## Usage

### Training

```bash
python scripts/train.py \
    --train-data datasets/raw/images/train \
    --train-ann datasets/raw/annotations/train.json \
    --val-data datasets/raw/images/val \
    --val-ann datasets/raw/annotations/val.json \
    --batch-size 16 \
    --epochs 120 \
    --lr 1e-3 \
    --backbone mobilenet_v2
```

**Arguments:**
- `--train-data`: Path to training images
- `--train-ann`: Path to training annotations
- `--val-data`: Path to validation images
- `--val-ann`: Path to validation annotations
- `--annotation-format`: Annotation format (coco, voc, yolo)
- `--batch-size`: Batch size (default: 16)
- `--epochs`: Number of epochs (default: 120)
- `--lr`: Learning rate (default: 1e-3)
- `--backbone`: Backbone network (mobilenet_v2, resnet50)
- `--resume`: Path to checkpoint to resume from

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint trainer/checkpoints/best_model.pth \
    --test-data datasets/raw/images/test \
    --test-ann datasets/raw/annotations/test.json \
    --batch-size 16
```

### Image Detection

```bash
# Single image
python scripts/detect_image.py \
    --checkpoint trainer/checkpoints/best_model.pth \
    --input image.jpg \
    --output detected_image.jpg \
    --conf-threshold 0.5

# Directory of images
python scripts/detect_image.py \
    --checkpoint trainer/checkpoints/best_model.pth \
    --input images/ \
    --output detected_images/
```

### Video Detection

```bash
# Process video file
python scripts/detect_video.py \
    --checkpoint trainer/checkpoints/best_model.pth \
    --input video.mp4 \
    --output detected_video.mp4

# Webcam detection
python scripts/detect_video.py \
    --checkpoint trainer/checkpoints/best_model.pth \
    --webcam 0
```

## Python API

### Training

```python
from config import Config
from models import create_ssd
from data import HumanDetectionDataset, create_train_val_dataloaders
from trainer import create_trainer, create_optimizer
from utils import set_seed, get_device

# Setup
set_seed(42)
device = get_device()

# Load data
train_dataset = HumanDetectionDataset(
    image_dir='datasets/raw/images/train',
    annotation_path='datasets/raw/annotations/train.json',
    annotation_format='coco'
)

val_dataset = HumanDetectionDataset(
    image_dir='datasets/raw/images/val',
    annotation_path='datasets/raw/annotations/val.json',
    annotation_format='coco'
)

train_loader, val_loader = create_train_val_dataloaders(
    train_dataset, val_dataset, batch_size=16
)

# Create model
model = create_ssd(num_classes=2, backbone='mobilenet_v2')
optimizer = create_optimizer(model, 'sgd', learning_rate=1e-3)

# Train
trainer = create_trainer(model, train_loader, val_loader, optimizer)
trainer.train(num_epochs=120)
```

### Inference

```python
from inference import load_detector

# Load detector
detector = load_detector('trainer/checkpoints/best_model.pth')

# Detect in image
results = detector.detect_image('image.jpg')
print(f"Detected {len(results['boxes'])} humans")

# Visualize
detector.visualize_results(
    'image.jpg',
    results,
    'output.jpg'
)
```

## Configuration

Edit [config/config.yaml](config/config.yaml) to modify training parameters:

```yaml
model:
  input_size: [300, 300]
  num_classes: 2
  backbone: mobilenet_v2

training:
  batch_size: 16
  num_epochs: 120
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0005

data:
  annotation_format: coco
  num_workers: 4

inference:
  conf_threshold: 0.5
  nms_threshold: 0.45
```

## Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir trainer/logs

# Open in browser
# http://localhost:6006
```

## Performance

Target metrics:
- **mAP@0.5**: > 0.5 on test set
- **Inference Speed**: > 20 FPS on GPU
- **Model Size**: ~15 MB (MobileNetV2 backbone)

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Use smaller backbone (`--backbone mobilenet_v2`)
- Enable gradient accumulation

### Poor Detection Quality

- Increase training epochs
- Adjust confidence threshold (`--conf-threshold`)
- Tune anchor box sizes in config
- Add more training data

### Slow Training

- Reduce `--num-workers` if I/O bound
- Use smaller input size
- Enable mixed precision training

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- See [requirements.txt](requirements.txt) for full list

## License

MIT License - see LICENSE file for details

## Acknowledgments

- SSD paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
- MobileNetV2 paper: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
