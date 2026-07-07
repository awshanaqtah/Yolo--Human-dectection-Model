# Face Attributes

Learning project. Point it at a photo → detect **human vs animal**, and for each human face read **gender, age, and expression**.

## Pipeline (two-stage)
```
frame → YOLO (person vs animal) → MTCNN face crop → Model A + Model B → annotated image
```

## Models (both ResNet-34, fine-tuned on Modal A100)
- **Model A** — gender + age, trained on **UTKFace**. Gender ~93%, age ±6 yrs.
- **Model B** — expression (7 emotions), trained on **RAF-DB** with weighted cross-entropy + label smoothing. ~80% accuracy, ~69% macro-F1.

**Known limit:** age is the weakest attribute — fragile on stylized / off-distribution faces. Gender and expression are robust.

## Layout
```
data_prep/
  download_data.py        fetch UTKFace / FER / RAF-DB from Kaggle
  build_manifest.py       unified face manifest (age/gender/expression) + splits
  upload_to_hf.py         push UTKFace + manifest to a private HF dataset
  upload_rafdb_to_hf.py   push RAF-DB to a private HF dataset
Model/Faces/
  FaceDataset.py          Model A loader (gender + age)
  FaceModel.py            Model A network (ResNet-34 + gender/age heads)
  ExpressionDataset.py    Model B loader (RAF-DB, 7 emotions + class weights)
  ExpressionModel.py      Model B network (ResNet-34 + 7-way head)
  ModalTrain.py           Model A training on Modal A100 (HF → volume → train → W&B)
  ModalTrainExpression.py Model B training on Modal A100 (weighted-CE / focal switch)
  DetectAndClassify.py    the wrapper: YOLO + MTCNN + Model A/B → annotated image
datasets/                 data + trained weights — git-ignored
```

## Run
```bash
pip install -r requirements.txt

# 1. data  (needs a Kaggle token at ~/.kaggle/access_token)
python data_prep/download_data.py
python data_prep/build_manifest.py

# 2. train on Modal A100  (needs `modal`, plus hf-secret + wandb-secret)
python -m modal run Model/Faces/ModalTrain.py             # Model A: gender + age
python -m modal run Model/Faces/ModalTrainExpression.py   # Model B: expression

# 3. run the full detector on any image
python Model/Faces/DetectAndClassify.py path/to/photo.jpg
```

## Stack
PyTorch + torchvision · Ultralytics YOLO + MTCNN (facenet-pytorch) · Modal (A100 training) · Hugging Face (datasets) · Weights & Biases (tracking)

## Data
- **UTKFace** — age + gender (filename-encoded)
- **RAF-DB** — 7 facial expressions (real color aligned faces)
