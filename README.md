# Face Attributes

Learning project. From a face crop, read **gender, age, and expression**.
Bigger plan is two-stage: detect **human vs animal**, then read attributes off the face.

## Models
- **Model A** — gender + age (UTKFace, ResNet-34)
- **Model B** — expression (FER-2013) — *next*

## Layout
```
data_prep/    download datasets + build the unified manifest
Model/Faces/  FaceDataset.py (loader) + FaceModel.py (ResNet-34 + heads)
datasets/     data — git-ignored (downloadable / regenerable)
```

## Run
```
pip install -r requirements.txt
python data_prep/download_data.py     # needs a Kaggle token at ~/.kaggle/access_token
python data_prep/build_manifest.py    # -> datasets/processed/faces_manifest.csv
python Model/Faces/FaceDataset.py      # smoke test
```

Training targets Modal.com (A100).
