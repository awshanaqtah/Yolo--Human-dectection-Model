from pathlib import Path

import numpy as np
import torch
from PIL import Image

from FaceModel import FaceGenderAgeModel

RepoRoot = Path(__file__).resolve().parents[2]
DemoRoot = RepoRoot / "datasets" / "real_world_demo"
CheckpointPath = Path(__file__).resolve().parent / "model_a_gender_age.pth"
ImageSize = 224
ImageNetMean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
ImageNetStd = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GenderNames = ["male", "female"]
BatchSize = 32


def LoadModel():
    Model = FaceGenderAgeModel()
    Model.load_state_dict(torch.load(CheckpointPath, map_location="cpu"))
    Model.eval()
    return Model


def LoadImageTensor(ImagePath):
    FaceImage = Image.open(ImagePath).convert("RGB").resize((ImageSize, ImageSize))
    Pixels = (np.asarray(FaceImage, np.float32) / 255.0 - ImageNetMean) / ImageNetStd
    return torch.from_numpy(Pixels).permute(2, 0, 1)


@torch.no_grad()
def PredictFolder(Model, FolderPath, TrueGender):
    ImagePaths = [p for p in FolderPath.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    CorrectGender = 0
    PredictedAges = []
    Samples = []
    for Start in range(0, len(ImagePaths), BatchSize):
        Tensors, ValidPaths = [], []
        for p in ImagePaths[Start:Start + BatchSize]:
            try:
                Tensors.append(LoadImageTensor(p))
                ValidPaths.append(p)
            except Exception:
                pass
        if not Tensors:
            continue
        GenderLogit, AgePrediction = Model(torch.stack(Tensors))
        GenderProb = torch.sigmoid(GenderLogit)
        for p, Prob, Age in zip(ValidPaths, GenderProb, AgePrediction):
            PredictedGender = GenderNames[int(Prob > 0.5)]
            CorrectGender += int(PredictedGender == TrueGender)
            PredictedAges.append(float(Age) * 100)
            if len(Samples) < 5:
                Samples.append((p.name, PredictedGender, float(Age) * 100))
    return CorrectGender, len(PredictedAges), PredictedAges, Samples


if __name__ == "__main__":
    Model = LoadModel()
    print(f"testing Model A on your real photos in {DemoRoot}\n")
    TotalCorrect = TotalCount = 0
    for FolderName, TrueGender in [("women", "female"), ("men", "male")]:
        CorrectGender, Count, PredictedAges, Samples = PredictFolder(Model, DemoRoot / FolderName, TrueGender)
        TotalCorrect += CorrectGender
        TotalCount += Count
        MeanAge = sum(PredictedAges) / len(PredictedAges) if PredictedAges else 0
        print(f"{FolderName:5s} (actual {TrueGender}): gender {CorrectGender}/{Count} = {CorrectGender/Count:.1%}   mean predicted age {MeanAge:.0f}")
        for Name, Gender, Age in Samples:
            print(f"      {Name[:34]:34s} -> {Gender:6s}  age {Age:.0f}")
    print(f"\noverall gender accuracy on real portraits: {TotalCorrect/TotalCount:.1%}")
