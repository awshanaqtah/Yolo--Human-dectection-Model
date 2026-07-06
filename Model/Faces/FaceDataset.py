import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DataRoot = Path(os.environ["FACE_DATA_ROOT"]) if os.environ.get("FACE_DATA_ROOT") else Path(__file__).resolve().parents[2]
ManifestPath = DataRoot / "datasets" / "processed" / "faces_manifest.csv"
ImageSize = 224
GenderToIndex = {"male": 0, "female": 1}
ImageNetMean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
ImageNetStd = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FaceGenderAgeDataset(Dataset):
    def __init__(self, DatasetSplit, UseAugmentation=False):
        with open(ManifestPath, newline="", encoding="utf-8") as ManifestFile:
            self.Rows = [Row for Row in csv.DictReader(ManifestFile)
                         if Row["source"] == "utkface" and Row["split"] == DatasetSplit]
        self.UseAugmentation = UseAugmentation

    def __len__(self):
        return len(self.Rows)

    def __getitem__(self, Index):
        Row = self.Rows[Index]
        FaceImage = Image.open(DataRoot / Row["image_path"]).convert("RGB").resize((ImageSize, ImageSize))
        if self.UseAugmentation and random.random() < 0.5:
            FaceImage = FaceImage.transpose(Image.FLIP_LEFT_RIGHT)
        NormalizedPixels = (np.asarray(FaceImage, np.float32) / 255.0 - ImageNetMean) / ImageNetStd
        ImageTensor = torch.from_numpy(NormalizedPixels).permute(2, 0, 1).contiguous()
        HumanGender = torch.tensor(GenderToIndex[Row["gender"]], dtype=torch.float32)
        HumanAge = torch.tensor(float(Row["age"]), dtype=torch.float32)
        return ImageTensor, HumanGender, HumanAge


if __name__ == "__main__":
    for SplitName in ("train", "val", "test"):
        SplitDataset = FaceGenderAgeDataset(SplitName)
        print(f"{SplitName:5s}: {len(SplitDataset):6d} faces")

    SampleDataset = FaceGenderAgeDataset("train", UseAugmentation=True)
    ImageTensor, HumanGender, HumanAge = SampleDataset[0]
    print(f"\nimage {tuple(ImageTensor.shape)}  range [{ImageTensor.min():.2f}, {ImageTensor.max():.2f}]")
    print(f"gender {HumanGender.item()} (0=male 1=female)   age {HumanAge.item()}")
