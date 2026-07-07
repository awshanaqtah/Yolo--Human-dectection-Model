import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DataRoot = Path(os.environ["FACE_DATA_ROOT"]) if os.environ.get("FACE_DATA_ROOT") else Path(__file__).resolve().parents[2]
RafdbRoot = DataRoot / "datasets" / "raw" / "rafdb" / "DATASET"
ImageSize = 224
ValFraction = 0.10
SplitSeed = 42
# RAF-DB folders 1..7 map to these emotions -> we store them as index 0..6
EmotionNames = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
ImageNetMean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
ImageNetStd = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ExpressionDataset(Dataset):
    def __init__(self, DatasetSplit, UseAugmentation=False):
        assert DatasetSplit in {"train", "val", "test"}, f"bad split: {DatasetSplit}"
        self.UseAugmentation = UseAugmentation
        FolderSplit = "test" if DatasetSplit == "test" else "train"

        self.Samples = []
        for ClassFolder in range(1, 8):
            Paths = sorted((RafdbRoot / FolderSplit / str(ClassFolder)).glob("*.jpg"))
            if DatasetSplit != "test":
                Paths = self._CarveTrainVal(Paths, DatasetSplit)
            self.Samples.extend((p, ClassFolder - 1) for p in Paths)

    def _CarveTrainVal(self, Paths, DatasetSplit):
        Shuffled = list(Paths)
        random.Random(SplitSeed).shuffle(Shuffled)  # fixed seed -> stable, stratified per class
        CutPoint = int(len(Shuffled) * (1 - ValFraction))
        return Shuffled[:CutPoint] if DatasetSplit == "train" else Shuffled[CutPoint:]

    def __len__(self):
        return len(self.Samples)

    def __getitem__(self, Index):
        ImagePath, EmotionIndex = self.Samples[Index]
        FaceImage = Image.open(ImagePath).convert("RGB").resize((ImageSize, ImageSize))
        if self.UseAugmentation and random.random() < 0.5:
            FaceImage = FaceImage.transpose(Image.FLIP_LEFT_RIGHT)
        NormalizedPixels = (np.asarray(FaceImage, np.float32) / 255.0 - ImageNetMean) / ImageNetStd
        ImageTensor = torch.from_numpy(NormalizedPixels).permute(2, 0, 1).contiguous()
        return ImageTensor, torch.tensor(EmotionIndex, dtype=torch.long)


def ComputeClassWeights():
    Counts = np.array([len(list((RafdbRoot / "train" / str(c)).glob("*.jpg"))) for c in range(1, 8)], dtype=np.float64)
    Weights = Counts.sum() / (len(Counts) * Counts)  # inverse-frequency, mean ~ 1.0
    return torch.tensor(Weights, dtype=torch.float32)


if __name__ == "__main__":
    for SplitName in ("train", "val", "test"):
        SplitDataset = ExpressionDataset(SplitName)
        print(f"{SplitName:5s}: {len(SplitDataset):6d} faces")

    ImageTensor, EmotionLabel = ExpressionDataset("train", UseAugmentation=True)[0]
    print(f"\nimage {tuple(ImageTensor.shape)}  range [{ImageTensor.min():.2f}, {ImageTensor.max():.2f}]")
    print(f"emotion index {EmotionLabel.item()} = {EmotionNames[EmotionLabel.item()]}")
    ClassWeights = ComputeClassWeights()
    print("class weights (surprise..neutral):", [round(w, 2) for w in ClassWeights.tolist()])
