import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from ultralytics import YOLO

from ExpressionModel import ExpressionModel
from FaceModel import FaceGenderAgeModel

HereDir = Path(__file__).resolve().parent
RepoRoot = HereDir.parents[1]
YoloModelName = "yolo11n.pt"
ImageSize = 224
AgeScale = 100.0
FaceConfidence = 0.97
CropTighten = 0.15
ImageNetMean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
ImageNetStd = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GenderNames = ["male", "female"]
EmotionNames = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
PersonClass = 0
AnimalClasses = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}  # COCO bird..giraffe


def LoadModels():
    Device = "cuda" if torch.cuda.is_available() else "cpu"
    GenderAgeModel = FaceGenderAgeModel()
    GenderAgeModel.load_state_dict(torch.load(HereDir / "model_a_gender_age.pth", map_location=Device))
    ExpressionNet = ExpressionModel()
    ExpressionNet.load_state_dict(torch.load(HereDir / "model_b_expression.pth", map_location=Device))
    Detector = YOLO(YoloModelName)
    FaceDetector = MTCNN(keep_all=True, device=Device)
    return GenderAgeModel.eval().to(Device), ExpressionNet.eval().to(Device), Detector, FaceDetector, Device


def PreprocessFace(FaceRgb, Device):
    FaceImage = Image.fromarray(FaceRgb).resize((ImageSize, ImageSize))
    Pixels = (np.asarray(FaceImage, np.float32) / 255.0 - ImageNetMean) / ImageNetStd
    return torch.from_numpy(Pixels).permute(2, 0, 1).unsqueeze(0).to(Device)


@torch.no_grad()
def ReadFaceAttributes(FaceRgb, GenderAgeModel, ExpressionNet, Device):
    Tensor = PreprocessFace(FaceRgb, Device)
    GenderLogit, AgePrediction = GenderAgeModel(Tensor)
    Gender = GenderNames[int(torch.sigmoid(GenderLogit) > 0.5)]
    Age = int(AgePrediction.item() * AgeScale)
    Emotion = EmotionNames[ExpressionNet(Tensor).argmax(1).item()]
    return f"{Gender}, {Age}, {Emotion}"


def DrawBox(Canvas, x1, y1, x2, y2, Label, Color):
    cv2.rectangle(Canvas, (x1, y1), (x2, y2), Color, 2)
    cv2.putText(Canvas, Label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Color, 2)


def DetectAndClassify(ImagePath, OutputPath):
    GenderAgeModel, ExpressionNet, Detector, FaceDetector, Device = LoadModels()

    BgrImage = cv2.imread(str(ImagePath))
    if BgrImage is None:
        raise FileNotFoundError(f"could not read image: {ImagePath}")
    RgbImage = cv2.cvtColor(BgrImage, cv2.COLOR_BGR2RGB)
    Height, Width = RgbImage.shape[:2]

    # 1. YOLO: person vs animal
    Result = Detector.predict(source=BgrImage, conf=0.4, verbose=False)[0]
    for Box, ClassId in zip(Result.boxes.xyxy.cpu().numpy().astype(int), Result.boxes.cls.cpu().numpy().astype(int)):
        if ClassId in AnimalClasses:
            DrawBox(BgrImage, *Box, f"animal: {Detector.names[ClassId]}", (0, 140, 255))
        elif ClassId == PersonClass:
            DrawBox(BgrImage, *Box, "person", (0, 200, 0))

    # 2. faces -> attributes
    FaceBoxes, FaceProbs = FaceDetector.detect(Image.fromarray(RgbImage))
    KeptFaces = 0
    if FaceBoxes is not None:
        for Box, Prob in zip(FaceBoxes, FaceProbs):
            if Prob is None or Prob < FaceConfidence:
                continue
            x1, y1, x2, y2 = int(Box[0]), int(Box[1]), int(Box[2]), int(Box[3])
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, Width), min(y2, Height)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            MarginX, MarginY = int((x2 - x1) * CropTighten / 2), int((y2 - y1) * CropTighten / 2)
            x1, y1, x2, y2 = x1 + MarginX, y1 + MarginY, x2 - MarginX, y2 - MarginY
            Attributes = ReadFaceAttributes(RgbImage[y1:y2, x1:x2], GenderAgeModel, ExpressionNet, Device)
            DrawBox(BgrImage, x1, y1, x2, y2, Attributes, (255, 60, 0))
            KeptFaces += 1

    cv2.imwrite(str(OutputPath), BgrImage)
    print(f"{len(Result.boxes)} objects, {KeptFaces} faces -> {OutputPath}")


if __name__ == "__main__":
    Input = Path(sys.argv[1]) if len(sys.argv) > 1 else RepoRoot / "datasets" / "real_world_demo" / "women" / "0001.jpg"
    Output = RepoRoot / "datasets" / "wrapper_output.jpg"
    DetectAndClassify(Input, Output)
