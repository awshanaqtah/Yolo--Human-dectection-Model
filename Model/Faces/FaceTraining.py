import torch
from torch.utils.data import DataLoader

from FaceDataset import FaceGenderAgeDataset
from FaceModel import FaceGenderAgeModel

Device = "cuda" if torch.cuda.is_available() else "cpu"
BatchSize = 128
EpochCount = 15
LearningRate = 1e-4
NumWorkers = 4
AgeScale = 100.0
CheckpointPath = "model_a_gender_age.pth"


def RunOneEpoch(Model, EpochLoader, GenderLoss, AgeLoss, Optimizer, IsTraining):
    Model.train() if IsTraining else Model.eval()
    RunningLoss = 0.0
    CorrectGenderCount = 0
    TotalAgeError = 0.0
    SampleCount = 0

    for ImageBatch, GenderBatch, AgeBatch in EpochLoader:
        ImageBatch = ImageBatch.to(Device)
        GenderBatch = GenderBatch.to(Device)
        AgeBatch = AgeBatch.to(Device)

        with torch.set_grad_enabled(IsTraining):
            GenderLogit, AgePrediction = Model(ImageBatch)
            BatchLoss = GenderLoss(GenderLogit, GenderBatch) + AgeLoss(AgePrediction, AgeBatch / AgeScale)

        if IsTraining:
            Optimizer.zero_grad()
            BatchLoss.backward()
            Optimizer.step()

        BatchCount = ImageBatch.size(0)
        RunningLoss += BatchLoss.item() * BatchCount
        PredictedGender = (torch.sigmoid(GenderLogit) > 0.5).float()
        CorrectGenderCount += (PredictedGender == GenderBatch).sum().item()
        TotalAgeError += (AgePrediction * AgeScale - AgeBatch).abs().sum().item()
        SampleCount += BatchCount

    return RunningLoss / SampleCount, CorrectGenderCount / SampleCount, TotalAgeError / SampleCount


def TrainModelA():
    TrainLoader = DataLoader(FaceGenderAgeDataset("train", UseAugmentation=True),
                             batch_size=BatchSize, shuffle=True, num_workers=NumWorkers, pin_memory=True)
    ValLoader = DataLoader(FaceGenderAgeDataset("val"),
                           batch_size=BatchSize, shuffle=False, num_workers=NumWorkers, pin_memory=True)

    Model = FaceGenderAgeModel().to(Device)
    GenderLoss = torch.nn.BCEWithLogitsLoss()
    AgeLoss = torch.nn.L1Loss()
    Optimizer = torch.optim.Adam(Model.parameters(), lr=LearningRate)

    print(f"training on {Device}")
    for Epoch in range(1, EpochCount + 1):
        TrainLoss, TrainGender, TrainAge = RunOneEpoch(Model, TrainLoader, GenderLoss, AgeLoss, Optimizer, True)
        ValLoss, ValGender, ValAge = RunOneEpoch(Model, ValLoader, GenderLoss, AgeLoss, Optimizer, False)
        print(f"epoch {Epoch:2d}  train[loss {TrainLoss:.3f} gender {TrainGender:.3f} age±{TrainAge:.1f}]  "
              f"val[loss {ValLoss:.3f} gender {ValGender:.3f} age±{ValAge:.1f}]")

    torch.save(Model.state_dict(), CheckpointPath)
    print(f"saved {CheckpointPath}")


if __name__ == "__main__":
    TrainModelA()
