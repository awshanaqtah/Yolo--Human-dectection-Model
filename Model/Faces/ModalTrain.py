import modal

App = modal.App("face-gender-age")

TrainingImage = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "pillow", "tqdm",
                 "wandb", "huggingface_hub")
    .add_local_python_source("FaceDataset", "FaceModel")
)

DataVolume = modal.Volume.from_name("face-data", create_if_missing=True)
RunsVolume = modal.Volume.from_name("face-runs", create_if_missing=True)

HfRepoId = "AwsHanaqtah/utkface-gender-age"
DataRootOnVolume = "/data"
RunsRoot = "/runs"
Hour = 60 * 60


@App.function(
    image=TrainingImage,
    volumes={DataRootOnVolume: DataVolume},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=Hour,
)
def PrepareData():
    import os
    import shutil
    import zipfile
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    Root = Path(DataRootOnVolume)
    UtkDir = Root / "datasets" / "raw" / "utkface" / "UTKFace"
    ManifestDst = Root / "datasets" / "processed" / "faces_manifest.csv"
    if UtkDir.exists() and ManifestDst.exists():
        print("data already on volume, skipping download")
        return

    Token = os.environ["HF_TOKEN"]
    ZipPath = hf_hub_download(HfRepoId, "utkface.zip", repo_type="dataset", token=Token)
    CsvPath = hf_hub_download(HfRepoId, "faces_manifest.csv", repo_type="dataset", token=Token)
    print("extracting utkface.zip onto the volume...")
    with zipfile.ZipFile(ZipPath) as Archive:
        Archive.extractall(Root)
    ManifestDst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(CsvPath, ManifestDst)
    DataVolume.commit()
    print(f"data ready: {sum(1 for _ in UtkDir.iterdir())} images on /data")


@App.function(
    image=TrainingImage,
    gpu="A100",
    volumes={DataRootOnVolume: DataVolume, RunsRoot: RunsVolume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=Hour,
)
def TrainOnGpu():
    import os

    os.environ["FACE_DATA_ROOT"] = DataRootOnVolume

    import torch
    import wandb
    from torch.utils.data import DataLoader

    from FaceDataset import FaceGenderAgeDataset
    from FaceModel import FaceGenderAgeModel

    BatchSize = 256
    MaxEpochs = 40
    LearningRate = 1e-4
    AgeScale = 100.0
    AgeWeight = 3.0
    EarlyStopPatience = 5
    Device = "cuda"
    BestCheckpointPath = f"{RunsRoot}/model_a_gender_age.pth"

    wandb.init(project="face-gender-age", config={
        "batch_size": BatchSize, "max_epochs": MaxEpochs, "lr": LearningRate,
        "backbone": "resnet34", "age_weight": AgeWeight, "early_stop_on": "val_loss", "patience": EarlyStopPatience})

    TrainLoader = DataLoader(FaceGenderAgeDataset("train", UseAugmentation=True),
                             batch_size=BatchSize, shuffle=True, num_workers=8, pin_memory=True)
    ValLoader = DataLoader(FaceGenderAgeDataset("val"),
                           batch_size=BatchSize, shuffle=False, num_workers=8, pin_memory=True)

    Model = FaceGenderAgeModel().to(Device).to(memory_format=torch.channels_last)
    GenderLoss = torch.nn.BCEWithLogitsLoss()
    AgeLoss = torch.nn.L1Loss()
    Optimizer = torch.optim.Adam(Model.parameters(), lr=LearningRate)
    Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Optimizer, mode="min", factor=0.5, patience=2)

    def RunEpoch(Loader, IsTraining):
        Model.train() if IsTraining else Model.eval()
        LossSum = GenderLossSum = AgeErrorSum = Seen = 0.0
        TruePos = FalsePos = FalseNeg = CorrectGender = 0
        for ImageBatch, GenderBatch, AgeBatch in Loader:
            ImageBatch = ImageBatch.to(Device, memory_format=torch.channels_last, non_blocking=True)
            GenderBatch = GenderBatch.to(Device, non_blocking=True)
            AgeBatch = AgeBatch.to(Device, non_blocking=True)
            with torch.set_grad_enabled(IsTraining):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    GenderLogit, AgePrediction = Model(ImageBatch)
                GenderLogit = GenderLogit.float()
                AgePrediction = AgePrediction.float()
                GenderLossValue = GenderLoss(GenderLogit, GenderBatch)
                BatchLoss = GenderLossValue + AgeWeight * AgeLoss(AgePrediction, AgeBatch / AgeScale)
            if IsTraining:
                Optimizer.zero_grad()
                BatchLoss.backward()
                Optimizer.step()
            N = ImageBatch.size(0)
            LossSum += BatchLoss.item() * N
            GenderLossSum += GenderLossValue.item() * N
            AgeErrorSum += (AgePrediction * AgeScale - AgeBatch).abs().sum().item()
            Predicted = (torch.sigmoid(GenderLogit) > 0.5).float()
            CorrectGender += (Predicted == GenderBatch).sum().item()
            TruePos += ((Predicted == 1) & (GenderBatch == 1)).sum().item()
            FalsePos += ((Predicted == 1) & (GenderBatch == 0)).sum().item()
            FalseNeg += ((Predicted == 0) & (GenderBatch == 1)).sum().item()
            Seen += N
        Precision = TruePos / (TruePos + FalsePos + 1e-9)
        Recall = TruePos / (TruePos + FalseNeg + 1e-9)
        return {"loss": LossSum / Seen, "gender_loss": GenderLossSum / Seen,
                "age_mae": AgeErrorSum / Seen, "gender_acc": CorrectGender / Seen,
                "gender_f1": 2 * Precision * Recall / (Precision + Recall + 1e-9)}

    BestValLoss = float("inf")
    EpochsSinceBest = 0
    for Epoch in range(1, MaxEpochs + 1):
        Train = RunEpoch(TrainLoader, True)
        Val = RunEpoch(ValLoader, False)
        Scheduler.step(Val["loss"])
        CurrentLr = Optimizer.param_groups[0]["lr"]
        print(f"epoch {Epoch:2d}  val[loss {Val['loss']:.3f} f1 {Val['gender_f1']:.3f} "
              f"acc {Val['gender_acc']:.3f} age {Val['age_mae']:.1f}y]  lr {CurrentLr:.1e}")
        wandb.log({"epoch": Epoch, "lr": CurrentLr,
                   "train_loss": Train["loss"], "val_loss": Val["loss"],
                   "train_gender_loss": Train["gender_loss"], "val_gender_loss": Val["gender_loss"],
                   "train_age_mae": Train["age_mae"], "val_age_mae": Val["age_mae"],
                   "train_gender_f1": Train["gender_f1"], "val_gender_f1": Val["gender_f1"],
                   "val_gender_acc": Val["gender_acc"]})

        if Val["loss"] < BestValLoss:
            BestValLoss = Val["loss"]
            EpochsSinceBest = 0
            torch.save(Model.state_dict(), BestCheckpointPath)
            RunsVolume.commit()
            print(f"  new best val loss {BestValLoss:.3f} (epoch {Epoch}) -> saved")
        else:
            EpochsSinceBest += 1
            if EpochsSinceBest >= EarlyStopPatience:
                print(f"early stop at epoch {Epoch} (no val-loss gain for {EarlyStopPatience} epochs)")
                break

    wandb.finish()
    print(f"done. best val loss {BestValLoss:.3f}; checkpoint {BestCheckpointPath}")


@App.local_entrypoint()
def main():
    PrepareData.remote()
    TrainOnGpu.remote()
