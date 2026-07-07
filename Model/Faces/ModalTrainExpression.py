import modal

App = modal.App("face-expression")

TrainingImage = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "pillow", "tqdm",
                 "wandb", "huggingface_hub")
    .add_local_python_source("ExpressionDataset", "ExpressionModel")
)

DataVolume = modal.Volume.from_name("face-data", create_if_missing=True)
RunsVolume = modal.Volume.from_name("face-runs", create_if_missing=True)

HfRepoId = "AwsHanaqtah/rafdb-expression"
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
    import zipfile
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    RafdbDir = Path(DataRootOnVolume) / "datasets" / "raw" / "rafdb" / "DATASET"
    if RafdbDir.exists():
        print("rafdb already on volume, skipping download")
        return

    ZipPath = hf_hub_download(HfRepoId, "rafdb.zip", repo_type="dataset", token=os.environ["HF_TOKEN"])
    print("extracting rafdb.zip onto the volume...")
    with zipfile.ZipFile(ZipPath) as Archive:
        Archive.extractall(DataRootOnVolume)
    DataVolume.commit()
    print(f"rafdb ready under {RafdbDir}")


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

    from ExpressionDataset import ExpressionDataset, ComputeClassWeights
    from ExpressionModel import ExpressionModel, EmotionCount

    BatchSize = 256
    MaxEpochs = 40
    LearningRate = 1e-4
    LabelSmoothing = 0.1
    LossType = "focal"          # "weighted_ce" or "focal"
    FocalGamma = 2.0
    EarlyStopPatience = 5
    Device = "cuda"
    BestCheckpointPath = f"{RunsRoot}/model_b_expression_{LossType}.pth"

    wandb.init(project="face-expression", name=f"{LossType}-g{FocalGamma}", config={
        "batch_size": BatchSize, "max_epochs": MaxEpochs, "lr": LearningRate,
        "backbone": "resnet34", "loss": LossType, "focal_gamma": FocalGamma,
        "label_smoothing": LabelSmoothing, "early_stop_on": "val_loss", "patience": EarlyStopPatience})

    TrainLoader = DataLoader(ExpressionDataset("train", UseAugmentation=True),
                             batch_size=BatchSize, shuffle=True, num_workers=8, pin_memory=True)
    ValLoader = DataLoader(ExpressionDataset("val"),
                           batch_size=BatchSize, shuffle=False, num_workers=8, pin_memory=True)

    Model = ExpressionModel().to(Device).to(memory_format=torch.channels_last)
    ClassWeights = ComputeClassWeights().to(Device)
    def ComputeLoss(Logits, Targets):
        PerSampleCe = torch.nn.functional.cross_entropy(Logits, Targets, weight=ClassWeights,
                                                        label_smoothing=LabelSmoothing, reduction="none")
        if LossType != "focal":
            return PerSampleCe.mean()
        TrueProb = Logits.softmax(1).gather(1, Targets.unsqueeze(1)).squeeze(1)
        return ((1 - TrueProb) ** FocalGamma * PerSampleCe).mean()
    Optimizer = torch.optim.Adam(Model.parameters(), lr=LearningRate)
    Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Optimizer, mode="min", factor=0.5, patience=2)

    def RunEpoch(Loader, IsTraining):
        Model.train() if IsTraining else Model.eval()
        LossSum = Seen = 0.0
        Confusion = torch.zeros(EmotionCount, EmotionCount, dtype=torch.long)
        for ImageBatch, EmotionBatch in Loader:
            ImageBatch = ImageBatch.to(Device, memory_format=torch.channels_last, non_blocking=True)
            EmotionBatch = EmotionBatch.to(Device, non_blocking=True)
            with torch.set_grad_enabled(IsTraining):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    Logits = Model(ImageBatch)
                BatchLoss = ComputeLoss(Logits.float(), EmotionBatch)
            if IsTraining:
                Optimizer.zero_grad()
                BatchLoss.backward()
                Optimizer.step()
            N = ImageBatch.size(0)
            LossSum += BatchLoss.item() * N
            Predicted = Logits.argmax(1)
            Pairs = (EmotionBatch * EmotionCount + Predicted).cpu()
            Confusion += torch.bincount(Pairs, minlength=EmotionCount * EmotionCount).reshape(EmotionCount, EmotionCount)
            Seen += N

        Diag = Confusion.diag().float()
        Accuracy = Diag.sum().item() / Confusion.sum().clamp(min=1).item()
        Precision = Diag / Confusion.sum(0).float().clamp(min=1)
        Recall = Diag / Confusion.sum(1).float().clamp(min=1)
        MacroF1 = (2 * Precision * Recall / (Precision + Recall).clamp(min=1e-9)).mean().item()
        return {"loss": LossSum / Seen, "accuracy": Accuracy, "macro_f1": MacroF1}

    BestValLoss = float("inf")
    EpochsSinceBest = 0
    for Epoch in range(1, MaxEpochs + 1):
        Train = RunEpoch(TrainLoader, True)
        Val = RunEpoch(ValLoader, False)
        Scheduler.step(Val["loss"])
        CurrentLr = Optimizer.param_groups[0]["lr"]
        print(f"epoch {Epoch:2d}  val[loss {Val['loss']:.3f} acc {Val['accuracy']:.3f} "
              f"macroF1 {Val['macro_f1']:.3f}]  lr {CurrentLr:.1e}")
        wandb.log({"epoch": Epoch, "lr": CurrentLr,
                   "train_loss": Train["loss"], "val_loss": Val["loss"],
                   "train_accuracy": Train["accuracy"], "val_accuracy": Val["accuracy"],
                   "train_macro_f1": Train["macro_f1"], "val_macro_f1": Val["macro_f1"]})

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
