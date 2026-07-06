import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class FaceGenderAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        FeatureCount = self.Backbone.fc.in_features
        self.Backbone.fc = nn.Identity()
        self.GenderHead = nn.Linear(FeatureCount, 1)
        self.AgeHead = nn.Linear(FeatureCount, 1)

    def forward(self, ImageTensor):
        Features = self.Backbone(ImageTensor)
        GenderLogit = self.GenderHead(Features).squeeze(1)
        AgePrediction = self.AgeHead(Features).squeeze(1)
        return GenderLogit, AgePrediction


if __name__ == "__main__":
    import torch

    Model = FaceGenderAgeModel()
    Model.eval()
    DummyBatch = torch.randn(4, 3, ImageSize := 224, ImageSize)
    GenderLogit, AgePrediction = Model(DummyBatch)
    print(f"gender logits {tuple(GenderLogit.shape)}   age predictions {tuple(AgePrediction.shape)}")
    ParameterCount = sum(Parameter.numel() for Parameter in Model.parameters())
    print(f"parameters: {ParameterCount:,}")
