import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

EmotionCount = 7


class ExpressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        FeatureCount = self.Backbone.fc.in_features
        self.Backbone.fc = nn.Identity()
        self.ExpressionHead = nn.Linear(FeatureCount, EmotionCount)

    def forward(self, ImageTensor):
        Features = self.Backbone(ImageTensor)
        return self.ExpressionHead(Features)


if __name__ == "__main__":
    import torch

    Model = ExpressionModel()
    Model.eval()
    DummyBatch = torch.randn(4, 3, 224, 224)
    EmotionLogits = Model(DummyBatch)
    print(f"emotion logits {tuple(EmotionLogits.shape)}")
    ParameterCount = sum(Parameter.numel() for Parameter in Model.parameters())
    print(f"parameters: {ParameterCount:,}")
