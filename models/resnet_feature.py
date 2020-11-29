import torch
from torchvision.models.resnet import resnet50


class ResnetFeature(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._model = resnet50(pretrained=True)

    def forward(self, input_):
        x = self._model.conv1(input_)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)
        x = self._model.layer1(x)
        x = self._model.layer2(x)
        x = self._model.layer3(x)
        x = self._model.layer4(x)
        return torch.sigmoid(x.mean([2, 3]))

if __name__ == "__main__":
    model = ResnetFeature()
    print(model(torch.rand(2, 3, 128, 128)).shape)
