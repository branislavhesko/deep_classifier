from torchvision.models import densenet169
import torch


class DenseNetClassifier(torch.nn.Module):

    def __init__(self, num_classes, pretrained=True, fine_tune=False):
        super(DenseNetClassifier, self).__init__()

        self.densenet = densenet169(pretrained)
        self._num_classes = num_classes
        set_parameter_requires_grad(self.densenet, True)
        self._build_last_layer()

    def _build_last_layer(self):
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = torch.nn.Linear(in_features, self._num_classes)

    def forward(self, x):
        return self.densenet(x)


def set_parameter_requires_grad(model, fine_tune):
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
