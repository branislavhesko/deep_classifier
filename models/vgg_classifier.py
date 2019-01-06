from torchvision.models import vgg19_bn, vgg
from torch.nn import Dropout, Module, Linear, ReLU, Sequential
import torch


class VGGClassifier(Module):

    def __init__(self, number_of_classes=2, load_pretrained_weights=True, freeze_weights=True):
        super(VGGClassifier, self).__init__()
        self._number_of_classes = number_of_classes
        self.vgg = vgg19_bn()
        path = "./pretrained_weights/vgg19_bn-c79401a0.pth"
        if load_pretrained_weights:
            self.vgg.load_state_dict(torch.load(path))
        if freeze_weights:
            self._freeze_vgg_layers_weights()
        self._change_output_layer_size()

    def forward(self, x):
        return self.vgg(x)

    def _change_output_layer_size(self):
        in_features = self.vgg.classifier[-1].in_features
        features = list(self.vgg.classifier.children())[:-1]
        features.append(Linear(in_features, self._number_of_classes))
        # features.append(torch.nn.ReLU(inplace=True))
        self.vgg.classifier = Sequential(*features)

    def _freeze_vgg_layers_weights(self):
        for param in self.vgg.features.parameters():
            param.require_grad = False

    def build_whole_classifier(self, input_shape):
        layers = [
            Linear(input_shape[0] * input_shape[1] // 2, 4096, bias=True),
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(4096, 4096, bias=True),
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(4096, self._number_of_classes, bias=True)
        ]
        self.vgg.classifier = torch.nn.Sequential(*layers)        

if __name__ == "__main__":
    v = VGGClassifier(2)
    v.build_whole_classifier((512,512))
    print(v)
