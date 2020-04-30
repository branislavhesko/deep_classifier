import torch


class ResNestClassifier(torch.nn.Module):

    def __init__(self, num_classes, weights=None, use_cuda=True):
        super().__init__()
        self._num_classes = num_classes
        self._model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
        self._model.fc = torch.nn.Linear(2048, self._num_classes)
        if weights:
            self._load(weights)
        if use_cuda:
            self._model = self._model.to("cuda")

    def _load(self, weights):
        self._model.load_state_dict(weights)

    def forward(self, input_):
        return self._model(input_)


if __name__ == "__main__":
    classifier = ResNestClassifier(10, use_cuda=False)
    out = classifier(torch.rand(2, 3, 128, 128))
    print(classifier)