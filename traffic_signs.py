from torchvision.transforms import Compose, RandomRotation, Normalize, Resize, ToTensor
import torch
from tqdm import tqdm

from models.densenet_classifier import DenseNetClassifier
from utils.dataset_traffic_signs import get_dataloader


INPUT_SIZE = (112, 112)
NUM_CLASSES = 25
WITH_CUDA = True

train_transform = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomRotation(10),
    Resize(INPUT_SIZE),
])

test_transform = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    Resize(INPUT_SIZE)
])

train_loader = get_dataloader("./data/train/", NUM_CLASSES, train_transform)
test_loader = get_dataloader("./data/test/", NUM_CLASSES, test_transform)

model = DenseNetClassifier(num_classes=NUM_CLASSES)
model = model.cuda() if WITH_CUDA else model
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.001, weight_decay=1e-4)
loss = torch.nn.CrossEntropyLoss()


def train(num_epochs, model, optimizer, loss_fn, data_loader_train, 
          data_loader_test, with_cuda, validation_frequency):
    
    for epoch in range(num_epochs):
        train_acc = 0
        train_loss = 0
        actual_loss = 0

        t = tqdm(data_loader_train)
        for image, label in t:
            t.set_description("ACTUAL LOSS: {}".format(actual_loss))

            optimizer.zero_grad()

            image = image.cuda() if with_cuda else image
            label = label.cuda() if with_cuda else label

            output = model(image)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            actual_loss = loss.detach().cpu().item()
            train_loss += actual_loss
            prediction = torch.argmax(output, 1)
            train_acc += torch.sum(prediction == label).item()

    if epoch % validation_frequency == 0:
        validate(model, optimizer, loss_fn, data_loader_test, with_cuda)

def validate(model, optimizer, loss, data_loader, with_cuda):
    pass
