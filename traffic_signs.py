from torchvision.transforms import Compose, RandomRotation, Normalize, Resize, ToTensor
import torch
from tqdm import tqdm

from models.densenet_classifier import DenseNetClassifier
from utils.dataset_traffic_signs import get_dataloader


INPUT_SIZE = (112, 112)
NUM_CLASSES = 46
WITH_CUDA = True

train_transform = Compose([
    RandomRotation(10),
    Resize(INPUT_SIZE),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

test_transform = Compose([
    Resize(INPUT_SIZE),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_loader = get_dataloader("/home/brani/DATA/datasets/traffic_signs/sorted/",
                              NUM_CLASSES, train_transform, shuffle=True, batch_size=4)
test_loader = get_dataloader("/home/brani/DATA/datasets/traffic_signs/sorted/",
                             NUM_CLASSES, test_transform, batch_size=4)

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
        model.densenet.train()
        t = tqdm(data_loader_train)
        for image, label in t:
            t.set_description("ACTUAL LOSS: {}".format(actual_loss))

            optimizer.zero_grad()

            image = image.cuda() if with_cuda else image
            label = label.cuda() if with_cuda else label

            output = model(image)
            loss = loss_fn(output, torch.squeeze(label))
            loss.backward()
            optimizer.step()

            actual_loss = loss.detach().cpu().item()
            train_loss += actual_loss
            prediction = torch.argmax(output, 1)
            train_acc += torch.sum(prediction == label).item()
        del image, label, output, loss
        torch.cuda.empty_cache()
        if epoch % validation_frequency == 0:
            validate(model, optimizer, loss_fn, data_loader_test, with_cuda)


def validate(model, optimizer, loss, data_loader, with_cuda):
    total_samples = 1
    total_correct = 0
    total_loss = 0
    model.densenet.eval()
    t = tqdm(data_loader)
    for image, label in t:
        t.set_description("ACTUAL_PRECISION: {}".format(total_correct / total_samples))

        image = image.cuda() if with_cuda else image
        label = label.cuda() if with_cuda else label

        output = model(image)
        total_loss += loss(output, label.squeeze()).item()

        prediction = torch.argmax(output, dim=1)
        total_correct += torch.sum(prediction == label.squeeze()).item()
        total_samples += prediction.size()[0]
        del output, prediction, image, label
        torch.cuda.empty_cache()

    print("PRECISION {}, LOSS {}".format(total_correct / total_samples, total_loss / total_samples))


train(5, model, optimizer, loss, train_loader, test_loader, True, 2)
