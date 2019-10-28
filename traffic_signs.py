from torchvision.transforms import Compose, RandomRotation, Normalize, Resize, ToTensor
import torch
from tqdm import tqdm

from models.densenet_classifier import DenseNetClassifier
from utils.dataset_traffic_signs import TrafficSignDataset, get_dataloader
from utils.show_predictions import show_predictions


INPUT_SIZE = (112, 112)
NUM_CLASSES = 46
WITH_CUDA = True

train_transform = Compose([
    RandomRotation(10),
    Resize(INPUT_SIZE),
    ToTensor(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

test_transform = Compose([
    Resize(INPUT_SIZE),
    ToTensor(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_loader, _ = get_dataloader("/home/brani/DATA/datasets/traffic_signs/sorted/",
                                 NUM_CLASSES, train_transform, shuffle=True, batch_size=8)
test_loader, test_set = get_dataloader("/home/brani/DATA/datasets/traffic_signs/sorted/",
                                       NUM_CLASSES, test_transform, batch_size=8, shuffle=True)

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
        for image, label, _ in t:
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


def validate(model, optimizer, loss, data_loader, with_cuda, with_visualization=True):
    images_visualization = []
    predictions_visualization = []
    ground_truths_visualization = []
    total_samples = 1
    total_correct = 0
    total_loss = 0
    model.densenet.eval()
    t = tqdm(data_loader)
    counter = 0
    for image, label, label_str in t:
        t.set_description("ACTUAL_PRECISION: {}".format(total_correct / total_samples))

        image = image.cuda() if with_cuda else image
        label = label.cuda() if with_cuda else label

        output = model(image)
        total_loss += loss(output, label.squeeze()).item()

        prediction = torch.argmax(output, dim=1)
        total_correct += torch.sum(prediction == label.squeeze()).item()
        total_samples += prediction.size()[0]

        if with_visualization:
            images_visualization.append(image.detach().cpu()[0, :, :, :].permute([1, 2, 0]).numpy())
            predictions_visualization.append(test_set.folders[int(prediction[0].cpu().numpy())])
            ground_truths_visualization.append(label_str[0])

        if len(images_visualization) == 5:
            show_predictions(images_visualization, predictions_visualization,
                             ground_truths_visualization, f"{counter}.png")
            counter += 1
            images_visualization = []
            predictions_visualization = []
            ground_truths_visualization = []

        del output, prediction, image, label
        torch.cuda.empty_cache()

    print("PRECISION {}, LOSS {}".format(total_correct / total_samples, total_loss / total_samples))


train(5, model, optimizer, loss, train_loader, test_loader, True, 2)
