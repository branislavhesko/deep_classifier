import numpy as np
import torch
from utils.dataloader_OCT import get_dataloaders_and_sizes
from torchvision.models import alexnet
from tqdm import tqdm
from utils.show_predictions import show_predictions
from models.densenet_classifier import DenseNetClassifier
from models.vgg_classifier import VGGClassifier


def predict(model, test_loader, loss_fn, cuda_available, available_classes=None):
    model.eval()

    pred_acc = 0
    pred_loss = 0
    unsuccessful = {
        "image": [],
        "gt": [],
        "pred": []
    }

    for data in tqdm(test_loader):
        image, labels = data

        if cuda_available:
            image = torch.autograd.Variable(image).cuda()
            labels = torch.autograd.Variable(labels).cuda()

        output = model(image)

        pred_loss += loss_fn(output, labels).cpu().item()
        prediction = torch.argmax(output, 1)
        pred_acc += torch.sum(prediction == labels)

        if available_classes is not None:
            for i in range(prediction.shape[0]):
                if prediction[i] != labels[i]:
                    unsuccessful["image"].append(np.transpose(image[i, :, :, :].cpu().numpy()))
                    unsuccessful["gt"].append(available_classes[labels[i].item()])
                    unsuccessful["pred"].append(available_classes[prediction[i].item()])

            if len(unsuccessful["image"]) == 5:
                show_predictions(unsuccessful["image"], unsuccessful["pred"], unsuccessful["gt"], "result.png")
                break

        torch.cuda.empty_cache()
    total_num_images = test_loader.batch_size * len(test_loader)
    print("Mean loss per image: {}, prediction_accuracy: {}/{}".format(
        pred_loss / total_num_images, pred_acc, total_num_images))


if __name__ == "__main__":
    available_classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    folders = ["train", "val", "test"]
    data_loaders, data_sizes = get_dataloaders_and_sizes(
        (224, 224), folders)
    test_loader = data_loaders[folders[2]]
    weight_path = "./model_weights/13_03_19-02_09_42.pth"
    model = VGGClassifier(4)
    model.build_whole_classifier((224, 224))
    model.load_state_dict(torch.load(weight_path))
    model.cuda()

    predict(model, test_loader, torch.nn.CrossEntropyLoss(), True, None)
