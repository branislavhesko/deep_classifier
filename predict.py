import torch
from utils.dataloader_OCT import get_dataloaders_and_sizes
from torchvision.models import alexnet
from tqdm import tqdm


def predict(model, test_loader, loss_fn, cuda_available):
    model.eval()

    pred_acc = 0
    pred_loss = 0

    for data in tqdm(test_loader):
        image, labels = data

        if cuda_available:
            image = torch.autograd.Variable(image).cuda()
            labels = torch.autograd.Variable(labels).cuda()

        output = model(image)

        pred_loss += loss_fn(output, labels).cpu().item()
        prediction = torch.argmax(output, 1)
        pred_acc += torch.sum(prediction == labels)
        torch.cuda.empty_cache()
    total_num_images = test_loader.batch_size * len(test_loader)
    print("Mean loss per image: {}, prediction_accuracy: {}/{}".format(
        loss_fn / total_num_images, pred_acc, total_num_images))


if __name__ == "__main__":
    data_loaders, data_sizes = get_dataloaders_and_sizes(
        (224, 224), ["train", "val", "test"])
    test_loader = data_loaders[2]
    weight_path = "./model_weights/"
    model = alexnet(True)
    model.load_state_dict(weight_path)
