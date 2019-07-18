import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tfs


def get_dataloaders_and_sizes(image_size, folders):
    root = "./data/PALM_clasification"

    tranform = {
        folders[0]: tfs.Compose([
            tfs.Resize(256),
            tfs.CenterCrop(224),
            tfs.ToTensor()
        ]),
        folders[1]: tfs.Compose([
            tfs.Resize(256),
            tfs.CenterCrop(224),
            tfs.ToTensor(),
        ]),
    }

    image_folders = {x: ImageFolder(os.path.join(root, x), transform=tranform[x]) for x in folders}
    dataloaders = {x: DataLoader(image_folders[x], batch_size=16, shuffle=True, num_workers=4) for x in folders}
    data_sizes = [len(image_folders[x]) for x in folders]
    return dataloaders, data_sizes


if __name__ == "__main__":
    folders = ["train", "validate"]
    loaders, sizes = get_dataloaders_and_sizes((224, 224), folders)

    for index, size in enumerate(sizes):
        print("Dataset for {} has size {} images.".format(folders[index], size))



