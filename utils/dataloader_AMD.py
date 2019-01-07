import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision


class AMDloader(Dataset):

    def __init__(self, image_files, possible_labels, transform=None):
        self._transform = transform
        self._image_files = image_files
        self.possible_labels = possible_labels
        self._labels = self._get_labels()

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, item_index):
        # print("Image:" + self._image_files[item_index] + " labels: " + str(self._labels[item_index]))
        image = Image.open(self._image_files[item_index])#.resize((224, 224), Image.ANTIALIAS)
        if self._transform:
            image = self._transform(image)
        return torch.Tensor(image).unsqueeze(0), torch.Tensor(np.array(self._labels[item_index])).unsqueeze(0).long()

    def _get_labels(self):
        labels = np.zeros((len(self._image_files)))
        for index, image_file in enumerate(self._image_files):
            if os.path.basename(image_file)[0] == "A":
                labels[index] = 1
            else:
                labels[index] = 0
        return labels


if __name__ == "__main__":
    import glob
    images = glob.glob("./data/AMD/*.jpg")
    loader = AMDloader(images, ("nonAMD", "AMD"))

    for sample in loader:
        pass
