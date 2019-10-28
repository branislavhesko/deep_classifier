from enum import auto, Enum
import glob
import numpy as np
import os
from PIL import Image
import torch


class DatasetKeys:
    LABEL = auto()
    INDEX = auto()
    IMAGE = auto()


class TrafficSignDataset(torch.utils.data.Dataset):
    EXTENSION = "png"

    def __init__(self, base_path: str, num_classes=2, transform=lambda x: x):
        super().__init__()
        self._base_path = base_path
        self._num_classes = num_classes
        self._dataset = []
        self._transform = transform

    def load(self):
        folders = next(os.walk(self._base_path))[1]
        assert self._num_classes == len(folders)
        dataset = {}
        for folder in folders:
            dataset[folder] = glob.glob(os.path.join(self._base_path, folder, "*." + self.EXTENSION))
        counter = 0
        for idx, class_ in enumerate(dataset.keys()):
            for item in dataset[class_]:
                self._dataset.append({
                    DatasetKeys.LABEL: class_,
                    DatasetKeys.IMAGE: item,
                    DatasetKeys.INDEX: idx
                })
                counter += 1
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        label = torch.Tensor([self._dataset[index][DatasetKeys.INDEX]]).long()
        image = Image.open(self._dataset[index][DatasetKeys.IMAGE])
        image = self._transform(image)
        return image, label


def get_dataloader(base_path, num_classes, transform, 
                   num_workers=4, batch_size=8, shuffle=False):
    dataset = TrafficSignDataset(base_path, num_classes, transform)
    dataset.load()
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=shuffle)
    return dataloader

