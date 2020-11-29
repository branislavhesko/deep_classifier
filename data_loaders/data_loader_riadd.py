import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset


class DataLoaderRiadd(Dataset):
    SUBFOLDER = "Training"
    ANNOTATIONS_FILE = "RFMiD_Training_Labels.csv"

    def __init__(self, path) -> None:
        super().__init__()
        self._images = sorted(glob.glob(os.path.join(path, self.SUBFOLDER, "*.png")))
        self._annotations = pd.read_csv(os.path.join(path, self.ANNOTATIONS_FILE))
        self.classes = self._annotations.columns

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        random_index = self._random_index(index)
        image1, annotation1 = self._get_image_annotation(index)
        image2, annotation2 = self._get_image_annotation(random_index)
        return (
            torch.from_numpy(image1).permute([2, 0, 1]).float(), 
            torch.from_numpy(image2).permute([2, 0, 1]).float(), 
            annotation1.values[0, 2:], annotation2.values[0, 2:])

    def _get_image_annotation(self, index):
        image = cv2.resize(np.array(Image.open(self._images[index])) / 255., (256, 256))
        basename = os.path.basename(self._images[index])
        image_id = int(basename.split(".")[0])
        annotation = self._annotations.loc[self._annotations["ID"] == image_id, :]
        return image, annotation

    def _random_index(self, index):
        random_int = index
        for _ in range(5):
            random_int = random.randint(0, len(self) - 1)
            if random_int != index:
                break
        return random_int


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    dataset = DataLoaderRiadd("/home/brani/STORAGE/DATA/RFMID/Training_Set/")

    for image_, _, _, annot in dataset:
        plt.imshow(image_.permute(1, 2, 0).numpy())
        plt.show()