import glob
import os
from random import shuffle

import cv2
import torch
from torch.utils.data import Dataset


class DataSetRefuge(Dataset):
    CLASSES = {"GLAUCOMA": 1, "NON-GLAUCOMA": 2}

    def __init__(self, path, transforms):
        super().__init__()
        self._transforms = transforms
        images_glaucoma = [(img,  self.CLASSES["GLAUCOMA"]) for img in glob.glob(
            os.path.join(path, "GLAUCOMA", "*.jpg"))]
        images_non_glaucoma = [(img, self.CLASSES["NON-GLAUCOMA"]) for img in glob.glob(
            os.path.join(path, "NON-GLAUCOMA", "*.jpg"))]
        self._annotations = images_glaucoma + images_non_glaucoma
        shuffle(self._annotations)

    def __len__(self):
        return len(self._annotations)

    def __getitem__(self, item):
        img_file, label = self._annotations[item]
        image = cv2.cvtColor(cv2.imread(img_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.
        image = self._transforms(image)
        return torch.from_numpy(image).permute([2, 0, 1]), torch.tensor(label)
