from enum import auto, Enum
import glob
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch


class ODIRDataset:
    class DataProps(Enum):
        AGE = auto()
        ID = auto()
        IMAGE_PATH = auto()
        LABEL = auto()
        LEFT_DIAGNOSIS = auto()
        RIGHT_DIAGNOSIS = auto()
        SEX = auto()

    LEFT_SUFFIX = "left"
    RIGHT_SUFFIX = "right"
    IMAGES_FOLDER = "images"
    NUM_CLASSES = 8

    class Pathology(Enum):
        NORMAL = 0
        DIABETES = 1
        GLAUCOMA = 2
        CATARACT = 3
        AGE_RELATED_MACULAR_DEGENERATION = 4
        HYPERTENSRION = 5
        MYOPIA = 6
        OTHER = 7

    def __init__(self, image_size):
        self._base_path = None
        self._image_size = image_size
        self.data = {
            self.DataProps.IMAGE_PATH: [],
            self.DataProps.LABEL: [],
            self.DataProps.LEFT_DIAGNOSIS: [],
            self.DataProps.RIGHT_DIAGNOSIS: [],
            self.DataProps.AGE: [],
            self.DataProps.ID: [],
            self.DataProps.SEX: []
        }

    def load(self, base_path):
        self._base_path = base_path
        print(glob.glob(os.path.join(base_path, "ODIR*.xlsx"))[0])
        label_spreadsheet = pd.read_excel(glob.glob(os.path.join(base_path, "ODIR*.xlsx"))[0])
        self.data[self.DataProps.IMAGE_PATH] = label_spreadsheet.loc[:, ("Left-Fundus", "Right-Fundus")]
        self.data[self.DataProps.SEX] = label_spreadsheet["Patient Sex"]
        self.data[self.DataProps.LABEL] = label_spreadsheet.loc[:, ('N', 'D', 'G', 'C', 'A', 'H', 'M', 'O')]
        self.data[self.DataProps.LEFT_DIAGNOSIS] = label_spreadsheet["Left-Diagnostic Keywords"]
        self.data[self.DataProps.RIGHT_DIAGNOSIS] = label_spreadsheet["Right-Diagnostic Keywords"]
        self.data[self.DataProps.ID] = label_spreadsheet["ID"]
        self.data[self.DataProps.AGE] = label_spreadsheet["Patient Age"]

    def get_diagnosis(self, index):
        label = np.array(self.data[self.DataProps.LABEL].iloc[index, :])
        print(label)
        print(label.nonzero()[0])
        diagnosis = [self.Pathology(idx).name for idx in label.nonzero()[0]]
        return "Image number: {}, diagnosis: {}".format(index, " & ".join(diagnosis))

    def show_images_at_index(self, index):
        imgs, label = self[index]
        fig = plt.figure(1, figsize=(10, 4), dpi=100)
        fig.suptitle(self.get_diagnosis(index))
        plt.subplot(1, 2, 1)
        plt.imshow(imgs[:, :, :3])
        plt.title("LEFT")
        plt.subplot(1, 2, 2)
        plt.imshow(imgs[:, :, 3:])
        plt.title("RIGHT")
        plt.show()

    def __len__(self):
        return len(self.data[self.DataProps.IMAGE_PATH])

    def __getitem__(self, index):
        left = cv2.imread(os.path.join(self._base_path, self.IMAGES_FOLDER, self.data[self.DataProps.IMAGE_PATH].loc[index, "Left-Fundus"]), cv2.IMREAD_COLOR)[:, :, ::-1]
        right = cv2.imread(os.path.join(self._base_path, self.IMAGES_FOLDER, self.data[self.DataProps.IMAGE_PATH].loc[index, "Right-Fundus"]), cv2.IMREAD_COLOR)[:, :, ::-1]
        label = np.array(self.data[self.DataProps.LABEL].iloc[index, :])
        label = np.argmax(label)
        left = cv2.resize(left, self._image_size)
        right = cv2.resize(right, self._image_size)
        image = np.concatenate((left, right), axis=2)
        return torch.from_numpy(image).permute([2, 0, 1]).float(), torch.tensor(label)


if __name__ == "__main__":
    odir = ODIRDataset((224, 224))
    odir.load("./data")
    odir.show_images_at_index(1000)