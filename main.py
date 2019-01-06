import os
import glob
import random
from time import gmtime, strftime
from models.vgg_classifier import VGGClassifier
from models.densenet_classifier import DenseNetClassifier
from torchvision.models.densenet import densenet169
from utils.dataloader import AMDloader
import torch
import torchvision

from train import train

images = glob.glob("./data/AMD/*.jpg")
random.shuffle(images)
classes = ["nonAMD", "AMD"]

image_size = (224, 224)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Resize(size=image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

transforms_val = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=image_size),
    torchvision.transforms.ToTensor()
])

dataloader = AMDloader(images[:320], ["nonAMD", "AMD"], transforms)
dataloader_val = AMDloader(images[320:], ["nonAMD", "AMD"], transforms_val)

# model = VGGClassifier(len(classes))
model = DenseNetClassifier(num_classes=2, pretrained=True)
# model.build_whole_classifier(image_size)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

train(501, model, dataloader, dataloader_val, optimizer, loss_fn, None, False)

model_save_path = "./model_weights"
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

torch.save(os.path.join(model_save_path, strftime("%d_%d_%y-%H_%M_%S.pth", gmtime())))