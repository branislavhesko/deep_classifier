import os
import glob
import random
from time import gmtime, strftime
from models.vgg_classifier import VGGClassifier
from models.densenet_classifier import DenseNetClassifier
from torchvision.models.densenet import densenet169
from utils.dataloader_AMD import AMDloader
from utils.dataloader_PALM import get_dataloaders_and_sizes
import torch
from torchvision.models import alexnet

from train import train


"""
images = glob.glob("./data/AMD/*.jpg")
random.shuffle(images)
classes = ["nonAMD", "AMD"]


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
"""

#model = VGGClassifier(4)
#weight_path = "./model_weights/13_03_19-11_04_12.pth"
model = DenseNetClassifier(2, pretrained=True)
#model.load_state_dict(torch.load(weight_path))# model = alexnet(True)
image_size = (224, 224)

#model.build_whole_classifier(image_size)
print(model)
optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
loss_fn = torch.nn.CrossEntropyLoss()
folders = ["train", "validate"]

dataloaders, datasizes = get_dataloaders_and_sizes((224, 224), folders)

train(50, model, dataloaders[folders[0]], dataloaders[folders[1]], optimizer_ft, loss_fn, None, True)

model_save_path = "./model_weights"
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

torch.save(model.state_dict(), os.path.join(model_save_path, strftime("%d_%m_%y-%H_%M_%S.pth", gmtime())))