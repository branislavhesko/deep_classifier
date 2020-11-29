import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loaders.data_loader_riadd import DataLoaderRiadd
from models.resnet_feature import ResnetFeature


def _distance(output1, output2):
    return torch.sqrt(EPS + (output1 - output2).pow(2).sum(-1))

def calculate_loss(output1, output2, annotation1, annotation2):
    distance = _distance(output1, output2)
    signs = torch.ones(annotation1.shape[0])
    for idx, (a1, a2) in enumerate(zip(annotation1, annotation2)):
        if torch.sum(a1 * a2) == 0:
            signs[idx] *= -1
    loss_ = 0
    for idx, sign in enumerate(signs):
        if sign < 0:
            loss_ += torch.relu(6 - distance[idx])
        else:
            loss_ += distance[idx]
    return loss_

DEVICE = "cuda"
EPS = 1e-7


data_loader = DataLoader(dataset=DataLoaderRiadd("/home/brani/STORAGE/DATA/RFMID/Training_Set/"),
                         shuffle=False, batch_size=2, num_workers=2)

model = ResnetFeature().to(DEVICE)
optimizer = SGD(params=model.parameters(), lr=1e-3)
writer = SummaryWriter()

for epoch in range(4):
    model.train()
    t = tqdm(data_loader)
    for idx, data in enumerate(t):
        optimizer.zero_grad()
        img1, img2, annotation1, annotation2 = data
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)

        output1 = model(img1)
        output2 = model(img2)
        loss = calculate_loss(output1, output2, annotation1, annotation2)
        loss.backward()
        writer.add_scalar("Loss", loss, idx)
        t.set_description("Loss: {}".format(loss.item()))
        optimizer.step()
    torch.save(model.state_dict(), "w.pth")