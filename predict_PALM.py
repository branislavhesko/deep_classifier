import torch
import torchvision.transforms as tfs
from tqdm import tqdm
from utils.show_predictions import show_predictions
from models.densenet_classifier import DenseNetClassifier
from models.vgg_classifier import VGGClassifier
import glob
from PIL import Image

weight_path = "./model_weights/17_03_19-13_59_04.pth"

transform = tfs.Compose([
            tfs.Resize(256),
            tfs.CenterCrop(224),
            tfs.ToTensor()])
model = DenseNetClassifier(num_classes=2, pretrained=True)
# model.build_whole_classifier((224, 224))
model.load_state_dict(torch.load(weight_path))
model.cuda()
model.eval()

for image in glob.glob("./data/PALM_clasification/test/PATHOLOGY/*.jpg"):
    image_loaded = Image.open(image)

    transformed = torch.autograd.Variable(transform(image_loaded).unsqueeze(dim=0)).cuda()

    output = model(transformed)

    print("FILE: {}, OUTPUT: {}".format(image, output))
