import numpy as np
import cv2
import torch
import torchvision
from model import *
from argparse import *


parser = ArgumentParser()
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--input_img",default="35_1_0_20170113005254692.jpg.chip.jpg", type=str)
args = parser.parse_args()
width=height=224
device=torch.device(args.device)

transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize((0), (1)),
])

img = cv2.imread(args.input_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (width, height))

input_img = transform(img).unsqueeze(0).to(device)
model = Model()
model = model.to(device)
model.load_state_dict(torch.load("age_prediction.pth"))
model.eval()

pred = model(input_img)
print("age predicted by model:", pred[0])