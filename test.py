import torch
import torchvision
import torch.nn as nn
from model import *
import os
from data_prepare import *
from argparse import *

parser = ArgumentParser()
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--data_path",default="UTKFace", type=str)
args = parser.parse_args()


os.system("gdown --id 1MSUTHE2yoYrW_S0rinbtDn_bVJnsWblK")#pretrain weights

batch_size = 64
epoch = 20
lr = 0.001

all_data = data_loader(args.data_path)

torch.manual_seed(0)
dataset_size = int(0.8 * len(all_data))
test_dataset_size = len(all_data) - dataset_size
_, test_dataset = torch.utils.data.random_split(all_data, [dataset_size, test_dataset_size])
test_dataset = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

device=torch.device(args.device)
model = Model()
model=model.to(device)


model.load_state_dict(torch.load("age_prediction.pth"))
model.eval()

def calc_loss(y_pred, labels):
    acc=torch.abs(y_pred - labels.data) / len(y_pred)
    return acc


test_loss=0.0
for img, label in test_dataset:

    img = img.to(device)
    label = label.to(device)

    pred = model(img)
    test_loss += calc_loss(pred, label)

total_loss = test_loss / len(test_dataset)
print("test loss:",total_loss)
