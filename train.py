import torch
import torchvision
import torch.nn as nn
from model import Model
from argparse import *
from data_prepare import *
import os



parser = ArgumentParser()
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--data_path",default="UTKFace", type=str)
args = parser.parse_args()

device = torch.device(args.device)
model = Model()
model = model.to(device)
model.train(True)

batch_size = 64
epoch = 20
lr = 0.001
# os.system("pip install gdown")
os.system("gdown --id 0BxYys69jI14kYVM3aVhKS1VhRUk")
os.system("tar -xf UTKFace.tar.gz")
all_data = data_loader(args.data_path)

torch.manual_seed(0)

dataset_size = int(0.8 * len(all_data))
train_dataset_size = len(all_data) - dataset_size

train_data, _ = torch.utils.data.random_split(all_data, [dataset_size, train_dataset_size])
train_data = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.L1Loss()

for e in range(epoch):
  train_loss = 0.0
  for img, labels in train_data:
    img, labels = img.to(device), labels.to(device)
    optimizer.zero_grad()
    img = img.float()
    preds = model(img)

    loss = loss_function(preds, labels.float())
    loss.backward()
    optimizer.step()
    train_loss += loss

  total_loss = train_loss / len(train_data)
  print(f"Epoch: {e+1}, Loss: {total_loss}")
torch.save(model.state_dict(), "age_prediction.pth")
