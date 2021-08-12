from runner import train_model
#from binary_runner import train_model
import torch
import torch.nn as nn
import torch.optim as optim
from aux_dataset import CustomDataset
from aux_dataset import data_transforms0 #, data_transforms_anuar

#from bin_dataset import CustomDataset
#from bin_dataset import data_transforms0 #, data_transforms_anuar

from torch.utils.data import WeightedRandomSampler
import numpy as np
import argparse
from efficientnet_pytorch import EfficientNet
import sys
sys.path.append("pytorch-image-models")
#sys.path.append("LovaszSoftmax/pytorch")
import timm
from model import Net, Net_v2l, Net_v2m, Net_v2s
from timm.models.efficientnet import *
from catalyst.data.sampler import BalanceClassSampler


parser = argparse.ArgumentParser()

parser.add_argument('--scheduler', type=str, default="plateau")
parser.add_argument('--model', type=str, default="b7")
parser.add_argument('--input_feats', type=str, default="2560")
parser.add_argument('--transforms_num', type=str, default="0")
parser.add_argument('--aux', type=str, default="567")
parser.add_argument('--penalize', type=bool, default=True)
parser.add_argument('--fold', type=str, default="2")
parser.add_argument('--v2_size', type=str, default="l")
parser.add_argument('--batch_size', type=int, default=5*6)
parser.add_argument('--bin', type=bool, default=False)

args = parser.parse_args()

data_root = "path to input"

transform = data_transforms0

if args.v2_size == "l":
    model = Net_v2l()
    if args.bin:
        model.logit = nn.Linear(1280, 1)
elif args.v2_size == "m":
    model = Net_v2m()
    if args.bin:
        model.logit = nn.Linear(1280, 1)
elif args.v2_size == "s":
    model = Net_v2s()
elif args.v2_size == "b7":
    model = Net()


print(args.v2_size)

model = torch.nn.DataParallel(model)
model = model.cuda()
criterion = [nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()]
transforms = transform()

lr = 3e-4

optimizer_ft = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)

image_datasets = {x: CustomDataset(transforms[x], x, args.fold, data_root) for x in ['train', 'val']}


dataloaders_dict = {"train":
                        torch.utils.data.DataLoader(image_datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=8),
                    "val": torch.utils.data.DataLoader(image_datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=8)
                    }

train_model(model, dataloaders_dict, criterion, optimizer_ft, fold=args.fold, aux=args.aux, penalize=args.penalize, num_epochs=20, scheduler=args.scheduler, v2_size=args.v2_size)

