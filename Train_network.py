import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from PIL import ImageFile
from utils.CBAMresnet import resnet18
from utils.load_data import XRayDataSet
from torch.utils.data import DataLoader
from collections.abc import Iterable
from torchvision import datasets, models, transforms
from torchvision.transforms.transforms import Compose, Resize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, ToTensor
from tensorboardX import SummaryWriter

data_path = './COVID19_dataset'
metadata = './metadata.csv'
PATH = 'X:/covid19/CBAM_model_pkls'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_aug = Compose([Resize((224, 224)), 
					 RandomRotation(180), 
					 RandomHorizontalFlip(), 
					 RandomVerticalFlip(), 
					 ToTensor()
					 ])
val_aug   = Compose([Resize((224, 224)), 
					 ToTensor()
					 ])

BS = 24
pre_epoch = 0
EPOCH = 50

writer = SummaryWriter(comment = 'Linear')

trainset = XRayDataSet(os.path.join(data_path, 'train'), metadata, transforms = train_aug)
testset = XRayDataSet(os.path.join(data_path, 'test'), metadata, transforms = val_aug)
trainloader = DataLoader(trainset, batch_size = BS, shuffle = True, num_workers = 0)
testloader = DataLoader(testset, batch_size = BS, shuffle = True, num_workers = 0)

net = resnet18(pretrained = True).to(device)

optimizer = optim.Adam(net.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
criterion.to(device = device)

def set_freeze_by_names(model, layer_names, freeze = True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

freeze_by_names(net, ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'))

y_predict = []
y_true = []

ImageFile.LOAD_TRUNCATED_IMAGES = True

print("Start Training!")
for epoch in range(pre_epoch, EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    accumulation_steps = 4
    length = len(trainloader)
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps
        loss.backward()
        sum_loss += loss.item()

        if ((i + 1) % accumulation_steps) == 0:
            optimizer.step()
            writer.add_scalar('Train', loss, epoch)
            optimizer.zero_grad()

        maxk = max((1, 1))
        label_resize = labels.view(-1, 1)
        _, predicted = outputs.topk(maxk, 1, True, True)
        total += labels.size(0)
        correct += torch.eq(predicted, label_resize).cpu().sum().float().item()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

    print("Waiting Test!")
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            
            maxk = max((1, 1))
            label_resize = labels.view(-1, 1)
            _, predicted = outputs.topk(maxk, 1, True, True)
            total += labels.size(0)
            correct += torch.eq(predicted, label_resize).cpu().sum().float().item()
            
            y_predict.append(predicted)
            y_true.append(labels)
        print('Test Accuracy:%.3f%%' % (100 * correct / total))
        acc = 100. * correct / total
    filename = 'CBAM(freezemore)_model_epoch_' + str(epoch) + '.pkl'
    torch.save(net.state_dict(), os.path.join(PATH, filename))
print("Training Finished, TotalEPOCH = %d" % EPOCH)
writer.close()