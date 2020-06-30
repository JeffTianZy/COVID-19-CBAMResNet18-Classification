import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import xgboost as xgb
import winsound
import time
import pickle
import h5py
import numpy as np
import scipy.io

from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose, Resize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, ToTensor
from load_data import XRayDataSet
from CBAMresnet import resnet18
from iteration import Iterator
from sklearn.model_selection import train_test_split as TTS
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
duration = 2000
freq = 440

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = './COVID19_dataset'
metadata = './metadata.csv'
PATH = './'

model = resnet18(pretrained=True).to(device)
model.load_state_dict(torch.load('./CBAM_model_pkls/CBAM(freezemore)_model_epoch_48.pkl'))

train_aug = Compose([Resize((224, 224)), 
					 ToTensor()
					 ])
val_aug   = Compose([Resize((224, 224)), 
					 ToTensor()
					 ])

BS = 1
Train_features = []
Train_classes = []
Test_features = []
Test_classes = []
trainset = XRayDataSet(os.path.join(data_path, 'train'), metadata, transforms = train_aug)
testset = XRayDataSet(os.path.join(data_path, 'test'), metadata, transforms = val_aug)
trainloader = DataLoader(trainset, batch_size = BS, shuffle = True, num_workers = 0)
testloader = DataLoader(testset, batch_size = BS, shuffle = True, num_workers = 0)

correct = 0
total = 0
y_predict = []
y_true = []

for data in testloader:
    model.eval()
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    
    maxk = max((1, 1))
    label_resize = labels.view(-1, 1)
    _, predicted = outputs.topk(maxk, 1, True, True)
    total += labels.size(0)
    correct += torch.eq(predicted, label_resize).cpu().sum().float().item()
    
    y_predict.append(int(predicted))
    y_true.append(int(labels))

print(y_true)
y_predict = np.array(y_predict)
y_true = np.array(y_true)
print(classification_report(y_true, y_predict))
acc = 100. * correct / total