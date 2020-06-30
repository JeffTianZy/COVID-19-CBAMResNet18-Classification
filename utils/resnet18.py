import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ResNet(nn.Module):
    def __init__(self, num_classes = 3):
        super(ResNet, self).__init__()
        net = models.resnet18(pretrained = False)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
                nn.Linear(1000, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x_out = self.features(x)
        x = x_out.view(x_out.size(0), -1)
        x = self.classifier(x)
        return x