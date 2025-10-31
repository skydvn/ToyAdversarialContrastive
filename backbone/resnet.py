import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np


# ResNet encoder for CIFAR-10 (smaller version)
class ResNetEncoder(nn.Module):
    def __init__(self, base_model='resnet18', out_dim=128):
        super().__init__()
        # Load pretrained ResNet
        self.encoder = torchvision.models.resnet18(pretrained=False)
        # Modify first conv for CIFAR-10 (32x32 images)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()

        # Get the dimension of the encoder output
        self.n_features = self.encoder.fc.in_features

        # Remove the final classification layer
        self.encoder.fc = nn.Identity()

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

        # Adversarial Projection head
        self.adv_projection = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z1 = self.projection(h)
        z2 = self.adv_projection(h)
        return h, z1, z2
