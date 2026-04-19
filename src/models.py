import torch
import torch.nn as nn


# one conv block to be used in CNNs- 2 conv layers with batch norm and relu, followed by max pooling
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            #using two 3x3 kernels instead of the traditional one 5x5 allows for more non-linearity (2 ReLUs) and a larger effective receptive field with fewer parameters (VGG-style)
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            #normalize layer output
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            #downsample image (number of filters will increase to compensate)
            #useful for hierarchical feature extraction but also for optimization and memory
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, filters=(32, 64, 128), dropout=0.4):
        super().__init__()
        layers = []
        in_ch = 1 #one channel for grayscale
        for f in filters:
            layers.append(ConvBlock(in_ch, f))
            in_ch = f
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), #average pool to get one feature per filter
            nn.Flatten(), #flatten the 1x1 feature maps into a vector
            nn.Dropout(dropout), #randomly zero out some features to prevent overfitting
            nn.Linear(filters[-1], num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x