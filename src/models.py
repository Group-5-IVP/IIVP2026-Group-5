import torch.nn as nn
import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import train_csv_path, DigitDataset, train_img_folder_path, _build_train_transform

nn_models_dir = f"{Path(__file__).parent.parent}/outputs/nn_models"


def save_model(model, file_name):
    torch.save(model, f"{nn_models_dir}/{file_name}.pth")

def load_model(path, weights_only=True):
    return torch.load(path, weights_only=weights_only)

def train_model(model_class, epochs=10, df=None, batch_size=128, device=None):
    if df is None:
        train_df = pd.read_csv(train_csv_path)
    else:
        train_df = df

    if device is None:
        device = torch.device(
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )

    train_set = DigitDataset(csv_path=train_csv_path,
                             img_path=train_img_folder_path,
                             transform=_build_train_transform(),
                             df=train_df)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = model_class().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for images, targets in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
    return model


# one conv block to be used in CNNs- 2 conv layers with batch norm and relu, followed by max pooling
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_convs=2, use_stride=False, dilation=1):
        super().__init__()
        layers=[]
        for i in range(n_convs):
            in_ch = in_ch if i == 0 else out_ch
            d=dilation if i>0 else 1
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ]
        if use_stride:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
        else:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, filters=(32, 64, 128), dropout=0.4,n_convs=2, use_stride=False, dilation=1):
        super().__init__()
        layers = []
        #testing out blocks with different numbers of conv layers, but default is 2 convs per block
        n_list = [n_convs] * len(filters) if isinstance(n_convs, int) else n_convs
        in_ch = 1 #one channel for grayscale
        for f, n in zip(filters, n_list):
            layers.append(ConvBlock(in_ch, f, n_convs=n, use_stride=use_stride, dilation=dilation))
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

class FastCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))