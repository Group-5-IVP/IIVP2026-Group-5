import torch.nn as nn
import torch
from torchvision.models import resnet18
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import train_csv_path, DigitDataset, train_img_folder_path
from augmentation import _build_train_transform
from predict import _predict_probs

nn_models_dir = f"{Path(__file__).parent.parent}/outputs/nn_models"


def save_model(model, file_name):
    torch.save(model, f"{nn_models_dir}/{file_name}.pth")

def load_model(path, weights_only=True):
    return torch.load(path, weights_only=weights_only)

def train_model(model_fn, epochs=10, df=None, lr = 0.01, batch_size=128, device=None):
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
    model = model_fn().to(device)
    if device.type == 'cuda':
        try:
            model = torch.compile(model)
        except Exception:
            pass  # torch.compile not supported (e.g., Windows)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    model.train()
    for epoch in range(epochs):
        for images, targets in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return model

def evaluate_model(model, val_df:pd.DataFrame, batch_size, use_tta, device=None):
    if device is None:
        device = torch.device(
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )
    avg_probs = _predict_probs(
        model, val_df, train_img_folder_path, train_csv_path,
        device=device, batch_size=batch_size, use_tta=use_tta,
    )
    pred = avg_probs.argmax(dim=1)
    targets = torch.tensor(val_df['Category'].tolist())
    return (pred == targets).float().mean().item() * 100

def train_and_evaluate(model_fn, train_df:pd.DataFrame, val_df, epochs, batch_size, lr, use_tta, device=None):
    if device is None:
        device = torch.device(
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )
    model = train_model(
        model_fn, df=train_df, epochs=epochs, lr=lr,
        batch_size=batch_size, device=device,
    )
    return evaluate_model(model=model, val_df=val_df, batch_size=batch_size, use_tta=use_tta, device=device)

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

def build_resnet18(num_classes=10):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False, se_reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

        self.se = SEBlock(out_ch, se_reduction) if use_se else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, num_classes=10, filters=(32, 64, 128), dropout=0.4,use_se=False, se_reduction=16, blocks_per_stage=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, filters[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
        )

        stages = []
        in_ch = filters[0]
        for out_ch in filters:
            for _ in range(blocks_per_stage):
                stages.append(ResBlock(in_ch, out_ch, use_se=use_se, se_reduction=se_reduction))
                in_ch = out_ch
            stages.append(nn.MaxPool2d(2))
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(filters[-1], num_classes),
        )

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))