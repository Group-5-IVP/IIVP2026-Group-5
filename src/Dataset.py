from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from PIL import Image
from torchvision.models.quantization import resnet50
from tqdm import tqdm
from torch.utils.data import random_split



test_csv_path = f"{Path(__file__).parent.parent}/resources/test.csv"
train_csv_path = f"{Path(__file__).parent.parent}/resources/train.csv"
test_img_folder_path = f"{Path(__file__).parent.parent}/resources/test/test"
trian_img_folder_path = f"{Path(__file__).parent.parent}/resources/train/train"

class DigitDataset:
    def __init__(self, csv_path:str, img_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.csv_path = csv_path
        self.img_path = img_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = -1
        if "Category" in self.df.columns:
            label = int(row["Category"])
        img_path = f"{self.img_path}/{label}/{row["Id"]}.png"
        img = Image.open(img_path).convert("L")  # "L" = grayscale

        if self.transform is not None:
            img = self.transform(img)

        return img, label #image is a tensor if transform defined

def _build_train_transform():
    return v2.Compose([v2.Grayscale(num_output_channels=1),
        v2.Resize((32, 32)),
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        v2.ElasticTransform(alpha=8.0, sigma=4.0),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
       ])

def _build_eval_transform():
    return v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((32, 32)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])

def _build_model():
    model = resnet50()

    # replace input layer to accept grayscale (1 channel) instead of RGB (3 channels)
    model.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64,kernel_size=7,stride=2, padding=3, bias=False)

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 10)
    return model




if __name__ == '__main__':
    train_ds = DigitDataset(train_csv_path, trian_img_folder_path, transform=_build_train_transform())

    model = _build_model()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_split, val_split = random_split(train_ds, [train_size, val_size])

    train_dl = DataLoader(train_split, batch_size=64, num_workers=0)

    num_epochs = 10
    for epoch in range(num_epochs):  # loop over epochs
        loop = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in loop:  # loop over batches
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model.eval()

    val_dl = DataLoader(val_split, batch_size=64, num_workers=0)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_dl:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
