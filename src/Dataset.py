from pathlib import Path
import pandas as pd
import torch
from torchvision.transforms import v2
from PIL import Image

import datasetStats as stats

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
        img_path = f"{self.img_path}/{label}/{row['Id']}.png"
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
        v2.ToDtype(torch.float32, scale=True), # it performs scaling from [0,255] to [0, 1]
        v2.Normalize(mean=[stats.mean], std=[stats.std]), # normalize for NN for training to [-1, 1] with mean 0
        v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
       ])

def _build_eval_transform():
    return v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((32, 32)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[stats.mean], std=[stats.std]),
    ])
