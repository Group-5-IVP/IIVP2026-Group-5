
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image


class HindiDigitsDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["filename"]  # adjust column name if different
        img = Image.open(img_path).convert("L")  # "L" = grayscale; use "RGB" if 3-channel
        label = int(row["label"])  # adjust column name if different

        if self.transform is not None:
            img = self.transform(img)

        return img, label


