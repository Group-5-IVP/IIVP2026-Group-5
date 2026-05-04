from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import stats as stats

test_csv_path = f"{Path(__file__).parent.parent}/resources/test.csv"
train_csv_path = f"{Path(__file__).parent.parent}/resources/train.csv"
test_img_folder_path = f"{Path(__file__).parent.parent}/resources/test/test"
train_img_folder_path = f"{Path(__file__).parent.parent}/resources/train/train"



class DigitDataset(Dataset):
    def __init__(self, csv_path:str, img_path: str, transform=None, df=None, preload=True):
        if df is None:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = df
        if preload and 'Category' in df.columns:
            self.images = [Image.open(f"{img_path}/{int(r.Category)}/{r.Id}.png").convert("L").copy()
                           for r in self.df.itertuples()]
        else:
            self.images = None
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

        # Use preloaded image if available, otherwise load from disk
        if self.images is not None:
            img = self.images[idx].copy()  # copy to avoid transform mutating cached image
        else:
            if label != -1:
                img_path = f"{self.img_path}/{label}/{row['Id']}.png"
            else:
                img_path = f"{self.img_path}/{row['Id']}.png"
            img = Image.open(img_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


