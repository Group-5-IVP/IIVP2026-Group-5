import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

from datasetStats import mean, std
from Dataset import _build_train_transform, _build_eval_transform, train_csv_path, trian_img_folder_path, DigitDataset

DATASET_MEAN = [mean]
DATASET_STD  = [std]


class SimpleCNN(nn.Module):
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

def kFoldValidation(dataset: DigitDataset, model_class: type[nn.Module], k_folds=4, epochs=5, batch_size=32, lr=0.01):
    y = dataset.df['Category'].tolist()
    k_fold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(np.zeros(len(dataset)), y)):
        train_df = dataset.df.iloc[train_idx].reset_index(drop=True)
        val_df = dataset.df.iloc[val_idx].reset_index(drop=True)

        train_subset = DigitDataset(
            dataset.csv_path,
            dataset.img_path,
            transform=_build_train_transform(),
            df=train_df
        )
        val_subset = DigitDataset(
            dataset.csv_path,
            dataset.img_path,
            transform=_build_eval_transform(),
            # ← clean transform
            df=val_df
        )

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                optimizer.zero_grad()
                loss = loss_fn(model(images), targets)
                loss.backward()
                optimizer.step()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                preds = model(images).argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        acc = correct / total * 100
        fold_accuracies.append(acc)
    mean_acc = sum(fold_accuracies)/len(fold_accuracies)
    standard_dev = np.array(fold_accuracies).std()
    print(f"Mean acc: {mean_acc} and STD {standard_dev}")
    return fold_accuracies

if __name__ == '__main__':
    dataset = DigitDataset(
        train_csv_path,
        trian_img_folder_path,
        transform=_build_train_transform()
    )

    results = kFoldValidation(dataset, SimpleCNN, k_folds=5, epochs=5)