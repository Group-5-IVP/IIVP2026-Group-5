import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from Dataset import _build_train_transform, _build_eval_transform, DigitDataset, train_csv_path, train_img_folder_path
import torch
import torch.nn as nn

def k_fold_validation(df: pd.DataFrame, model_class: type[nn.Module], k_folds=4, epochs=5, batch_size=128, lr=0.01, device=None):
    if device is None:
        device = torch.device("cpu")

    y = df['Category'].tolist()
    k_fold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(np.zeros(len(df)), y)):
        print(f"Fold {fold+1}/{k_folds}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_subset = DigitDataset(
            train_csv_path,
            train_img_folder_path,
            transform=_build_train_transform(),
            df=train_df
        )
        val_subset = DigitDataset(
            train_csv_path,
            train_img_folder_path,
            transform=_build_eval_transform(),
            df=val_df
        )

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(images), targets)
                loss.backward()
                optimizer.step()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds = model(images).argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        acc = correct / total * 100
        fold_accuracies.append(acc)
        print(f"Fold {fold+1} Accuracy: {acc:.2f}%")
    mean_acc = sum(fold_accuracies)/len(fold_accuracies)
    standard_dev = np.array(fold_accuracies).std()
    print(f"Mean acc: {mean_acc} and STD {standard_dev}")
    return fold_accuracies