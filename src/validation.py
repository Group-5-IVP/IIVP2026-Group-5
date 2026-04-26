import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from models import train_and_evaluate
from predict import _get_device

def simple_validation(df: pd.DataFrame, model_fn, use_tta=False, epochs=10, train_size=0.8, batch_size=128, lr = 0.01):
    device = _get_device()
    y = df['Category'].tolist()
    train_idx, val_idx = train_test_split(
        np.arange(len(df)), train_size=train_size, stratify=y, random_state=42,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    acc = train_and_evaluate(model_fn, train_df, val_df, epochs, batch_size, lr, use_tta, device)
    print(f"Validation accuracy = {acc}")
    return acc


def k_fold_validation(df: pd.DataFrame, model_fn, use_tta=False, k_folds=4, epochs=10, batch_size=128, lr=0.01):
    device = _get_device()

    y = df['Category'].tolist()
    k_fold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(np.zeros(len(df)), y)):
        print(f"Fold {fold+1}/{k_folds}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        acc = train_and_evaluate(model_fn, train_df, val_df, epochs, batch_size, lr, use_tta, device)
        fold_accuracies.append(acc)
        print(f"Fold {fold+1} accuracy = {acc}")
    mean_acc = sum(fold_accuracies)/len(fold_accuracies)
    standard_dev = np.array(fold_accuracies).std()
    print(f"Mean acc: {mean_acc} and STD {standard_dev}")
    return fold_accuracies