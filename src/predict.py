import pandas as pd
import torch
from torch.utils.data import DataLoader
from Dataset import test_csv_path, test_img_folder_path, DigitDataset, _build_eval_transform
from pathlib import Path

csv_results_dir = f"{Path(__file__).parent.parent}/outputs/csv_results"

def predict_test(model, file_name_csv, test_dataset=None, device=None, batch_size=128):
    if device is None:
        device = torch.device(
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )

    model = model.to(device)
    model.eval()

    if test_dataset:
        test_df = test_dataset
    else:
        test_df = pd.read_csv(test_csv_path)

    dataset_test = DigitDataset(test_csv_path, test_img_folder_path, transform=_build_eval_transform(),df=test_df)
    loader = DataLoader(dataset_test, batch_size=batch_size)
    predictions = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            predictions.extend(preds)
    ids = dataset_test.df['Id'].tolist()

    df = pd.DataFrame({
        'Id': ids,
        'Category': predictions
    })
    df.to_csv(f"{csv_results_dir}/{file_name_csv}", index=False)
    print(f"Saved {len(df)} predictions to {file_name_csv}")
    return df