import torch
from pathlib import Path
import pandas as pd

csv_results_dir = Path(__file__).parent.parent/'outputs'/'csv_results'

def save_pred_to_csv(df: pd.DataFrame, csv_file_name: str):
    if not isinstance(csv_file_name, str):
        raise TypeError(
            f"csv_file_name must be a string, got {type(csv_file_name).__name__}"
        )
    if not csv_file_name.endswith(".csv"):
        csv_file_name += ".csv"

    csv_results_dir.mkdir(parents=True, exist_ok=True)
    file_path = csv_results_dir / csv_file_name
    if file_path.exists():
        stem = file_path.stem  # filename without .csv
        suffix = file_path.suffix  # ".csv"
        counter = 1
        while file_path.exists():
            file_path = csv_results_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    df.to_csv(file_path, index=False)
    return file_path

def _get_device():
    return torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )