import pandas as pd
import torch
from torch.utils.data import DataLoader
from Dataset import test_csv_path, test_img_folder_path, DigitDataset, _build_eval_transform, _build_tta_transform
from pathlib import Path

csv_results_dir = Path(__file__).parent.parent/'outputs'/'csv_results'

def predict_test(model, test_df=None, use_tta=False, batch_size=128):
    #device selection and model loading
    device = _get_device()
    model = model.to(device)
    #dataset selection
    avg_probs = _predict_probs(model,
                               test_df,
                               test_img_folder_path,
                               test_csv_path, device,
                               batch_size=batch_size,
                               use_tta=use_tta)
    df = pd.DataFrame({
        'Id' : test_df['Id'].tolist()
        'Category' :
    })

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
    return df

def predict_test_tta(model, test_df=None, batch_size = 128):
    device = _get_device()

    model = model.to(device)
    model.eval()

    if test_df is None:
        test_df = pd.read_csv(test_csv_path)

    loaders = []
    all_probs = []
    # range = 5 => there r 5 transforms in _build_tta_transform(variant)
    for i in range(5):
        dataset = DigitDataset(test_csv_path, test_img_folder_path, transform=_build_tta_transform(i), df=test_df)
        loader = DataLoader(dataset=dataset, batch_size=batch_size)
        loaders.append(loader)

    for loader in loaders:
        loader_probs = []
        with torch.no_grad():
            for images,_ in loader:
                images = images.to(device)
                outputs = model(images) # logits, shape: [batch, 10]
                probs = torch.softmax(input=outputs, dim=1) # probabilities, shape: [batch, 10]
                loader_probs.append(probs.cpu())
        all_probs.append(torch.cat(loader_probs, dim=0))

    #stacking and averaging
    stacked = torch.stack(all_probs, dim=0)  # shape: [5, num_samples, 10]
    avg_probs = stacked.mean(dim=0)  # shape: [num_samples, 10]
    predictions = avg_probs.argmax(dim=1).tolist()
    ids = test_df['Id'].tolist()
    df_full = pd.DataFrame({
        'Id': ids,
        'Category': predictions,
        'prob_0': avg_probs[:, 0].tolist(),
        'prob_1': avg_probs[:, 1].tolist(),
        'prob_2': avg_probs[:, 2].tolist(),
        'prob_3': avg_probs[:, 3].tolist(),
        'prob_4': avg_probs[:, 4].tolist(),
        'prob_5': avg_probs[:, 5].tolist(),
        'prob_6': avg_probs[:, 6].tolist(),
        'prob_7': avg_probs[:, 7].tolist(),
        'prob_8': avg_probs[:, 8].tolist(),
        'prob_9': avg_probs[:, 9].tolist(),
    })
    df_pred = df_full[['Id', 'Category']]
    return df_full, df_pred

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

def _predict_probs(model, df, img_path, csv_path, device,
                   batch_size=128, use_tta=False):
    """Returns averaged softmax probabilities tensor of shape [N, num_classes]."""
    transforms = (
        [_build_tta_transform(i) for i in range(5)] if use_tta
        else [_build_eval_transform()]
    )

    model.eval()
    all_probs = []
    for transform in transforms:
        dataset = DigitDataset(csv_path, img_path, transform=transform, df=df)
        loader = DataLoader(dataset, batch_size=batch_size)

        chunks = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                probs = torch.softmax(model(images), dim=1)
                chunks.append(probs.cpu())
        all_probs.append(torch.cat(chunks, dim=0))

    return torch.stack(all_probs, dim=0).mean(dim=0)