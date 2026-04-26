import pandas as pd
import torch
from torch.utils.data import DataLoader
from Dataset import test_csv_path, test_img_folder_path, DigitDataset
from augmentation import _build_eval_transform, _build_tta_transform
from pathlib import Path
from utils import _get_device

csv_results_dir = Path(__file__).parent.parent/'outputs'/'csv_results'

def predict_test(model, test_df=None, use_tta=False, batch_size=128):
    #device selection and model loading
    device = _get_device()
    model = model.to(device)
    if test_df is None:
        test_df = pd.read_csv(test_csv_path)
    #dataset selection
    avg_probs = _predict_probs(model,
                               test_df,
                               test_img_folder_path,
                               test_csv_path, device,
                               batch_size=batch_size,
                               use_tta=use_tta)
    ids = test_df['Id'].tolist()
    predictions = avg_probs.argmax(dim=1).tolist()
    df = pd.DataFrame({
        'Id' : ids,
        'Category' : predictions
    })
    df_probs = pd.DataFrame({
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
    return df, df_probs

def _predict_probs(model, df, img_path, csv_path, device,
                   batch_size=128, use_tta=False):
    """Returns averaged softmax probabilities tensor of shape [N, num_classes]."""

    transforms = (
        [_build_tta_transform(i) for i in range(5)] if use_tta
        else [_build_eval_transform()]
    )
    print(f"TTA enabled: {use_tta}, using {len(transforms)} transforms")  # ADD THIS

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