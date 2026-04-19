from pathlib import Path
import Dataset as ds
from torchvision.transforms import v2
import torch

test_csv_path = f"{Path(__file__).parent.parent}/resources/test.csv"
train_csv_path = f"{Path(__file__).parent.parent}/resources/train.csv"
test_img_folder_path = f"{Path(__file__).parent.parent}/resources/test/test"
trian_img_folder_path = f"{Path(__file__).parent.parent}/resources/train/train"
mean=0.2423
std=0.3887

def compute_stats():
    stats_transform = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((32, 32)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    dataset = ds.DigitDataset(csv_path=train_csv_path, img_path=trian_img_folder_path, transform=stats_transform)
    total_sum, total_sq_sum, total_pixels = 0.0, 0.0, 0
    for i in range(len(dataset)):
        img, _ = dataset[i]  # img is a tensor of shape [1, 32, 32]
        total_sum += img.sum().item()
        total_sq_sum += (img ** 2).sum().item()
        total_pixels += img.numel()
    mean = total_sum / total_pixels
    std = (total_sq_sum / total_pixels - mean ** 2) ** 0.5
    print(f"mean={mean:.4f}, std={std:.4f}")
    return mean, std

if __name__ == '__main__':
    compute_stats()