from torchvision.transforms import v2
import torch
import torch.nn.functional as F
from stats import mean, std
import random

rng = random.Random(42)
torch.manual_seed(42)


def _build_train_transform():
    return v2.Compose([v2.Grayscale(num_output_channels=1),
        v2.Resize((32, 32)),
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        v2.ElasticTransform(alpha=4.0, sigma=4.0),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        #v2.Lambda(lambda x : random_stroke(x, prob=0.5, k=3)),
        v2.Normalize(mean=[mean], std=[std]),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
       ])

def _build_eval_transform():
    return v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((32, 32)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean], std=[std]),
    ])

def _build_tta_transform(variant):
    #variant == 0 (no augmentation) is handled in the else section
    if variant == 1:
        return v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((32, 32)),
        v2.RandomAffine(degrees=(-5,-5)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean], std=[std]),
    ])
    elif variant == 2:
        return v2.Compose([
            v2.Grayscale(num_output_channels=1),
            v2.Resize((32, 32)),
            v2.RandomAffine(degrees=(5, 5)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[mean], std=[std]),
        ])
    elif variant == 3:
        return v2.Compose([
            v2.Grayscale(num_output_channels=1),
            v2.Resize((32, 32)),
            v2.RandomAffine(degrees=0, scale=(0.95, 0.95)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[mean], std=[std]),
        ])
    elif variant == 4:
        return v2.Compose([
            v2.Grayscale(num_output_channels=1),
            v2.Resize((32, 32)),
            v2.RandomAffine(degrees=0, scale=(1.05, 1.05)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[mean], std=[std]),
        ])
    else:
        return _build_eval_transform()


def random_stroke(img, prob=0.5, k=3):
    if rng.random() > prob:
        return img
    else:
        x = img.unsqueeze(0)
        if rng.random() < 0.5:
            out = F.max_pool2d(x, k, stride=1, padding = k // 2) # dilate bright = thicker strokes
        else:
            out = -F.max_pool2d(-x, k, stride=1, padding= k // 2) # erode bright = narrower strokes
        return out.squeeze(0)





