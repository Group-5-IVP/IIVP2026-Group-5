from torchvision.transforms import v2
import torch
import torch.nn.functional as F
from stats import mean, std
import random

rng = random.Random(42)
torch.manual_seed(42)

def build_train_transform_mild(stroke=False):
    return v2.Compose([v2.Grayscale(num_output_channels=1),
                       v2.Resize((32, 32)),
                       v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
                       v2.ElasticTransform(alpha=4.0, sigma=4.0),
                       v2.ColorJitter(brightness=0.2, contrast=0.2),
                       v2.ToImage(),
                       v2.ToDtype(torch.float32, scale=True),
                       RandomStrokeTransform(prob=0.1 if stroke else 0.0, k=3),
                       v2.Normalize(mean=[mean], std=[std]),
                       v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
                       ])

def build_train_transform_medium(stroke=False):
    return v2.Compose([v2.Grayscale(num_output_channels=1),
                       v2.Resize((32, 32)),
                       v2.RandomAffine(degrees=17.5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
                       v2.ElasticTransform(alpha=4.0, sigma=4.0),
                       v2.ColorJitter(brightness=0.2, contrast=0.2),
                       v2.ToImage(),
                       v2.ToDtype(torch.float32, scale=True),
                       RandomStrokeTransform(prob=0.25 if stroke else 0.0, k=3),
                       v2.Normalize(mean=[mean], std=[std]),
                       v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
                       ])

def build_train_transform_aggressive(stroke=False):
    return v2.Compose([v2.Grayscale(num_output_channels=1),
                       v2.Resize((32, 32)),
                       v2.RandomAffine(degrees=25, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5),
                       v2.ElasticTransform(alpha=4.0, sigma=4.0),
                       v2.ColorJitter(brightness=0.2, contrast=0.2),
                       v2.ToImage(),
                       v2.ToDtype(torch.float32, scale=True),
                       RandomStrokeTransform(prob=0.4 if stroke else 0.0, k=3),
                       v2.Normalize(mean=[mean], std=[std]),
                       v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
                       ])

def _build_train_transform(augm_level):
    if augm_level not in ('baseline','medium', 'aggressive'):
        raise IOError('type not correct. select from baseline,medium, aggressive')
    if augm_level == 'aggressive':
        return v2.Compose([v2.Grayscale(num_output_channels=1),
            v2.Resize((32, 32)),
            v2.RandomAffine(degrees=25, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5),
            v2.ElasticTransform(alpha=4.0, sigma=4.0),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            #v2.Lambda(lambda x : random_stroke(x, prob=0.5, k=3)),
            v2.Normalize(mean=[mean], std=[std]),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
           ])
    elif augm_level == 'medium':
        return v2.Compose([v2.Grayscale(num_output_channels=1),
           v2.Resize((32, 32)),
           v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
           v2.ElasticTransform(alpha=4.0, sigma=4.0),
           v2.ColorJitter(brightness=0.2, contrast=0.2),
           v2.ToImage(),
           v2.ToDtype(torch.float32, scale=True),
           # v2.Lambda(lambda x : random_stroke(x, prob=0.5, k=3)),
           v2.Normalize(mean=[mean], std=[std]),
           v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
           ])
    #Baseline
    else:
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


class RandomStrokeTransform:
    """Picklable transform wrapper for random_stroke."""
    def __init__(self, prob=0.5, k=3):
        self.prob = prob
        self.k = k

    def __call__(self, x):
        return random_stroke(x, prob=self.prob, k=self.k)


augments = {
    'mild_stroke' : build_train_transform_mild(stroke=True),
    'medium_stroke' : build_train_transform_medium(stroke=True),
    'aggressive_stroke' : build_train_transform_aggressive(stroke=True),
    'mild_no_stroke' : build_train_transform_mild(stroke=False),
    'medium_no_stroke' : build_train_transform_medium(stroke=False),
    'aggressive_no_stroke' : build_train_transform_aggressive(stroke=False)
}

def get_transform_steps_report(degree=10, translate = (0.1, 0.1), scale=1.0, stroke_prob=1.0, stroke_k=3, shear=5, elastic_alpha=0.4,
                               elastic_sigma=4.0, brightness=1.2, contrast=1.2, erasing_p=0.25, erasing_scale=0.08):
    return [
        ("grayscale", v2.Grayscale(num_output_channels=1)),
        ("resize", v2.Resize((32, 32))),
        ("affine", v2.RandomAffine(degrees=(degree,degree), translate=translate, scale=(scale,scale), shear=(shear,shear))),
        ("elastic", v2.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma)),
        ("jitter", v2.ColorJitter(brightness=(brightness,brightness), contrast=(contrast,contrast))),
        ("to_tensor", v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])),
        ("stroke", RandomStrokeTransform(prob=stroke_prob, k=stroke_k)),
        ("normalize", v2.Normalize(mean=[mean], std=[std])),
        ("erasing", v2.RandomErasing(p=erasing_p, scale=(erasing_scale,erasing_scale))),
    ]





