import copy
import torchvision
from torch.utils.data import Dataset, ConcatDataset

def augment_dataset(ds: Dataset, transformations: torchvision.transforms, path) -> Dataset:
    augmented_ds = copy.copy(ds)
    augmented_ds.transform = transformations
    return ConcatDataset([ds, augmented_ds])





