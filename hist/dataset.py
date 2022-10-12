from __future__ import annotations

from pathlib import Path
import random
from typing import NamedTuple
from typing import Sequence
import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset

from hist.io_utils import simplify_label

CELL_FEATURE_SIZE = 256
MaskMap = dict[str, torch.Tensor]


class CellInstance(NamedTuple):
    name: str
    label: str
    feature: list[float]


LABEL_DIR = Path("D:/code/Docs")
FEAT_DIR = Path("D:/DATA/feats")


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: CustomImageDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.labels = dataset.labels[indices]
        self.targets = dataset.targets[indices]
        self.slide_names = dataset.slide_names[indices]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class CustomImageDataset(Dataset):
    def __init__(
        self,
        features,
        labels,
        slide_names,
        bag_size=96,
        
    ):
        self.features = [torch.as_tensor(feature, dtype=torch.float32) for feature in features]
        self.slide_names = slide_names
        self.bag_size = bag_size
        self.labels = labels
        self.le = LabelEncoder()
        self.targets = self.le.fit_transform(self.labels)
        
        self.mask_tensor = torch.zeros(CELL_FEATURE_SIZE, dtype=torch.float32)
        self._mask_map = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        label = self.targets[idx]
        feature_pool = self._get_features_by_idx(idx)
        feature_bag = torch.index_select(
            feature_pool,
            0,
            torch.as_tensor(random.sample(range(len(feature_pool)), self.bag_size)),
        )
        # print(feature_bag.shape)
        return feature_bag, label

    def _get_features_by_idx(self, idx: int):
        tensor = self.features[idx]
        if self._mask_map is None:
            return tensor
        else:
            copy = tensor.clone()
            mask = self._mask_map[self.slide_names[idx]]
            copy[mask] = self.mask_tensor
            return copy
