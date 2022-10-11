from itertools import chain
import json
from pathlib import Path
from typing import Sequence
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random
import torch

import hist.dataset  as data
from hist.model import encoder_training
random.seed(42)

def sample_features(features: Sequence, n: int):
    indices = random.sample(range(len(features)), n)
    return (features[i] for i in indices)

def mk_pca(train_feat_pool):
    pca = PCA(n_components=2)
    feature_samples = list(chain.from_iterable((sample_features(feat_pool, 4) for feat_pool in train_feat_pool)))
    pca.fit(feature_samples)
    return pca

BASE_DIR = "experiments0"

class Planner:
    def __init__(self, threshold=96) -> None:
        self.base = Path(BASE_DIR)
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        features = np.load("Data/features.npy")
        with open("Data/y_true.json", 'r') as f:
            labels_raw = json.load(f)
            labels = [item["single_label"] for item in labels_raw]
        with open("Data/slide_names.json", 'r') as f:
            slide_names = json.load(f)
            
        _features = []
        _labels = []
        _slide_names = []
        for index, feature in enumerate(features):
            if len(feature) >= threshold:
                _features.append(feature)
                _labels.append(labels[index])
                _slide_names.append(slide_names[index])
        self.features = _features
        self.slide_names = _slide_names
        self.labels = _labels
    
    def run(self, n=5):
        sss = StratifiedShuffleSplit(n_splits=n, test_size=0.5, random_state=42)
        x = list(range(len(self.slide_names)))
        y = x
        for trial, (train_index, test_index) in enumerate(sss.split(x, y)):
            marker = str(trial)
            dst_dir = self.base / f"trial{marker}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            split_json_p = dst_dir / f"split{marker}.json"
            with open(split_json_p, "w") as f:
                json.dump(
                    dict(
                        train=train_index.tolist(),
                        val=test_index.tolist(),
                    ),
                    f,
                )

        for trial in range(n):
            dst_dir = self.base / f"trial{trial}"
            split_json_p = dst_dir / f"split{trial}.json"
            model_path, pca = self.train_model(split_json_p, dst_dir)
            make_embeddings(
                model_path,
                split_json_p,
                dataset=self.dataset,
                dst_dir=dst_dir,
                trial=str(trial),
            )
        
    def train_model(self, split_json_p: Path, dst):
        in_dim = 256
        with open(split_json_p, "r") as f:
            cache = json.load(f)
            train_indices = cache["train"]
        train_feature = [self.features[i] for i in train_indices]
        pca = mk_pca(train_feature)
        train_x = [pca.transform(feat) for feat in train_feature]
        train_y = [self.labels[i] for i in train_indices]
        train_slide_names = [self.slide_names[i] for i in train_indices]
        train_set = data.CustomImageDataset(train_x, train_y, train_slide_names)
        model_path = encoder_training(
            train_set,
            in_dim=in_dim,
            num_epochs=250,
            num_workers=1,
            dst_dir=dst,
        )
        return model_path, pca
        