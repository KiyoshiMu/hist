from itertools import chain
import json
from pathlib import Path
from typing import Sequence
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random
import torch
import pickle
from sklearn.preprocessing import StandardScaler

import hist.dataset as data
from hist.embed import ds_avg, ds_embed, ds_project
from hist.io_utils import pkl_dump, simplify_label
from hist.model import BagPooling, encoder_training
from hist.plot import measure_slide_vectors

random.seed(42)
PCA_N = 64
THRESHOLD = 96
def sample_features(features: Sequence, n: int):
    indices = random.sample(range(len(features)), n)
    return (features[i] for i in indices)


def mk_scaler(train_feat_pool, pca):
    scaler = StandardScaler()
    feature_samples = list(
        chain.from_iterable(
            (sample_features(feat_pool, 96) for feat_pool in train_feat_pool)
        )
    )
    scaler.fit(pca.transform(feature_samples))
    return scaler

def norm_prop(wsi_feats, scaler, pca):
    return scaler.transform(pca.transform(wsi_feats)).mean(axis=0)

def norm_pipe(wsi_feats):
    pca = mk_pca(wsi_feats)
    scaler = mk_scaler(wsi_feats, pca)
    features_normed = [norm_prop(feats, scaler, pca) for feats in wsi_feats]
    return features_normed, scaler, pca

def mk_pca(train_feat_pool):
    pca = PCA(n_components=PCA_N)
    feature_samples = list(
        chain.from_iterable(
            (sample_features(feat_pool, THRESHOLD) for feat_pool in train_feat_pool)
        )
    )
    pca.fit(feature_samples)
    explain_sum = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance ratio: {explain_sum}")
    return pca


BASE_DIR = "experiments1"


class Planner:
    def __init__(self, threshold=THRESHOLD) -> None:
        self.base = Path(BASE_DIR)
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        features = np.load("Data/features.npy", allow_pickle=True)
        with open("Data/y_true.json", "r") as f:
            labels_raw = json.load(f)
            y_all = [i["label"] for i in labels_raw]
            y_simple = [simplify_label(ls) for ls in y_all]
        with open("Data/slide_names.json", "r") as f:
            slide_names = json.load(f)

        _features = []
        _labels = []
        _slide_names = []
        for index, feature in enumerate(features):
            if len(feature) >= threshold and y_simple[index] != "OTHER":
                _features.append(feature)
                _labels.append(y_simple[index])
                _slide_names.append(slide_names[index])
        self.features = _features
        self.slide_names = _slide_names
        self.labels = _labels

    def run(self, n=5):
        sss = StratifiedShuffleSplit(n_splits=n, test_size=0.5, random_state=42)
        x = list(range(len(self.slide_names)))
        for trial, (train_index, test_index) in enumerate(sss.split(x, self.labels)):
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
            self.make_embeddings(
                model_path,
                split_json_p,
                dst_dir,
                str(trial),
                pca,
            )

    def train_model(self, split_json_p: Path, dst):
        with open(split_json_p, "r") as f:
            cache = json.load(f)
            train_indices = cache["train"]
        train_feature = [self.features[i] for i in train_indices]
        pca = mk_pca(train_feature)
        # save pca
        with open(dst / "pca.pkl", "wb") as f:
            pickle.dump(pca, f)
            
        in_dim = PCA_N
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

    def make_embeddings(
        self,
        model_path: str,
        split_json_p: Path,
        dst_dir: Path,
        trial: str,
        pca,
    ):
        with open(split_json_p, "r") as f:
            cache = json.load(f)
            train_indices = cache["train"]
            val_indices = cache["val"]
        train_feature = [self.features[i] for i in train_indices]
        val_feature = [self.features[i] for i in val_indices]
        in_dim = PCA_N

        train_x = [pca.transform(feat) for feat in train_feature]
        train_y = [self.labels[i] for i in train_indices]
        train_slide_names = [self.slide_names[i] for i in train_indices]
        train_set = data.CustomImageDataset(train_x, train_y, train_slide_names)

        val_x = [pca.transform(feat) for feat in val_feature]
        val_y = [self.labels[i] for i in val_indices]
        val_slide_names = [self.slide_names[i] for i in val_indices]
        val_set = data.CustomImageDataset(val_x, val_y, val_slide_names)

        model = BagPooling.from_checkpoint(model_path, in_dim=in_dim)
        embed_func = lambda ds: ds_embed(ds, model)
        train_pkl_dst, _ = ds_project(
            train_set,
            embed_func,
            dst_dir,
            name_mark=f"train{trial}",
        )
        val_pkl_dst, _ = ds_project(
            val_set,
            embed_func,
            dst_dir,
            name_mark=f"val{trial}",
        )
        measure_slide_vectors(
            train_pkl_dst,
            val_pkl_dst,
            mark="pool",
            trial=trial,
            dummy_baseline=True,
            dst=dst_dir,
        )

        avg_func = lambda ds: ds_avg(ds)
        train_avg_pkl_dst, _ = ds_project(
            train_set,
            avg_func,
            dst_dir,
            name_mark=f"train_avg{trial}",
        )
        val_avg_pkl_dst, _ = ds_project(
            val_set,
            avg_func,
            dst_dir,
            name_mark=f"val_avg{trial}",
        )
        measure_slide_vectors(
            train_avg_pkl_dst,
            val_avg_pkl_dst,
            mark="avg",
            trial=trial,
            dst=dst_dir,
            dummy_baseline=False,
        )

def norm_exp(features, labels, slide_names):
    BASE_DIR = "experiments1"
    base = Path(BASE_DIR)
    for trial in range(5):
        dst_dir = base / f"trial{trial}"
        split_json_p = dst_dir / f"split{trial}.json"
        with open(split_json_p, "r") as f:
            cache = json.load(f)
            train_indices = cache["train"]
            val_indices = cache["val"]
        train_feature = [features[i] for i in train_indices]
        val_feature = [features[i] for i in val_indices]
        
        train_x, scaler, pca = norm_pipe(train_feature)
        train_y = [labels[i] for i in train_indices]
        train_slide_names = [slide_names[i] for i in train_indices]

        pkl_dst = str(dst_dir / "normTrain_pool.pkl")
        pkl_dump(dict(embed_pool=train_x, labels=train_y, index=train_slide_names), pkl_dst)
        
        val_x = [norm_prop(feats, scaler, pca) for feats in val_feature]
        val_y = [labels[i] for i in val_indices]
        val_slide_names = [slide_names[i] for i in val_indices]

        pkl_dst = str(dst_dir / "normVal_pool.pkl")
        pkl_dump(dict(embed_pool=val_x, labels=val_y, index=val_slide_names), pkl_dst)
        
        measure_slide_vectors(
                dst_dir / "normTrain_pool.pkl",
                dst_dir / "normVal_pool.pkl",
                mark="norm",
                trial=trial,
                dummy_baseline=False,
                dst=dst_dir,
            )
        
if __name__ == "__main__":
    planner = Planner()
    planner.run()