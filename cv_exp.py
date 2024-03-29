from collections import Counter
import json
from pathlib import Path
from typing import Optional
from joblib import load
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random
import torch

import hist.dataset as data
from hist.embed import ds_avg, ds_embed, ds_project
from hist.io_utils import pkl_dump, simplify_label
from hist.model import BagPooling, encoder_training
from hist.plot import measure_slide_vectors
from k_means_filter import kmeans_filter

random.seed(42)
THRESHOLD = 64


def data_loading(
    feature_p="Data/all_vit_feats.npy",
    kmeans_p="Data/vit_kmeans.joblib",
    threshold=THRESHOLD,
    kmeans_target: Optional[int] = 0,
):
    features = np.load(feature_p, allow_pickle=True)
    with open("Data/y_true.json", "r") as f:
        labels_raw = json.load(f)
        y_all = [i["label"] for i in labels_raw]
        y_simple = [simplify_label(ls) for ls in y_all]
    with open("Data/slide_names.json", "r") as f:
        slide_names = json.load(f)
    _keep_indices = [
        index for index, feature in enumerate(features) if len(feature) >= threshold
    ]
    print("total WSI count: ", len(_keep_indices))
    keep_indices = [index for index in _keep_indices if y_simple[index] != "OTHER"]
    _features = [features[index] for index in keep_indices]
    _labels = [y_simple[index] for index in keep_indices]
    _slide_names = [slide_names[index] for index in keep_indices]
    print("filtered WSI count: ", len(_features))

    if kmeans_target is not None:
        kmeans = load(kmeans_p)
        _features = [
            _padding_feature(
                _feat[kmeans.predict(_feat.astype(float)) == kmeans_target], threshold
            )
            for _feat in _features
        ]

    return _features, _labels, _slide_names


def _padding_feature(feature, threshold):
    dim = feature.shape[1]
    if len(feature) < threshold:
        feature = np.concatenate(
            [feature, np.zeros((threshold - len(feature), dim))], axis=0
        )
    return feature


class Planner:
    def __init__(
        self,
        base_dir: Path,
        feature_p: str,
        kmeans_p: str,
        threshold=THRESHOLD,
        kmeans_target: Optional[int] = None,
    ) -> None:
        """

        Args:
            threshold (_type_, optional): _description_. Defaults to THRESHOLD.
            test_normal (bool, optional): _description_. Defaults to True.
            kmeans_target (Optional[int], optional): 0 is ROI, 1 is non-ROI. Defaults to None.
        """
        self.base = base_dir
        self.base.mkdir(parents=True, exist_ok=True)
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        _features, _labels, _slide_names = data_loading(
            feature_p=feature_p,
            kmeans_p=kmeans_p,
            threshold=threshold,
            kmeans_target=kmeans_target,
        )
        self.features = _features
        self.slide_names = _slide_names
        self.labels = _labels
        print(Counter(self.labels))
        print(f"Slide count: {len(self.slide_names)}")

    def run(self, n=5):
        self.make_split(n)
        self.run_modeling(n)

    def make_split(self, n=5):
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

    def run_modeling(self, n=5):
        for trial in range(n):
            dst_dir = self.base / f"trial{trial}"
            split_json_p = dst_dir / f"split{trial}.json"
            model_path = self.train_model(split_json_p, dst_dir)
            self.make_embeddings(
                model_path,
                split_json_p,
                dst_dir,
                str(trial),
            )

    def train_model(self, split_json_p: Path, dst):
        train_x, train_y, train_slide_names = self._load_xy_with_json(split_json_p)

        in_dim = train_x[0].shape[1]
        print(f"in_dim: {in_dim}")
        train_set = data.CustomImageDataset(
            train_x, train_y, train_slide_names, bag_size=THRESHOLD
        )
        model_path = encoder_training(
            train_set,
            in_dim=in_dim,
            num_epochs=128,
            num_workers=1,
            dst_dir=dst,
        )
        return model_path

    def _load_xy_with_json(self, split_json_p: Path, target="train"):
        with open(split_json_p, "r") as f:
            cache = json.load(f)
            indices = cache[target]
        x = [self.features[i] for i in indices]
        y = [self.labels[i] for i in indices]
        slide_names = [self.slide_names[i] for i in indices]
        return x, y, slide_names

    def make_embeddings(
        self,
        model_path: str,
        split_json_p: Path,
        dst_dir: Path,
        trial: str,
    ):
        train_x, train_y, train_slide_names = self._load_xy_with_json(
            split_json_p, target="train"
        )
        val_x, val_y, val_slide_names = self._load_xy_with_json(
            split_json_p, target="val"
        )

        in_dim = train_x[0].shape[1]

        train_set = data.CustomImageDataset(
            train_x, train_y, train_slide_names, bag_size=THRESHOLD
        )
        val_set = data.CustomImageDataset(
            val_x, val_y, val_slide_names, bag_size=THRESHOLD
        )

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
    BASE_DIR = "experiments4"
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

        train_x = train_feature
        train_y = [labels[i] for i in train_indices]
        train_slide_names = [slide_names[i] for i in train_indices]

        pkl_dst = str(dst_dir / "normTrain_pool.pkl")
        pkl_dump(
            dict(embed_pool=train_x, labels=train_y, index=train_slide_names), pkl_dst
        )

        val_x = val_feature
        val_y = [labels[i] for i in val_indices]
        val_slide_names = [slide_names[i] for i in val_indices]

        pkl_dst = str(dst_dir / "normVal_pool.pkl")
        pkl_dump(dict(embed_pool=val_x, labels=val_y, index=val_slide_names), pkl_dst)

        measure_slide_vectors(
            dst_dir / "normTrain_pool.pkl",
            dst_dir / "normVal_pool.pkl",
            mark="norm",
            trial=str(trial),
            dummy_baseline=False,
            dst=dst_dir,
        )


def main(
    label_mappings,
    feature_p="Data/all_vit_feats.npy",
    dst: Path = Path("lab_vit"),
    kmeans_p="Data/kmeans_test/all_vit_feats/kmeans.joblib",
):
    dst.mkdir(exist_ok=True)
    features: np.ndarray = np.load(feature_p, allow_pickle=True)
    # drop the first one as the first one is used for showing samples in [kmeans_filter]
    features = features[1:]

    for exp, kmeans_target in label_mappings.items():
        base_dir = dst / exp
        planner = Planner(
            base_dir=base_dir,
            feature_p=feature_p,
            kmeans_p=str(kmeans_p),
            kmeans_target=kmeans_target,
        )
        planner.run()


if __name__ == "__main__":
    # if kmean_g0 is ROI, kmean_g1 is non-ROI, then set {pos: 0, neg: 1, pos_neg: None}
    # if kmean_g0 is non-ROI, kmean_g1 is ROI, then set {pos: 1, neg: 0, pos_neg: None}

    # main(
    #     {"pos": 0, "neg": 1, "pos_neg": None},
    #     feature_p="Data/all_featuresK.npy",
    #     kmeans_p="Data/kmeans_test/all_featuresK/kmeans.joblib",
    #     dst=Path("lab_denseK"),
    # )
    # main(
    #     {"pos": 0, "neg": 1, "pos_neg": None},
    #     feature_p="Data/all_vit_feats.npy",
    #     kmeans_p="Data/kmeans_test/all_vit_feats/kmeans.joblib",
    #     dst=Path("lab_vit"),
    # )
    # main(
    #     {"pos": 0, "neg": 1, "pos_neg": None},
    #     feature_p="Data/features.npy",
    #     kmeans_p="Data/kmeans_test/features/kmeans.joblib",
    #     dst=Path("lab_dense"),
    # )
    main(
        {"pos": 0, "neg": 1, "pos_neg": None},
        feature_p="Data/all_dino_feats.npy",
        kmeans_p="Data/kmeans_test/all_dino_feats/kmeans.joblib",
        dst=Path("lab_dino0"),
    )
