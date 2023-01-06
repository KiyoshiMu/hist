"""Take bags of features, sample a subset of them, use them to train a 
k-means model, and then use the model to cluster the rest of the bags.
Show samples from each cluster.
Save the k-means model
"""

from itertools import chain
import random
import joblib

random.seed(42)
from sklearn.cluster import KMeans
from pathlib import Path
from numpy import ndarray
import numpy as np
from PIL import Image, ImageOps


def kmeans_filter(train_feat_pool, patch_ps, dst: Path):
    case0_feat = train_feat_pool[0].astype(float)
    case0_ps = patch_ps[0]
    np.save(dst / "case0_feat.npy", case0_feat)
    np.save(dst / "case0_ps.npy", case0_ps)
    # case0_feat = np.load("Data/kmeans_test/case0_feat.npy")
    # case0_ps = np.load("Data/kmeans_test/case0_ps.npy")
    
    k_means = mk_kmean(train_feat_pool[1:])
    # save the kmeans model
    k_means_path = dst / "kmeans.joblib"
    joblib.dump(k_means, k_means_path)
    # show samples from each cluster

    case_pred = k_means.predict(case0_feat)
    g0 = [case0_ps[i] for i in range(len(case0_ps)) if case_pred[i] == 0]
    g1 = [case0_ps[i] for i in range(len(case0_ps)) if case_pred[i] == 1]
    _show_kmean_group(g0).save(dst / "kmean_g0.jpg")
    _show_kmean_group(g1).save(dst / "kmean_g1.jpg")
    return k_means_path

def mk_kmean(train_feat_pool: ndarray):
    kmeans = KMeans(n_clusters=2, random_state=0)
    feature_samples = list(
        chain.from_iterable(
            (_sample_features(feat_pool, 96) for feat_pool in train_feat_pool)
        )
    )
    print(len(train_feat_pool), len(feature_samples))
    kmeans = kmeans.fit(feature_samples)
    return kmeans


def _sample_features(features, n: int):
    if n >= len(features):
        return features
    indices = random.sample(range(len(features)), n)
    return (features[i] for i in indices)


def _show_kmean_group(g0_names):
    base = Path("Data/histo_tiles/19_0563_TR")
    sample_g0_ps = [base / n.split("/")[-1] for n in g0_names]
    return _sample_patches(sample_g0_ps)


def _sample_patches(patch_ps):
    NUM = 8
    PATCH_SIZE = 64
    PADDING = 8
    WIDTH = PATCH_SIZE * NUM + PADDING * (NUM + 1)
    HEIGHT = WIDTH

    canvas = Image.new("RGB", (WIDTH, HEIGHT), color=(255, 255, 255))
    selected_patch_ps = random.sample(patch_ps, NUM * NUM)

    for idx, patch_p in enumerate(selected_patch_ps):
        cell_img = Image.open(patch_p)
        cell_img = ImageOps.fit(
            cell_img, (PATCH_SIZE, PATCH_SIZE), method=Image.ANTIALIAS
        )
        col_loc = idx % NUM
        row_loc = idx // NUM
        canvas.paste(
            cell_img,
            (
                PADDING + col_loc * (PATCH_SIZE + PADDING),
                PADDING + row_loc * (PATCH_SIZE + PADDING),
            ),
        )

    return canvas


if __name__ == "__main__":
    features: np.ndarray = np.load("Data/all_vit_feats.npy", allow_pickle=True)
    patch_ps = np.load("Data/all_vit_patch_ps.npy", allow_pickle=True)
    kmeans_filter(features,patch_ps, Path("Data/kmeans_test"), )
