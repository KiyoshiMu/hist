"""Take bags of features, sample a subset of them, use them to train a 
k-means model, and then use the model to cluster the rest of the bags.
Show samples from each cluster.
Save the k-means model
"""

from itertools import chain
import json
import random
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.metrics import pairwise_distances, silhouette_score
from pathlib import Path
from numpy import ndarray
import numpy as np
from PIL import Image, ImageOps
import umap
from sklearn.utils import _safe_indexing
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
import plotly.graph_objects as go
import plotly.io as pio
pio.kaleido.scope.mathjax = None




def pca_check(feat):
    """use pca the diversity of the features"""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=0.95)
    pca.fit(feat)
    return int(pca.n_components_)


def kmeans_filter(train_feat_pool, patch_ps, dst: Path):
    case0_feat = train_feat_pool[0].astype(float)
    case0_ps = patch_ps[0]
    patch_projection(case0_feat, case0_ps).save(dst / "case0_proj.jpg")

    np.save(dst / "case0_feat.npy", case0_feat)
    np.save(dst / "case0_ps.npy", case0_ps)
    # case0_feat = np.load("Data/kmeans_test/case0_feat.npy")
    # case0_ps = np.load("Data/kmeans_test/case0_ps.npy")

    k_means, feature_samples = mk_kmean(train_feat_pool[1:])
    
    # save the kmeans model
    k_means_path = dst / "kmeans.joblib"
    joblib.dump(k_means, k_means_path)
    n_components = pca_check(feature_samples)
    n_components_ratio = n_components / len(case0_feat[0])
    # show samples from each cluster

    case_pred = k_means.predict(case0_feat)

    scores = {
        "n_components": n_components,
        "n_components_ratio": n_components_ratio,
    }
    print(scores)
    with open(dst / "cluster_scores.json", "w") as f:
        json.dump(scores, f)

    g0 = [case0_ps[i] for i in range(len(case0_ps)) if case_pred[i] == 0]
    feat0 = [case0_feat[i] for i in range(len(case0_feat)) if case_pred[i] == 0]
    g1 = [case0_ps[i] for i in range(len(case0_ps)) if case_pred[i] == 1]
    feat1 = [case0_feat[i] for i in range(len(case0_feat)) if case_pred[i] == 1]
    patch_projection(feat0, g0).save(dst / "kmean_g0_proj.jpg")
    patch_projection(feat1, g1).save(dst / "kmean_g1_proj.jpg")

    _show_kmean_group(g0).save(dst / "kmean_g0.jpg")
    _show_kmean_group(g1).save(dst / "kmean_g1.jpg")

    return k_means_path


def mk_kmean(train_feat_pool: ndarray):
    random.seed(42)

    kmeans = KMeans(n_clusters=2, random_state=0)
    feature_samples = list(
        chain.from_iterable(
            (_sample_features(feat_pool, 96) for feat_pool in train_feat_pool)
        )
    )
    print(len(train_feat_pool), len(feature_samples))
    kmeans = kmeans.fit(feature_samples)
    return kmeans, feature_samples

def proj_dots(train_feat_pool: ndarray, dst_p: str):
    # project the embedding to 2d as dots
    random.seed(42)
    feature_samples = list(
        chain.from_iterable(
            (_sample_features(feat_pool, 16) for feat_pool in train_feat_pool[1:])
        )
    )
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(feature_samples)
    # draw the dots
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode="markers",
            marker=dict(color="black", size=1),
        )
    )
    # no axis, no background
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    fig.write_image(dst_p)
        
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
    selected_patch_ps = random.sample(patch_ps, min(NUM * NUM, len(patch_ps)))

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


def patch_projection(
    feat_pool: list, ps: list[str], base=Path("Data/histo_tiles/19_0563_TR")
):
    random.seed(42)

    SAMPLE_SIZE = 64
    PATCH_SIZE = 64
    CANVAS_SIZE = 1024
    PLOT_MAX = CANVAS_SIZE - PATCH_SIZE
    _ps = [base / n.split("/")[-1] for n in ps]

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(feat_pool)
    embedding = minmax_scale(embedding)
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), color=(255, 255, 255))
    # sample
    sample_indices = random.sample(range(len(ps)), min(SAMPLE_SIZE, len(ps)))
    sample_ps = [_ps[i] for i in sample_indices]
    sample_embedding = embedding[sample_indices]
    for idx, patch_p in enumerate(sample_ps):
        cell_img = Image.open(patch_p)
        cell_img = ImageOps.fit(
            cell_img, (PATCH_SIZE, PATCH_SIZE), method=Image.ANTIALIAS
        )
        col_loc = round(sample_embedding[idx][0] * PLOT_MAX)
        row_loc = round(sample_embedding[idx][1] * PLOT_MAX)
        canvas.paste(
            cell_img,
            (
                col_loc,
                row_loc,
            ),
        )

    return canvas


if __name__ == "__main__":
    for p in ["Data/all_vit_feats.npy", "Data/all_featuresK.npy", "Data/all_dino_feats.npy", "Data/features.npy"]:
        features: np.ndarray = np.load(p, allow_pickle=True)
        dst_p = Path("Data/kmeans_test") /f'{p.split("/")[-1].split(".")[0]}.pdf'
        proj_dots(features, str(dst_p))
    patch_ps = np.load("Data/all_vit_patch_ps.npy", allow_pickle=True)
    kmeans_filter(
        features,
        patch_ps,
        Path("Data/kmeans_test"),
    )
    