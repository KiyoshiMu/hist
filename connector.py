#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from pathlib import Path
from joblib import load


def main(feature_p, ps_p, kmeans_p, threshold=64):
    """
    Process feature data, predict clusters using k-means, and save the results.

    Args:
        feature_p (str): Path to the feature file.
        ps_p (str): Path to the patch file.
        kmeans_p (str): Path to the k-means model file.
        threshold (int): min Threshold for patch count in a WSI (default is 64).
    """
    features = np.load(feature_p, allow_pickle=True)
    ps = np.load(ps_p, allow_pickle=True)
    kmeans = load(kmeans_p)

    base = []
    print(f"Start processing {len(features)} WSIs")
    for idx, wsi_feats in enumerate(features):
        if len(ps[idx]) < threshold:
            continue
        wsi_p = Path(ps[idx][0])
        wsi_name = wsi_p.parent.name
        pred = kmeans.predict(wsi_feats.astype(float))
        base.append({"wsi_name": wsi_name, "kmean_preds": pred, "features": wsi_feats})
        # wsi_name is a string; 
        # kmean_preds is a list of cluster predict (0 or 1);
        # features is a list of list, [patch0_features, patch1_features ..]

    feature_p = Path(feature_p)
    output_path = feature_p.parent / f"{feature_p.stem}_clean.npy"
    
    np.save(output_path, base)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process feature, patch, and kmeans file paths."
    )
    parser.add_argument("--feature_path", type=str, help="e.g. all_vit_feats.npy")
    parser.add_argument("--ps_path", type=str, help="e.g. all_vit_patch_ps.npy")
    parser.add_argument("--kmeans_path", type=str, help="e.g. kmeans.joblib")
    args = parser.parse_args()
    main(args.feature_path, args.ps_path, args.kmeans_path)
    
    # e.g.
    # python connector.py --feature_path Data/all_vit_feats.npy --ps_path Data/all_vit_patch_ps.npy --kmeans_path Data/kmeans_test/all_vit_feats/kmeans.joblib
    # data = np.load("Data/all_vit_feats_clean.npy", allow_pickle=True)
    # print(data[0]["features"].shape)
