from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier



def create_knn(refer_embed: np.ndarray, labels, use_all=False):
    if use_all:
        n_neighbors = len(refer_embed)
    else:
        n_neighbors = round(math.sqrt(len(refer_embed)))
    print(f"n_neighbors: {n_neighbors}")
    knn: KNeighborsClassifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",
    ).fit(
        refer_embed,
        labels,
    )
    return knn


def proba_to_dfDict(pred_probs, classes_, val_labels):

    pred_probs_argsort = np.argsort(pred_probs, axis=1)[:, ::-1]
    prob_top0 = [
        f"{classes_[indices[0]]}:{pred_probs[row_idx, indices[0]]:.2f}"
        for row_idx, indices in enumerate(pred_probs_argsort)
    ]
    prob_top1 = [
        f"{classes_[indices[1]]}:{pred_probs[row_idx, indices[1]]:.2f}"
        for row_idx, indices in enumerate(pred_probs_argsort)
    ]
    prob_top2 = [
        f"{classes_[indices[2]]}:{pred_probs[row_idx, indices[2]]:.2f}"
        for row_idx, indices in enumerate(pred_probs_argsort)
    ]
    top3_corrects = [
        any(
            e
            for e in (prob_top0[idx], prob_top1[idx], prob_top2[idx])
            if ("0.00" not in e and val_labels[idx] in e)
        )
        for idx in range(len(val_labels))
    ]
    weighted_acc = [
        cal_weighted_acc(
            val_labels[idx],
            prob_top0[idx],
            prob_top1[idx],
            prob_top2[idx],
        )
        for idx in range(len(val_labels))
    ]
    _df = {
        "label": val_labels,
        "prob_top0": prob_top0,
        "prob_top1": prob_top1,
        "prob_top2": prob_top2,
        "top3_correct": top3_corrects,
        "weighted_acc": weighted_acc,
    }
    return _df


def cal_weighted_acc(label, *preds):
    acc = 0
    for _, pred in enumerate(preds, start=1):
        confident = float(pred.split(":")[-1])
        acc += int(label in pred and "0.00" not in pred) * confident
    return acc


def dump_metric(y_true, y_pred, unique_labels, dst, to_csv=True):
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=unique_labels,
    )
    # print(precision, recall, fscore)
    if to_csv:
        metric_df = pd.DataFrame(
            dict(precision=precision, recall=recall, fscore=fscore),
            index=unique_labels,
        )

        metric_df.to_csv(dst)


def dummy_exp(refer_embed, refer_labels, test_embed, test_labels, dst):
    dummy = DummyClassifier(strategy="stratified", random_state=42).fit(
        refer_embed,
        refer_labels,
    )
    classes_ = dummy.classes_
    pred = dummy.predict(test_embed)
    dump_metric(test_labels, pred, classes_, dst)
