
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