"""Measure search Top10 query accuracy"""

from collections import defaultdict
import math
import numpy as np
import pandas as pd
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from pathlib import Path
from itertools import chain

from hist.io_utils import pkl_load, simplify_label
from hist.plot import box_plot

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

def cal_micro_f1(
    query,
    reference,
    query_labels,
    reference_labels,
):
    knn = create_knn(reference, reference_labels)
    y_pred = knn.predict(query)
    f1_micro = f1_score(
        query_labels,
        y_pred,
        average="weighted",
    )
    return f1_micro

def cal_search_quality(
    query,
    reference,
    query_labels,
    reference_labels,
    k=10
):
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    accuracy_calculator = AccuracyCalculator(
        include=("mean_average_precision",),
        exclude=(),
        avg_of_avgs=False,
        return_per_class=False,
        k=k,
        label_comparison_fn=None,
        device=None,
        knn_func=None,
        kmeans_func=None,
    )
    ret = accuracy_calculator.get_accuracy(
        query, reference, query_labels, reference_labels, False
    )
    return ret

MARKS = ["Attention Pooling", "Average Pooling",]

def measure(base_dir:Path):
    df_search_raw = defaultdict(list)
    df_f1_micro_raw = defaultdict(list)
    for trial in range(5):
        marks = MARKS
        dst_dir = base_dir / f"trial{trial}"
        for idx, mark in enumerate(marks) :
            if mark == "Attention Pooling":
                train_pkl_p = dst_dir / f"train{trial}_pool.pkl"
                val_pkl_p = dst_dir /  f"val{trial}_pool.pkl"
            elif mark == "Average Pooling":
                train_pkl_p =dst_dir /  f"train_avg{trial}_pool.pkl"
                val_pkl_p =dst_dir /  f"val_avg{trial}_pool.pkl"
            else:
                raise ValueError("Unknown mark")
            
            train = pkl_load(train_pkl_p)
            val = pkl_load(val_pkl_p)
            train_label = [simplify_label(l) for l in train["labels"]]
            val_label = [simplify_label(l) for l in val["labels"]]
            le = LabelEncoder()
            reference_labels = le.fit_transform(train_label)
            query_labels = le.transform(val_label)
            reference = train["embed_pool"]
            query = val["embed_pool"]
            ret = cal_search_quality(
                query,
                reference,
                query_labels,
                reference_labels,
            )
            f1_micro = cal_micro_f1(
                query,
                reference,
                query_labels,
                reference_labels,
            )

            df_search_raw[mark].append(ret["mean_average_precision"])
            df_f1_micro_raw[mark].append(f1_micro)
            
            if idx == len(marks) - 1:
                random_query = np.random.rand(*query.shape)
                random_reference = np.random.rand(*reference.shape)
                random_ret = cal_search_quality(
                    random_query,
                    random_reference,
                    query_labels,
                    reference_labels,
                )
                df_search_raw["Random"].append(random_ret["mean_average_precision"])
                f1_micro = cal_micro_f1(
                    random_query,
                    random_reference,
                    query_labels,
                    reference_labels,
                )
                df_f1_micro_raw["Random"].append(f1_micro)
                
    df_search = pd.DataFrame(df_search_raw)
    df_search = pd.melt(df_search, var_name="Agg Method", value_name="mAP@10")
    
    df_f1_micro = pd.DataFrame(df_f1_micro_raw)
    df_f1_micro = pd.melt(df_f1_micro, var_name="Agg Method", value_name="Micro F1")
    return df_search, df_f1_micro

def step(lab_dir:Path):
    out_search = []
    out_f1 = []
    base_dirs = list(chain((p for p in lab_dir.iterdir() if p.is_dir())))
    print(base_dirs)
    order = {"neg": 2, "pos": 0, "pos_neg": 1}
    base_dirs.sort(key=lambda x: order[x.name])
    for base_dir in base_dirs:
        name = base_dir.name
        if name == "neg":
            _key = "Negative K-Means"
        elif name == "pos":
            _key = "With K-Means"
        elif name == "pos_neg":
            _key = "Without K-Means"
        else:
            raise ValueError("Unknown name")
        part_search_df, part_f1_df = measure(base_dir)
        part_search_df["Setting"] = _key
        part_f1_df["Setting"] = _key
        out_search.append(part_search_df)
        out_f1.append(part_f1_df)
    search_df = pd.concat(out_search)
    search_df.to_csv(lab_dir / "search_quality.csv", index=False)
    
    f1_df = pd.concat(out_f1)
    f1_df.to_csv(lab_dir / "f1_micro.csv", index=False)
    
    for mark in MARKS + ["Random"]:
        search_fig = box_plot(search_df.loc[search_df['Agg Method'] == mark], x="Setting", y="mAP@10")
        search_fig.write_image(lab_dir / f"{mark} search quality K.jpg", scale=3)

        f1_fig = box_plot(f1_df.loc[f1_df['Agg Method'] == mark], x="Setting", y="Micro F1")
        f1_fig.write_image(lab_dir / f"{mark} f1 micro K.jpg", scale=3)
        
if __name__ == "__main__":
    lab_dirs = [Path("lab_dense"), Path("lab_vit"), Path("lab_denseK")]
    for lab_dir in lab_dirs:
        step(lab_dir)