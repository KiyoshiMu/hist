from pathlib import Path

import numpy as np
import pandas as pd
from hist.io_utils import pkl_load, simplify_label
from hist.knn import create_knn
from hist.performance import dummy_exp, dump_metric
import random
import plotly.express as px

random.seed(0)


def norm_vs(
    train_pkl_p,
    val_pkl_p,
    mark="pool",
    trial="",
    dummy_baseline=True,
    dst=Path("MULTIpos"),
):
    dst.mkdir(exist_ok=True)
    train = pkl_load(train_pkl_p)
    refer_embed = train["embed_pool"]
    labels = np.array([simplify_label(l) for l in train["labels"]])
    val = pkl_load(val_pkl_p)
    val_embed = val["embed_pool"]
    val_full_label = val["labels"]
    val_labels = np.array([simplify_label(l) for l in val_full_label])
    # run experiments on NORMAL vs. all other classes
    targets = set(labels) - {"NORMAL"}
    for target in targets:
        # val_keep0 = [i for i, l in enumerate(val_labels) if l == target]
        # val_keep1 = random.sample([i for i, l in enumerate(val_labels) if l == "NORMAL"], len(val_keep0))
        # refer_keep0 = [i for i, l in enumerate(labels) if l == target]
        # refer_keep1 = [i for i, l in enumerate(labels) if l == "NORMAL"]
        val_keep = [i for i, l in enumerate(val_labels) if l == target or l == "NORMAL"]
        refer_keep = [i for i, l in enumerate(labels) if l == target or l == "NORMAL"]
        refer_embed_ = refer_embed[refer_keep]
        labels_ = labels[refer_keep]
        val_embed_ = val_embed[val_keep]
        val_labels_ = val_labels[val_keep]
        knn = create_knn(refer_embed_, labels_)
        classes_ = knn.classes_
        preds_ = knn.predict(val_embed_)

        dump_metric(
            val_labels_,
            preds_,
            classes_,
            dst / f"{mark}_{target}_{trial}_metric.csv",
        )
        if dummy_baseline:
            dummy_exp(
                refer_embed_,
                labels_,
                val_embed_,
                val_labels_,
                dst / f"dummy_{target}_{trial}_metric.csv",
            )

TARGETS = set(("ACL", "LPD", "MDS", "MPN", "PCN"))

def plot(dst=Path("MULTI")):
    csvs = list(dst.rglob("*.csv"))
    targets = []
    methods = []
    value = []
    exps = []
    for csv in csvs:
        parts = csv.stem.split("_")
        target = parts[1]
        if target not in TARGETS:
            continue
        method = parts[0]
        df = pd.read_csv(csv, index_col=0)
        fscore = df["fscore"].to_dict()

        value.append(fscore["NORMAL"])
        targets.append("NORMAL")

        value.append(fscore[target])
        targets.append(target)

        methods.append(method)
        methods.append(method)
        exps.append(target)
        exps.append(target)

    df = pd.DataFrame(
        {"target": targets, "method": methods, "f1-score": value, "exp": exps}
    )
    unique_exp = df["exp"].unique()
    for exp in unique_exp:
        exp_df = df[df["exp"] == exp]
        box_fig = px.box(exp_df, x="target", y="f1-score", color="method")
        # save to jpg
        box_fig.write_image(str(dst / f"{exp}.jpg"), scale=2)

def main(base_dir:Path):
    for trial in range(5):
        dst_dir = base_dir / f"trial{trial}"

        norm_vs(
            dst_dir / f"train{trial}_pool.pkl",
            dst_dir / f"val{trial}_pool.pkl",
            mark="pool",
            trial=str(trial),
            dummy_baseline=True,
            dst=dst_dir,
        )
    plot(dst=base_dir)

if __name__ == "__main__":
    from itertools import chain
    base_dirs = list(chain((p for p in Path("lab_vit").iterdir() if p.is_dir()),
                           (p for p in Path("lab_dense").iterdir() if p.is_dir())))
    print(base_dirs)
    for base_dir in base_dirs:
        main(base_dir)
    
