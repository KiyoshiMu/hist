import pandas as pd
from pathlib import Path
def step(p, mark=None, metric="mAP@10"):
    df = pd.read_csv(p)
    # group by Agg Method, calculate the mean, std between different Setting
    groups = df.groupby(["Agg Method", "Setting"]).agg(["mean", "std"])
    # remove the column levels
    groups.columns = groups.columns.droplevel(1)
    groups = groups.reset_index()
    groups[metric] = [f"{v:.3f}±{s:.3f}" for v, s in zip(groups.iloc[:, 2],groups.iloc[:, 3])]
    groups = groups.iloc[:, 0: 3]
    groups = groups.sort_values(by=metric, ascending=False)
    if mark is not None:
        groups["Extraction"] = mark
    # groups.to_latex(Path(p).parent / "search_quality.tex", index=False)
    return groups

    
if __name__ == "__main__":
    labs = ["lab_vit", "lab_denseK", "lab_dense"]
    marks = ["ViT", "KimiaNet", "DenseNet"]
    names = ["search_quality.csv", "f1_micro.csv"]
    metrics  = ["mAP@10", "Micro F1"]
    for name, metric in zip(names, metrics):
        out = [step(Path(lab) / name, mark, metric=metric) for lab, mark in zip(labs, marks)]
        df = pd.concat(out)
        # sort the columns to [Extraction, Agg Method, Setting, metric]
        df = df[["Extraction", "Agg Method", "Setting", metric]]
        df.sort_values(by=metric, ascending=False, inplace=True)
        df.to_latex(f"{name[:-4]}.lax", index=False)
