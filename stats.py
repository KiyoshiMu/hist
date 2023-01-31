import pandas as pd
from pathlib import Path
from scipy.stats import ttest_1samp
import numpy as np
def step(p, mark=None, metric="mAP@10"):
    _df = pd.read_csv(p)
    # group by Agg Method, calculate the mean, std between different Setting
    groups = _df.groupby(["Agg Method", "Setting"]).agg(["mean", "std"])
    # remove the column levels
    groups.columns = groups.columns.droplevel(1)
    groups = groups.reset_index()
    groups[metric] = [f"{v:.3f}±{s:.3f}" for v, s in zip(groups.iloc[:, 2],groups.iloc[:, 3])]
    groups = groups.iloc[:, 0: 3]
    groups = groups.sort_values(by=metric, ascending=False)
    if mark is not None:
        groups["Extraction"] = mark
        _df["Extraction"] = mark
    # groups.to_latex(Path(p).parent / "search_quality.tex", index=False)
    return groups, _df

def merge():
    labs = ["lab_vit", "lab_denseK", "lab_dense"]
    marks = ["ViT", "KimiaNet", "DenseNet"]
    names = ["search_quality.csv", "f1_micro.csv"]
    metrics  = ["mAP@10", "Micro F1"]
    
    for name, metric in zip(names, metrics):
        out = []
        records_ = []
        for lab, mark in zip(labs, marks):
            p = Path(lab) / name
            metrics_part, record_part = step(p, mark, metric=metric)
            out.append(metrics_part)
            records_.append(record_part)
        df_metrics = pd.concat(out)
        # sort the columns to [Extraction, Agg Method, Setting, metric]
        df_metrics = df_metrics[["Extraction", "Agg Method", "Setting", metric]]
        df_pivot = df_metrics.pivot_table(index=["Extraction", "Agg Method"], columns=["Setting"], values=metric, aggfunc=lambda x: x)
        # sort the columns to [With K-Means, Without K-Means, Negative K-Means]
        df_pivot = df_pivot[["With K-Means", "Without K-Means", "Negative K-Means"]]
        df_pivot.to_latex(f"{name[:-4]}.lax", index=True)
        
        df_record = pd.concat(records_)
        df_record.to_csv(f"{name[:-4]}_record.csv", index=False)

def p_test():
    names = ["search_quality", "f1_micro"]
    metrics  = ["mAP@10", "Micro F1"]
    agg_methods = ["Average Pooling", "Attention Pooling"]
    extraction_methods = ["ViT", "KimiaNet", "DenseNet"]
    settings = ["With K-Means", "Without K-Means"]
    for name, metric in zip(names, metrics):
        print(f"=================={name}==================")
        df = pd.read_csv(f"{name}_record.csv")
        # stat t test cross between different Agg Method and Setting
        for agg_method in agg_methods:
            for extraction in extraction_methods:
                gS0 = df[(df["Agg Method"] == agg_method) & (df["Extraction"] == extraction) & (df["Setting"] == settings[0])][metric].to_numpy()
                gS1 = df[(df["Agg Method"] == agg_method) & (df["Extraction"] == extraction) & (df["Setting"] == settings[1])][metric].to_numpy()
                diff_s0_s1 = gS0 - gS1
                print(f"{agg_method} {extraction} {settings[0]} vs {settings[1]}: {ttest_1samp(diff_s0_s1, 0, alternative='greater')}")
                effect = (diff_s0_s1 > 0).sum()
                mean_diff = (gS0.mean() - gS1.mean()) / gS1.mean()
                print(f"Effect: {effect}, Mean Diff: {mean_diff:.3f}")
                

if __name__ == "__main__":
    p_test()
