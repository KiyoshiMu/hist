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

    return groups, _df

def merge():
    labs = ["lab_vit", "lab_denseK", "lab_dense", "lab_dino"]
    marks = ["ViT-16/256", "KimiaNet", "DenseNet", "DINO"]
    names = ["search_quality.csv", "f1_weighted.csv", "recall.csv", "precision.csv"]
    metrics  = ["mAP@10", "Weighted-F1", "Recall", "Precision"]
    
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
        df_pivot = df_pivot[["With BPG", "Without BPG", "With BPG-"]]
        df_pivot.to_latex(f"{name[:-4]}.lax", index=True)
        
        df_record = pd.concat(records_)
        df_record.to_csv(f"{name[:-4]}_record.csv", index=False)

def p_test_bgp():
    names = ["search_quality", "f1_weighted", "recall", "precision"]
    metrics  = ["mAP@10", "Weighted-F1", "Recall", "Precision"]
    agg_methods = ["Average Pooling", "Attention Pooling"]
    extraction_methods = ["ViT-16/256", "KimiaNet", "DenseNet", "DINO"]
    settings = ["With BPG", "Without BPG"]
    
    # settings = ["With BPG", "With BPG-"]
    for name, metric in zip(names, metrics):
        means = []
        diffs = []
        print(f"=================={name}==================")
        df = pd.read_csv(f"{name}_record.csv")
        # stat t test cross between different Agg Method and Setting
        for agg_method in agg_methods:
            for extraction in extraction_methods:
                gS0 = df[(df["Agg Method"] == agg_method) & (df["Extraction"] == extraction) & (df["Setting"] == settings[0])][metric].to_numpy()
                gS1 = df[(df["Agg Method"] == agg_method) & (df["Extraction"] == extraction) & (df["Setting"] == settings[1])][metric].to_numpy()
                diff_s0_s1 = gS0 - gS1
                p = ttest_1samp(diff_s0_s1, 0, alternative='greater').pvalue
                if p < 0.05:
                    print(f"{agg_method} {extraction} {settings[0]} vs {settings[1]}: {p}")
                diff = diff_s0_s1.mean()
                diffs.append(diff)
                mean_diff = (gS0.mean() - gS1.mean()) / gS1.mean()
                means.append(mean_diff)
                
        print(f"Abs Diff mean: {np.mean(diffs):.3f}")
        print(f"Mean Diff%: {np.mean(means):.3f}")
        
def p_test_dino():
    names = ["search_quality", "f1_weighted"]
    metrics  = ["mAP@10", "Weighted-F1"]
    agg_method = "Attention Pooling"
    target = "DINO"
    extraction_methods = ["ViT-16/256", "KimiaNet", "DenseNet"]    
    setting = "With BPG" 
    for name, metric in zip(names, metrics):
        means = []
        diffs = []
        print(f"=================={name}==================")
        df = pd.read_csv(f"{name}_record.csv")
        # stat t test cross between different Agg Method and Setting
        for extraction in extraction_methods:
            gS0 = df[(df["Agg Method"] == agg_method) & (df["Extraction"] == target) & (df["Setting"] == setting)][metric].to_numpy()
            gS1 = df[(df["Agg Method"] == agg_method) & (df["Extraction"] == extraction) & (df["Setting"] == setting)][metric].to_numpy()
            diff_s0_s1 = gS0 - gS1
            p = ttest_1samp(diff_s0_s1, 0, alternative='greater').pvalue
            if p < 0.05:
                print(f"{agg_method} {target}  vs {extraction}: {p}")
            effect = (diff_s0_s1 > 0).sum()
            diff = diff_s0_s1.mean()
            diffs.append(diff)
            mean_diff = (gS0.mean() - gS1.mean()) / gS1.mean()
            means.append(mean_diff)
            print(f"Effect: {effect}, Mean Diff: {mean_diff:.3f}, Abs diff: {diff:.3f}")
        print(f"Abs Diff mean: {np.mean(diffs):.3f}")
        print(f"Mean Diff%: {np.mean(means):.3f}")
if __name__ == "__main__":
    merge()
    p_test_bgp()
    p_test_dino()
