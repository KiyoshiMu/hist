import pandas as pd
import json

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    labs = ["lab_vit", "lab_denseK", "lab_dense", "lab_dino"]
    names = ["ViT-16/256", "KimiaNet", "DenseNet", "DINO"]
    feature_size = [384, 1024, 1024, 384]
    columns = ["davies bouldin score", "silhouette score", "n components (0.95 explained variance)", "Ratio"]
    dfs = [load_json(f"{lab}/cluster_scores.json") for lab in labs]
    df = pd.DataFrame(dfs, index=names)
    df.columns = columns
    df["Original feature size"] = feature_size
    df.drop(columns=["davies bouldin score", "silhouette score"], inplace=True)
    df = df[["n components (0.95 explained variance)", "Original feature size", "Ratio"]]
    df = df.loc[["DINO", "KimiaNet", "ViT-16/256", "DenseNet"]]
    df.to_latex("cluster_table.tex", index=True)

if __name__ == "__main__":
    main()
        