from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional
from unicodedata import name

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import umap
from plotly.validators.scatter.marker import SymbolValidator
from sklearn.manifold import TSNE
import plotly.io as pio

from hist.io_utils import pkl_load, simplify_label
from hist.knn import create_knn
from hist.performance import dummy_exp, dump_metric, proba_to_dfDict


pio.kaleido.scope.mathjax = None

TEMPLATE = "plotly_white"
FONT = "Arial"


def arr_project(arrays, method="umap") -> np.ndarray:
    if method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=0)
    else:
        reducer = TSNE(n_components=2, random_state=0)

    return reducer.fit_transform(arrays)


def make_projection(embed, labels, index, source):

    projection = arr_project(embed)
    plot_df = pd.DataFrame(
        {
            "D1": projection[:, 0],
            "D2": projection[:, 1],
            "label": labels,
            "index": index,
            "source": source,
        },
    )
    plot_df.to_json("slide_embed.json")


def avg_pool(arr):
    arr_norm = np.linalg.norm(arr, axis=1, keepdims=True)
    arr_normalized = arr / arr_norm
    arr_avg = arr_normalized.mean(axis=0)
    return arr_avg


def make_plot(embed, labels, index, mark="", write=True, dst=Path(".")):

    projection = arr_project(embed)
    plot_df = pd.DataFrame(
        {
            "D1": projection[:, 0],
            "D2": projection[:, 1],
            "label": labels,
            "index": index,
        },
    )
    fig = plot_embedding_sketch(plot_df, thresh=10, use_simplify=True)
    if write:
        plot_df.to_csv(str(dst / f"{mark}_pre_plot_df.csv"))
        fig.write_html(str(dst / f"{mark}_pre_umap.html"))
    return fig


def plot_embedding_sketch(df: pd.DataFrame, thresh=15, use_simplify=False):
    if use_simplify:
        df["marker_base"] = df["label"].map(simplify_label)
    else:
        df["marker_base"] = df["label"]
    markers = df["marker_base"].value_counts()
    label_value_counts = markers >= thresh
    show_legends = set(label_value_counts.index.to_numpy()[label_value_counts])
    df["show_legends"] = df["marker_base"].map(lambda x: x in show_legends)

    raw_symbols = SymbolValidator().values
    colors = px.colors.qualitative.Plotly
    symbol_dict = {}
    color_dict = {}
    color_len = len(colors)
    for idx, label in enumerate(markers.index):
        symbol_idx = idx // color_len
        color_idx = idx % color_len
        symbol_dict[label] = raw_symbols[symbol_idx]
        color_dict[label] = colors[color_idx]
    df["color"] = df["marker_base"].map(color_dict)
    df["symbol"] = df["marker_base"].map(symbol_dict)

    fig = go.Figure()
    sel_labels = sorted(show_legends, key=len)
    for sel_label in sel_labels:
        tmp_df = df.loc[df["marker_base"] == sel_label, :]
        fig.add_trace(
            go.Scatter(
                x=tmp_df["D1"],
                y=tmp_df["D2"],
                mode="markers",
                marker_color=tmp_df["color"],
                marker_symbol=tmp_df["symbol"],
                text=tmp_df["label"].to_list(),
                showlegend=True,
                name=sel_label,
            ),
        )

    no_legend_df = df.loc[~df["show_legends"]]
    fig.add_trace(
        go.Scatter(
            x=no_legend_df["D1"],
            y=no_legend_df["D2"],
            mode="markers",
            opacity=0.5,
            marker_color=no_legend_df["color"],
            marker_symbol=no_legend_df["symbol"],
            showlegend=False,
            text=no_legend_df["label"].to_list(),
            name="",
        ),
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        legend=dict(
            orientation="h",
        ),
        width=1280,
        height=600,
    )
    return fig


def measure_slide_vectors(
    train_pkl_p,
    val_pkl_p,
    val_entropy: Optional[list] = None,
    mark="pool",
    trial="",
    dummy_baseline=True,
    dst=Path("."),
):
    train = pkl_load(train_pkl_p)
    refer_embed = train["embed_pool"]
    labels = [simplify_label(l) for l in train["labels"]]
    knn = create_knn(refer_embed, labels)
    classes_ = knn.classes_
    val = pkl_load(val_pkl_p)
    val_embed = val["embed_pool"]
    val_full_label = val["labels"]
    val_labels = [simplify_label(l) for l in val_full_label]
    val_names = val["index"]
    projection = arr_project(val_embed)
    pred_probs: np.ndarray = knn.predict_proba(val_embed)
    _df = proba_to_dfDict(pred_probs, classes_, val_labels)
    preds = knn.predict(val_embed)

    _df.update(
        {
            "D1": projection[:, 0],
            "D2": projection[:, 1],
            "label": val_labels,
            "full_label": val_full_label,
            "index": val_names,
            "pred": preds,
            "correct": [
                val_labels == preds for (val_labels, preds) in zip(val_labels, preds)
            ],
        },
    )
    if val_entropy is not None:
        _df["pred_entropy"] = val_entropy

    plot_df = pd.DataFrame(_df)
    dump_metric(val_labels, preds, classes_, dst / f"{mark}{trial}_metric.csv")
    if dummy_baseline:
        dummy_exp(
            refer_embed,
            labels,
            val_embed,
            val_labels,
            dst / f"dummy{trial}_metric.csv",
        )
    plot_df.to_json(str(dst / f"{mark}{trial}.json"), orient="records")
    fig = plot_embedding(plot_df)
    fig.write_html(str(dst / f"{mark}{trial}_umap.html"))
    return fig

def plot_embedding(df: pd.DataFrame, marks = None):
    df["label"] = df["label"]
    fig = px.scatter(
        df,
        x="D1",
        y="D2",
        color="label",
        hover_name="index",
        # symbol="correct",
        hover_data=[
            "pred",
            "prob_top0",
            "prob_top1",
            "prob_top2",
            "full_label",
            "correct",
        ],
    )
    if marks != None:
        df.set_index("index", inplace=True)
        for mark in marks:
            case = df.loc[mark.slide_name]
            x = case["D1"]
            y = case["D2"]
            fig.add_annotation(
                x=x,
                y=y,
                text=mark.mark_name,
                showarrow=True,
                arrowhead=1,
            )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        legend=dict(
            orientation="h",
        ),
        width=1280,
        height=600,
    )
    return fig