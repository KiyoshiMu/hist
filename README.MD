# Improved whole slide image representation with minimal human intervention

This a Github Repo hosting custom codes for the paper "*Improved whole slide image representation with minimal human intervention*".

## System requirements

### Hardware Requirements

For optimal performance, we recommend a computer with the following specs:

* RAM: 16+ GB
* CPU: 2+ cores, 2.2+ GHz/core
* GPU: 16+ GB

The runtimes below are generated using a computer with the recommended specs:
 * RAM: 64 GB
 * CPU: Intel(R) Xeon(R) W-10885M CPU @ 2.40GHz 
 * GPU: NVIDIA Quadro RTX 4000 with Max-Q Design, CUDA Version: 11.6

### Software Requirements

The package development version is tested on Linux operating system (Ubuntu 18.04.5 LTS).

Python Dependencies:

    python = ">=3.9,<3.11"
    faiss-cpu = ">=1.7.2"
    torch = ">=1.12.1"
    numpy = ">=1.21.5"
    pandas = ">=1.4.1"
    record-keeper = ">=0.9.31"
    tensorboard = ">=2.8.0"
    plotly = ">=5.6.0"
    scikit-learn = ">=1.0.2"
    numba = ">=0.55.1"
    scipy = "^1.8.0"
    umap-learn = ">=0.5.2"
    pytorch-metric-learning = ">=1.5.1"
    tqdm = "^4.63.0"
    kaleido = "^0.2.1"


*It takes about 5-10 mins.*

## Demo

> The data that support the findings of this study are available on reasonable request from the corresponding author, pending local REB and privacy office approval. The data are not publicly available because they contain information that could compromise research participant privacy/consent.


## Instructions for use


### Scripts

1. WSIs -> Patches

    [tile_batch.py](hist/tile_batch.py)

1. Patch -> Patch Feature

    DenseNet-121: [feature_extract.py](hist/feature_extract.py) (`from_basic`)

    KimiaNet: [feature_extract.py](hist/feature_extract.py) (`from_kimia`)

    ViT-16/256: [feature_extract_vit.py](hist/feature_extract_vit.py)

    DINO: [feature_extract_dino.py](hist/feature_extract_dino.py)

1. Aggregation

    KMF: [k_means_filter.py](k_means_filter.py)

    Pipeline: [cv_exp.py](cv_exp.py)


    PS. Patch Feature collection: [feature_gather.py](hist/feature_gather.py); Experiment Preparation: [data_prep.py](hist/data_prep.py)
    

1.  Metrics

    [measure_search.py](measure_search.py)

### Patch features

**You need first to contact the corresponding author to get the files**

all_vit_feats.npy:  https://github.com/mahmoodlab/HIPT ViT on 256x256

all_featuresK.npy: KimiaNet on 512x512

features.npy: DenseNet121 on 512x512

all_dino_feats.npy: https://github.com/facebookresearch/dino ViT on 256x256