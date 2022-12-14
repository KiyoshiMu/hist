
from pathlib import Path
import random
import numpy as np
import json
from tqdm.auto import tqdm
random.seed(42)
from hist.io_utils import simplify_label
def main():
    with open("Data/y_true.json", "r") as f:
        labels_raw = json.load(f)
        _y_all = [i["label"] for i in labels_raw]
        
    _features = np.load("Data/all_vit_feats.npy", allow_pickle=True)
    _paths = np.load("Data/all_vit_patch_ps.npy", allow_pickle=True)
    keep_idx = [i for i, ps in tqdm(enumerate(_paths))if len(ps) >= 96]
    path = [_paths[i] for i in keep_idx]
    features = [_features[i] for i in keep_idx]
    y_all = [_y_all[i] for i in keep_idx]
    y_simple = [simplify_label(ls) for ls in y_all]
    sampled_indices = sample_by_groups(y_simple, n=10)
    
    dst = Path("Data/vit_feat")
    dst.mkdir(exist_ok=True)
    print(sampled_indices.keys())
    name_map =  {}
    for key, indices in tqdm(sampled_indices.items()):
        if key == "OTHER":
            continue
        name_map[key] =  []
        for idx in indices:
            ps = path[idx]
            name = Path(ps[0]).parent.stem
            name_map[key].append(name)
            feats = features[idx]
            label = y_all[idx]
            simple_label = y_simple[idx]
            records = [{"feat": feat.tolist(), "name": Path(p).name, "label":label, "simple_label":simple_label} for p, feat in zip(ps, feats)]
            with open(dst / f"{name}.json", "w") as f:
                json.dump(records, f)
    
    with open(dst / "vit_review_name_map.json", "w") as f:
        json.dump(name_map, f)
    
def sample_by_groups(labels: list[str], n: int = 10):
    """Randomly sample [n] cases from each group.
    Return a dict with groups as keys and the indices of the cases as values"""       

    groups = set(labels)
    indices: dict[str, list[int]] = {g: [] for g in groups}
    for i, l in enumerate(labels):
        indices[l].append(i)
    indices = {k: random.sample(v, n) for k, v in indices.items() if len(v) >= n}
    return indices
    
if __name__ == "__main__":
    main()