import numpy as np
from pathlib import Path

SIZE_THRESH = 80 * 1024
def get_img_ps(dir_p):
    img_ps = list(dir_p.glob("*.jpg"))
    img_ps = [p for p in img_ps if p.stat().st_size > SIZE_THRESH]
    return img_ps

feat_ps = [dp/"features.npy" for dp in Path("histo_tiles").iterdir() if (dp/"features.npy").exists()]

all_feats = []
all_patch_ps = []
for feat_p in feat_ps:
    features = np.load(feat_p)
    patch_ps = get_img_ps(feat_p.parent)
    all_feats.append(features)
    all_patch_ps.extend(patch_ps)
    break