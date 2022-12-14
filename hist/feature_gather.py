import numpy as np
from pathlib import Path

SIZE_THRESH = 80 * 1024


def get_img_ps(dir_p):
    img_ps = list(dir_p.glob("*.jpg"))
    img_ps = [p for p in img_ps if p.stat().st_size > SIZE_THRESH]
    return img_ps


def main(mark="features.npy", src=Path("histo_tiles"), dst=Path(".")):
    feat_ps = [dp / mark for dp in src.iterdir() if (dp / mark).exists()]

    all_feats = []
    all_patch_ps = []
    for feat_p in feat_ps:
        features = np.load(feat_p)
        patch_ps = get_img_ps(feat_p.parent)
        all_feats.append(features)
        all_patch_ps.append([str(p) for p in patch_ps])

    # save
    np.save(dst / "all_feats.npy", all_feats, allow_pickle=True)
    np.save(dst / "all_patch_ps.npy", all_patch_ps, allow_pickle=True)


if __name__ == "__main__":
    main()
