"""Adapted from https://github.com/Richarizardd/Self-Supervised-ViT-Path
and https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/hipt_model_utils.py
"""

import numpy as np
import torch
from tqdm.auto import tqdm

from torchvision import transforms as pth_transforms
from vision_transformer import vit_small

from pathlib import Path
from PIL import Image


SIZE_THRESH = 80 * 1024
TARGET_SIZE = 256


class FeatureExtractor:
    def __init__(self, ckpt_path):
        model = vit_small(patch_size=16)
        state_dict = torch.load(ckpt_path, map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.transform = eval_transforms()

    def extract_features(self, imgs):
        imgs = torch.stack([self.transform(img) for img in imgs]).to(self.device)
        with torch.no_grad():
            features = self.model(imgs)
        return features

    def extract_features_from_dir(self, dir_p: Path, batch_size=64):
        img_ps = list(dir_p.glob("*.jpg"))
        img_ps = [p for p in img_ps if p.stat().st_size > SIZE_THRESH]
        features = []
        for i in tqdm(range(0, len(img_ps), batch_size)):
            imgs = [Image.open(p) for p in img_ps[i : i + batch_size]]
            features.extend(self.extract_features(imgs).cpu().numpy())
        return np.array(features)


# https://github.com/facebookresearch/dino/blob/main/eval_knn.py#L36
def eval_transforms():
    transform = pth_transforms.Compose(
        [
            # resize to 256x256 as Dino does
            pth_transforms.Resize(
                TARGET_SIZE, interpolation=pth_transforms.InterpolationMode.BICUBIC
            ),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transform


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    extractor = FeatureExtractor("dino_deitsmall16_pretrain_full_checkpoint.pth")
    batch_size = args.batch_size
    _dir = args.dir
    for slide_dir in _dir.iterdir():
        if not slide_dir.is_dir() or not slide_dir.name.endswith("TR"):
            continue
        features = extractor.extract_features_from_dir(slide_dir, batch_size)
        with open(slide_dir / "dino_features.npy", "wb") as f:
            np.save(f, features)
        print(features.shape)
