import torch
import timm
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

SIZE_THRESH = 80 * 1024


class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model("densenet121", pretrained=True, num_classes=0)
        model.to(self.device)
        model.eval()
        config = resolve_data_config({}, model=model)
        self.model = model
        self.transform = create_transform(**config)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    extractor = FeatureExtractor()
    batch_size = args.batch_size
    _dir = args.dir
    for slide_dir in _dir.iterdir():
        if (
            not slide_dir.is_dir()
            or not slide_dir.name.endswith("TR")
            or (slide_dir / "features.npy").exists()
        ):
            continue
        features = extractor.extract_features_from_dir(slide_dir, batch_size)
        with open(slide_dir / "features.npy", "wb") as f:
            np.save(f, features)
        print(features.shape)
