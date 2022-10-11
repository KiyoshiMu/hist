import json
import shutil
from sklearn.decomposition import PCA
import random
from tqdm.auto import tqdm
random.seed(0)


class PatchExporter:
    def __init__(self, n_components=0.95, diff_fit=False, copy_img=True) -> None:
        self.pca = PCA(n_components=n_components)
        self.diff_fit = diff_fit
        self.copy_img = copy_img
        
    def export(self, feats, ps_list, dst, size=200):
        pca = self.pca
        pca.fit(feats[0])

        for feat, ps in tqdm(zip(feats, ps_list)):
            slide_name = ps[0].parent.name
            if self.diff_fit:
                pca.fit(feat)
            feat = pca.transform(feat).tolist()
            if len(feat) > size:
                idxs = random.sample(range(len(feat)), size)
                feat = [feat[i] for i in idxs]
                ps = [ps[i] for i in idxs]
            exp = [{"feat": f, "name": p.name} for f, p in zip(feat, ps)]
            with open(dst / f"{slide_name}.json", "w") as f:
                json.dump(exp, f)
                
            if self.copy_img:
                img_dir = dst / slide_name
                img_dir.mkdir(exist_ok=True)
                for p in ps:
                    shutil.copy(p, img_dir)

if __name__ == "__main__":
    exporter = PatchExporter()
    import pickle
    with open("histo_tiles/feats.pkl", "rb") as f:
        feats, ps_list = pickle.load(f)