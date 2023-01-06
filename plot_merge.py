"""merge the jpg files in to a large jpg file, 3 per row
"""
from pathlib import Path
from PIL import Image

def merge_jpgs(src, dst, row=3):
    """merge the jpg files in to a large jpg file, 3 per row
    """

    files = [f for f in Path(src).glob("*.jpg")]
    print(f"merge {len(files)} jpg files")
    files.sort()
    imgs = [Image.open(f) for f in files]
    w, h = imgs[0].size
    new_img = Image.new("RGB", (w * row, h * (len(imgs) // row + 1)), "white")
    for i, img in enumerate(imgs):
        new_img.paste(img, (i % row * w, i // row * h))
    new_img.save(dst)
    
if __name__ == "__main__":
    labs = [Path("lab_dense"), Path("lab_vit")]
    types = ["pos", "neg", "pos_neg"]
    for lab in labs:
        for t in types:
            merge_jpgs(lab/t, lab/ f"{t}.jpeg")
