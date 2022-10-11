import PIL
import random
import umap

random.seed(0)


def plot_slide_patch(case_feat, case_ps, idx=0, save=True):

    slide_name = case_ps[0].parent.name
    print(slide_name)
    print(len(case_feat))

    case_patches = [PIL.Image.open(p) for p in case_ps]
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(case_feat)
    cc = plot_scatter_patch(embedding[:, 0], embedding[:, 1], case_patches)
    if save:
        cc.save(f"{slide_name}_patch.jpg")
    for img_f in case_patches:
        img_f.close()
    return cc, embedding


def plot_scatter_patch(xs, yx, imgs, w=3072, h=3072, patch_size=128, max_patches=200):
    assert (
        len(xs) == len(yx) == len(imgs)
    ), f"xs, yx, imgs must be same length {len(xs)} {len(yx)} {len(imgs)}"

    if len(xs) > max_patches:
        idxs = random.sample(range(len(xs)), max_patches)
        xs = [xs[i] for i in idxs]
        yx = [yx[i] for i in idxs]
        imgs = [imgs[i] for i in idxs]
    canvas = PIL.Image.new("RGB", (w, h))
    scaled_xs = scale(xs, 0, w - patch_size)
    scaled_yx = scale(yx, 0, h - patch_size)
    for x, y, img in zip(scaled_xs, scaled_yx, imgs):
        canvas.paste(img.resize((patch_size, patch_size)), (int(x), int(y)))

    return canvas


def scale(seq, new_min, new_max):
    old_min, old_max = min(seq), max(seq)
    old_range = old_max - old_min
    new_range = new_max - new_min
    return [((x - old_min) * new_range / old_range) + new_min for x in seq]
