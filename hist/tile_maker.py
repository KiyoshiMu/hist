import logging
import math
import random
from dataclasses import dataclass
from itertools import tee
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional

import cv2
import numpy as np
import tifffile
from openslide import OpenSlide
from tqdm.auto import tqdm


@dataclass
class Grid:
    w_num: int
    h_num: int


@dataclass
class Area:
    x: int
    y: int
    width: int
    height: int
    is_tile = False

    @property
    def center(self):
        x, y = self.x + int(self.width / 2), self.y + int(self.height / 2)
        return (x, y)

    def loc(self, half_tile_size: int):
        if self.is_tile:
            return (self.x, self.y)
        center_x, center_y = self.center
        return (center_x - half_tile_size, center_y - half_tile_size)


def mask_from_thumbnail(img: np.ndarray):
    gaussian_blur = cv2.GaussianBlur(src=img, ksize=(7, 7), sigmaX=0, sigmaY=0)
    img_gray = cv2.cvtColor(gaussian_blur, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_gray, 205, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh
    mask: np.ndarray = cv2.erode(thresh, (5, 5))
    return mask


def area_from_mask(mask: np.ndarray, thumbnail_step: int):
    thumbnail_height, thumbnail_width = mask.shape
    # exclude the edge
    for loc_h in range(0, thumbnail_height - thumbnail_step, thumbnail_step):
        for loc_w in range(0, thumbnail_width - thumbnail_step, thumbnail_step):
            if (
                mask[
                    loc_h : loc_h + thumbnail_step,
                    loc_w : loc_w + thumbnail_step,
                ].mean()
                < 0.5
            ):
                continue
            yield Area(loc_w, loc_h, thumbnail_step, thumbnail_step)


class TileMaker:
    def __init__(
        self,
        tile_size: int = 512,
    ) -> None:
        self.tile_size = tile_size
        self.logger = logging.getLogger("TileMaker")
        self.half_tile_size = int(self.tile_size / 2)
        random.seed(42)

    def scan_slide(
        self,
        wsi_p: str,
        search_area: Optional[Area] = None,
    ):
        slide_arr = tifffile.imread(wsi_p)
        height, width, *_ = slide_arr.shape
        if search_area is None:
            search_area = Area(0, 0, width, height)
        area_gen = self._iter_area(search_area)
        for roi_imgs, roi_cells in tqdm(self._step_on_array(area_gen, slide_arr)):
            if len(roi_imgs) == 0:
                continue
            locs = [cell.loc(self.half_tile_size) for cell in roi_cells]
            yield from zip(roi_imgs, locs)

    def scan_slide_by_mask(
        self,
        wsi_p: Path,
        dst_dir: Optional[Path] = None,
    ):
        areas, mask_img = self.slide_mask(wsi_p)
        if dst_dir is not None:
            cv2.imwrite(str(dst_dir / f"{wsi_p.stem}_mask.jpg"), mask_img[:, :, ::-1])

        open_slide = OpenSlide(str(wsi_p))
        for search_area in areas:
            area_list = list(self._iter_area(search_area))
            roi_imgs, roi_cells = self._step(area_list, open_slide)
            if len(roi_imgs) == 0:
                continue
            locs = [cell.loc(self.half_tile_size) for cell in roi_cells]
            yield from zip(roi_imgs, locs)
        open_slide.close()

    def slide_mask(self, wsi_p: Path, slide_step=512 * 8):
        slide = tifffile.TiffFile(str(wsi_p))
        thumbnail = slide.pages[1].asarray()
        thumbnail_height, thumbnail_width, *_ = thumbnail.shape
        slide_height, slide_width, *_ = slide.pages[0].shape
        w_scale = thumbnail_width / slide_width
        h_scale = thumbnail_height / slide_height
        mask = mask_from_thumbnail(
            thumbnail,
        )
        slide_step = self.tile_size * 10
        areas, areas_copy = tee(
            area_from_mask(mask, int(math.ceil(slide_step * w_scale))),
            2,
        )
        img_copy = np.zeros_like(thumbnail)
        for patch in areas_copy:
            img_copy[
                patch.y : patch.y + patch.height,
                patch.x : patch.x + patch.width,
            ] = thumbnail[
                patch.y : patch.y + patch.height,
                patch.x : patch.x + patch.width,
            ]
        slide.close()
        return [
            Area(
                int(patch.x / w_scale),
                int(patch.y / h_scale),
                slide_step,
                slide_step,
            )
            for patch in areas
        ], img_copy

    def _step(
        self,
        areas: Iterable[Area],
        slide: OpenSlide,
    ):
        probe_areas: List[Area] = []
        probe_imgs: List[np.ndarray] = []
        for area in areas:
            img = slide.read_region((area.x, area.y), 0, (area.width, area.height))
            img = np.array(img)[:, :, :3]
            if self._is_blank(img):
                continue
            probe_areas.append(area)
            probe_imgs.append(img)
        return probe_imgs, probe_areas

    def _step_on_array(
        self,
        areas: Iterable[Area],
        slide_arr: np.ndarray,
    ):
        probe_areas: List[Area] = []
        probe_imgs: List[np.ndarray] = []
        for area in areas:
            img = slide_arr[area.y : area.y + area.height, area.x : area.x + area.width]
            if self._is_blank(img):
                continue
            probe_areas.append(area)
            probe_imgs.append(img)
        return probe_imgs, probe_areas

    def _iter_area(self, area: Area):
        for loc_h in range(
            area.y,
            area.y + area.height - self.tile_size,
            self.tile_size,
        ):
            for loc_w in range(
                area.x,
                area.x + area.width - self.tile_size,
                self.tile_size,
            ):
                yield Area(loc_h, loc_w, self.tile_size, self.tile_size)

    def _is_blank(self, img: np.ndarray):
        return self._isBlackPatch(img) or self._isWhitePatch(img)

    @staticmethod
    def _isWhitePatch(patch: np.ndarray, satThresh=5):
        # https://github.com/mahmoodlab/CLAM
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        return True if np.mean(patch_hsv[:, :, 1]) < satThresh else False

    @staticmethod
    def _isBlackPatch(patch: np.ndarray, rgbThresh=40):
        # https://github.com/mahmoodlab/CLAM
        return True if np.all(np.mean(patch, axis=(0, 1)) < rgbThresh) else False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tile_size = 512
    tile_maker = TileMaker(tile_size=tile_size)

    slide_name_p = "slides.txt"
    slide_dir = Path("/storage/create_local/campbell/OBIO/Images")
    dst_dir = Path("histo_tiles")

    done = {f"{item.name}.tif" for item in dst_dir.iterdir() if item.name.endswith("TR")}
    print(f"Done: {len(done)}")
    with open(slide_name_p, "r") as f:
        slides = [l.strip() for l in f.readlines()]
    in_que = [s for s in slides if s not in done]
    print(f"Que: {len(in_que)}")

    mask_dst = dst_dir / "masks"
    mask_dst.mkdir(exist_ok=True, parents=True)

    for slide_f in tqdm(in_que):
        slide_p = slide_dir / slide_f
        slide_name = slide_p.stem
        slide_dst = dst_dir / slide_name
        slide_dst.mkdir(exist_ok=True, parents=True)

        for img, (x_shift, y_shift) in tile_maker.scan_slide_by_mask(
            slide_p,
            dst_dir=mask_dst,
        ):
            cv2.imwrite(
                str(slide_dst / f"{x_shift}_{y_shift}.jpg"),
                img[:, :, ::-1],
            )
