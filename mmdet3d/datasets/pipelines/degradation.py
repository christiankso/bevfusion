import zlib

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from mmdet.datasets.builder import PIPELINES

__all__ = ["SensorDegradation"]


@PIPELINES.register_module()
class SensorDegradation:
    """Controlled camera/LiDAR degradation for robustness tests.

    Insert after the image and point loaders and before ImageAug3D, so images
    are still full-resolution PIL images and points include all sweeps.
    Randomness is seeded per sample, so a given condition drops the same
    points on every run.

    Args:
        point_drop (float): Fraction of LiDAR points removed at random.
        max_range (float): Remove points farther than this (m, in the xy plane).
        brightness (float): Image brightness factor; 1.0 = unchanged, 0.0 = black.
        contrast (float): Image contrast factor; 1.0 = unchanged, 0.0 = flat gray.
        blur (float): Gaussian blur radius in pixels on the original image.
        drop_cams (list[str]): Cameras replaced by black frames, e.g. ["CAM_FRONT"].
        seed (int): Base random seed.
    """

    def __init__(
        self,
        point_drop=0.0,
        max_range=None,
        brightness=1.0,
        contrast=1.0,
        blur=0.0,
        drop_cams=(),
        seed=0,
    ):
        assert 0.0 <= point_drop < 1.0, "point_drop must be in [0, 1)"
        self.point_drop = point_drop
        self.max_range = max_range
        self.brightness = brightness
        self.contrast = contrast
        self.blur = blur
        self.drop_cams = list(drop_cams)
        self.seed = seed

    def _rng(self, results):
        key = str(results.get("token", results.get("lidar_path", "")))
        return np.random.default_rng(self.seed + zlib.crc32(key.encode()))

    def _degrade_points(self, results):
        points = results["points"]
        keep = torch.ones(len(points), dtype=torch.bool)
        if self.max_range is not None:
            xy = points.tensor[:, :2]
            keep &= torch.linalg.norm(xy, dim=1) <= self.max_range
        if self.point_drop > 0:
            rng = self._rng(results)
            keep &= torch.from_numpy(rng.random(len(points)) >= self.point_drop)
        results["points"] = points[keep]

    def _degrade_images(self, results):
        images = []
        for path, img in zip(results["image_paths"], results["img"]):
            # nuScenes paths look like .../samples/CAM_FRONT/xxx.jpg; match the
            # folder so CAM_FRONT does not also match CAM_FRONT_LEFT.
            if any(f"/{cam}/" in path for cam in self.drop_cams):
                img = Image.new(img.mode, img.size)
            else:
                if self.brightness != 1.0:
                    img = ImageEnhance.Brightness(img).enhance(self.brightness)
                if self.contrast != 1.0:
                    img = ImageEnhance.Contrast(img).enhance(self.contrast)
                if self.blur > 0:
                    img = img.filter(ImageFilter.GaussianBlur(self.blur))
            images.append(img)
        results["img"] = images

    def __call__(self, results):
        if self.point_drop > 0 or self.max_range is not None:
            self._degrade_points(results)
        if (
            self.brightness != 1.0
            or self.contrast != 1.0
            or self.blur > 0
            or self.drop_cams
        ):
            self._degrade_images(results)
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(point_drop={self.point_drop}, "
            f"max_range={self.max_range}, brightness={self.brightness}, "
            f"contrast={self.contrast}, blur={self.blur}, "
            f"drop_cams={self.drop_cams})"
        )
