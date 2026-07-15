from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml


@dataclass
class CameraModel:
    K: np.ndarray              # 3x3 camera matrix
    dist: np.ndarray           # distortion coefficients
    size: Optional[Tuple[int, int]] = None  # (width, height)


def load_camera_yaml(path: str | Path) -> Optional[CameraModel]:
    path = Path(path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(
        data.get("dist_coefficients") or data.get("dist_coeffs"),
        dtype=np.float64
    )

    w = data.get("image_width")
    h = data.get("image_height")
    size = (int(w), int(h)) if w and h else None

    return CameraModel(K=K, dist=dist, size=size)


def undistort_image(img: np.ndarray, cam: CameraModel) -> np.ndarray:
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        cam.K, cam.dist, (w, h), alpha=0
    )
    return cv2.undistort(img, cam.K, cam.dist, None, new_K)
