from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from fastmosaic.io.dataset import FrameMeta


@dataclass
class StitchConfig:
    orb_features: int = 3000
    ratio_test: float = 0.75
    max_input_dim: int = 1200
    max_canvas_dim: int = 16000
    hard_canvas_dim: int = 12000
    affine_ransac_thresh_px: float = 3.0
    min_inliers: int = 25
    blend_feather: bool = True


def _match_orb(img_prev, img_cur, cfg: StitchConfig):
    orb = cv2.ORB_create(nfeatures=cfg.orb_features)
    k1, d1 = orb.detectAndCompute(img_prev, None)
    k2, d2 = orb.detectAndCompute(img_cur, None)

    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None

    bf    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(d1, d2, k=2)
    good  = [m for m, n in pairs if m.distance < cfg.ratio_test * n.distance]

    if len(good) < 12:
        return None

    pts_prev = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_cur  = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return pts_prev, pts_cur


def _feather_blend(canvas: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """
    Distance-transform tabanlı feather blending.
    canvas ve patch aynı boyutta olmalidir.
    """
    m_new   = (cv2.cvtColor(patch,  cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    m_old   = (cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    overlap = (m_new & m_old).astype(bool)

    if not overlap.any():
        return np.where(m_new[:, :, None], patch, canvas).astype(np.uint8)

    d_new = cv2.distanceTransform(m_new, cv2.DIST_L2, 5)
    d_old = cv2.distanceTransform(m_old, cv2.DIST_L2, 5)

    alpha = np.zeros(canvas.shape[:2], dtype=np.float32)
    denom = d_new + d_old + 1e-6
    alpha[overlap] = d_new[overlap] / denom[overlap]
    alpha = alpha[:, :, None]

    blended  = (canvas.astype(np.float32) * (1 - alpha) +
                patch.astype(np.float32)  * alpha).astype(np.uint8)
    only_new = m_new & ~m_old
    return np.where(only_new[:, :, None], patch, blended).astype(np.uint8)


def stitch_sequence(metas: List[FrameMeta], out_path: str | Path, cfg: StitchConfig) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Starting ORB stitch: {len(metas)} frames  "
          f"(feather={'on' if cfg.blend_feather else 'off'})")

    def _resize_if_needed(img):
        h, w = img.shape[:2]
        if max(h, w) <= cfg.max_input_dim:
            return img
        s = cfg.max_input_dim / float(max(h, w))
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    base = cv2.imread(str(metas[0].path))
    if base is None:
        raise RuntimeError(f"Cannot read: {metas[0].path}")

    base     = _resize_if_needed(base)
    canvas   = base.copy()
    H_canvas = np.eye(3, dtype=np.float64)
    prev     = base
    stitched = 0
    skipped  = 0

    for i in range(1, len(metas)):
        cur = cv2.imread(str(metas[i].path))
        if cur is None:
            skipped += 1
            continue
        cur = _resize_if_needed(cur)

        match = _match_orb(prev, cur, cfg)
        if match is None:
            print(f"  [{i+1:03d}/{len(metas)}] no match — skipped")
            prev = cur
            skipped += 1
            continue

        pts_prev, pts_cur = match
        M, inliers = cv2.estimateAffinePartial2D(
            pts_cur, pts_prev,
            method=cv2.RANSAC,
            ransacReprojThreshold=cfg.affine_ransac_thresh_px,
        )

        if M is None or inliers is None or int(inliers.sum()) < cfg.min_inliers:
            print(f"  [{i+1:03d}/{len(metas)}] weak match — skipped")
            prev = cur
            skipped += 1
            continue

        A               = np.vstack([M, [0, 0, 1]])
        H_cur_to_canvas = H_canvas @ A

        h1, w1 = canvas.shape[:2]
        h2, w2 = cur.shape[:2]

        corners_cur    = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        warped_corners = cv2.transform(corners_cur, H_cur_to_canvas[:2])
        corners_canvas = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
        all_pts        = np.concatenate([corners_canvas, warped_corners], axis=0)

        xmin, ymin = np.floor(all_pts.min(axis=0).ravel()).astype(int)
        xmax, ymax = np.ceil(all_pts.max(axis=0).ravel()).astype(int)

        tx    = -xmin if xmin < 0 else 0
        ty    = -ymin if ymin < 0 else 0
        T     = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)
        new_w = int(xmax - xmin)
        new_h = int(ymax - ymin)

        if new_w > cfg.hard_canvas_dim or new_h > cfg.hard_canvas_dim or new_w <= 0 or new_h <= 0:
            prev = cur
            skipped += 1
            continue

        new_canvas   = cv2.warpAffine(canvas, T[:2], (new_w, new_h))
        H_cur_to_new = T @ H_cur_to_canvas
        cur_warp     = cv2.warpAffine(cur, H_cur_to_new[:2], (new_w, new_h))

        if cfg.blend_feather:
            new_canvas = _feather_blend(new_canvas, cur_warp)
        else:
            mask       = (cv2.cvtColor(cur_warp, cv2.COLOR_BGR2GRAY) != 0)[:,:,None]
            new_canvas = np.where(mask, cur_warp, new_canvas).astype(np.uint8)

        canvas   = new_canvas
        H_canvas = H_cur_to_new

        ch, cw = canvas.shape[:2]
        if max(ch, cw) > cfg.max_canvas_dim:
            s2       = cfg.max_canvas_dim / float(max(ch, cw))
            canvas   = cv2.resize(canvas, (int(cw*s2), int(ch*s2)), interpolation=cv2.INTER_AREA)
            S        = np.array([[s2,0,0],[0,s2,0],[0,0,1]], dtype=np.float64)
            H_canvas = S @ H_canvas

        stitched += 1
        print(f"  [{i+1:03d}/{len(metas)}] stitched  "
              f"inliers={int(inliers.sum())}  "
              f"canvas={canvas.shape[1]}x{canvas.shape[0]}")
        prev = cur

    print(f"\n  Summary: {stitched} stitched, {skipped} skipped")
    cv2.imwrite(str(out_path), canvas)
    print(f"✓ Saved → {out_path}")