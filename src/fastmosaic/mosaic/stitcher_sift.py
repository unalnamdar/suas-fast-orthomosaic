from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from fastmosaic.io.dataset import FrameMeta


@dataclass
class StitchConfig:
    sift_features: int = 5000
    ratio_test: float = 0.7

    # Safety limits to avoid huge canvases / memory issues
    max_input_dim: int = 1200  # resize each input image if larger than this
    max_canvas_dim: int = 16000  # downscale canvas if it grows beyond this
    hard_canvas_dim: int = 12000  # if canvas would exceed this, skip the frame

    # RANSAC / quality
    affine_ransac_thresh_px: float = 5.0
    min_inliers: int = 30

    # Multi-frame matching
    search_window: int = 10  # Look back at last N frames for best match
    use_clahe: bool = True  # Adaptive histogram equalization


@dataclass
class FrameInfo:
    """Store frame with its transformation to canvas"""
    img: np.ndarray
    idx: int
    H_to_canvas: np.ndarray


def _match_sift(img_prev: np.ndarray, img_cur: np.ndarray, cfg: StitchConfig) -> Optional[
    Tuple[np.ndarray, np.ndarray]]:
    """
    Match features between two images using SIFT
    Returns: (pts_prev, pts_cur) or None if matching fails
    """
    sift = cv2.SIFT_create(nfeatures=cfg.sift_features)

    # Preprocessing with CLAHE for better feature detection
    if cfg.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY) if len(img_prev.shape) == 3 else img_prev
        gray_cur = cv2.cvtColor(img_cur, cv2.COLOR_BGR2GRAY) if len(img_cur.shape) == 3 else img_cur
        gray_prev = clahe.apply(gray_prev)
        gray_cur = clahe.apply(gray_cur)
    else:
        gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY) if len(img_prev.shape) == 3 else img_prev
        gray_cur = cv2.cvtColor(img_cur, cv2.COLOR_BGR2GRAY) if len(img_cur.shape) == 3 else img_cur

    k1, d1 = sift.detectAndCompute(gray_prev, None)
    k2, d2 = sift.detectAndCompute(gray_cur, None)

    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None

    # FLANN matcher (faster and better for SIFT)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(d1, d2, k=2)

    # Lowe's ratio test
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < cfg.ratio_test * n.distance:
                good.append(m)

    if len(good) < 20:
        return None

    pts_prev = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_cur = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return pts_prev, pts_cur


def _find_best_match_visual(frame_buffer: List[FrameInfo], cur_img: np.ndarray,
                            cfg: StitchConfig) -> Tuple[Optional[Tuple], int]:
    """
    Try matching current frame against last N frames in buffer
    Returns: ((pts_prev, pts_cur, M), ref_idx) or (None, -1)
    """
    best_match = None
    best_idx = -1
    best_score = 0

    # Try last N frames
    search_range = min(len(frame_buffer), cfg.search_window)

    for i in range(search_range):
        ref_frame = frame_buffer[-(i + 1)]  # Start from most recent

        match = _match_sift(ref_frame.img, cur_img, cfg)
        if match is None:
            continue

        pts_prev, pts_cur = match

        # Estimate transformation and score based on inliers
        M, inliers = cv2.estimateAffinePartial2D(
            pts_cur, pts_prev,
            method=cv2.RANSAC,
            ransacReprojThreshold=cfg.affine_ransac_thresh_px,
        )

        if M is not None and inliers is not None:
            score = int(inliers.sum())
            if score > best_score and score >= cfg.min_inliers:
                best_score = score
                best_match = (pts_prev, pts_cur, M)
                best_idx = ref_frame.idx

    return best_match, best_idx


def _blend_images(canvas: np.ndarray, cur_warp: np.ndarray) -> np.ndarray:
    """
    Blend images with distance-based feathering for smooth transitions
    """
    h, w = canvas.shape[:2]

    mask_cur = (cv2.cvtColor(cur_warp, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    mask_canvas = (cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

    # Distance transform for smooth blending weights
    dist_cur = cv2.distanceTransform(mask_cur, cv2.DIST_L2, 5)
    dist_cur = cv2.normalize(dist_cur, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    dist_canvas = cv2.distanceTransform(mask_canvas, cv2.DIST_L2, 5)
    dist_canvas = cv2.normalize(dist_canvas, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # Calculate blending weights in overlapping regions
    overlap = (mask_cur & mask_canvas).astype(bool)
    alpha = np.zeros((h, w, 1), dtype=np.float32)

    # Blend based on distance from edges
    denominator = dist_cur[overlap] + dist_canvas[overlap] + 1e-6
    alpha[overlap] = (dist_cur[overlap] / denominator).reshape(-1, 1)
    alpha = np.repeat(alpha, 3, axis=2)

    # Perform blending
    result = canvas.astype(np.float32) * (1 - alpha) + cur_warp.astype(np.float32) * alpha

    # Where only cur exists, use cur directly
    only_cur = mask_cur[:, :, None].astype(bool) & ~mask_canvas[:, :, None].astype(bool)
    result = np.where(only_cur, cur_warp, result)

    return result.astype(np.uint8)


def stitch_sequence(metas: List[FrameMeta], out_path: str | Path, cfg: StitchConfig) -> None:
    """
    SIFT-based robust stitching with multi-frame matching
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _resize_if_needed(img):
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= cfg.max_input_dim:
            return img
        s = cfg.max_input_dim / float(m)
        new_w = int(w * s)
        new_h = int(h * s)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Initialize with first frame
    base = cv2.imread(str(metas[0].path))
    if base is None:
        raise RuntimeError(f"Cannot read: {metas[0].path}")
    base = _resize_if_needed(base)

    canvas = base
    H_canvas_identity = np.eye(3, dtype=np.float64)

    # Buffer to store recent frames with their transformations
    frame_buffer: List[FrameInfo] = [
        FrameInfo(img=base, idx=0, H_to_canvas=H_canvas_identity.copy())
    ]

    successful_stitches = 0
    skipped_frames = []

    for i in range(1, len(metas)):
        print(f"Processing frame {i}/{len(metas) - 1}...", end='\r')

        cur = cv2.imread(str(metas[i].path))
        if cur is None:
            skipped_frames.append(i)
            continue
        cur = _resize_if_needed(cur)

        # Try to match against multiple reference frames
        match_result, ref_idx = _find_best_match_visual(frame_buffer, cur, cfg)

        if match_result is None:
            skipped_frames.append(i)
            continue

        pts_prev, pts_cur, M = match_result

        # Find the reference frame's transformation to canvas
        H_ref = next(f.H_to_canvas for f in frame_buffer if f.idx == ref_idx)

        # Compute cur → canvas transformation
        A = np.vstack([M, [0, 0, 1]])  # cur → ref
        H_cur_to_canvas = H_ref @ A  # (cur → ref) @ (ref → canvas)

        # --- Compute new canvas bounds ---
        h1, w1 = canvas.shape[:2]
        h2, w2 = cur.shape[:2]

        # Transform current frame corners to canvas space
        corners_cur = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        warped_corners = cv2.transform(corners_cur, H_cur_to_canvas[:2])

        # Existing canvas corners
        corners_canvas = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)

        # Find bounding box
        all_pts = np.concatenate([corners_canvas, warped_corners], axis=0)
        xmin, ymin = np.floor(all_pts.min(axis=0).ravel()).astype(int)
        xmax, ymax = np.ceil(all_pts.max(axis=0).ravel()).astype(int)

        # Translation to ensure positive coordinates
        tx = -xmin if xmin < 0 else 0
        ty = -ymin if ymin < 0 else 0

        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
        new_w = int(xmax - xmin)
        new_h = int(ymax - ymin)

        # Safety check
        if new_w > cfg.hard_canvas_dim or new_h > cfg.hard_canvas_dim or new_w <= 0 or new_h <= 0:
            skipped_frames.append(i)
            continue

        # Translate canvas to new coordinate system
        new_canvas = cv2.warpAffine(canvas, T[:2], (new_w, new_h))

        # Warp current frame into new canvas space
        H_cur_to_new_canvas = T @ H_cur_to_canvas
        cur_warp = cv2.warpAffine(cur, H_cur_to_new_canvas[:2], (new_w, new_h))

        # Blend images
        new_canvas = _blend_images(new_canvas, cur_warp)
        canvas = new_canvas

        # Update transformation for next iteration
        H_updated = H_cur_to_new_canvas

        # Add current frame to buffer (keep last N frames)
        frame_buffer.append(FrameInfo(img=cur, idx=i, H_to_canvas=H_updated))
        if len(frame_buffer) > cfg.search_window:
            frame_buffer.pop(0)

        # Update all remaining frames' transformations in buffer to account for T
        for frame_info in frame_buffer[:-1]:  # Exclude the one we just added
            frame_info.H_to_canvas = T @ frame_info.H_to_canvas

        # Downscale if canvas gets too large
        ch, cw = canvas.shape[:2]
        cm = max(ch, cw)
        if cm > cfg.max_canvas_dim:
            s2 = cfg.max_canvas_dim / float(cm)
            new_cw = int(cw * s2)
            new_ch = int(ch * s2)
            canvas = cv2.resize(canvas, (new_cw, new_ch), interpolation=cv2.INTER_AREA)

            S = np.array([[s2, 0, 0], [0, s2, 0], [0, 0, 1]], dtype=np.float64)

            # Update all transformations in buffer
            for frame_info in frame_buffer:
                frame_info.H_to_canvas = S @ frame_info.H_to_canvas

        successful_stitches += 1

    print(f"\n✓ Successfully stitched {successful_stitches}/{len(metas) - 1} frames")
    if skipped_frames:
        print(
            f"⚠ Skipped {len(skipped_frames)} frames: {skipped_frames[:10]}{'...' if len(skipped_frames) > 10 else ''}")

    cv2.imwrite(str(out_path), canvas)