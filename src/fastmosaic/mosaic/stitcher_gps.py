"""
GPS-ordered vision stitcher.

GPS / CSV is used ONLY to:
  1. Match image filenames to waypoint labels
  2. Sort frames in correct flight order
  3. Filter out frames taken during sharp manoeuvres

Actual image placement is done by ORB feature matching —
the same proven logic as stitcher.py, with multi-frame search.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GpsStitchConfig:
    # Frame filtering
    max_heading_change_deg: float = 30.0   # skip frames where drone was turning hard
    min_inliers: int = 20                  # minimum ORB inliers to accept a match

    # ORB matching
    orb_features: int = 4000
    ratio_test: float = 0.72
    ransac_thresh: float = 5.0
    use_clahe: bool = True

    # Multi-frame search window (look back N frames if current match is weak)
    search_window: int = 5

    # Canvas
    max_input_dim: int = 1200              # resize input images if larger
    max_canvas_dim: int = 16000

    # Blending
    blend_feather: bool = True


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GpsFrame:
    path: Path
    lat: float
    lon: float
    alt: float
    heading: float
    timestamp: str = ""


# ---------------------------------------------------------------------------
# CSV loader — GPS used only for ordering & filtering
# ---------------------------------------------------------------------------

def load_gps_frames_from_csv(frames_dir: str | Path,
                              csv_path: str | Path,
                              cfg: GpsStitchConfig) -> List[GpsFrame]:
    frames_dir = Path(frames_dir)
    csv_path   = Path(csv_path)

    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    # Build label -> image path map
    img_map: dict[str, Path] = {}
    for p in frames_dir.iterdir():
        if p.suffix.lower() not in img_exts:
            continue
        parts = p.stem.split("_")
        if len(parts) >= 2:
            label = f"{parts[0]}_{parts[1]}"   # e.g. "wp_002"
            img_map[label.lower()] = p

    frames: List[GpsFrame] = []
    missing: List[str] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = row.get("event", "").strip()
            if "PHOTO" not in event.upper():
                continue

            parts = event.split()
            label = parts[-1].lower() if len(parts) >= 2 else ""
            img_path = img_map.get(label)

            if img_path is None:
                missing.append(label)
                continue

            frames.append(GpsFrame(
                path      = img_path,
                lat       = float(row["lat"]),
                lon       = float(row["lon"]),
                alt       = float(row["alt_m"]),
                heading   = float(row["heading_deg"]),
                timestamp = row.get("timestamp", ""),
            ))

    # Sort by timestamp (flight order)
    frames.sort(key=lambda f: f.timestamp)

    if missing:
        print(f"  [WARN] No image found for: {missing}")
    print(f"  Found {len(frames)} frames from CSV")

    # Filter out frames taken during sharp turns
    filtered: List[GpsFrame] = [frames[0]]
    skipped = 0
    for i in range(1, len(frames)):
        prev_h = frames[i - 1].heading % 360
        curr_h = frames[i].heading % 360
        diff   = abs(curr_h - prev_h)
        if diff > 180:
            diff = 360 - diff
        if diff > cfg.max_heading_change_deg:
            print(f"  [SKIP] {frames[i].path.name}  "
                  f"heading jump {diff:.1f}° > {cfg.max_heading_change_deg}°")
            skipped += 1
        else:
            filtered.append(frames[i])

    print(f"  Kept {len(filtered)} frames  (skipped {skipped} manoeuvre frames)")
    return filtered


# ---------------------------------------------------------------------------
# Fallback JSON loader
# ---------------------------------------------------------------------------

def load_gps_frames(frames_dir: str | Path,
                    waypoints_dir: str | Path,
                    cfg: GpsStitchConfig) -> List[GpsFrame]:
    import json
    frames_dir    = Path(frames_dir)
    waypoints_dir = Path(waypoints_dir)

    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    imgs  = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in img_exts)
    jsons = {p.stem: p for p in waypoints_dir.iterdir() if p.suffix == ".json"}

    frames: List[GpsFrame] = []
    for img_path in imgs:
        parts  = img_path.stem.split("_")
        prefix = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else img_path.stem
        matched = next((v for k, v in jsons.items() if k.startswith(prefix)), None)
        if matched is None:
            print(f"  [WARN] No JSON for {img_path.name} — skipping")
            continue
        with open(matched) as f:
            meta = json.load(f)
        frames.append(GpsFrame(
            path      = img_path,
            lat       = float(meta.get("latitude",  meta.get("lat",  0))),
            lon       = float(meta.get("longitude", meta.get("lon",  0))),
            alt       = float(meta.get("altitude",  meta.get("alt", 20))),
            heading   = float(meta.get("heading",   meta.get("yaw",  0))),
            timestamp = str(meta.get("timestamp", "")),
        ))

    frames.sort(key=lambda f: f.timestamp or str(f.path))
    print(f"  Loaded {len(frames)} frames (JSON mode)")
    return frames


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_if_needed(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# ORB feature matching
# ---------------------------------------------------------------------------

def _match_orb(img1: np.ndarray, img2: np.ndarray,
               cfg: GpsStitchConfig) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns (pts_in_img1, pts_in_img2, affine_M) or None.
    """
    def gray(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if cfg.use_clahe:
            g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        return g

    orb = cv2.ORB_create(nfeatures=cfg.orb_features)
    k1, d1 = orb.detectAndCompute(gray(img1), None)
    k2, d2 = orb.detectAndCompute(gray(img2), None)

    if d1 is None or d2 is None or len(k1) < 15 or len(k2) < 15:
        return None

    bf   = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw  = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < cfg.ratio_test * n.distance]

    if len(good) < cfg.min_inliers:
        return None

    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(
        pts2, pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_thresh,
    )
    if M is None or inliers is None or int(inliers.sum()) < cfg.min_inliers:
        return None

    return pts1, pts2, M


# ---------------------------------------------------------------------------
# Canvas expansion helper
# ---------------------------------------------------------------------------

def _expand_canvas(canvas: np.ndarray,
                   H_canvas: np.ndarray,
                   cur: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand canvas to fit the new warped frame.
    Returns (new_canvas, T_3x3, H_cur_to_new_canvas).
    """
    ch, cw = canvas.shape[:2]
    fh, fw = cur.shape[:2]

    # Corners of cur in canvas coords
    corners = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]).reshape(-1, 1, 2)
    warped_corners = cv2.transform(corners, H_canvas[:2])

    all_x = np.concatenate([[0, cw], warped_corners[:, 0, 0]])
    all_y = np.concatenate([[0, ch], warped_corners[:, 0, 1]])

    xmin, xmax = int(math.floor(all_x.min())), int(math.ceil(all_x.max()))
    ymin, ymax = int(math.floor(all_y.min())), int(math.ceil(all_y.max()))

    tx = max(0, -xmin)
    ty = max(0, -ymin)

    new_w = min(xmax - xmin + tx, 16000)
    new_h = min(ymax - ymin + ty, 16000)

    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    new_canvas = cv2.warpAffine(canvas, T[:2], (new_w, new_h))

    return new_canvas, T


# ---------------------------------------------------------------------------
# Feather blending
# ---------------------------------------------------------------------------

def _feather_blend(canvas: np.ndarray, patch: np.ndarray,
                   x0: int, y0: int) -> np.ndarray:
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]

    x1 = min(x0 + pw, cw)
    y1 = min(y0 + ph, ch)
    rx, ry = x1 - x0, y1 - y0
    if rx <= 0 or ry <= 0:
        return canvas

    p_roi = patch[:ry, :rx]
    c_roi = canvas[y0:y1, x0:x1]

    m_new = (cv2.cvtColor(p_roi, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    m_old = (cv2.cvtColor(c_roi, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    overlap = (m_new & m_old).astype(bool)

    if not overlap.any():
        canvas[y0:y1, x0:x1] = np.where(m_new[:, :, None], p_roi, c_roi)
        return canvas

    d_new = cv2.distanceTransform(m_new, cv2.DIST_L2, 5)
    d_old = cv2.distanceTransform(m_old, cv2.DIST_L2, 5)
    alpha = np.zeros((ry, rx), dtype=np.float32)
    denom = d_new + d_old + 1e-6
    alpha[overlap] = d_new[overlap] / denom[overlap]
    alpha = alpha[:, :, None]

    blended  = (c_roi.astype(np.float32) * (1 - alpha) +
                p_roi.astype(np.float32) * alpha).astype(np.uint8)
    only_new = m_new & ~m_old
    canvas[y0:y1, x0:x1] = np.where(only_new[:, :, None], p_roi, blended)
    return canvas


# ---------------------------------------------------------------------------
# Main stitcher
# ---------------------------------------------------------------------------

def stitch_gps(frames_dir:    str | Path,
               waypoints_dir: str | Path,
               out_path:      str | Path,
               cfg:           Optional[GpsStitchConfig] = None,
               csv_path:      Optional[str | Path] = None) -> None:
    """
    GPS orders the frames; ORB vision places them on the canvas.
    """
    cfg      = cfg or GpsStitchConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load & order frames via GPS/CSV
    if csv_path is not None:
        frames = load_gps_frames_from_csv(frames_dir, csv_path, cfg)
    else:
        frames = load_gps_frames(frames_dir, waypoints_dir, cfg)

    if len(frames) < 2:
        raise ValueError("Need at least 2 frames after filtering")

    # 2. Read base frame
    base = cv2.imread(str(frames[0].path))
    if base is None:
        raise RuntimeError(f"Cannot read base image: {frames[0].path}")
    base = _resize_if_needed(base, cfg.max_input_dim)

    canvas   = base.copy()
    H_canvas = np.eye(3, dtype=np.float64)   # current frame -> canvas coords

    # Frame buffer for multi-frame search
    # Each entry: (image, H_to_canvas)
    buffer: List[Tuple[np.ndarray, np.ndarray]] = [(base, H_canvas.copy())]

    successful = 0
    skipped    = 0

    for i in range(1, len(frames)):
        cur = cv2.imread(str(frames[i].path))
        if cur is None:
            print(f"  [{i+1:02d}/{len(frames)}] WARN: cannot read {frames[i].path.name}")
            skipped += 1
            continue

        cur = _resize_if_needed(cur, cfg.max_input_dim)

        # Try matching against recent frames in buffer (best match wins)
        best_result = None
        best_inliers = 0
        best_ref_H   = None

        search = buffer[-cfg.search_window:]
        for ref_img, ref_H in reversed(search):
            result = _match_orb(ref_img, cur, cfg)
            if result is None:
                continue
            _, _, M = result
            # Count inliers by re-running RANSAC (already done inside _match_orb)
            # Use translation magnitude as proxy for quality
            n_inliers = cfg.min_inliers  # already filtered above
            if n_inliers >= best_inliers:
                best_inliers = n_inliers
                best_result  = M
                best_ref_H   = ref_H
                break  # take first (most recent) good match

        if best_result is None or best_ref_H is None:
            print(f"  [{i+1:02d}/{len(frames)}] {frames[i].path.name}  "
                  f"⚠ no vision match — skipped")
            skipped += 1
            continue

        M = best_result
        A = np.vstack([M, [0, 0, 1]])          # cur -> ref
        H_cur_to_canvas = best_ref_H @ A       # cur -> canvas

        # Expand canvas to fit new frame
        ch, cw = canvas.shape[:2]
        fh, fw = cur.shape[:2]
        corners = np.float32([[0,0],[fw,0],[fw,fh],[0,fh]]).reshape(-1,1,2)
        wc = cv2.transform(corners, H_cur_to_canvas[:2])

        all_x = np.concatenate([[0, cw], wc[:,0,0]])
        all_y = np.concatenate([[0, ch], wc[:,0,1]])
        xmin  = int(math.floor(all_x.min()))
        ymin  = int(math.floor(all_y.min()))
        xmax  = int(math.ceil(all_x.max()))
        ymax  = int(math.ceil(all_y.max()))

        tx = max(0, -xmin)
        ty = max(0, -ymin)
        new_w = min(xmax + tx, cfg.max_canvas_dim)
        new_h = min(ymax + ty, cfg.max_canvas_dim)

        T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)
        new_canvas = cv2.warpAffine(canvas, T[:2], (new_w, new_h))

        H_cur_to_new = T @ H_cur_to_canvas
        cur_warp = cv2.warpAffine(cur, H_cur_to_new[:2], (new_w, new_h))

        if cfg.blend_feather:
            # Use feather blend via mask
            mask = (cv2.cvtColor(cur_warp, cv2.COLOR_BGR2GRAY) > 0)
            m_new = mask.astype(np.uint8)
            m_old = (cv2.cvtColor(new_canvas, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
            overlap = (m_new & m_old).astype(bool)

            if overlap.any():
                d_new = cv2.distanceTransform(m_new, cv2.DIST_L2, 5)
                d_old = cv2.distanceTransform(m_old, cv2.DIST_L2, 5)
                alpha = np.zeros((new_h, new_w), dtype=np.float32)
                denom = d_new + d_old + 1e-6
                alpha[overlap] = d_new[overlap] / denom[overlap]
                alpha = alpha[:, :, None]
                blended = (new_canvas.astype(np.float32) * (1 - alpha) +
                           cur_warp.astype(np.float32) * alpha).astype(np.uint8)
                only_new = m_new & ~m_old
                new_canvas = np.where(only_new[:,:,None], cur_warp, blended)
            else:
                new_canvas = np.where(mask[:,:,None], cur_warp, new_canvas)
        else:
            mask = (cv2.cvtColor(cur_warp, cv2.COLOR_BGR2GRAY) > 0)
            new_canvas = np.where(mask[:,:,None], cur_warp, new_canvas)

        canvas   = new_canvas.astype(np.uint8)
        H_canvas = H_cur_to_new   # next frame matches against this one

        # Update buffer
        buffer.append((cur, H_cur_to_new))
        if len(buffer) > cfg.search_window:
            buffer.pop(0)

        successful += 1
        tx_mag = math.hypot(M[0,2], M[1,2])
        print(f"  [{i+1:02d}/{len(frames)}] {frames[i].path.name}  "
              f"✓ stitched  shift={tx_mag:.1f}px  "
              f"canvas={canvas.shape[1]}x{canvas.shape[0]}")

    print(f"\n  Summary: {successful} stitched, {skipped} skipped")

    # Trim black border
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        canvas = canvas[y:y+h, x:x+w]
        print(f"  Trimmed to {w}x{h} px")

    cv2.imwrite(str(out_path), canvas)
    print(f"✓ Saved -> {out_path}")