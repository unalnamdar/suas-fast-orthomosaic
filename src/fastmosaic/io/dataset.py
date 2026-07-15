from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import csv

@dataclass
class FrameMeta:
    frame_id: str
    path: Path
    timestamp_ns: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_m: Optional[float] = None
    yaw_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None


def _read_telemetry_csv(csv_path: Path) -> dict[str, dict]:
    """
    Expected columns (flexible):
    - frame_id (or frame)
    - timestamp_ns (optional)
    - lat, lon, alt_m, yaw_deg, roll_deg, pitch_deg (optional)
    """
    if not csv_path.exists():
        return {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = {}
        for r in reader:
            fid = (r.get("frame_id") or r.get("frame") or "").strip()
            if not fid:
                continue
            def to_float(x):
                try: return float(x)
                except: return None
            def to_int(x):
                try: return int(float(x))
                except: return None

            rows[fid] = {
                "timestamp_ns": to_int(r.get("timestamp_ns")),
                "lat": to_float(r.get("lat")),
                "lon": to_float(r.get("lon")),
                "alt_m": to_float(r.get("alt_m") or r.get("alt")),
                "yaw_deg": to_float(r.get("yaw_deg") or r.get("yaw")),
                "roll_deg": to_float(r.get("roll_deg") or r.get("roll")),
                "pitch_deg": to_float(r.get("pitch_deg") or r.get("pitch")),
            }
        return rows


def load_dataset(input_dir: str | Path) -> List[FrameMeta]:
    """
    Dataset layout:
      input_dir/
        frames/ frame_000001.jpg ...
        telemetry.csv (optional)
    """
    input_dir = Path(input_dir)
    frames_dir = input_dir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames/ folder not found: {frames_dir}")

    # Collect images
    img_paths = sorted([*frames_dir.glob("*.jpg"), *frames_dir.glob("*.png"), *frames_dir.glob("*.jpeg")])
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {frames_dir}")

    telemetry = _read_telemetry_csv(input_dir / "telemetry.csv")

    metas: List[FrameMeta] = []
    for p in img_paths:
        # expected: frame_000001.jpg -> 000001
        stem = p.stem
        fid = stem.split("_")[-1] if "_" in stem else stem
        meta = FrameMeta(frame_id=fid, path=p)

        t = telemetry.get(fid)
        if t:
            meta.timestamp_ns = t.get("timestamp_ns")
            meta.lat = t.get("lat")
            meta.lon = t.get("lon")
            meta.alt_m = t.get("alt_m")
            meta.yaw_deg = t.get("yaw_deg")
            meta.roll_deg = t.get("roll_deg")
            meta.pitch_deg = t.get("pitch_deg")

        metas.append(meta)

    return metas
