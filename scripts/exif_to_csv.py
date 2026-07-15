"""
Reads GPS EXIF from DJI drone images and creates flight_log_exif.csv
"""
import csv
from pathlib import Path
import piexif

def _to_decimal(dms, ref):
    d, m, s = dms
    decimal = d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600
    if ref in [b'S', b'W']:
        decimal = -decimal
    return decimal

def extract_exif_gps(frames_dir: str, out_csv: str):
    frames_dir = Path(frames_dir)
    imgs = sorted(p for p in frames_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg"})

    rows = []
    for img_path in imgs:
        try:
            exif = piexif.load(str(img_path))
            gps  = exif.get("GPS", {})

            lat = _to_decimal(
                gps[piexif.GPSIFD.GPSLatitude],
                gps[piexif.GPSIFD.GPSLatitudeRef])
            lon = _to_decimal(
                gps[piexif.GPSIFD.GPSLongitude],
                gps[piexif.GPSIFD.GPSLongitudeRef])

            alt_raw = gps.get(piexif.GPSIFD.GPSAltitude, ((0, 1),))
            alt     = alt_raw[0] / alt_raw[1]

            ts_raw = exif["Exif"].get(piexif.ExifIFD.DateTimeOriginal, b"")
            ts     = ts_raw.decode("utf-8", errors="ignore").replace(" ", "T")

            label = img_path.stem   # DJI_0001
            rows.append({
                "timestamp":   ts,
                "lat":         round(lat, 8),
                "lon":         round(lon, 8),
                "alt_m":       round(alt, 2),
                "heading_deg": 0.0,
                "event":       f"PHOTO {label}",
            })
            print(f"  ✓ {img_path.name}  lat={lat:.6f}  lon={lon:.6f}  alt={alt:.1f}m")

        except Exception as e:
            print(f"  ✗ {img_path.name}  — {e}")

    rows.sort(key=lambda r: r["timestamp"])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f,
            fieldnames=["timestamp","lat","lon","alt_m","heading_deg","event"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ {len(rows)} frames -> {out_csv}")

if __name__ == "__main__":
    extract_exif_gps(
        frames_dir="data/raw/gps_mission/frames",
        out_csv="data/raw/gps_mission/flight_log_exif.csv",
    )