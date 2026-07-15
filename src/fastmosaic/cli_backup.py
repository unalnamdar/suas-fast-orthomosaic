import argparse

from fastmosaic.preprocess import preprocess_dataset
from fastmosaic.io.dataset import load_dataset
from fastmosaic.mosaic.stitcher import stitch_sequence, StitchConfig
from fastmosaic.mosaic import stitcher_sift

def main():
    parser = argparse.ArgumentParser(prog="fastmosaic")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------- preprocess --------
    p_pre = subparsers.add_parser(
        "preprocess",
        help="Prepare dataset (undistort, copy frames, telemetry)"
    )
    p_pre.add_argument("--in", dest="inp", default="data/raw")
    p_pre.add_argument("--processed", default="data/processed")
    p_pre.add_argument("--camera", default="configs/camera.yaml")

    # -------- run (ORB) --------
    p_run = subparsers.add_parser("run", help="ORB-based mosaic pipeline")
    p_run.add_argument("--in", dest="inp", default="data/processed")
    p_run.add_argument("--out", default="data/outputs/map_orb.png")

    # -------- run-sift --------
    p_sift = subparsers.add_parser("run-sift", help="SIFT-based mosaic pipeline")
    p_sift.add_argument("--in", dest="inp", default="data/processed")
    p_sift.add_argument("--out", default="data/outputs/map_sift.png")

    # -------- run-gps --------
    p_gps = subparsers.add_parser("run-gps", help="GPS-guided stitching using flight log CSV + images")
    p_gps.add_argument("--frames",    default="data/raw/gps_mission/frames",    help="Folder containing images")
    p_gps.add_argument("--waypoints", default="data/raw/gps_mission/waypoints", help="Folder with JSON files (fallback if no --csv)")
    p_gps.add_argument("--csv",       default=None,                             help="Path to flight log CSV (recommended)")
    p_gps.add_argument("--out",       default="data/outputs/map_gps.png",       help="Output map path")
    p_gps.add_argument("--res",       type=float, default=0.05,                 help="Ground resolution m/px (default 0.05)")
    p_gps.add_argument("--fov",       type=float, default=80.0,                 help="Camera horizontal FOV degrees (default 80)")
    p_gps.add_argument("--no-refine", action="store_true",                      help="GPS placement only, skip feature refinement")

    # -------- realtime --------
    p_rt = subparsers.add_parser("realtime", help="Real-time RTSP mapping (SIYI A8)")
    p_rt.add_argument("--url",        default="rtsp://192.168.144.25:8554/main.264")
    p_rt.add_argument("--out",        default="data/outputs/realtime_mosaic.png")
    p_rt.add_argument("--duration",   type=float, default=None)
    p_rt.add_argument("--fps",        type=int,   default=3)
    p_rt.add_argument("--no-preview", action="store_true")

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_dataset(raw_dir=args.inp, processed_dir=args.processed, camera_yaml=args.camera)
        print("✓ Preprocess done")
        return

    if args.command == "run-gps":
        from fastmosaic.mosaic.stitcher_gps import stitch_gps, GpsStitchConfig
        cfg = GpsStitchConfig()
        stitch_gps(
            frames_dir=args.frames,
            waypoints_dir=args.waypoints,
            out_path=args.out,
            cfg=cfg,
            csv_path=args.csv,
        )
        print("✓ GPS stitching done  →", args.out)
        return

    if args.command == "run-sift":
        metas = load_dataset(args.inp)
        if len(metas) < 2:
            raise SystemExit("Need at least 2 images.")
        cfg = stitcher_sift.StitchConfig(
            sift_features=5000, ratio_test=0.7, min_inliers=30,
            search_window=10, use_clahe=True, affine_ransac_thresh_px=5.0)
        stitcher_sift.stitch_sequence(metas, args.out, cfg)
        print("✓ SIFT stitching done  →", args.out)
        return

    if args.command == "run-gps":
        from fastmosaic.mosaic.stitcher_gps import stitch_gps, GpsStitchConfig
        cfg = GpsStitchConfig(
            ground_res_m_per_px=args.res,
            camera_hfov_deg=args.fov,
            refine_with_features=not args.no_refine,
        )
        stitch_gps(
            frames_dir=args.frames,
            waypoints_dir=args.waypoints,
            out_path=args.out,
            cfg=cfg,
            csv_path=args.csv,
        )
        print("✓ GPS stitching done  →", args.out)
        return

    if args.command == "realtime":
        from fastmosaic.realtime_mapper import RealtimeMapper, RealtimeConfig
        config = RealtimeConfig(
            rtsp_url=args.url,
            process_every_n_frames=args.fps,
            show_preview=not args.no_preview,
            resize_width=800, orb_features=2000, preview_scale=0.5)
        print(f"Starting real-time mapping from {args.url}")
        print("Press 'q' to quit, 's' to save intermediate mosaic")
        mapper = RealtimeMapper(config)
        mapper.run(duration=args.duration, save_path=args.out)
        return

if __name__ == "__main__":
    main()
