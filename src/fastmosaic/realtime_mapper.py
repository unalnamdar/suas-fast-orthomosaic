"""
Real-time mapping from RTSP camera stream (SIYI A8)
Based on the existing fastmosaic structure
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque

import cv2
import numpy as np


@dataclass
class RealtimeConfig:
    """Configuration for real-time mapping"""
    # RTSP connection
    rtsp_url: str = "rtsp://192.168.144.25:8554/main.264"
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0

    # Frame processing
    process_every_n_frames: int = 10  # Process every 10th frame (was 5)
    resize_width: int = 480  # Smaller = faster (was 640)

    # Feature matching (ORB for speed)
    orb_features: int = 1000  # Fewer features = faster (was 1500)
    ratio_test: float = 0.75
    min_inliers: int = 15  # Lower threshold (was 20)
    ransac_thresh: float = 5.0  # More tolerant (was 4.0)

    # Canvas management
    max_canvas_dim: int = 3000  # Limit canvas size (was 4000)

    # Display
    show_preview: bool = True
    preview_scale: float = 0.6  # Reasonable window size (was 10.0!!!)

    # Frame buffer for matching
    buffer_size: int = 3  # Smaller buffer = less processing (was 5)


class RealtimeMapper:
    """Real-time mosaic builder from RTSP stream"""

    def __init__(self, config: RealtimeConfig):
        self.cfg = config
        self.canvas = None
        self.H_canvas = np.eye(3, dtype=np.float64)
        self.frame_buffer = deque(maxlen=config.buffer_size)

        # Statistics
        self.frame_count = 0
        self.processed_count = 0
        self.stitched_count = 0
        self.fps = 0.0
        self.last_time = time.time()

        # ORB detector (reuse for efficiency)
        self.orb = cv2.ORB_create(nfeatures=self.cfg.orb_features)

    def connect_stream(self) -> cv2.VideoCapture:
        """Connect to RTSP stream with retry logic"""
        print(f"Connecting to {self.cfg.rtsp_url}...")

        for attempt in range(self.cfg.reconnect_attempts):
            cap = cv2.VideoCapture(self.cfg.rtsp_url, cv2.CAP_FFMPEG)

            # Set buffer size to 1 for lowest latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                print("✓ Connected to camera stream")
                # Read a test frame
                ret, frame = cap.read()
                if ret:
                    print(f"  Stream resolution: {frame.shape[1]}x{frame.shape[0]}")
                    return cap

            print(f"  Attempt {attempt + 1}/{self.cfg.reconnect_attempts} failed")
            cap.release()
            time.sleep(self.cfg.reconnect_delay)

        raise RuntimeError("Failed to connect to RTSP stream")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize and prepare frame for processing"""
        h, w = frame.shape[:2]
        if w > self.cfg.resize_width:
            scale = self.cfg.resize_width / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (self.cfg.resize_width, new_h))
        return frame

    def match_frame(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Match current frame against buffer
        Returns: (pts_prev, pts_cur, M) or None
        """
        if len(self.frame_buffer) == 0:
            return None

        # Detect features in current frame
        kp_cur, desc_cur = self.orb.detectAndCompute(frame, None)
        if desc_cur is None or len(kp_cur) < 20:
            return None

        # Try matching against recent frames (most recent first)
        best_match = None
        best_score = 0

        for prev_frame, prev_kp, prev_desc in reversed(list(self.frame_buffer)):
            if prev_desc is None:
                continue

            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(prev_desc, desc_cur, k=2)

            # Ratio test
            good = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.cfg.ratio_test * n.distance:
                        good.append(m)

            if len(good) < 20:
                continue

            # Extract points
            pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_cur = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Estimate transformation
            M, inliers = cv2.estimateAffinePartial2D(
                pts_cur, pts_prev,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.ransac_thresh
            )

            if M is not None and inliers is not None:
                score = int(inliers.sum())
                if score > best_score and score >= self.cfg.min_inliers:
                    best_score = score
                    best_match = (pts_prev, pts_cur, M)

        return best_match

    def stitch_frame(self, frame: np.ndarray) -> bool:
        """
        Stitch frame into canvas
        Returns: True if stitched successfully
        """
        if self.canvas is None:
            # Initialize canvas with first frame
            self.canvas = frame.copy()
            self.H_canvas = np.eye(3, dtype=np.float64)
            return True

        # Match frame
        match_result = self.match_frame(frame)
        if match_result is None:
            return False

        pts_prev, pts_cur, M = match_result

        # Update transformation
        A = np.vstack([M, [0, 0, 1]])
        H_cur_to_canvas = self.H_canvas @ A

        # Compute new canvas bounds
        h1, w1 = self.canvas.shape[:2]
        h2, w2 = frame.shape[:2]

        corners_cur = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        warped_corners = cv2.transform(corners_cur, H_cur_to_canvas[:2])

        corners_canvas = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        all_pts = np.concatenate([corners_canvas, warped_corners], axis=0)

        xmin, ymin = np.floor(all_pts.min(axis=0).ravel()).astype(int)
        xmax, ymax = np.ceil(all_pts.max(axis=0).ravel()).astype(int)

        tx = -xmin if xmin < 0 else 0
        ty = -ymin if ymin < 0 else 0

        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
        new_w = int(xmax - xmin)
        new_h = int(ymax - ymin)

        # Safety check
        if new_w > self.cfg.max_canvas_dim or new_h > self.cfg.max_canvas_dim:
            print("⚠ Canvas too large, resetting...")
            self.canvas = frame.copy()
            self.H_canvas = np.eye(3, dtype=np.float64)
            return True

        # Warp and blend
        new_canvas = cv2.warpAffine(self.canvas, T[:2], (new_w, new_h))
        H_cur_to_new_canvas = T @ H_cur_to_canvas
        cur_warp = cv2.warpAffine(frame, H_cur_to_new_canvas[:2], (new_w, new_h))

        # Simple alpha blending in overlap regions
        mask_cur = (cv2.cvtColor(cur_warp, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        mask_canvas = (cv2.cvtColor(new_canvas, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        overlap = (mask_cur & mask_canvas).astype(bool)

        # Blend with 50/50 in overlaps, otherwise take new content
        new_canvas[overlap] = (new_canvas[overlap].astype(float) * 0.5 +
                               cur_warp[overlap].astype(float) * 0.5).astype(np.uint8)
        new_canvas[mask_cur & ~mask_canvas] = cur_warp[mask_cur & ~mask_canvas]

        self.canvas = new_canvas
        self.H_canvas = H_cur_to_new_canvas

        return True

    def update_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed > 0:
            self.fps = 1.0 / elapsed
        self.last_time = current_time

    def create_preview(self) -> np.ndarray:
        """Create preview window with canvas and stats"""
        if self.canvas is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Scale canvas for preview
        h, w = self.canvas.shape[:2]
        scale = self.cfg.preview_scale
        preview_w = int(w * scale)
        preview_h = int(h * scale)

        if preview_w > 0 and preview_h > 0:
            preview = cv2.resize(self.canvas, (preview_w, preview_h))
        else:
            preview = self.canvas.copy()

        # Add statistics overlay
        stats_text = [
            f"FPS: {self.fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Processed: {self.processed_count}",
            f"Stitched: {self.stitched_count}",
            f"Canvas: {self.canvas.shape[1]}x{self.canvas.shape[0]}"
        ]

        y_offset = 30
        for i, text in enumerate(stats_text):
            cv2.putText(preview, text, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return preview

    def run(self, duration: Optional[float] = None, save_path: Optional[str] = None):
        """
        Main loop for real-time mapping

        Args:
            duration: Run for specified seconds (None = run until 'q' pressed)
            save_path: Path to save final mosaic
        """
        cap = self.connect_stream()

        start_time = time.time()

        try:
            print("\nReal-time mapping started...")
            print("Press 'q' to quit, 's' to save current mosaic")

            while True:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    print(f"\n✓ Duration limit ({duration}s) reached")
                    break

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("\n⚠ Lost connection, attempting to reconnect...")
                    cap.release()
                    time.sleep(1)
                    cap = self.connect_stream()
                    continue

                self.frame_count += 1
                self.update_fps()

                # Process every Nth frame
                if self.frame_count % self.cfg.process_every_n_frames != 0:
                    continue

                self.processed_count += 1

                # Preprocess
                processed = self.preprocess_frame(frame)

                # Detect features for buffer
                kp, desc = self.orb.detectAndCompute(processed, None)
                self.frame_buffer.append((processed.copy(), kp, desc))

                # Try to stitch
                if self.stitch_frame(processed):
                    self.stitched_count += 1

                # Display preview
                if self.cfg.show_preview:
                    preview = self.create_preview()
                    cv2.imshow("Real-time Mosaic", preview)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n✓ User quit")
                        break
                    elif key == ord('s'):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"mosaic_{timestamp}.png"
                        cv2.imwrite(filename, self.canvas)
                        print(f"\n✓ Saved mosaic to {filename}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            # Save final mosaic
            if save_path and self.canvas is not None:
                cv2.imwrite(save_path, self.canvas)
                print(f"\n✓ Final mosaic saved to {save_path}")

            # Print statistics
            print("\n=== Mapping Statistics ===")
            print(f"Total frames received: {self.frame_count}")
            print(f"Frames processed: {self.processed_count}")
            print(f"Frames stitched: {self.stitched_count}")
            print(f"Success rate: {100 * self.stitched_count / max(1, self.processed_count):.1f}%")
            if self.canvas is not None:
                print(f"Final mosaic size: {self.canvas.shape[1]}x{self.canvas.shape[0]}")


def main():
    """Demo entry point"""
    config = RealtimeConfig(
        rtsp_url="rtsp://192.168.144.25:8554/main.264",
        process_every_n_frames=3,  # Process every 3rd frame
        resize_width=800,
        orb_features=2000,
        show_preview=True,
        preview_scale=0.5
    )

    mapper = RealtimeMapper(config)
    mapper.run(
        duration=None,  # Run until user quits
        save_path="data/outputs/realtime_mosaic.png"
    )


if __name__ == "__main__":
    main()