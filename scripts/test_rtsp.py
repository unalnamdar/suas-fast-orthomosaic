"""
Simple script to test RTSP connection to SIYI A8 camera
Run this first to verify the camera is accessible
"""

import cv2
import time


def test_rtsp_connection(url: str = "rtsp://192.168.144.25:8554/main.264"):
    """Test basic RTSP connection and display stream"""

    print(f"Testing connection to: {url}")
    print("Make sure:")
    print("  1. Camera is powered on")
    print("  2. You're connected to camera's WiFi network")
    print("  3. Camera IP is 192.168.144.25")
    print()

    # Try to connect
    print("Connecting...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("❌ Failed to connect!")
        print("\nTroubleshooting:")
        print("  - Ping the camera: ping 192.168.144.25")
        print("  - Check camera is streaming")
        print("  - Try VLC: vlc " + url)
        return False

    print("✓ Connected!")

    # Read and display frames
    frame_count = 0
    start_time = time.time()

    print("\nStreaming video... (Press 'q' to quit)")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Lost connection or end of stream")
            break

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Add info overlay
        h, w = frame.shape[:2]
        info_text = [
            f"Resolution: {w}x{h}",
            f"FPS: {fps:.1f}",
            f"Frames: {frame_count}"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("RTSP Stream Test", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✓ Test complete")
    print(f"  Total frames: {frame_count}")
    print(f"  Average FPS: {fps:.1f}")
    print(f"  Resolution: {w}x{h}")

    return True


if __name__ == "__main__":
    import sys

    # Allow custom URL as argument
    url = sys.argv[1] if len(sys.argv) > 1 else "rtsp://192.168.144.25:8554/main.264"

    test_rtsp_connection(url)