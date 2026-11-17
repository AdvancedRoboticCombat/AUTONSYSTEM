#!/usr/bin/env python3
"""Headless camera check: list GigE cameras, then grab frames for a few seconds
through the same GStreamer pipeline the dashboard backend uses.

    scripts/camera.sh --check [seconds]
"""
import re
import subprocess
import sys
import time

import cv2

sys.path.insert(0, "nemesis_dashboard")

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0


def main():
    if not re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()):
        sys.exit("FAIL  OpenCV has no GStreamer support (run scripts/setup_env.sh)")

    try:
        found = subprocess.run(["arv-tool-0.8"], capture_output=True, text=True, timeout=15)
        listing = (found.stdout + found.stderr).strip()
    except FileNotFoundError:
        sys.exit("FAIL  arv-tool-0.8 not found (sudo apt install aravis-tools)")
    print("Cameras on the network:", listing or "(none)")
    if not listing or "No device found" in listing:
        sys.exit("FAIL  no camera found. Check PoE power and both Ethernet cables, "
                 "and give the camera about a minute after plugging it in.")

    from backend.config import VIDEO_SOURCE  # the backend's default pipeline

    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit("FAIL  could not open the camera stream:\n      " + VIDEO_SOURCE)
    frames, shape, start = 0, None, time.perf_counter()
    try:
        while time.perf_counter() - start < SECONDS:
            ok, frame = cap.read()
            if ok:
                frames += 1
                shape = frame.shape
    finally:
        cap.release()
    if not frames:
        sys.exit("FAIL  stream opened but no frames arrived")
    print(f"PASS  {frames} frames in {SECONDS:.0f} s = {frames / SECONDS:.1f} FPS at {shape[1]}x{shape[0]}")


if __name__ == "__main__":
    main()
