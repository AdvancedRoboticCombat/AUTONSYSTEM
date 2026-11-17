"""Settings of the dashboard backend.

The constants are the defaults; edit them here. Config carries one run's
values: main.py fills it from the command line and camera_setup.py from its
window. Nothing heavy is imported here, so scripts can read these cheaply.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

DASHBOARD_DIR = Path(__file__).resolve().parent.parent  # nemesis_dashboard/
DIST_DIR = DASHBOARD_DIR / "dist"                       # the built page (npm run build)
LOG_DIR = Path(os.environ.get("NEMESIS_LOG_DIR") or DASHBOARD_DIR.parent / "logs")

# =========================
# Camera and vision
# =========================
CAPTURE_FPS = 120  # camera-dependent; should MATCH OR EXCEED target VISION_FPS
VISION_FPS = 115   # detector FPS; BEST EFFORT; SHOULD NOT EXCEED CAPTURE_FPS!!! (0 = unlimited)

VIDEO_SOURCE = (
    "aravissrc camera-name=\"169.254.211.45\" ! "
    "video/x-bayer,format=rggb ! "
    "bayer2rgb ! videoconvert ! "
    "video/x-raw,format=RGBA ! glupload ! "
    "glcolorconvert ! gldownload ! "
    "video/x-raw,format=BGR ! "
    "appsink max-buffers=1 drop=true sync=false"
)

# Detector config (passed to the selected vision method via vision.get_detector).
# Two classes (robot, nemesis), so Nemesis is told from the opponent by class, not by position.
# It is ZACVISIONQUEST's all_fixes run, trained at 1280 px. The old one-class models
# (MediumBest.pt, MAYBEGRAYWORKS.pt, 640 px) are still in YOLO_models/.
VISION_MODEL_PATH = "YOLO_models/all_fixes_1280.pt"
VISION_MODEL_IMGSZ = 1280  # the size the model was trained at
VISION_MODE = "seg"  # "seg" (mask polygons) or "box"

# =========================
# Stream to the browser (independent of capture and vision)
# =========================
STREAM_FPS = 30      # most frames per second sent to the page
STREAM_WIDTH = 1280  # frames wider than this are shrunk before JPEG encoding; 0 = camera size
JPEG_QUALITY = 40

# =========================
# Fight-video mode (--fight-video, or the dashboard's "Run fight video" button):
# a recorded fight is played through the detector instead of the camera.
# =========================
FIGHT_VIDEO_PATH = "/home/arc-lab-pc/battle_gamma08_fight.mp4"
FIGHT_MODEL_PATH = "../ZACVISIONQUEST/runs/sweep/yolo26m-seg_1280_all_fixes/weights/chosen.pt"  # classes: robot, nemesis
FIGHT_MODEL_IMGSZ = 1280
# Arena points for that video: {"pixel_points": [[x, y], ...], "arena_points_ft": [[x, y], ...]}
FIGHT_HOMOGRAPHY_PATH = "fight_video_homography.json"

# =========================
# Homography (pixel <-> arena coordinates)
# TODO: refill these - cleared for now, to be re-derived later.
# =========================
DEFAULT_CLICKED_POINTS: List[Tuple[float, float]] = []  # pixel-space points seeding the default homography
HOMOG_DEST: List[Tuple[float, float]] = []              # arena-space destination points (planar)
HOMOG_DEST3D: List[Tuple[float, float]] = []            # arena-space destination points (3D pose)
NEMESIS_CENTER_HEIGHT = 1.0  # TODO

# =========================
# ROS and status
# =========================
FRAME_ID = "image"
WATCHDOG_TIMEOUT = 3.0    # TODO: tune this timeout amount
OVERRIDE_PUBLISH_HZ = 20  # rate of /nemesis/override (manual/auto switch)
STATUS_PRINT_S = 5.0      # seconds between FPS / latency lines in the terminal

# =========================
# Web server
# =========================
HOST = "127.0.0.1"  # this computer only: the page carries the manual/auto switch
PORT = 8080
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def resolve(path: str) -> str:
    """A path from this file, made absolute: relative ones start at nemesis_dashboard/."""
    return str(path if os.path.isabs(path) else DASHBOARD_DIR / path)


@dataclass
class Config:
    """The values of one run."""

    # Vision
    vision: str = "yolo"                         # vision method (see vision.VISION_METHODS)
    video_source: Optional[str] = VIDEO_SOURCE   # GStreamer pipeline; None = no camera
    capture_fps: float = CAPTURE_FPS
    vision_fps: float = VISION_FPS
    vision_model_path: str = VISION_MODEL_PATH
    vision_mode: str = VISION_MODE
    yolo_imgsz: Optional[int] = VISION_MODEL_IMGSZ  # None = the detector's default

    # Stream
    stream_fps: float = STREAM_FPS
    stream_width: int = STREAM_WIDTH
    jpeg_quality: int = JPEG_QUALITY

    # Fight video
    start_with_fight_video: bool = False
    fight_video_path: str = FIGHT_VIDEO_PATH
    fight_model_path: str = FIGHT_MODEL_PATH
    fight_model_imgsz: int = FIGHT_MODEL_IMGSZ
    fight_homography_path: str = FIGHT_HOMOGRAPHY_PATH

    # Recording
    record_dir: Optional[str] = None
    record_scale: float = 1.0

    # Web server
    host: str = HOST
    port: int = PORT
    allow_hosts: Tuple[str, ...] = ()            # extra host names the page may be opened under

    # ROS; start_manual True = the driver has the robot until OVERRIDE is pressed
    use_ros: bool = True
    start_manual: bool = field(default_factory=lambda: os.environ.get("NEMESIS_START_MANUAL") == "1")

    # Homography
    homog_dest: Sequence = field(default_factory=lambda: list(HOMOG_DEST))
    homog_dest_3d: Sequence = field(default_factory=lambda: list(HOMOG_DEST3D))
    default_clicked_points: Sequence = field(default_factory=lambda: list(DEFAULT_CLICKED_POINTS))
    nemesis_center_height: float = NEMESIS_CENTER_HEIGHT

    def allowed_hosts(self) -> frozenset:
        """Host names a request may be addressed to (see server.py)."""
        return frozenset(name.lower() for name in (*LOCAL_HOSTS, *self.allow_hosts))
