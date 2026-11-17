"""Dashboard: one run of the backend. Owns the settings, the threads, the
detector and the homography, and carries out what the page asks for."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import signal
import threading
from typing import List, Optional, Tuple

import vision
from utils.homography import Homography

from . import pipeline
from .config import LOCAL_HOSTS, LOG_DIR, Config, resolve
from .recorder import FrameRecorder
from .server import WebServer
from .state import DashboardState, FrameStore

log = logging.getLogger("nemesis")
audit = logging.getLogger("nemesis.audit")


class NullRos:
    """Stands in for ros_bridge.RosBridge with --no-ros: positions go to the page only."""

    def publish_centroid(self, robot, cx, cy, coasting=False):
        pass

    def publish_image(self, frame):
        pass

    def spin(self):
        """Waits for Ctrl-C (or SIGTERM)."""
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass

    def shutdown(self):
        pass


def setup_logging():
    """Terminal, plus logs/dashboard.log (kept: who switched OVERRIDE and when)."""
    root = logging.getLogger("nemesis")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    root.propagate = False
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    root.addHandler(console)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file = logging.handlers.RotatingFileHandler(
            LOG_DIR / "dashboard.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(file)
    except OSError as error:
        root.warning("Not writing %s: %s", LOG_DIR / "dashboard.log", error)


class Dashboard:
    def __init__(self, config: Config, detector=None):
        """detector: an already loaded detector (camera_setup.py hands over its
        own); None loads config.vision."""
        self.config = config
        self.stop = threading.Event()
        self.frames = FrameStore()
        self.state = DashboardState(override=config.start_manual)

        # Read by other threads without a lock: each is replaced whole, never mutated
        self.override = config.start_manual  # True = the driver has the robot
        self.swap = False                    # swap which detection is Nemesis
        self.detector = detector
        self.homography = Homography(config.homog_dest, config.homog_dest_3d, config.nemesis_center_height)
        if config.default_clicked_points:
            self.homography.compute(config.default_clicked_points)
        self.fight_run = 0                   # 0 = live camera; +1 each time the fight video (re)starts
        self.frame_size: Optional[Tuple[int, int]] = None  # (width, height) of the frames vision sees

        self.recorder: Optional[FrameRecorder] = None
        self.server = WebServer(self)
        self.ros = NullRos()

        self._lock = threading.Lock()  # override, clicked points
        self._clicked_points: List[Tuple[float, float]] = []
        self._fight_lock = threading.Lock()  # one fight-video start at a time
        self._fight_thread: Optional[threading.Thread] = None
        self._threads: List[threading.Thread] = []

    # ---------- run ----------

    def run(self):
        """Runs until Ctrl-C."""
        cfg = self.config
        self._check_host()
        if cfg.record_dir:
            self.recorder = FrameRecorder(cfg.record_dir, cfg.record_scale)
        try:
            # The page first, so it shows "Link up" while the model loads
            self.server.start()
            shown = "localhost" if cfg.host in LOCAL_HOSTS else cfg.host
            log.info("✅ Dashboard: http://%s:%d", shown, cfg.port)

            if cfg.use_ros:
                from .ros_bridge import RosBridge  # rclpy only when ROS is wanted
                self.ros = RosBridge(self)
            else:
                signal.signal(signal.SIGTERM, signal.default_int_handler)
                log.info("ROS is off (--no-ros): positions go to the page only.")

            if cfg.start_with_fight_video:
                self._fight_lock.acquire()
                self._start_fight_video()
            else:
                if self.detector is None:
                    self.detector = self._load_detector()
                self._start(pipeline.capture_loop, "capture")
            log.info("✅ Vision method: %s", cfg.vision)

            self._start(pipeline.stream_loop, "stream")
            self._start(pipeline.vision_loop, "vision")
            self.ros.spin()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            # Let the threads finish their current frame: exiting while one is
            # still inside OpenCV or the detector aborts the process.
            for thread in [*self._threads, self._fight_thread]:
                if thread:
                    thread.join(timeout=2)
            self.ros.shutdown()
            self.server.stop()
            if self.recorder:
                self.recorder.close()

    def _check_host(self):
        cfg = self.config
        if cfg.host in LOCAL_HOSTS:
            return
        if not cfg.allow_hosts:
            raise SystemExit(
                f"--host {cfg.host} would put the dashboard, OVERRIDE button included, on the network, "
                "and it has no password. If that is really wanted, also give --allow-host NAME for each "
                "address the page will be opened under.")
        log.warning("⚠️ Listening on %s: anyone who can reach this computer as %s can press OVERRIDE.",
                    cfg.host, ", ".join(cfg.allow_hosts))

    def _start(self, loop_function, name: str):
        thread = threading.Thread(target=loop_function, args=(self,), name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _load_detector(self):
        cfg = self.config
        options = {"model_path": resolve(cfg.vision_model_path), "mode": cfg.vision_mode}
        if cfg.yolo_imgsz:
            options["imgsz"] = cfg.yolo_imgsz
        return vision.get_detector(cfg.vision, **options)

    # ---------- what the page asks for (called from the web server's thread) ----------

    def set_override(self, manual: bool, who: str):
        with self._lock:
            before = self.override
            self.override = manual
        self.state.update(override=manual)
        self.server.push_state()
        mode = "MANUAL (driver has the robot)" if manual else "AUTONOMOUS (policy has the robot)"
        if manual != before:
            audit.info("OVERRIDE -> %s, from %s", mode, who)
        else:
            audit.info("OVERRIDE pressed from %s, already %s", who, mode)

    def handle_action(self, key: str, who: str):
        if key == "swap_btn_pressed":
            self.swap = not self.swap
            audit.info("Swap detection %s, from %s", "ON" if self.swap else "OFF", who)
        elif key == "fight_video_btn_pressed":
            audit.info("Run fight video, from %s", who)
            self.start_fight_video()

    def canvas_click(self, x: float, y: float):
        """A click on the live feed, in camera pixels: one homography point."""
        cfg = self.config
        needed = len(cfg.homog_dest)
        if needed < 4:
            log.warning("⚠️ Canvas click ignored: fill in HOMOG_DEST (arena points) in backend/config.py first.")
            return
        size = self.frame_size
        if size is None or not (0 <= x <= size[0] and 0 <= y <= size[1]):
            log.warning("Canvas click outside the camera frame ignored: (%.0f, %.0f)", x, y)
            return
        with self._lock:
            self._clicked_points.append((x, y))
            log.info("📍 Canvas clicks (%d/%d): %s", len(self._clicked_points), needed, self._clicked_points)
            if len(self._clicked_points) < needed:
                return
            points = self._clicked_points[:needed]
            self._clicked_points = []  # reset for next calibration
        # A new object, swapped in whole: the vision loop never sees half an update
        homography = Homography(cfg.homog_dest, cfg.homog_dest_3d, cfg.nemesis_center_height)
        homography.compute(points)
        if homography.matrix is not None:
            self.homography = homography
        else:
            log.warning("No homography from those points; keeping the previous one.")

    # ---------- fight video ----------

    def start_fight_video(self):
        """Play the fight video through the fight model instead of the camera.
        Calling it again restarts the video from its first frame. Loading the
        model takes seconds, so this happens in a thread of its own."""
        if not self._fight_lock.acquire(blocking=False):
            log.info("The fight video is already starting.")
            return
        threading.Thread(target=self._start_fight_video, name="fight-start", daemon=True).start()

    def _start_fight_video(self):
        """Holds _fight_lock on entry; releases it."""
        cfg = self.config
        try:
            # The two-class fight model for YOLO; another method ("none" in the
            # smoke test) runs as it is
            if self.fight_run == 0 and cfg.vision == "yolo":
                self.detector = vision.get_detector(
                    "yolo", model_path=resolve(cfg.fight_model_path), mode=cfg.vision_mode,
                    imgsz=cfg.fight_model_imgsz)
            elif self.detector is None:
                self.detector = self._load_detector()
            path = resolve(cfg.fight_homography_path)
            if os.path.exists(path):
                with open(path) as f:
                    points = json.load(f)
                homography = Homography(points["arena_points_ft"], cfg.homog_dest_3d, cfg.nemesis_center_height)
                homography.compute(points["pixel_points"])
                self.homography = homography
            else:
                log.info("No %s: playing the fight video without arena positions.", path)
            self.fight_run += 1
            thread = threading.Thread(target=pipeline.fight_video_loop, args=(self, self.fight_run),
                                      name="fight-video", daemon=True)
            self._fight_thread = thread
            thread.start()
        except Exception:
            log.exception("❌ Could not start the fight video")
        finally:
            self._fight_lock.release()


def run(config: Config, detector=None):
    setup_logging()
    Dashboard(config, detector).run()
