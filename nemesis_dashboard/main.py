"""Nemesis dashboard backend.

Runs the real-time vision + telemetry pipeline that feeds the web GUI:

  * a Socket.IO server (aiohttp) the browser connects to on port 3001,
  * a capture thread pulling frames from the GigE camera via GStreamer,
  * a stream thread JPEG-encoding frames out to the browser,
  * a vision thread running the selected detector, homography, and
    constant-velocity tracking, then publishing centroids to ROS and the
    frontend,
  * a watchdog thread flagging components that have gone silent.

The vision method is pluggable: ``--vision <method>`` selects a detector from
the ``vision`` package (e.g. ``yolo``). main.py itself contains no
detector-specific code; it only orchestrates the pipeline around whatever
detector is returned by ``vision.get_detector``.

Usage:
    python3 main.py --vision yolo
    python3 main.py --vision yolo --fight-video [VIDEO]   # recorded fight instead of the camera
    python3 main.py --vision yolo --record DIR            # also save the frames and their grab times
"""

import argparse
import asyncio
import json
import math
import os
import queue
import subprocess
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64, Bool, Float32MultiArray
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Imu, Image

import socketio
from aiohttp import web

import vision
from vision import get_detector, VISION_METHODS
from utils.geometry import adjust_center
from utils.homography import Homography

# =========================
# Config
# =========================
STREAM_W, STREAM_H = 0, 0  # resolution of OUTGOING stream to frontend; just leave this alone!!
CAPTURE_FPS = 120  # camera-dependent; should MATCH OR EXCEED target VISION_FPS
STREAM_FPS = 30  # fps of OUTGOING stream to frontend (independent of capture/vision)
VISION_FPS = 115  # detector FPS; BEST EFFORT; SHOULD NOT EXCEED CAPTURE_FPS!!!
JPEG_QUALITY = 40

VIDEO_SOURCE = (
    "aravissrc camera-name=\"169.254.211.45\" ! "
    "video/x-bayer,format=rggb ! "
    "bayer2rgb ! videoconvert ! "
    "video/x-raw,format=RGBA ! glupload ! "
    "glcolorconvert ! gldownload ! "
    "video/x-raw,format=BGR ! "
    "appsink max-buffers=1 drop=true sync=false"
)

# Detector config (passed to the selected vision method via get_detector).
VISION_MODEL_PATH = "YOLO_models/MediumBest.pt"  # CHANGE FOR UPDATED MODELS!!
VISION_MODE = "seg"  # "seg" (mask polygons) or "box"

# Fight-video mode (--fight-video, or the dashboard's "Run fight video" button):
# a recorded fight is played through the detector instead of the camera.
FIGHT_VIDEO_PATH = "/home/arc-lab-pc/battle_gamma08_fight.mp4"
FIGHT_MODEL_PATH = "../ZACVISIONQUEST/runs/sweep/yolo26m-seg_1280_fightbg/weights/chosen.pt"  # classes: robot, nemesis
FIGHT_MODEL_IMGSZ = 1280
# Arena points for that video: {"pixel_points": [[x, y], ...], "arena_points_ft": [[x, y], ...]}
FIGHT_HOMOGRAPHY_PATH = "fight_video_homography.json"

# Recording (--record DIR): ffmpeg options for DIR/video.mp4. On the CPU, so
# the detector keeps the GPU to itself. Measured on this PC: 1920x1440 at
# 70 FPS uses about one CPU core, and the encoder can take 320 FPS. No B-frames
# (-bf 0), so frame N of the file is exactly the N-th frame written.
# GPU alternative (half the CPU, a third of the file size, but it shares the GPU):
#   ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", "23", "-bf", "0"]
RECORD_ENCODER = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-bf", "0"]
RECORD_QUEUE_FRAMES = 64  # grabbed frames that may wait for the encoder (8 MB each); beyond that they are dropped

FRAME_ID = "image"

DETECTION_W = 640
DETECTION_H = 360

WATCHDOG_TIMEOUT = 3.0  # TODO: tune this timeout amount
OVERRIDE_PUBLISH_HZ = 20  # rate of /nemesis/override (manual/auto switch)
STATUS_PRINT_S = 5.0      # seconds between FPS / latency lines in the terminal

# Manual override switch (True = driver has control). NEMESIS_START_MANUAL=1
# makes the stack come up in manual until the dashboard button is pressed.
override_pressed = os.environ.get("NEMESIS_START_MANUAL") == "1"

# =========================
# Tracker config (arena-coordinate interpolation)
# =========================
VELOCITY_DECAY = 0.9    # per-step velocity bias when coasting (< 1 → slight deceleration)
ASSOC_MAX_DIST = 100      # max arena-unit distance to match a detection to a predicted track
TRACK_TIMEOUT = 2.0       # seconds without a matched detection before dropping a track


# =========================
# Dashboard state
# =========================
dashboard_state = {
    "systemHealth": {
        "vision": {"fps": 0, "captureFps": 0, "latencyMs": 0, "nemConfidence": 0, "oppConfidence": 0, "track": "ok"},
        "path": {"validity": True},
    },
    "imu": {"rotation": 0, "track": 0, "linSpeed": 0},
    "telemetry": {
        "driveL": {"escVolt": 0, "escCurr": 0, "escTemp": 0, "motorRPM": 0},
        "driveR": {"escVolt": 0, "escCurr": 0, "escTemp": 0, "motorRPM": 0},
        "arm":    {"escVolt": 0, "escCurr": 0, "escTemp": 0, "motorRPM": 0},
        "weapon": {"escVolt": 0, "escCurr": 0, "escTemp": 0, "motorRPM": 0},
    },
    "controls": {"vision": [0, 0, 0], "path": [0, 0]},
    "machineCondition": {"motors": [], "wheels": []},
    "calculations": {
        "WheelSlip": {"odometrySpeed": 0, "absVelDiff": 0},
        "weapon": {"kineticEnergy": 0, "kineticETransfer": 0},
    },
    "motorAlerts": {
        "weapon_motor": False, "weapon_esc": False,
        "arm_motor": False, "arm_esc": False,
        "leftWheel_motor": False, "leftWheel_esc": False,
        "rightWheel_motor": False, "rightWheel_esc": False,
    },
    # True = manual (driver) control, False = autonomous. Mirrors /nemesis/override.
    "override": override_pressed,
}

# track last time message received from ROS
time_last_received = {
    "weapon_motor": 0.0,
    "weapon_esc": 0.0,
    "arm_motor": 0.0,
    "arm_esc": 0.0,
    "leftWheel_motor": 0.0,
    "leftWheel_esc": 0.0,
    "rightWheel_motor": 0.0,
    "rightWheel_esc": 0.0,
}


# =========================
# Socket.IO server
# =========================
sio = socketio.AsyncServer(
    async_mode="aiohttp",
    cors_allowed_origins="*",
    transports=["websocket"],  # disable polling entirely
)
app = web.Application()
sio.attach(app)
server_loop: Optional[asyncio.AbstractEventLoop] = None

producer_sids = {}


@sio.on("register-producer")
async def register_producer(sid, data):
    name = data.get("name") if isinstance(data, dict) else None
    if not name:
        print(f"Producer {sid} registration failed")
        return
    producer_sids[name] = sid
    print(f"✅ Registered producer '{name}' sid={sid}")


@sio.event
async def connect(sid, environ):
    print(f"🔗 Client connected: sid={sid}")


@sio.event
async def disconnect(sid):
    dead = [name for name, psid in producer_sids.items() if psid == sid]
    for name in dead:
        producer_sids.pop(name, None)


@sio.on("send-command")
async def handle_command(sid, data):
    print(f"Received Command: {data}")
    if not isinstance(data, dict):
        return

    target = data.get("target")
    if not target:
        return

    target_sid = producer_sids.get(target)
    if target_sid:
        await sio.emit("command", data, to=target_sid)
        print(f"➡️ Forwarded command to {target} (sid={target_sid})")
    else:
        print(f"⚠️ No producer registered as '{target}'")


# =========================
# Direct hub actions
# (handled server-side, not forwarded to any producer)
# =========================
@sio.on("trigger-action")
async def handle_trigger_action(sid, data):
    if not isinstance(data, dict):
        return

    key = data.get("key")

    # match the keys to functions found in this script!!!
    if key == "swap_btn_pressed":
        await set_swap_detect()
    elif key == "override_btn_pressed":
        await swap_override()
    elif key == "fight_video_btn_pressed":
        start_fight_video()
    # Add more elif branches here for future buttons


async def set_swap_detect():
    current = state.get_swap()
    state.set_swap(not current)


async def swap_override():
    global override_pressed
    override_pressed = not override_pressed
    dashboard_state["override"] = override_pressed
    print(f"Override {'ON (manual)' if override_pressed else 'OFF (autonomous)'}")
    await sio.emit("dashboard-update", dashboard_state)


def start_background_server(host="0.0.0.0", port=3001):
    global server_loop
    server_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(server_loop)

    runner = web.AppRunner(app)
    server_loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, host, port)
    server_loop.run_until_complete(site.start())
    print(f"✅ SERVER STARTED on port {port}")
    server_loop.run_forever()


def emit_async(event: str, data):
    """Thread-safe emit from any thread."""
    if server_loop is None:
        return
    asyncio.run_coroutine_threadsafe(sio.emit(event, data), server_loop)


# =========================
# Homography (pixel ↔ arena coordinates)
# The math lives in utils.homography.Homography; main.py keeps the calibration
# config variables below and delegates compute/transform to that class.
# =========================
clicked_points: list = []  # list of (x, y) tuples in canvas pixel space

# TODO: refill these — cleared for now, to be re-derived later.
DEFAULT_CLICKED_POINTS: list = []                  # pixel-space points seeding the default homography
HOMOG_DEST = np.empty((0, 2), dtype=np.float32)    # arena-space destination points (planar)
HOMOG_DEST3D = np.empty((0, 2), dtype=np.float32)  # arena-space destination points (3D pose)
NEMESIS_CENTER_HEIGHT = 1.0  # TODO

# Shared homography state: read by the vision thread, updated by canvas clicks.
homography = Homography(HOMOG_DEST, HOMOG_DEST3D, NEMESIS_CENTER_HEIGHT)

# Compute default homography at import time (skipped while no default points set)
if DEFAULT_CLICKED_POINTS:
    homography.compute(DEFAULT_CLICKED_POINTS)


@sio.on("canvas-click")
async def handle_canvas_click(sid, data):
    global clicked_points
    if not isinstance(data, dict):
        return
    needed = len(HOMOG_DEST)
    if needed < 4:
        print("⚠️ Canvas click ignored: fill in HOMOG_DEST (arena points) in main.py first.")
        return
    x = float(data.get("x", 0.0))
    y = float(data.get("y", 0.0))
    clicked_points.append((x, y))
    print(f"📍 Canvas clicks ({len(clicked_points)}/{needed}): {clicked_points}")

    if len(clicked_points) >= needed:
        homography.compute(clicked_points[:needed])
        clicked_points = []  # reset for next calibration


# =========================
# Shared state (thread-safe)
# =========================
class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._frame_seq: int = 0

        self._nem_centroid: Optional[Tuple[float, float]] = None
        self._opp_centroid: Optional[Tuple[float, float]] = None
        self._det_ts: float = 0.0

        self._swap_centroids: bool = False

    def set_frame(self, frame: np.ndarray, ts: float):
        with self._lock:
            self._frame = frame
            self._frame_ts = ts
            self._frame_seq += 1

    def get_frame(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self._lock:
            return self._frame, self._frame_ts, self._frame_seq

    def set_centroids(self, nem, opp, ts: float):
        with self._lock:
            self._nem_centroid = nem
            self._opp_centroid = opp
            self._det_ts = ts

    def get_centroids(self):
        with self._lock:
            return self._nem_centroid, self._opp_centroid, self._det_ts

    def set_swap(self, value: bool):
        with self._lock:
            self._swap_centroids = value

    def get_swap(self) -> bool:
        with self._lock:
            return self._swap_centroids


# =========================
# Constant-velocity tracker (arena coordinates)
# =========================
class RobotTrack:
    """Single-robot constant-velocity track in arena (homography) coordinates."""

    def __init__(self, pos, t):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.t = t
        self.last_det_t = t

    def predict(self, t_now):
        dt = t_now - self.t
        if dt <= 0:
            return self.pos.copy()
        return self.pos + self.vel * dt

    def update_with_detection(self, measurement, t_now):
        dt = t_now - self.t
        if dt > 0:
            self.vel = (np.array(measurement) - self.pos) / dt
        self.pos = np.array(measurement, dtype=np.float64)
        self.t = t_now
        self.last_det_t = t_now

    def coast(self, t_now):
        dt = t_now - self.t
        if dt > 0:
            self.pos = self.pos + self.vel * dt
            self.vel *= VELOCITY_DECAY
        self.t = t_now

    def is_expired(self, t_now):
        return (t_now - self.last_det_t) > TRACK_TIMEOUT


class DualTracker:
    """Tracks nemesis and opponent in arena coordinates using constant-velocity interpolation."""

    def __init__(self):
        self.nem: Optional[RobotTrack] = None
        self.opp: Optional[RobotTrack] = None

    def step(self, arena_positions, t_now):
        """
        arena_positions: list of (x, y) arena-coord positions for each detection.

        Returns (nem_info, opp_info) where each is:
            (arena_pos_tuple, det_index) if track exists, else None.
            det_index is int if matched to a detection, None if coasting.
        """
        # Drop expired tracks
        if self.nem and self.nem.is_expired(t_now):
            self.nem = None
        if self.opp and self.opp.is_expired(t_now):
            self.opp = None

        nem_pred = self.nem.predict(t_now) if self.nem else None
        opp_pred = self.opp.predict(t_now) if self.opp else None

        n_dets = len(arena_positions)

        # No tracks yet — initialise from detections
        if self.nem is None and self.opp is None:
            if n_dets == 0:
                return (None, None)
            if n_dets == 1:
                self.nem = RobotTrack(arena_positions[0], t_now)
                return ((tuple(arena_positions[0]), 0), None)
            # >= 2 detections: leftmost = nemesis
            indices = sorted(range(n_dets), key=lambda i: arena_positions[i][0])
            self.nem = RobotTrack(arena_positions[indices[0]], t_now)
            self.opp = RobotTrack(arena_positions[indices[1]], t_now)
            return (
                (tuple(arena_positions[indices[0]]), indices[0]),
                (tuple(arena_positions[indices[1]]), indices[1]),
            )

        # Build greedy nearest-neighbour assignment (tracks ↔ detections)
        tracks = []
        track_names = []
        preds = []
        if self.nem is not None:
            tracks.append(self.nem)
            track_names.append('nem')
            preds.append(nem_pred)
        if self.opp is not None:
            tracks.append(self.opp)
            track_names.append('opp')
            preds.append(opp_pred)

        assignments = {}  # track_name -> det_index
        if n_dets > 0 and len(tracks) > 0:
            dists = np.zeros((len(tracks), n_dets))
            for ti, pred in enumerate(preds):
                for di in range(n_dets):
                    dists[ti, di] = np.linalg.norm(pred - np.array(arena_positions[di]))

            used_tracks = set()
            used_dets = set()
            for flat_idx in np.argsort(dists, axis=None):
                ti = int(flat_idx // n_dets)
                di = int(flat_idx % n_dets)
                if ti in used_tracks or di in used_dets:
                    continue
                if dists[ti, di] > ASSOC_MAX_DIST:
                    continue
                assignments[track_names[ti]] = di
                used_tracks.add(ti)
                used_dets.add(di)

        # Update matched tracks / coast unmatched
        nem_info = None
        opp_info = None

        if self.nem is not None:
            if 'nem' in assignments:
                di = assignments['nem']
                self.nem.update_with_detection(arena_positions[di], t_now)
                nem_info = (tuple(self.nem.pos), di)
            else:
                self.nem.coast(t_now)
                nem_info = (tuple(self.nem.pos), None)

        if self.opp is not None:
            if 'opp' in assignments:
                di = assignments['opp']
                self.opp.update_with_detection(arena_positions[di], t_now)
                opp_info = (tuple(self.opp.pos), di)
            else:
                self.opp.coast(t_now)
                opp_info = (tuple(self.opp.pos), None)

        # Create new tracks from unassigned detections
        assigned_dets = set(assignments.values())
        for di in range(n_dets):
            if di not in assigned_dets:
                if self.nem is None:
                    self.nem = RobotTrack(arena_positions[di], t_now)
                    nem_info = (tuple(arena_positions[di]), di)
                elif self.opp is None:
                    self.opp = RobotTrack(arena_positions[di], t_now)
                    opp_info = (tuple(arena_positions[di]), di)

        return (nem_info, opp_info)

    def step_known(self, arena_positions, nem_idx, opp_idx, t_now):
        """step() for a model that tells the robots apart itself: nem_idx and
        opp_idx are the detection index of each robot (None = not detected).
        Same return value, same coasting and expiry."""
        infos = []
        for name, di in (('nem', nem_idx), ('opp', opp_idx)):
            track = getattr(self, name)
            if track and track.is_expired(t_now):
                track = None
            if di is None:
                if track:
                    track.coast(t_now)
            elif track:
                track.update_with_detection(arena_positions[di], t_now)
            else:
                track = RobotTrack(arena_positions[di], t_now)
            setattr(self, name, track)
            infos.append((tuple(track.pos), di) if track else None)
        return tuple(infos)


# =========================
# ROS node
# =========================
class NemesisNode(Node):
    def __init__(self):
        super().__init__("nemesis_backend_node")

        # --- CENTROID PUBLISHERS (from video processing) ---
        self.nem_centroid_pub = self.create_publisher(PointStamped, "/nemesis/centroid", 10)
        self.opp_centroid_pub = self.create_publisher(PointStamped, "/opponent/centroid", 10)
        self.nem_override_pub = self.create_publisher(Bool, "/nemesis/override", 10)

        # --- IMAGE PUBLISHER (raw BGR frames for pose tracker) ---
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", image_qos)

        # --- Subscribers (incoming from robot) ---
        # TODO: confirm real topic names
        self.create_subscription(Float64, "/nemesis/weapon_motor/cmd_vel", self._on_weapon_motor_data, 10)
        self.create_subscription(Float64, "/nemesis/arm_motor/cmd_vel", self._on_arm_motor_data, 10)
        self.create_subscription(Float64, "/nemesis/left_wheels/cmd_vel", self._on_left_wheel_data, 10)
        self.create_subscription(Float64, "/nemesis/right_wheels/cmd_vel", self._on_right_wheel_data, 10)
        self.create_subscription(PoseStamped, "/nemesis/pose", self._on_pose_data, 10)
        self.create_subscription(Imu, "/nemesis/imu", self._on_imu_data, 10)
        self.create_subscription(Float32MultiArray, "/nemesis/predicted_arc", self._on_predicted_arc, 10)

        # Publish the override state at a fixed rate, independent of the camera:
        # manual takeover must still work when no frames are arriving.
        self.create_timer(1.0 / OVERRIDE_PUBLISH_HZ, self._publish_override_state)

        self.get_logger().info("--- NEMESIS BACKEND NODE READY ---")
        self.KEDataThresh = 20  # TUNE IF NEEDED COULD BE TOO SLOW
        self.lastAvgKE = 0
        self.dataCounter = 0
        self.currAvgKE = 0
        self.currAvgSum = 0

    def _publish_override_state(self):
        self.publish_override(self.nem_override_pub, override_pressed)

    # --- Centroid publishing (called from vision thread) ---
    def publish_centroid(self, pub, cx: float, cy: float, coasting: bool = False):
        """Arena position of one robot. point.z is 0.0 when the position comes
        from a detection in this frame and 1.0 when the tracker is coasting
        (extrapolating a robot the detector did not find), so consumers that
        want real fixes only can tell them apart."""
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.point.x = float(cx)
        msg.point.y = float(cy)
        msg.point.z = 1.0 if coasting else 0.0
        pub.publish(msg)

    def publish_override(self, pub, override: bool):
        msg = Bool()
        msg.data = override
        pub.publish(msg)

    # --- Raw image publishing (called from capture thread) ---
    def publish_image(self, frame):
        # Only the (currently parked) 6D pose tracker consumes raw frames.
        # Serialising ~8 MB per frame is expensive, so skip it until something
        # actually subscribes to /camera/image_raw.
        if self.image_pub.get_subscription_count() == 0:
            return
        h, w = frame.shape[:2]
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.height = h
        msg.width = w
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = frame.tobytes()
        self.image_pub.publish(msg)

    # --- Incoming robot data handlers ---
    def _on_weapon_motor_data(self, msg):
        rpm = msg.data
        dashboard_state["telemetry"]["weapon"]["motorRPM"] = rpm
        angvel = rpm * (2 * math.pi / 60)
        MOI = 0.0990301366  # kgm^2
        KE = (1 / 2) * MOI * (angvel ** 2)
        time_last_received["weapon_motor"] = time.time()
        if self.dataCounter >= self.KEDataThresh:
            # Window full: keep its mean as the previous average, start a new one.
            self.lastAvgKE = self.currAvgKE
            self.currAvgSum = 0.0
            self.dataCounter = 0
        self.currAvgSum += KE
        self.dataCounter += 1
        self.currAvgKE = self.currAvgSum / self.dataCounter
        dashboard_state["calculations"]["weapon"]["kineticEnergy"] = self.currAvgKE
        if self.lastAvgKE >= 0.2 * self.currAvgKE:
            dashboard_state["calculations"]["weapon"]["kineticETransfer"] = self.lastAvgKE - self.currAvgKE
        self._push_update()

    def _on_arm_motor_data(self, msg):
        # TODO: not yet wired up — arm motor RPM → telemetry
        time_last_received["arm_motor"] = time.time()

    def _on_left_wheel_data(self, msg):
        # TODO: not yet wired up — left wheel speed → calculations
        time_last_received["leftWheel_motor"] = time.time()

    def _on_right_wheel_data(self, msg):
        # TODO: not yet wired up — right wheel speed → calculations
        time_last_received["rightWheel_motor"] = time.time()

    def _on_pose_data(self, msg: PoseStamped):
        # TODO: not yet wired up — pose → dashboard
        return

    def _on_imu_data(self, msg: Imu):
        # TODO: not yet wired up — IMU orientation → imu.rotation (utils.geometry.quat_to_yaw_deg)
        return

    def _on_predicted_arc(self, msg: Float32MultiArray):
        """Receives flat [x0,y0, x1,y1, ...] arc from policy engine, forwards to frontend."""
        data = msg.data
        arc = []
        for i in range(0, len(data) - 1, 2):
            arc.append({"x": float(data[i]), "y": float(data[i + 1])})
        emit_async("predicted-arc", arc)

    def _push_update(self):
        emit_async("dashboard-update", dashboard_state)


# =========================
# Threads
# =========================
def capture_thread_fn(state: SharedState, ros_node: "NemesisNode", source, target_fps=30):
    global STREAM_W, STREAM_H
    if source is None:
        print("No camera selected: the dashboard is running without video or detections.")
        return
    cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ Could not open video source:", source)
        return

    initialized = False

    try:
        frame_count = 0
        fps_timer = time.perf_counter()
        while rclpy.ok() and fight_video_run == 0:  # the fight video takes over from the camera
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t_grab = time.time()  # Unix time, for --record
            if not ok:
                time.sleep(0.005)
                continue
            if not initialized:
                STREAM_H, STREAM_W = frame.shape[:2]
                print(f"✅ Camera resolution detected: {STREAM_W}x{STREAM_H}")
                initialized = True

            # Camera frame rate, for the dashboard and the vision thread's status line
            frame_count += 1
            if frame_count >= 30:
                elapsed = time.perf_counter() - fps_timer
                dashboard_state["systemHealth"]["vision"]["captureFps"] = round(frame_count / elapsed, 1)
                frame_count = 0
                fps_timer = time.perf_counter()

            state.set_frame(frame, t0)
            if recorder:
                recorder.add(frame, t_grab, target_fps)
            ros_node.publish_image(frame)
    finally:
        cap.release()


# =========================
# Recording (--record DIR)
# =========================
recorder = None  # FrameRecorder when --record is given


class FrameRecorder:
    """Saves the grabbed frames to DIR/video.mp4 and, for each frame in that
    file, the Unix time it was grabbed to DIR/frames.csv (columns frame, t).

    The frames wait in a bounded queue for a writer thread that pipes them to
    ffmpeg. When the encoder cannot keep up the queue fills and new frames are
    dropped (no row for them) rather than slowing the capture down."""

    def __init__(self, folder: str, scale: float = 1.0):
        self.video_path = os.path.join(folder, "video.mp4")
        self.csv_path = os.path.join(folder, "frames.csv")
        if os.path.exists(self.video_path):
            raise SystemExit(f"{self.video_path} already exists: give --record another folder.")
        os.makedirs(folder, exist_ok=True)
        self.scale = scale
        self.dropped = 0
        self._last_drop_print = 0.0
        self._queue = queue.Queue(maxsize=RECORD_QUEUE_FRAMES)
        self._thread = threading.Thread(target=self._writer_thread_fn, daemon=True)
        self._thread.start()

    def add(self, frame: np.ndarray, t: float, fps: float):
        """Queue one grabbed frame; never blocks. t: time.time() at the grab.
        fps: nominal rate of the source, the playback speed of the video."""
        try:
            self._queue.put_nowait((frame, t, fps))
        except queue.Full:
            self.dropped += 1
            if t - self._last_drop_print >= STATUS_PRINT_S:
                self._last_drop_print = t
                print(f"⚠️ Recording cannot keep up: {self.dropped} frames dropped so far")

    def close(self):
        """Encode what is still queued and finish both files."""
        self._queue.put(None)
        self._thread.join()

    def _writer_thread_fn(self):
        ffmpeg = None
        written = 0
        try:
            with open(self.csv_path, "w", buffering=1) as rows:  # line-buffered: complete after a crash too
                rows.write("frame,t\n")
                while (item := self._queue.get()) is not None:
                    frame, t, fps = item
                    if ffmpeg is None:
                        # The first frame sets the size (H.264 needs even numbers)
                        h, w = frame.shape[:2]
                        size = (int(w * self.scale) // 2 * 2, int(h * self.scale) // 2 * 2)
                        # Own session: a Ctrl-C in the terminal must not reach ffmpeg
                        # before the queued frames are written.
                        ffmpeg = subprocess.Popen(
                            ["ffmpeg", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{size[0]}x{size[1]}", "-framerate", str(fps), "-i", "-",
                             *RECORD_ENCODER, "-pix_fmt", "yuv420p", self.video_path],
                            stdin=subprocess.PIPE, start_new_session=True,
                        )
                        print(f"⏺ Recording {size[0]}x{size[1]} to {self.video_path}")
                    if (frame.shape[1], frame.shape[0]) != size:
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    ffmpeg.stdin.write(frame)
                    ffmpeg.stdin.flush()
                    rows.write(f"{written},{t:.6f}\n")
                    written += 1
        except Exception as error:  # ffmpeg missing or stopped (its own message is above), disk full...
            print(f"❌ Recording stopped after {written} frames: {error!r}")
            while self._queue.get() is not None:  # keep the queue empty until close()
                pass
        if ffmpeg:
            # Closing its input makes ffmpeg finish the mp4; without this it would not play.
            try:
                ffmpeg.stdin.close()
            except OSError:
                pass
            if ffmpeg.wait() != 0:
                print(f"❌ ffmpeg failed: {self.video_path} is not usable")
                return
        print(f"⏹ Recording: {written} frames in {self.video_path}, {self.dropped} dropped")


# =========================
# Fight-video mode
# =========================
detector = None            # the vision thread's detector; start_fight_video() swaps it
fight_video_run = 0        # 0 = live camera; goes up by one each time the fight video is (re)started
fight_video_thread = None  # the thread playing the video


def start_fight_video():
    """Play FIGHT_VIDEO_PATH through the two-class fight model instead of the
    camera. Calling it again restarts the video from its first frame."""
    global detector, homography, fight_video_run, fight_video_thread
    if fight_video_run == 0:
        # vision.get_detector, not get_detector: camera_setup.py replaces the
        # latter with one that hands back its own (camera) model.
        detector = vision.get_detector(
            "yolo", model_path=FIGHT_MODEL_PATH, mode=VISION_MODE, imgsz=FIGHT_MODEL_IMGSZ
        )
    if os.path.exists(FIGHT_HOMOGRAPHY_PATH):
        with open(FIGHT_HOMOGRAPHY_PATH) as f:
            points = json.load(f)
        fight_homography = Homography(points["arena_points_ft"], HOMOG_DEST3D, NEMESIS_CENTER_HEIGHT)
        fight_homography.compute(points["pixel_points"])
        homography = fight_homography
    else:
        print(f"No {FIGHT_HOMOGRAPHY_PATH}: playing the fight video without arena positions.")
    fight_video_run += 1
    fight_video_thread = threading.Thread(target=fight_video_thread_fn, args=(state, fight_video_run), daemon=True)
    fight_video_thread.start()


def fight_video_thread_fn(state: SharedState, run: int):
    """Feeds the fight video to the pipeline at its own frame rate, then holds
    the last frame. Gives way as soon as the video is started again."""
    global STREAM_W, STREAM_H
    cap = cv2.VideoCapture(FIGHT_VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Could not open fight video:", FIGHT_VIDEO_PATH)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dashboard_state["systemHealth"]["vision"]["captureFps"] = round(fps, 1)
    print(f"▶ Fight video: {FIGHT_VIDEO_PATH} ({fps:.0f} FPS)")

    start = time.perf_counter()
    frame_no = 0
    while rclpy.ok() and run == fight_video_run:
        ok, frame = cap.read()
        if not ok:
            print("Fight video finished; holding the last frame.")
            break
        wait = start + frame_no / fps - time.perf_counter()  # real time, no faster
        if wait > 0:
            time.sleep(wait)
        STREAM_H, STREAM_W = frame.shape[:2]
        state.set_frame(frame, time.perf_counter())
        if recorder:
            recorder.add(frame, time.time(), fps)
        frame_no += 1
    cap.release()


def stream_thread_fn(state: SharedState, fps=30, jpeg_quality=55):
    while server_loop is None and rclpy.ok():
        time.sleep(0.05)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    period = 1.0 / float(fps) if fps > 0 else 0.0
    last_sent_seq = -1

    while rclpy.ok():
        t0 = time.perf_counter()
        frame, _, seq = state.get_frame()

        if frame is None or seq == last_sent_seq:
            time.sleep(0.002)
            continue

        last_sent_seq = seq

        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            emit_async("video-frame", buf.tobytes())

        elapsed = time.perf_counter() - t0
        sleep_for = period - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)


def vision_thread_fn(state: SharedState, ros_node: NemesisNode, fps=6):
    period = 1.0 / float(fps) if fps > 0 else 0.0
    last_frame_seq = -1
    tracker = DualTracker()

    vision_frame_count = 0
    vision_fps_timer = time.perf_counter()
    skipped_frames = 0  # frames produced by capture thread but never seen by the detector
    latency_acc = 0.0   # accumulated capture→output latency over current batch
    last_status_print = 0.0

    while rclpy.ok():
        t0 = time.perf_counter()
        frame, frame_ts, seq = state.get_frame()

        if frame is None:
            time.sleep(0.01)
            continue
        if seq == last_frame_seq:
            time.sleep(0.005)
            continue

        # Track skipped frames (seq jumped by more than 1)
        if last_frame_seq != -1 and seq > last_frame_seq + 1:
            skipped_frames += seq - last_frame_seq - 1

        last_frame_seq = seq

        # ── Run pluggable detector, collect raw detections ─────────────
        current_detector = detector  # read once: the fight-video button can swap it
        raw_detections = []  # each: {pixel, conf, seg, cls}
        for det in current_detector.detect(frame):
            x1, y1, x2, y2 = det.bbox
            adjusted_cx, adjusted_cy = adjust_center(
                det.center[0], det.center[1], y1, y2, frame_height=frame.shape[0]
            )
            raw_detections.append(
                {'pixel': (adjusted_cx, adjusted_cy), 'conf': det.confidence, 'seg': det.mask,
                 'cls': det.class_id}
            )

        # A model with a "nemesis" class tells the robots apart itself: Nemesis is
        # its most confident nemesis detection, the opponent the most confident other.
        nemesis_class = current_detector.nemesis_class
        nem_idx = opp_idx = None
        if nemesis_class is not None:
            def most_confident(indices):
                return max(indices, key=lambda i: raw_detections[i]['conf'], default=None)
            nem_idx = most_confident(i for i, d in enumerate(raw_detections) if d['cls'] == nemesis_class)
            opp_idx = most_confident(i for i, d in enumerate(raw_detections) if d['cls'] != nemesis_class)

        nem = None
        opp = None
        nem_h = None
        opp_h = None
        nem_conf = None
        opp_conf = None
        nem_seg = None
        opp_seg = None
        nem_seen = False  # matched to a detection this frame (False = tracker is coasting)
        opp_seen = False

        if homography.matrix is not None:
            # ── Homography FIRST: pixel → arena for every detection ──
            arena_positions = []
            for det in raw_detections:
                pt = np.array([[[det['pixel'][0], det['pixel'][1]]]], dtype=np.float32)
                res = cv2.perspectiveTransform(pt, homography.matrix)
                arena_positions.append((float(res[0][0][0]), float(res[0][0][1])))

            # ── Tracker assigns labels in arena coords ──────────────
            if nemesis_class is not None:
                nem_info, opp_info = tracker.step_known(arena_positions, nem_idx, opp_idx, t0)
            else:
                nem_info, opp_info = tracker.step(arena_positions, t0)

            # Extract nemesis result
            if nem_info is not None:
                arena_pos, det_idx = nem_info
                nem_h = arena_pos
                if det_idx is not None:
                    det = raw_detections[det_idx]
                    nem = det['pixel']
                    nem_conf = det['conf']
                    nem_seg = det['seg']
                    nem_seen = True
                elif homography.inv_matrix is not None:
                    # Coasting — project interpolated arena pos back to pixel
                    pt = np.array([[[arena_pos[0], arena_pos[1]]]], dtype=np.float32)
                    px = cv2.perspectiveTransform(pt, homography.inv_matrix)
                    nem = (float(px[0][0][0]), float(px[0][0][1]))

            # Extract opponent result
            if opp_info is not None:
                arena_pos, det_idx = opp_info
                opp_h = arena_pos
                if det_idx is not None:
                    det = raw_detections[det_idx]
                    opp = det['pixel']
                    opp_conf = det['conf']
                    opp_seg = det['seg']
                    opp_seen = True
                elif homography.inv_matrix is not None:
                    pt = np.array([[[arena_pos[0], arena_pos[1]]]], dtype=np.float32)
                    px = cv2.perspectiveTransform(pt, homography.inv_matrix)
                    opp = (float(px[0][0][0]), float(px[0][0][1]))
        elif nemesis_class is not None:
            # ── No homography, two-class model: identity from the class ──
            if nem_idx is not None:
                d = raw_detections[nem_idx]
                nem, nem_conf, nem_seg = d['pixel'], d['conf'], d['seg']
            if opp_idx is not None:
                d = raw_detections[opp_idx]
                opp, opp_conf, opp_seg = d['pixel'], d['conf'], d['seg']
        else:
            # ── No homography: fallback sort-by-x (old behaviour) ───
            raw_detections.sort(key=lambda d: d['pixel'][0])
            if len(raw_detections) >= 1:
                d = raw_detections[0]
                nem, nem_conf, nem_seg = d['pixel'], d['conf'], d['seg']
            if len(raw_detections) >= 2:
                d = raw_detections[1]
                opp, opp_conf, opp_seg = d['pixel'], d['conf'], d['seg']

        # Swap applied atomically before any publish/emit
        if state.get_swap():
            nem, opp = opp, nem
            nem_conf, opp_conf = opp_conf, nem_conf
            nem_h, opp_h = opp_h, nem_h
            nem_seg, opp_seg = opp_seg, nem_seg
            nem_seen, opp_seen = opp_seen, nem_seen

        # Publish confidences to frontend (after the swap, so labels match)
        dashboard_state["systemHealth"]["vision"]["nemConfidence"] = round(nem_conf * 100, 1) if nem_conf is not None else 0
        dashboard_state["systemHealth"]["vision"]["oppConfidence"] = round(opp_conf * 100, 1) if opp_conf is not None else 0
        emit_async("dashboard-update", dashboard_state)

        # Publish centroids to ROS (unless Ctrl-C shut ROS down during this frame)
        if nem_h is not None and rclpy.ok():
            ros_node.publish_centroid(ros_node.nem_centroid_pub, nem_h[0], nem_h[1], coasting=not nem_seen)
        if opp_h is not None and rclpy.ok():
            ros_node.publish_centroid(ros_node.opp_centroid_pub, opp_h[0], opp_h[1], coasting=not opp_seen)

        # Emit centroids to frontend
        payload = {
            "detW": STREAM_W,
            "detH": STREAM_H,
            "nemesis": None if nem is None else {"cx": nem[0], "cy": nem[1]},
            "opponent": None if opp is None else {"cx": opp[0], "cy": opp[1]},
            "nemesisHomog": None if nem_h is None else {"x": nem_h[0], "y": nem_h[1]},
            "opponentHomog": None if opp_h is None else {"x": opp_h[0], "y": opp_h[1]},
            "nemesisSegment": nem_seg,
            "opponentSegment": opp_seg,
            "invHomog": homography.inv_matrix.flatten().tolist() if homography.inv_matrix is not None else None,
            "ts": t0,
        }
        emit_async("vision-centroids", payload)
        state.set_centroids(nem, opp, t0)

        latency_acc += (time.perf_counter() - frame_ts) * 1000
        vision_frame_count += 1
        if vision_frame_count >= 20:
            now = time.perf_counter()
            vision_fps = vision_frame_count / (now - vision_fps_timer)
            avg_latency = latency_acc / vision_frame_count
            health = dashboard_state["systemHealth"]["vision"]
            health["fps"] = round(vision_fps, 1)
            health["latencyMs"] = round(avg_latency, 1)   # frame grabbed -> positions published
            # One status line every few seconds instead of one per batch of frames
            if now - last_status_print >= STATUS_PRINT_S:
                last_status_print = now
                print(f"Camera {health.get('captureFps', 0):.1f} FPS | Vision {vision_fps:.1f} FPS | "
                      f"grab-to-result {avg_latency:.1f} ms | skipped {skipped_frames} frames")
            vision_frame_count = 0
            vision_fps_timer = now
            skipped_frames = 0
            latency_acc = 0.0

        elapsed = time.perf_counter() - t0
        sleep_for = period - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)


def watchdog_thread_fn():
    # periodically checks if any component has gone silent ROS-side
    while rclpy.ok():
        now = time.time()

        for key in time_last_received:
            # TRUE means failure: no message within WATCHDOG_TIMEOUT seconds
            dashboard_state["motorAlerts"][key] = (now - time_last_received[key]) > WATCHDOG_TIMEOUT

        emit_async("dashboard-update", dashboard_state)
        time.sleep(0.25)


# =========================
# Main
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Nemesis dashboard backend.")
    parser.add_argument(
        "--vision",
        required=True,
        choices=VISION_METHODS,
        help="Vision/detection method to run (e.g. 'yolo').",
    )
    parser.add_argument(
        "--fight-video",
        nargs="?",
        const=FIGHT_VIDEO_PATH,
        metavar="VIDEO",
        help=f"Play a recorded fight through the detector instead of the camera (default: {FIGHT_VIDEO_PATH}).",
    )
    parser.add_argument(
        "--record",
        metavar="DIR",
        help="Save every grabbed frame to DIR/video.mp4 and its grab time (Unix seconds) to DIR/frames.csv.",
    )
    parser.add_argument(
        "--record-scale",
        type=float,
        default=1.0,
        metavar="S",
        help="Size of the recorded video relative to the camera frame (default: 1.0, full size).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global state, detector, recorder, FIGHT_VIDEO_PATH
    if args.record:
        recorder = FrameRecorder(args.record, args.record_scale)
    rclpy.init()
    state = SharedState()
    ros_node = NemesisNode()

    # Build the selected vision detector (only the vision thread uses it).
    if args.fight_video:
        FIGHT_VIDEO_PATH = args.fight_video
        start_fight_video()
    else:
        detector = get_detector(args.vision, model_path=VISION_MODEL_PATH, mode=VISION_MODE)
    print(f"✅ Vision method: {args.vision}")

    # Server thread
    server_t = threading.Thread(target=start_background_server, daemon=True)
    server_t.start()

    # Wait for server loop before starting threads that emit
    while server_loop is None:
        time.sleep(0.05)

    # Capture thread (camera; in fight-video mode the video is already playing)
    if not args.fight_video:
        cap_t = threading.Thread(target=capture_thread_fn, args=(state, ros_node, VIDEO_SOURCE, CAPTURE_FPS), daemon=True)
        cap_t.start()

    # Stream thread
    stream_t = threading.Thread(target=stream_thread_fn, args=(state, STREAM_FPS, JPEG_QUALITY), daemon=True)
    stream_t.start()

    # Vision thread
    vision_t = threading.Thread(target=vision_thread_fn, args=(state, ros_node, VISION_FPS), daemon=True)
    vision_t.start()

    # Watchdog thread
    watchdog_t = threading.Thread(target=watchdog_thread_fn, daemon=True)
    watchdog_t.start()

    try:
        rclpy.spin(ros_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Ctrl-C may already have shut the ROS context down
        if rclpy.ok():
            ros_node.destroy_node()
            rclpy.shutdown()
        # Let the threads finish their current frame: exiting while one is still
        # inside OpenCV or the detector aborts the process.
        for thread in (fight_video_thread, stream_t, vision_t):
            if thread:
                thread.join(timeout=2)
        if recorder:
            recorder.close()


if __name__ == "__main__":
    main()
