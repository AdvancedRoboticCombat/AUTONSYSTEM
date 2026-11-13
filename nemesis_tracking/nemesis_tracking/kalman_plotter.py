#!/usr/bin/env python3
"""
kalman_plotter.py
=================
Live 3-panel debug plotter for the Nemesis Kalman Fusion Node.

Graphs
------
1. Camera position vs Kalman filtered position (XY trajectory)
2. Velocity X & Y vs time  (/nemesis/kalman/twist)
3. Bias accel_x, accel_y, gyro_z vs time  (/nemesis/kalman/biases)

Requirements
------------
  pip install matplotlib
  ros-<distro>-rclpy  (already present in your ROS2 env)

Usage
-----
  # In your ROS2 workspace (source setup.bash first):
  python3 kalman_plotter.py

  # Optional: tune the window size (seconds of history shown)
  python3 kalman_plotter.py --window 15

  # Optional: change update rate (Hz)
  python3 kalman_plotter.py --hz 20

NOTE: You must also add the bias publisher to KalmanFusionNode._publish_nemesis().
      See the patch at the bottom of this file.
"""

from __future__ import annotations

import argparse
import math
import threading
from collections import deque
from typing import Deque, Optional

import matplotlib
matplotlib.use("TkAgg")  # change to "Qt5Agg" if you don't have Tk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np

# ── ROS2 imports ────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray

# ── CLI args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Kalman fusion live plotter")
parser.add_argument("--window", type=float, default=20.0, help="Time window to display (seconds)")
parser.add_argument("--hz",     type=float, default=15.0, help="Plot refresh rate (Hz)")
args, _ = parser.parse_known_args()

WINDOW_SEC  = args.window
REFRESH_HZ  = args.hz
MAX_POINTS  = 2000  # circular buffer cap


# ── Thread-safe data store ────────────────────────────────────────────────────────
class DataStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.t:       Deque[float] = deque(maxlen=MAX_POINTS)

        # Graph 1 — position
        self.cam_x:  Deque[float] = deque(maxlen=MAX_POINTS)
        self.cam_y:  Deque[float] = deque(maxlen=MAX_POINTS)
        self.kal_x:  Deque[float] = deque(maxlen=MAX_POINTS)
        self.kal_y:  Deque[float] = deque(maxlen=MAX_POINTS)

        # Graph 2 — velocity
        self.vel_t:  Deque[float] = deque(maxlen=MAX_POINTS)
        self.vel_x:  Deque[float] = deque(maxlen=MAX_POINTS)
        self.vel_y:  Deque[float] = deque(maxlen=MAX_POINTS)

        # Graph 3 — biases
        self.bias_t:  Deque[float] = deque(maxlen=MAX_POINTS)
        self.bias_ax: Deque[float] = deque(maxlen=MAX_POINTS)
        self.bias_ay: Deque[float] = deque(maxlen=MAX_POINTS)
        self.bias_gz: Deque[float] = deque(maxlen=MAX_POINTS)

        self._t0: Optional[float] = None

    def _rel(self, t_sec: float) -> float:
        if self._t0 is None:
            self._t0 = t_sec
        return t_sec - self._t0

    def push_cam_pose(self, t_sec: float, x: float, y: float):
        with self._lock:
            rt = self._rel(t_sec)
            self.t.append(rt)
            self.cam_x.append(x)
            self.cam_y.append(y)

    def push_kal_pose(self, t_sec: float, x: float, y: float):
        with self._lock:
            self.kal_x.append(x)
            self.kal_y.append(y)

    def push_velocity(self, t_sec: float, vx: float, vy: float):
        with self._lock:
            self.vel_t.append(self._rel(t_sec))
            self.vel_x.append(vx)
            self.vel_y.append(vy)

    def push_bias(self, t_sec: float, bax: float, bay: float, bgz: float):
        with self._lock:
            self.bias_t.append(self._rel(t_sec))
            self.bias_ax.append(bax)
            self.bias_ay.append(bay)
            self.bias_gz.append(bgz)

    def snapshot(self):
        """Return copies of all arrays under lock."""
        with self._lock:
            return dict(
                t      = list(self.t),
                cam_x  = list(self.cam_x),
                cam_y  = list(self.cam_y),
                kal_x  = list(self.kal_x),
                kal_y  = list(self.kal_y),
                vel_t  = list(self.vel_t),
                vel_x  = list(self.vel_x),
                vel_y  = list(self.vel_y),
                bias_t = list(self.bias_t),
                bias_ax= list(self.bias_ax),
                bias_ay= list(self.bias_ay),
                bias_gz= list(self.bias_gz),
            )


store = DataStore()


# ── ROS2 subscriber node ──────────────────────────────────────────────────────────
class PlotterNode(Node):
    def __init__(self):
        super().__init__("kalman_plotter")

        # Raw camera centroid (from the dashboard backend) vs filtered pose.
        self.create_subscription(
            PointStamped, "/nemesis/centroid",
            self._on_cam_pose, 20
        )
        self.create_subscription(
            PoseStamped, "/nemesis/kalman/pos",
            self._on_kal_pose, 20
        )
        self.create_subscription(
            TwistStamped, "/nemesis/kalman/twist",
            self._on_twist, 20
        )
        self.create_subscription(
            Float32MultiArray, "/nemesis/kalman/biases",
            self._on_biases, 20
        )

        self.get_logger().info("kalman_plotter subscribed and ready.")

    def _stamp_sec(self, stamp) -> float:
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def _on_cam_pose(self, msg: PointStamped):
        t = self._stamp_sec(msg.header.stamp)
        store.push_cam_pose(t, msg.point.x, msg.point.y)

    def _on_kal_pose(self, msg: PoseStamped):
        t = self._stamp_sec(msg.header.stamp)
        store.push_kal_pose(t, msg.pose.position.x, msg.pose.position.y)

    def _on_twist(self, msg: TwistStamped):
        t = self._stamp_sec(msg.header.stamp)
        store.push_velocity(t, msg.twist.linear.x, msg.twist.linear.y)

    def _on_biases(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        store.push_bias(t, msg.data[0], msg.data[1], msg.data[2])


# ── Matplotlib setup ──────────────────────────────────────────────────────────────
plt.style.use("dark_background")
FG   = "#e0e0e0"
GRID = "#1e1e1e"

fig = plt.figure(figsize=(13, 9), facecolor="#0a0d12")
fig.suptitle("Nemesis Kalman Fusion — Live Debug", fontsize=13,
             color=FG, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 1, hspace=0.52, top=0.93, bottom=0.06,
                       left=0.08, right=0.97)

ax1 = fig.add_subplot(gs[0])  # XY position
ax2 = fig.add_subplot(gs[1])  # velocity vs time
ax3 = fig.add_subplot(gs[2])  # bias vs time

def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor("#0d1117")
    ax.set_title(title, color=FG, fontsize=10, loc="left", pad=6)
    ax.set_xlabel(xlabel, color="#777", fontsize=8)
    ax.set_ylabel(ylabel, color="#777", fontsize=8)
    ax.tick_params(colors="#555", labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#222")

_style(ax1, "① Camera Position vs Kalman Filtered Position", "x (m)", "y (m)")
_style(ax2, "② Velocity vs Time  ─  /nemesis/kalman/twist", "time (s)", "m/s")
_style(ax3, "③ IMU Biases vs Time  ─  /nemesis/kalman/biases", "time (s)", "bias value")

# Pre-create line objects (faster than re-plotting each frame)
ln_cam,  = ax1.plot([], [], "o", ms=2.5, color="#3b82f6", alpha=0.45, label="/nemesis/pose (camera)")
ln_kal,  = ax1.plot([], [], "-", lw=1.8,  color="#f59e0b", alpha=0.9,  label="/nemesis/kalman (filtered)")
ax1.legend(fontsize=8, facecolor="#111", edgecolor="#333", labelcolor=FG)

ln_vx,   = ax2.plot([], [], "-", lw=1.4, color="#22d3ee", label="vel_x")
ln_vy,   = ax2.plot([], [], "-", lw=1.4, color="#a78bfa", label="vel_y")
ax2.legend(fontsize=8, facecolor="#111", edgecolor="#333", labelcolor=FG)
ax2.axhline(0, color="#333", linewidth=0.8, linestyle="--")

ln_bax,  = ax3.plot([], [], "-", lw=1.4, color="#f87171", label="bias accel_x  (x[5])")
ln_bay,  = ax3.plot([], [], "-", lw=1.4, color="#fb923c", label="bias accel_y  (x[6])")
ln_bgz,  = ax3.plot([], [], "-", lw=1.4, color="#4ade80", label="bias gyro_z   (x[7])")
ax3.legend(fontsize=8, facecolor="#111", edgecolor="#333", labelcolor=FG)
ax3.axhline(0, color="#333", linewidth=0.8, linestyle="--")

status_text = fig.text(0.99, 0.005, "waiting for data…",
                       ha="right", va="bottom", fontsize=7.5, color="#555")


def _trim_window(times, *arrays):
    """Keep only the last WINDOW_SEC of data."""
    if not times:
        return times, arrays
    t_max = times[-1]
    t_min = t_max - WINDOW_SEC
    idx = next((i for i, t in enumerate(times) if t >= t_min), 0)
    return times[idx:], tuple(a[idx:] for a in arrays)


def update(_frame):
    d = store.snapshot()

    # ── Graph 1: XY position ────────────────────────────────────────────────
    cx, cy = d["cam_x"], d["cam_y"]
    kx, ky = d["kal_x"], d["kal_y"]

    ln_cam.set_data(cx, cy)
    ln_kal.set_data(kx, ky)

    all_x = cx + kx
    all_y = cy + ky
    if all_x:
        pad = 0.3
        ax1.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax1.set_ylim(min(all_y) - pad, max(all_y) + pad)

    # ── Graph 2: velocity vs time ────────────────────────────────────────────
    vt, vx, vy = _trim_window(d["vel_t"], d["vel_x"], d["vel_y"])
    vx, vy = vx[0], vy[0]

    ln_vx.set_data(vt, vx)
    ln_vy.set_data(vt, vy)

    if vt:
        ax2.set_xlim(vt[0], max(vt[-1], vt[0] + 1))
        all_v = vx + vy
        pad = max(0.1, 0.1 * (max(all_v) - min(all_v))) if all_v else 0.5
        ax2.set_ylim(min(all_v) - pad, max(all_v) + pad)

    # ── Graph 3: bias vs time ────────────────────────────────────────────────
    bt, bax, bay, bgz = _trim_window(
        d["bias_t"], d["bias_ax"], d["bias_ay"], d["bias_gz"]
    )
    bax, bay, bgz = bax[0], bay[0], bgz[0]

    ln_bax.set_data(bt, bax)
    ln_bay.set_data(bt, bay)
    ln_bgz.set_data(bt, bgz)

    if bt:
        ax3.set_xlim(bt[0], max(bt[-1], bt[0] + 1))
        all_b = bax + bay + bgz
        pad = max(0.005, 0.1 * (max(all_b) - min(all_b))) if all_b else 0.05
        ax3.set_ylim(min(all_b) - pad, max(all_b) + pad)

    # Status line
    n_pts = len(d["t"])
    t_now = d["t"][-1] if d["t"] else 0.0
    status_text.set_text(f"t = {t_now:.1f}s  |  {n_pts} samples  |  window = {WINDOW_SEC}s")

    return ln_cam, ln_kal, ln_vx, ln_vy, ln_bax, ln_bay, ln_bgz


ani = animation.FuncAnimation(
    fig, update,
    interval=int(1000 / REFRESH_HZ),
    blit=False,
    cache_frame_data=False,
)


# ── Spin ROS in background thread ────────────────────────────────────────────────
def ros_thread():
    rclpy.init()
    node = PlotterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

t = threading.Thread(target=ros_thread, daemon=True)
t.start()

plt.show()


# ════════════════════════════════════════════════════════════════════════════════
# PATCH — add this to KalmanFusionNode in your existing script
# ════════════════════════════════════════════════════════════════════════════════
#
# In __init__, after the other publishers:
#
#   from std_msgs.msg import Float32MultiArray
#   self._bias_pub = self._node.create_publisher(
#       Float32MultiArray, "/nemesis/kalman/biases", 10
#   )
#
# At the END of _publish_nemesis(), add:
#
#   px, py, vx, vy, yaw, bax, bay, bgz = self._nemesis_filter.x
#   bias_msg = Float32MultiArray()
#   bias_msg.data = [float(bax), float(bay), float(bgz)]
#   self._bias_pub.publish(bias_msg)
#
# ════════════════════════════════════════════════════════════════════════════════