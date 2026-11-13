#!/usr/bin/env python3
"""Hardware-free check of the autonomy chain.

Starts the real saw_policy_node, channel_mux and kalman_node, feeds them fake
camera centroids and a fake IMU (no camera, robot, radio or audio needed) and
checks:

  1. the saw policy publishes tread commands in [-1, 1] on /control/channels,
     with the blade held stopped and the arm idle (weapon output is off),
  2. the Kalman filter still estimates velocity from the camera alone,
  3. with override on, only the manual sticks reach /control/channels,
  4. when the camera only has coasted (extrapolated) positions for us, the
     policy stops commanding,
  5. when the IMU goes quiet, the policy stops commanding.

Runs on its own ROS_DOMAIN_ID so it cannot talk to a live robot stack.
Run through scripts/test.sh (which sets up the environment).
"""

import os
import signal
import subprocess
import sys
import time

DOMAIN_ID = "87"
os.environ["ROS_DOMAIN_ID"] = DOMAIN_ID

import rclpy
from geometry_msgs.msg import PointStamped, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32MultiArray

NODES = [
    ["ros2", "run", "nemesis_policy_engine", "saw_policy_node"],
    ["ros2", "run", "nemesis_control", "channel_mux"],
    ["ros2", "run", "nemesis_tracking", "kalman_node"],
]
SPEED = 6.0          # ft/s, fake Nemesis velocity along x (arena feet, corner origin)
MANUAL = [0.25, -0.5, 0.0, 0.0]


class Probe(Node):
    def __init__(self):
        super().__init__("pipeline_smoke_test")
        self.nem_pub = self.create_publisher(PointStamped, "/nemesis/centroid", 10)
        self.opp_pub = self.create_publisher(PointStamped, "/opponent/centroid", 10)
        self.imu_pub = self.create_publisher(Imu, "/nemesis/imu", 10)
        self.override_pub = self.create_publisher(Bool, "/nemesis/override", 10)
        self.radio_pub = self.create_publisher(Float32MultiArray, "/radiomaster/channels", 10)
        self.create_subscription(TwistStamped, "/nemesis/kalman/twist", self._on_twist, 10)
        self.create_subscription(Float32MultiArray, "/control/channels", self._on_channels, 10)

        self.camera = "seen"         # "seen": real detections, "coasting": extrapolated only
        self.feed_imu = True
        self.override = False
        self.t0 = time.monotonic()
        self.twist_vx = None
        self.channels = []           # (monotonic time, [left, right, blade, arm])
        self.create_timer(1 / 60, self._tick)

    def _point(self, x, y, coasting=False):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "arena"
        msg.point.x, msg.point.y = float(x), float(y)
        msg.point.z = 1.0 if coasting else 0.0
        return msg

    def _tick(self):
        t = time.monotonic() - self.t0
        # Drive back and forth across the middle of the 48 ft arena at constant speed.
        leg, pos = divmod(t * SPEED, 16.0)
        x = 16.0 + pos if int(leg) % 2 == 0 else 32.0 - pos
        self.nem_pub.publish(self._point(x, 24.0, coasting=self.camera == "coasting"))
        self.opp_pub.publish(self._point(36.0, 32.0))
        if self.feed_imu:
            imu = Imu()
            imu.header.stamp = self.get_clock().now().to_msg()
            imu.orientation.w = 1.0
            self.imu_pub.publish(imu)
        self.override_pub.publish(Bool(data=self.override))
        if self.override:
            self.radio_pub.publish(Float32MultiArray(data=MANUAL))

    def _on_twist(self, msg):
        self.twist_vx = msg.twist.linear.x

    def _on_channels(self, msg):
        self.channels.append((time.monotonic(), list(msg.data)))


def spin_for(node, seconds, until=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
        if until is not None and until():
            return True
    return until is None


def main():
    logs = open(os.path.join(os.environ.get("NEMESIS_LOG_DIR", "/tmp"), "pipeline_smoke_test_nodes.log"), "w")
    procs = [subprocess.Popen(cmd, stdout=logs, stderr=subprocess.STDOUT, start_new_session=True) for cmd in NODES]
    rclpy.init()
    probe = Probe()
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + detail + ')' if detail else ''}", flush=True)

    def quiet_after(seconds_to_settle, label):
        """True if nothing reaches /control/channels once the policy has had time to notice."""
        spin_for(probe, seconds_to_settle)
        mark = time.monotonic()
        spin_for(probe, 1.5)
        late = [data for t, data in probe.channels if t > mark]
        check(label, not late, f"{len(late)} messages")

    try:
        # 1: autonomous
        got = spin_for(probe, 60, until=lambda: len(probe.channels) >= 30)
        check("saw policy publishes on /control/channels", got,
              f"{len(probe.channels)} messages" if got else f"none after 60 s, see {logs.name}")
        spin_for(probe, 2.0)
        auto = [data for _, data in probe.channels]
        treads = [v for data in auto for v in data[:2]]
        check("four channels, treads within [-1, 1]",
              bool(auto) and all(len(d) == 4 for d in auto) and all(-1.0 <= v <= 1.0 for v in treads),
              f"tread range {min(treads):+.2f}..{max(treads):+.2f}" if treads else "no data")
        check("weapon output off: blade held stopped, arm idle",
              bool(auto) and all(len(d) == 4 and d[2] == -1.0 and d[3] == 0.0 for d in auto))
        moving = any(abs(v) > 0.05 for v in treads)
        check("the policy actually drives", moving)

        # 2: Kalman velocity (sample over a few seconds; the fake path turns round now and then)
        speeds = []
        end = time.monotonic() + 3.0
        while time.monotonic() < end:
            rclpy.spin_once(probe, timeout_sec=0.05)
            if probe.twist_vx is not None:
                speeds.append(abs(probe.twist_vx))
        median = sorted(speeds)[len(speeds) // 2] if speeds else 0.0
        check("Kalman filter estimates velocity without help", 0.6 * SPEED < median < 1.4 * SPEED,
              f"median |vx| = {median:.1f} ft/s, true {SPEED:.1f}")

        # 3: manual override
        probe.override = True
        spin_for(probe, 1.0)
        mark = time.monotonic()
        spin_for(probe, 1.5)
        manual = [data for t, data in probe.channels if t > mark]
        only_manual = bool(manual) and all(
            all(abs(a - b) < 1e-6 for a, b in zip(data, MANUAL)) for data in manual)
        check("override on: only the manual sticks reach /control/channels", only_manual, f"{len(manual)} messages")
        probe.override = False
        back = len(probe.channels)
        spin_for(probe, 3.0, until=lambda: len(probe.channels) > back + 10)
        check("override off: the policy takes the robot back", len(probe.channels) > back + 10)

        # 4: vision has lost us (the tracker only extrapolates)
        probe.camera = "coasting"
        quiet_after(2.0, "our position only coasted: the policy stops commanding")
        probe.camera = "seen"
        back = len(probe.channels)
        spin_for(probe, 3.0, until=lambda: len(probe.channels) > back + 10)

        # 5: IMU lost
        probe.feed_imu = False
        quiet_after(2.0, "IMU silent: the policy stops commanding")
    finally:
        probe.destroy_node()
        rclpy.shutdown()
        for proc in procs:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        logs.close()

    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if results and all(results) else 1)


if __name__ == "__main__":
    main()
