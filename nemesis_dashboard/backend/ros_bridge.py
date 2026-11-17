"""The ROS side of the backend. The only module that imports rclpy: with
--no-ros the Dashboard uses NullRos (app.py) and this file is never loaded."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64, Bool, Float32MultiArray
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Imu, Image

from .config import FRAME_ID, OVERRIDE_PUBLISH_HZ

if TYPE_CHECKING:
    from .app import Dashboard


class NemesisNode(Node):
    def __init__(self, app: "Dashboard"):
        super().__init__("nemesis_backend_node")
        self.app = app

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
        msg = Bool()
        msg.data = self.app.override
        self.nem_override_pub.publish(msg)

    # --- Centroid publishing (called from the vision loop) ---
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

    # --- Raw image publishing (called from the capture loop) ---
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
        state = self.app.state
        rpm = msg.data
        state.update("telemetry", "weapon", motorRPM=rpm)
        angvel = rpm * (2 * math.pi / 60)
        MOI = 0.0990301366  # kgm^2
        KE = (1 / 2) * MOI * (angvel ** 2)
        state.heard("weapon_motor", time.time())
        if self.dataCounter >= self.KEDataThresh:
            # Window full: keep its mean as the previous average, start a new one.
            self.lastAvgKE = self.currAvgKE
            self.currAvgSum = 0.0
            self.dataCounter = 0
        self.currAvgSum += KE
        self.dataCounter += 1
        self.currAvgKE = self.currAvgSum / self.dataCounter
        state.update("calculations", "weapon", kineticEnergy=self.currAvgKE)
        if self.lastAvgKE >= 0.2 * self.currAvgKE:
            state.update("calculations", "weapon", kineticETransfer=self.lastAvgKE - self.currAvgKE)

    def _on_arm_motor_data(self, msg):
        # TODO: not yet wired up — arm motor RPM → telemetry
        self.app.state.heard("arm_motor", time.time())

    def _on_left_wheel_data(self, msg):
        # TODO: not yet wired up — left wheel speed → calculations
        self.app.state.heard("leftWheel_motor", time.time())

    def _on_right_wheel_data(self, msg):
        # TODO: not yet wired up — right wheel speed → calculations
        self.app.state.heard("rightWheel_motor", time.time())

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
        self.app.server.publish_arc(arc)


class RosBridge:
    """What the rest of the backend sees of ROS (NullRos in app.py is its stand-in)."""

    def __init__(self, app: "Dashboard"):
        rclpy.init()
        self.node = NemesisNode(app)

    def publish_centroid(self, robot: str, cx: float, cy: float, coasting: bool = False):
        """robot: "nemesis" or "opponent"."""
        if not rclpy.ok():  # Ctrl-C shut ROS down during this frame
            return
        pub = self.node.nem_centroid_pub if robot == "nemesis" else self.node.opp_centroid_pub
        self.node.publish_centroid(pub, cx, cy, coasting)

    def publish_image(self, frame):
        if rclpy.ok():
            self.node.publish_image(frame)

    def spin(self):
        """Runs the node in the calling (main) thread until Ctrl-C."""
        try:
            rclpy.spin(self.node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass

    def shutdown(self):
        # Ctrl-C may already have shut the ROS context down
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()
