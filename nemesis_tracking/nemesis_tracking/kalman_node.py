from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Deque, Optional, Tuple
import importlib

from collections import deque

import numpy as np


@dataclass(frozen=True)
class KalmanTuning:
    """Fill in measured values here before running."""

    # ── Topics ───────────────────────────────────────────────────────────
    # Subscriptions
    nemesis_pose_topic:    str = "/nemesis/centroid"
    nemesis_pose_6d_topic: str = "/nemesis/pose"
    opponent_pose_topic:   str = "/opponent/centroid"
    nemesis_imu_topic:     str = "/nemesis/imu"
    nemesis_rpy_topic:     str = "/opponent/rpy"

    # Publishers
    nemesis_out_topic:    str = "/nemesis/kalman/pos"
    opponent_out_topic:   str = "/opponent/kalman/pos"
    nemesis_twist_topic:  str = "/nemesis/kalman/twist"
    opponent_twist_topic: str = "/opponent/kalman/twist"
    world_frame:          str = "arena"

    # ── Timing ───────────────────────────────────────────────────────────
    max_dt_sec: float = 0.05
    min_dt_sec: float = 1e-3

    # ── Nemesis filter ───────────────────────────────────────────────────
    gyro_bias:               float = 0.0   # rad/s  — average angular_velocity.z at rest
    imu_gyro_sigma:          float = 0.15  # rad/s  — gyro noise at rest
    cam_pos_sigma:           float = 0.35  # m      — camera position noise at rest
    nem_process_vel_sigma:   float = 0.8   # m/s/√s — how fast nemesis can change velocity
    min_vel_for_heading_m_s: float = 0.30  # m/s    — min speed to trust velocity-vector heading
    nem_pose_yaw_sigma:      float = 0.05  # rad    — orientation noise from /nemesis/pose 6D source

    # ── Stationary detection ─────────────────────────────────────────────
    reset_movement_threshold: float = 0.25  # m      — max camera displacement to call it stopped
    reset_gyro_threshold:     float = 0.10  # rad/s  — max gyro magnitude when stopped

    # ── Opponent filter ──────────────────────────────────────────────────
    opp_process_vel_sigma: float = 1.5  # m/s/√s — higher = assume opponent moves more erratically




def _wrap_angle(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad



def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class NemesisImuPoseEkf:
    """Constant-velocity EKF with state [x, y, vx, vy, yaw].
    Gyro bias is a fixed constant (tuning.gyro_bias) subtracted before use.
    Heading is driven by gyro integration + IMU orientation + velocity-vector fusion.
    Accelerometer is NOT used for position/velocity prediction.
    """

    def __init__(self, tuning: KalmanTuning):
        self.tuning = tuning
        self.x = np.zeros((5,), dtype=np.float64)  # [x, y, vx, vy, yaw]
        self.P = np.eye(5, dtype=np.float64) * 1.0

        self._last_t_sec: Optional[float] = None
        self._last_u: Optional[float] = None  # gz (body)

    def set_state_from_pose(self, px: float, py: float, yaw: float):
        self.x[0] = px
        self.x[1] = py
        self.x[4] = _wrap_angle(yaw)

    def _q(self, dt: float) -> np.ndarray:
        # Constant-velocity model: velocity is a random walk; position inherits.
        q = np.zeros((5, 5), dtype=np.float64)

        v2 = self.tuning.nem_process_vel_sigma ** 2
        # Standard CV model (continuous-time discretised):
        #   Q_pos  = v2 * dt^3 / 3
        #   Q_vel  = v2 * dt
        q[0, 0] = v2 * (dt ** 3) / 3.0
        q[1, 1] = v2 * (dt ** 3) / 3.0
        q[2, 2] = v2 * dt
        q[3, 3] = v2 * dt

        # Gyro noise drives yaw.
        g2 = self.tuning.imu_gyro_sigma ** 2
        q[4, 4] = g2 * (dt ** 2)

        return q

    def predict(self, gz: float, dt: float):
        """Constant-velocity predict step.  Only gyro drives yaw; position
        integrates the current velocity estimate with no acceleration input."""
        dt = float(np.clip(dt, self.tuning.min_dt_sec, self.tuning.max_dt_sec))
        self._last_u = gz

        px, py, vx, vy, yaw = self.x
        yaw_rate = gz - self.tuning.gyro_bias  # fixed bias subtraction

        # State propagation (vx, vy unchanged — CV model)
        self.x[0] = px + vx * dt
        self.x[1] = py + vy * dt
        self.x[4] = _wrap_angle(yaw + yaw_rate * dt)

        # Jacobian F (5×5)
        F = np.eye(5, dtype=np.float64)
        F[0, 2] = dt  # x  w.r.t. vx
        F[1, 3] = dt  # y  w.r.t. vy

        Q = self._q(dt)
        self.P = F @ self.P @ F.T + Q

    def predict_to_time(self, target_t_sec: float):
        if self._last_t_sec is None:
            return
        dt = target_t_sec - self._last_t_sec
        if dt <= 0:
            return
        # With no gyro sample yet (camera-only operation) hold yaw and still
        # propagate position/velocity, otherwise the filter never predicts and
        # the velocity estimate stays at zero.
        self.predict(self._last_u if self._last_u is not None else 0.0, dt)
        self._last_t_sec = target_t_sec

    def set_time(self, t_sec: float):
        self._last_t_sec = t_sec

    def last_time(self) -> Optional[float]:
        return self._last_t_sec

    def set_gyro(self, gz: float):
        self._last_u = gz

    def update_pose(self, px_m: float, py_m: float):
        """Position-only update — heading comes from IMU and velocity vector."""
        H = np.zeros((2, 5), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        z = np.array([px_m, py_m], dtype=np.float64)
        y = z - np.array([self.x[0], self.x[1]], dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * self.tuning.cam_pos_sigma ** 2
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.x[4] = _wrap_angle(float(self.x[4]))
        I = np.eye(5, dtype=np.float64)
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T

    def update_yaw(self, yaw_m: float, sigma: float):
        """1-DOF yaw measurement update."""
        H = np.zeros((1, 5), dtype=np.float64)
        H[0, 4] = 1.0

        z = np.array([float(yaw_m)], dtype=np.float64)
        hx = np.array([float(self.x[4])], dtype=np.float64)
        y = z - hx
        y[0] = _wrap_angle(float(y[0]))

        R = np.array([[sigma ** 2]], dtype=np.float64)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        self.x[4] = _wrap_angle(float(self.x[4]))

        I = np.eye(5, dtype=np.float64)
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T

    def zero_reset(self):
        self.x[2] = 0.0
        self.x[3] = 0.0


class OpponentPoseKf:
    """Constant-velocity KF with state [x, y, vx, vy].
    No IMU on opponent — heading is derived from velocity at publish time."""

    def __init__(self, tuning: KalmanTuning):
        self.tuning = tuning
        self.x = np.zeros((4,), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 2.0

    def predict(self, dt: float):
        dt = float(np.clip(dt, self.tuning.min_dt_sec, self.tuning.max_dt_sec))
        self.x[0] += self.x[2] * dt
        self.x[1] += self.x[3] * dt

        v2 = self.tuning.opp_process_vel_sigma ** 2
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        Q = np.diag([v2 * dt ** 3 / 3, v2 * dt ** 3 / 3, v2 * dt, v2 * dt]).astype(np.float64)
        self.P = F @ self.P @ F.T + Q

    def update_pose(self, px_m: float, py_m: float):
        H = np.zeros((2, 4), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        z = np.array([px_m, py_m], dtype=np.float64)
        y = z - H @ self.x
        R = np.eye(2, dtype=np.float64) * self.tuning.cam_pos_sigma ** 2
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T


class StationaryDetector:
    def __init__(self, tuning: KalmanTuning):
        self.tuning = tuning
        self._cam_hist: Deque[Tuple[float, float, float]] = deque()  # (t, x, y)
        self._imu_hist: Deque[Tuple[float, float, float, float]] = deque()  # (t, ax, ay, gz)

    _WINDOW = 0.35  # seconds of history to keep

    def push_camera(self, t_sec: float, x: float, y: float):
        self._cam_hist.append((t_sec, x, y))
        cutoff = t_sec - self._WINDOW
        while self._cam_hist and self._cam_hist[0][0] < cutoff:
            self._cam_hist.popleft()

    def push_imu(self, t_sec: float, ax: float, ay: float, gz: float):
        self._imu_hist.append((t_sec, ax, ay, gz))
        cutoff = t_sec - self._WINDOW
        while self._imu_hist and self._imu_hist[0][0] < cutoff:
            self._imu_hist.popleft()

    def stationary(self) -> bool:
        if len(self._cam_hist) < 2 or len(self._imu_hist) < 2:
            return False

        cam_t_span = self._cam_hist[-1][0] - self._cam_hist[0][0]
        imu_t_span = self._imu_hist[-1][0] - self._imu_hist[0][0]
        if cam_t_span < 0.8 * self._WINDOW or imu_t_span < 0.8 * self._WINDOW:
            return False

        xs = [p[1] for p in self._cam_hist]
        ys = [p[2] for p in self._cam_hist]
        disp = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)

        g_mag_av = sum(abs(v[3]) for v in self._imu_hist) / len(self._imu_hist)

        return (
            disp     <= self.tuning.reset_movement_threshold
            and g_mag_av <= self.tuning.reset_gyro_threshold
        )


class KalmanFusionNode:
    def __init__(self, tuning: Optional[KalmanTuning] = None):
        self.tuning = tuning or KalmanTuning()

        self._rclpy = importlib.import_module("rclpy")
        Node = importlib.import_module("rclpy.node").Node
        _geom = importlib.import_module("geometry_msgs.msg")
        self._PoseStamped = _geom.PoseStamped
        self._PointStamped = _geom.PointStamped
        self._TwistStamped = _geom.TwistStamped
        self._RpyStamped = _geom.Vector3Stamped
        self._Imu = importlib.import_module("sensor_msgs.msg").Imu

        if not self._rclpy.ok():
            self._rclpy.init()

        self._node = Node("nemesis_kalman")

        self._nemesis_pub = self._node.create_publisher(
            self._PoseStamped, self.tuning.nemesis_out_topic, 10
        )
        self._opponent_pub = self._node.create_publisher(
            self._PoseStamped, self.tuning.opponent_out_topic, 10
        )
        self._nemesis_twist_pub = self._node.create_publisher(
            self._TwistStamped, self.tuning.nemesis_twist_topic, 10
        )
        self._opponent_twist_pub = self._node.create_publisher(
            self._TwistStamped, self.tuning.opponent_twist_topic, 10
        )

        self._nemesis_filter = NemesisImuPoseEkf(self.tuning)
        self._opponent_filter = OpponentPoseKf(self.tuning)
        self._stationary = StationaryDetector(self.tuning)

        self._last_nemesis_pose_t: Optional[float] = None
        self._last_opponent_pose_t: Optional[float] = None
        self._last_imu_t: Optional[float] = None
        self._last_imu_gz: Optional[float] = None
        self._last_opp_yaw: float = 0.0
        self._last_opp_measured_yaw: Optional[float] = None
        self._last_opp_rpy_t: Optional[float] = None

        self._node.create_subscription(
            self._PointStamped,
            self.tuning.nemesis_pose_topic,
            self._on_nemesis_pose,
            20,
        )
        self._node.create_subscription(
            self._PoseStamped,
            self.tuning.nemesis_pose_6d_topic,
            self._on_nemesis_pose_6d,
            20,
        )
        self._node.create_subscription(
            self._PointStamped,
            self.tuning.opponent_pose_topic,
            self._on_opponent_pose,
            20,
        )
        self._node.create_subscription(
            self._Imu, self.tuning.nemesis_imu_topic, self._on_imu, 100
        )
        self._node.create_subscription(
            self._RpyStamped,
            self.tuning.nemesis_rpy_topic,
            self._on_opponent_rpy,
            20,
        )

        # publish at fixed rate too (helps consumers)
        self._timer = self._node.create_timer(1.0 / 60.0, self._publish_timer)

        self._node.get_logger().info(
            f"Kalman fusion online. Sub: {self.tuning.nemesis_pose_topic}, {self.tuning.opponent_pose_topic}, {self.tuning.nemesis_imu_topic} | Pub: {self.tuning.nemesis_out_topic}, {self.tuning.opponent_out_topic}, {self.tuning.nemesis_twist_topic}, {self.tuning.opponent_twist_topic}"
        )
        self._node.get_logger().info(
            "Opponent heading: derived from velocity vector atan2(vy,vx) internally (no external heading node needed)."
        )

    def _now_sec(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9

    def destroy(self):
        # On Ctrl-C under `ros2 launch` the context is already shut down.
        if self._rclpy.ok():
            self._node.destroy_node()
            self._rclpy.shutdown()

    def spin(self):
        self._rclpy.spin(self._node)

    def _stamp_to_sec(self, stamp) -> float:
        # stamp is builtin_interfaces/msg/Time
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def _on_imu(self, msg):
        t = self._stamp_to_sec(msg.header.stamp)
        gz = float(msg.angular_velocity.z)

        if self._last_imu_t is None:
            self._last_imu_t = t
            if self._nemesis_filter.last_time() is None:
                self._nemesis_filter.set_time(t)
            self._nemesis_filter.set_gyro(gz)
            self._last_imu_gz = gz
            return

        if t - self._last_imu_t <= 0:
            return

        # The camera callbacks and the publish timer also advance the filter,
        # so predict only from the filter's own time. Predicting the whole
        # IMU interval again would integrate the same motion twice.
        filter_t = self._nemesis_filter.last_time()
        dt = t - filter_t if filter_t is not None else t - self._last_imu_t
        if dt > 0:
            self._nemesis_filter.predict(gz, dt)
        else:
            self._nemesis_filter.set_gyro(gz)

        # Heading is fused from /nemesis/pose orientation in _on_nemesis_pose_6d,
        # not from atan2(vy, vx) anymore — the 6D pose source is direct and
        # speed-independent.

        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        self._stationary.push_imu(t, ax, ay, gz)

        if dt > 0:
            self._nemesis_filter.set_time(t)
        self._last_imu_t = t
        self._last_imu_gz = gz

    def _on_nemesis_pose(self, msg):
        t = self._stamp_to_sec(msg.header.stamp)
        px = float(msg.point.x)
        py = float(msg.point.y)

        # Heading is NOT taken from this message; it comes from the IMU and
        # velocity vector alone.  We do a position-only EKF update.

        if self._last_nemesis_pose_t is None:
            # First camera message — initialize position regardless of IMU state.
            self._nemesis_filter.set_state_from_pose(px, py, 0.0)
            if self._last_imu_t is None:
                self._nemesis_filter.set_time(t)
        else:
            # Predict up to this camera timestamp, then correct.
            self._nemesis_filter.predict_to_time(t)
            self._nemesis_filter.update_pose(px, py)

        self._last_nemesis_pose_t = t
        self._stationary.push_camera(t, px, py)
        if self._stationary.stationary():
            self._nemesis_filter.zero_reset()
        self._publish_nemesis(t)

    def _on_opponent_pose(self, msg):
        t = self._stamp_to_sec(msg.header.stamp)
        px = float(msg.point.x)
        py = float(msg.point.y)

        if self._last_opponent_pose_t is None:
            self._opponent_filter.x[0] = px
            self._opponent_filter.x[1] = py
            self._last_opponent_pose_t = t
            self._publish_opponent(t)
            return

        dt = t - self._last_opponent_pose_t
        if dt > 0:
            self._opponent_filter.predict(dt)

        self._opponent_filter.update_pose(px, py)

        self._last_opponent_pose_t = t
        self._publish_opponent(t)

    def _on_nemesis_pose_6d(self, msg):
        t = self._stamp_to_sec(msg.header.stamp)
        q = msg.pose.orientation
        yaw_m = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))

        self._nemesis_filter.predict_to_time(t)

        self._nemesis_filter.update_yaw(yaw_m, sigma=self.tuning.nem_pose_yaw_sigma)

    def _on_opponent_rpy(self, msg):
        self._last_opp_measured_yaw = float(msg.vector.z)
        self._last_opp_rpy_t = self._stamp_to_sec(msg.header.stamp)

    def _publish_timer(self):
        # Publish latest states even if no new measurements.
        now = self._node.get_clock().now().to_msg()
        t = self._stamp_to_sec(now)

        if self._last_imu_gz is not None or self._last_nemesis_pose_t is not None:
            self._nemesis_filter.predict_to_time(t)
            self._publish_nemesis(t)
        if self._last_opponent_pose_t is not None:
            dt = t - self._last_opponent_pose_t
            if dt > 0:
                self._opponent_filter.predict(dt)
                self._last_opponent_pose_t = t
            self._publish_opponent(t)

    def _publish_nemesis(self, t_sec: float):
        msg = self._PoseStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self.tuning.world_frame

        px, py, vx, vy, yaw = self._nemesis_filter.x
        qx, qy, qz, qw = _quat_from_yaw(float(yaw))

        msg.pose.position.x = float(px)
        msg.pose.position.y = float(py)
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)

        self._nemesis_pub.publish(msg)

        tmsg = self._TwistStamped()
        tmsg.header = msg.header
        # Velocities are in the same frame as pose (arena/world frame)
        tmsg.twist.linear.x = float(vx)
        tmsg.twist.linear.y = float(vy)
        tmsg.twist.linear.z = 0.0
        yaw_rate = 0.0
        if self._last_imu_gz is not None:
            yaw_rate = float(self._last_imu_gz - self.tuning.gyro_bias)
        tmsg.twist.angular.x = 0.0
        tmsg.twist.angular.y = 0.0
        tmsg.twist.angular.z = float(yaw_rate)
        self._nemesis_twist_pub.publish(tmsg)

    def _publish_opponent(self, t_sec: float):
        msg = self._PoseStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self.tuning.world_frame

        px, py, vx, vy = self._opponent_filter.x
        speed = math.sqrt(vx ** 2 + vy ** 2)
        if self._last_opp_measured_yaw is not None:
            self._last_opp_yaw = self._last_opp_measured_yaw
        elif speed > 0.1:
            self._last_opp_yaw = math.atan2(vy, vx)
        yaw = self._last_opp_yaw
        # if 
        qx, qy, qz, qw = _quat_from_yaw(float(yaw))

        msg.pose.position.x = float(px)
        msg.pose.position.y = float(py)
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)

        self._opponent_pub.publish(msg)

        tmsg = self._TwistStamped()
        tmsg.header = msg.header
        tmsg.twist.linear.x = float(vx)
        tmsg.twist.linear.y = float(vy)
        tmsg.twist.linear.z = 0.0
        tmsg.twist.angular.x = 0.0
        tmsg.twist.angular.y = 0.0
        tmsg.twist.angular.z = 0.0
        self._opponent_twist_pub.publish(tmsg)


def main():
    node = KalmanFusionNode()
    shutdown_errors = (KeyboardInterrupt, importlib.import_module("rclpy.executors").ExternalShutdownException)
    try:
        node.spin()
    except shutdown_errors:
        pass
    finally:
        node.destroy()


if __name__ == "__main__":
    main()
