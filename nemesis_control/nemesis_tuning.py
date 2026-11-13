#!/usr/bin/env python3
"""
Nemesis Kalman Filter Tuning Procedure
=======================================
Run this script with the robot connected and ROS2 running.

    python3 nemesis_tuning.py [step]

Steps:
    1  gyro_bias            — robot sitting still, measure gyro drift
    2  imu_gyro_sigma       — robot sitting still, measure gyro noise
    3  cam_pos_sigma        — robot sitting still, measure camera noise
    4  stationary_thresh    — derive reset_movement_threshold and reset_gyro_threshold from rest data
    5  process_vel_sigma    — robot driving, derive nem_process_vel_sigma from acceleration

Run in order. Each step prints the recommended value to paste into KalmanTuning.
"""

import sys
import math
import time
import importlib
import numpy as np
from collections import deque

# ── Config ────────────────────────────────────────────────────────────────────
IMU_TOPIC    = "/nemesis/imu"
CAMERA_TOPIC = "/nemesis/centroid"
COLLECT_SEC  = 10.0   # how many seconds to collect data per step
DRIVE_SEC    = 15.0   # seconds for the drive step

# ── Helpers ───────────────────────────────────────────────────────────────────

def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def collect_imu(node, rclpy, duration_sec: float):
    """Collect raw IMU messages for `duration_sec` seconds.  Returns list of (t, gz, ax, ay)."""
    Imu = importlib.import_module("sensor_msgs.msg").Imu
    samples = []

    def cb(msg):
        t = stamp_to_sec(msg.header.stamp)
        samples.append((
            t,
            float(msg.angular_velocity.z),
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
        ))

    sub = node.create_subscription(Imu, IMU_TOPIC, cb, 200)
    deadline = time.time() + duration_sec
    while rclpy.ok() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.01)
        remaining = deadline - time.time()
        print(f"\r  collecting IMU … {remaining:4.1f}s left   ", end="", flush=True)
    print()
    node.destroy_subscription(sub)
    return samples


def collect_camera(node, rclpy, duration_sec: float):
    """Collect camera centroid messages.  Returns list of (t, x, y)."""
    PointStamped = importlib.import_module("geometry_msgs.msg").PointStamped
    samples = []

    def cb(msg):
        t = stamp_to_sec(msg.header.stamp)
        samples.append((t, float(msg.point.x), float(msg.point.y)))

    sub = node.create_subscription(PointStamped, CAMERA_TOPIC, cb, 50)
    deadline = time.time() + duration_sec
    while rclpy.ok() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.01)
        remaining = deadline - time.time()
        print(f"\r  collecting camera … {remaining:4.1f}s left   ", end="", flush=True)
    print()
    node.destroy_subscription(sub)
    return samples


# ── Steps ─────────────────────────────────────────────────────────────────────

def step1_gyro_bias(node, rclpy):
    print("\n=== STEP 1: gyro_bias ===")
    print("Place Nemesis on the floor, completely still.")
    input("Press Enter when ready …")
    print(f"Collecting IMU for {COLLECT_SEC:.0f}s …")

    samples = collect_imu(node, rclpy, COLLECT_SEC)
    if len(samples) < 10:
        print("ERROR: not enough IMU samples. Check topic:", IMU_TOPIC)
        return

    gz_vals = [s[1] for s in samples]
    bias = float(np.mean(gz_vals))

    print(f"\n  Samples collected : {len(gz_vals)}")
    print(f"  Mean gz           : {bias:+.6f} rad/s")
    print(f"\n  >>> gyro_bias = {bias:.6f}")
    print("  Paste this value into KalmanTuning.gyro_bias")


def step2_imu_gyro_sigma(node, rclpy):
    print("\n=== STEP 2: imu_gyro_sigma ===")
    print("Place Nemesis on the floor, completely still.")
    input("Press Enter when ready …")
    print(f"Collecting IMU for {COLLECT_SEC:.0f}s …")

    samples = collect_imu(node, rclpy, COLLECT_SEC)
    if len(samples) < 10:
        print("ERROR: not enough IMU samples. Check topic:", IMU_TOPIC)
        return

    gz_vals = [s[1] for s in samples]
    bias = float(np.mean(gz_vals))
    sigma = float(np.std(gz_vals))

    # 1-sigma is the noise floor.  Add a small margin (×1.5).
    recommended = round(sigma * 1.5, 4)

    print(f"\n  Samples collected : {len(gz_vals)}")
    print(f"  Gyro bias (mean)  : {bias:+.6f} rad/s")
    print(f"  Gyro std dev      : {sigma:.6f} rad/s")
    print(f"\n  >>> imu_gyro_sigma = {recommended}  (std × 1.5)")
    print("  Paste this value into KalmanTuning.imu_gyro_sigma")


def step3_cam_pos_sigma(node, rclpy):
    print("\n=== STEP 3: cam_pos_sigma ===")
    print("Place Nemesis on the floor, completely still, visible to overhead camera.")
    input("Press Enter when ready …")
    print(f"Collecting camera for {COLLECT_SEC:.0f}s …")

    samples = collect_camera(node, rclpy, COLLECT_SEC)
    if len(samples) < 5:
        print("ERROR: not enough camera samples. Check topic:", CAMERA_TOPIC)
        return

    xs = [s[1] for s in samples]
    ys = [s[2] for s in samples]
    sx = float(np.std(xs))
    sy = float(np.std(ys))
    sigma = float(np.sqrt(sx**2 + sy**2))
    recommended = round(max(sigma * 2.0, 0.05), 4)  # at least 5 cm, 2-sigma margin

    print(f"\n  Samples collected : {len(samples)}")
    print(f"  X std dev         : {sx:.4f} m")
    print(f"  Y std dev         : {sy:.4f} m")
    print(f"  Combined sigma    : {sigma:.4f} m")
    print(f"\n  >>> cam_pos_sigma = {recommended}  (combined × 2.0, min 0.05)")
    print("  Paste this value into KalmanTuning.cam_pos_sigma")


def step4_stationary_thresh(node, rclpy):
    print("\n=== STEP 4: stationary thresholds ===")
    print("Place Nemesis on the floor, completely still, visible to overhead camera.")
    input("Press Enter when ready …")
    print(f"Collecting both sensors for {COLLECT_SEC:.0f}s …")

    # collect in parallel via two callbacks
    Imu = importlib.import_module("sensor_msgs.msg").Imu
    PointStamped = importlib.import_module("geometry_msgs.msg").PointStamped
    imu_samples = []
    cam_samples = []

    def imu_cb(msg):
        t = stamp_to_sec(msg.header.stamp)
        imu_samples.append((t, float(msg.angular_velocity.z)))

    def cam_cb(msg):
        t = stamp_to_sec(msg.header.stamp)
        cam_samples.append((t, float(msg.point.x), float(msg.point.y)))

    sub_i = node.create_subscription(Imu, IMU_TOPIC, imu_cb, 200)
    sub_c = node.create_subscription(PointStamped, CAMERA_TOPIC, cam_cb, 50)

    deadline = time.time() + COLLECT_SEC
    while rclpy.ok() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.01)
        remaining = deadline - time.time()
        print(f"\r  collecting … {remaining:4.1f}s left   ", end="", flush=True)
    print()
    node.destroy_subscription(sub_i)
    node.destroy_subscription(sub_c)

    if len(imu_samples) < 10 or len(cam_samples) < 3:
        print("ERROR: not enough samples.")
        return

    # Camera: max displacement in any 0.35s sliding window
    window = 0.35
    max_disp = 0.0
    for i, (t0, x0, y0) in enumerate(cam_samples):
        window_pts = [(x, y) for (t, x, y) in cam_samples if t0 <= t <= t0 + window]
        if len(window_pts) < 2:
            continue
        xs = [p[0] for p in window_pts]
        ys = [p[1] for p in window_pts]
        disp = math.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
        max_disp = max(max_disp, disp)

    # Gyro: mean absolute gz over all samples
    gz_vals = [abs(s[1]) for s in imu_samples]
    gz_mean = float(np.mean(gz_vals))
    gz_max = float(np.max(gz_vals))

    # Recommend thresholds with 2× headroom over observed rest values
    rec_cam = round(max(max_disp * 2.0, 0.05), 4)
    rec_gyro = round(max(gz_max * 2.0, 0.05), 4)

    print(f"\n  Camera samples    : {len(cam_samples)}")
    print(f"  IMU samples       : {len(imu_samples)}")
    print(f"  Max cam disp (rest, 0.35s window): {max_disp:.4f} m")
    print(f"  Gyro mean abs     : {gz_mean:.4f} rad/s")
    print(f"  Gyro max abs      : {gz_max:.4f} rad/s")
    print(f"\n  >>> reset_movement_threshold = {rec_cam}  (max disp × 2.0, min 0.05)")
    print(f"  >>> reset_gyro_threshold     = {rec_gyro}  (max gz × 2.0, min 0.05)")
    print("  Paste these into KalmanTuning")


def step5_process_vel_sigma(node, rclpy):
    """
    Derive nem_process_vel_sigma from how fast velocity changes during hard driving.
    We differentiate camera position to get velocity, then differentiate again for
    acceleration.  The 1-sigma of acceleration × sqrt(dt) gives process_vel_sigma.

    Alternatively, a good rule of thumb:
        nem_process_vel_sigma ≈ max_acceleration_in_m_s2 × 0.1
    where 0.1 is ~1 filter time step.
    """
    print("\n=== STEP 5: nem_process_vel_sigma ===")
    print("Drive Nemesis aggressively — full speed, hard stops, sharp turns.")
    print("Keep robot visible to camera the whole time.")
    input("Press Enter when ready to start …")
    print(f"Collecting camera for {DRIVE_SEC:.0f}s. Drive now!")

    samples = collect_camera(node, rclpy, DRIVE_SEC)
    if len(samples) < 10:
        print("ERROR: not enough camera samples. Check topic:", CAMERA_TOPIC)
        return

    # Differentiate to get velocity, then acceleration
    vels = []
    for i in range(1, len(samples)):
        t0, x0, y0 = samples[i - 1]
        t1, x1, y1 = samples[i]
        dt = t1 - t0
        if dt < 1e-4:
            continue
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        speed = math.sqrt(vx**2 + vy**2)
        vels.append((t1, vx, vy, speed))

    accels = []
    for i in range(1, len(vels)):
        t0, vx0, vy0, _ = vels[i - 1]
        t1, vx1, vy1, _ = vels[i]
        dt = t1 - t0
        if dt < 1e-4:
            continue
        ax = (vx1 - vx0) / dt
        ay = (vy1 - vy0) / dt
        accels.append(math.sqrt(ax**2 + ay**2))

    if not accels:
        print("ERROR: could not compute accelerations.")
        return

    max_accel = float(np.percentile(accels, 95))  # 95th percentile to reject outliers
    max_speed = max(v[3] for v in vels)

    # Rule of thumb: sigma = max_accel * dt_typical (0.05s)
    recommended = round(max_accel * 0.05, 3)
    # Clamp to sensible range
    recommended = round(max(min(recommended, 5.0), 0.3), 3)

    print(f"\n  Camera samples    : {len(samples)}")
    print(f"  Max speed seen    : {max_speed:.2f} m/s")
    print(f"  95th pct accel    : {max_accel:.2f} m/s²")
    print(f"\n  >>> nem_process_vel_sigma = {recommended}  (accel_95 × 0.05)")
    print("  Paste this into KalmanTuning.nem_process_vel_sigma")
    print()
    print("  Note: min_vel_for_heading_m_s should be ~10–20% of max speed observed.")
    rec_heading = round(max(max_speed * 0.15, 0.15), 2)
    print(f"  >>> min_vel_for_heading_m_s = {rec_heading}  (max_speed × 0.15)")
    print("  Paste this into KalmanTuning.min_vel_for_heading_m_s")


# ── Main ──────────────────────────────────────────────────────────────────────

STEPS = {
    "1": step1_gyro_bias,
    "2": step2_imu_gyro_sigma,
    "3": step3_cam_pos_sigma,
    "4": step4_stationary_thresh,
    "5": step5_process_vel_sigma,
}

STEP_NAMES = {
    "1": "gyro_bias",
    "2": "imu_gyro_sigma",
    "3": "cam_pos_sigma",
    "4": "stationary thresholds (reset_movement_threshold, reset_gyro_threshold)",
    "5": "nem_process_vel_sigma + min_vel_for_heading_m_s",
}


def main():
    rclpy = importlib.import_module("rclpy")
    Node  = importlib.import_module("rclpy.node").Node

    if not rclpy.ok():
        rclpy.init()
    node = Node("nemesis_tuning")

    chosen = sys.argv[1] if len(sys.argv) > 1 else None

    if chosen is None:
        print("\nNemesis Kalman Tuning Procedure")
        print("─" * 40)
        for k, name in STEP_NAMES.items():
            print(f"  Step {k}: {name}")
        print()
        print("Usage:  python3 nemesis_tuning.py <step>")
        print("Run steps 1–5 in order, robot powered on, ROS2 active.")
        node.destroy_node()
        rclpy.shutdown()
        return

    if chosen not in STEPS:
        print(f"Unknown step '{chosen}'. Choose 1–5.")
        node.destroy_node()
        rclpy.shutdown()
        return

    try:
        STEPS[chosen](node, rclpy)
    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
