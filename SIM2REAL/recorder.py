#!/usr/bin/env python3
"""
Recorder: writes one run folder (README, "A run folder").

    python SIM2REAL/recorder.py --name floor_test                  # SIM2REAL/runs/<date_time>_floor_test/
    python SIM2REAL/recorder.py --dir SIM2REAL/runs/x --source fake

Every row is stamped with time.time() (Unix seconds) when the message arrived
here. Rows are flushed to disk twice a second, so a crash loses half a second.

What goes into meta.json and where it comes from:
    name, started, source          the arguments
    arena_ft, mixer, limits        the test pilot's /sim2real/session message
                                   (with --policy-params: that settings file of saw_policy_node)
    maneuvers                      each maneuver as it was played, from the pilot's maneuver_start
                                   events (with --policy-params: one entry, the policy)
    truth                          the fake robot's /sim2real/truth message
events.csv holds the pilot's events (/sim2real/events) and every change of
/nemesis/override, which the recorder notes itself (override_on, override_off).
"""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32MultiArray, String

from maneuvers import tread_k
from test_pilot import battlebox, latched

HERE = Path(__file__).resolve().parent
COLUMNS = {
    "commands": ["t", "left", "right", "blade", "arm"],
    "events": ["t", "event", "maneuver", "detail"],
    "position": ["t", "t_stamp", "x_ft", "y_ft", "coasting"],
    "opponent": ["t", "t_stamp", "x_ft", "y_ft", "coasting"],
    "imu": ["t", "t_stamp", "yaw_rad", "gyro_z", "accel_x", "accel_y"],
}


def _stamp(msg):
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def policy_meta(params_file, note):
    """arena_ft, mixer, limits and the one "maneuver" of a policy flight, from saw_policy_node's settings."""
    p = yaml.safe_load(Path(params_file).read_text())["/**"]["ros__parameters"]
    width, height = float(p.get("arena_width_ft", 0.0)), float(p.get("arena_height_ft", 0.0))
    half_track = float(p.get("half_track_ft", 1.48))
    return dict(
        arena_ft=[width, height] if width and height else battlebox()[0],
        mixer=dict(half_track_ft=half_track, tread_k=tread_k(half_track)),
        limits=dict(stale_s=float(p.get("stale_s", 1.0)), policy=True),
        maneuvers=[dict(name="policy", kind="policy", params=str(params_file), checkpoint=p.get("checkpoint", ""),
                        note=note)])


class Recorder(Node):
    def __init__(self, folder, meta):
        super().__init__("sim2real_recorder")
        self.folder, self.meta = Path(folder), meta
        self.folder.mkdir(parents=True, exist_ok=True)
        self.files, self.rows, self.count = {}, {}, {}
        for name, columns in COLUMNS.items():
            self.files[name] = open(self.folder / f"{name}.csv", "w", newline="")
            self.rows[name] = csv.writer(self.files[name])
            self.rows[name].writerow(columns)
            self.count[name] = 0
        self.override = None
        self._write_meta()

        self.create_subscription(Float32MultiArray, "/control/channels", self._on_command, 200)
        self.create_subscription(PointStamped, "/nemesis/centroid", lambda m: self._on_point("position", m), 200)
        self.create_subscription(PointStamped, "/opponent/centroid", lambda m: self._on_point("opponent", m), 200)
        self.create_subscription(Imu, "/nemesis/imu", self._on_imu, 500)
        self.create_subscription(Bool, "/nemesis/override", self._on_override, 50)
        self.create_subscription(String, "/sim2real/events", self._on_event, latched(1000))
        self.create_subscription(String, "/sim2real/session", self._on_session, latched())
        self.create_subscription(String, "/sim2real/truth", self._on_truth, latched())
        self.create_timer(0.5, self._flush)
        self.create_timer(10.0, self._report)
        print(f"[recorder] writing {self.folder}", flush=True)

    def _write_meta(self):
        tmp = self.folder / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2) + "\n")
        os.replace(tmp, self.folder / "meta.json")

    def _row(self, name, *values):
        self.rows[name].writerow(values)
        self.count[name] += 1

    # ── One row per message ───────────────────────────────────────────────

    def _on_command(self, msg):
        d = [float(v) for v in msg.data] + [math.nan] * 4
        self._row("commands", f"{time.time():.6f}", *(f"{v:.5f}" for v in d[:4]))

    def _on_point(self, name, msg):
        self._row(name, f"{time.time():.6f}", f"{_stamp(msg):.6f}", f"{msg.point.x:.4f}", f"{msg.point.y:.4f}",
                  1 if msg.point.z > 0.5 else 0)

    def _on_imu(self, msg):
        q = msg.orientation
        yaw = math.nan                         # no usable orientation in the message
        if 0.9 < math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z) < 1.1:
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._row("imu", f"{time.time():.6f}", f"{_stamp(msg):.6f}", f"{yaw:.5f}", f"{msg.angular_velocity.z:.5f}",
                  f"{msg.linear_acceleration.x:.4f}", f"{msg.linear_acceleration.y:.4f}")

    def _on_override(self, msg):
        if bool(msg.data) != self.override:
            first, self.override = self.override is None, bool(msg.data)
            self._row("events", f"{time.time():.6f}", "override_on" if self.override else "override_off", "",
                      "the switch as first heard" if first else "")

    # ── From the pilot and the fake robot ─────────────────────────────────

    def _on_event(self, msg):
        e = json.loads(msg.data)
        self._row("events", f"{e['t']:.6f}", e["event"], e.get("maneuver", ""), e.get("detail", ""))
        if e["event"] == "maneuver_start" and "settings" in e:
            self.meta["maneuvers"].append(e["settings"])
            self._write_meta()

    def _on_session(self, msg):
        self.meta.update({key: value for key, value in json.loads(msg.data).items() if key in ("arena_ft", "mixer", "limits")})
        self._write_meta()

    def _on_truth(self, msg):
        self.meta["truth"] = json.loads(msg.data)
        self._write_meta()

    # ── Housekeeping ──────────────────────────────────────────────────────

    def _flush(self):
        for f in self.files.values():
            f.flush()

    def _report(self):
        print("[recorder] " + ", ".join(f"{n} {name}" for name, n in self.count.items()), flush=True)

    def close(self):
        self._report()
        for f in self.files.values():
            f.close()


def main():
    ap = argparse.ArgumentParser(description="Record a SIM2REAL run folder (see SIM2REAL/README.md).")
    ap.add_argument("--name", default="run", help="what the session is, for the folder name and meta.json")
    ap.add_argument("--dir", help="the run folder (default: SIM2REAL/runs/<YYYYmmdd_HHMMSS>_<name>)")
    ap.add_argument("--source", choices=("robot", "fake"), default="robot")
    ap.add_argument("--policy-params", metavar="YAML",
                    help="the session is a flight of saw_policy_node with this settings file, not test drives")
    ap.add_argument("--note", default="", help="with --policy-params: what the policy flew against")
    args = ap.parse_args()

    started = time.time()
    folder = args.dir or HERE / "runs" / f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(started))}_{args.name}"
    meta = dict(name=args.name, started=started, source=args.source, arena_ft=None, maneuvers=[], mixer=None, limits=None)
    if args.policy_params:
        meta.update(policy_meta(args.policy_params, args.note))

    rclpy.init()
    node = Recorder(folder, meta)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        print(f"[recorder] closed {node.folder}", flush=True)
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
