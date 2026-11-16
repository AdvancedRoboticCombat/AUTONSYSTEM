#!/usr/bin/env python3
"""
Phantom opponent: publishes a made-up /opponent/centroid at 30 Hz, so the real
saw_policy_node can be flown with no enemy in the arena.

    python SIM2REAL/phantom_opponent.py --at 34 27                       # standing at a point
    python SIM2REAL/phantom_opponent.py --line 30 18 30 36 --speed 4     # back and forth between two points
    python SIM2REAL/phantom_opponent.py --circle 26 27 8 --speed 4       # round a circle (centre x y, radius)

Arena feet in the policy's frame, speed in ft/s (negative: the other way
round the circle). The message is what the dashboard would send for a robot it
sees: geometry_msgs/PointStamped, point.z = 0, stamped now.
"""
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node

RATE_HZ = 30.0


def place(args, t):
    """Where the phantom is t seconds after the start: (x, y)."""
    if args.line:
        x0, y0, x1, y1 = args.line
        length = math.hypot(x1 - x0, y1 - y0)
        along = (abs(args.speed) * t) % (2 * length) if length else 0.0
        share = (along if along <= length else 2 * length - along) / length if length else 0.0
        return x0 + share * (x1 - x0), y0 + share * (y1 - y0)
    if args.circle:
        cx, cy, radius = args.circle
        angle = args.speed * t / radius if radius else 0.0
        return cx + radius * math.cos(angle), cy + radius * math.sin(angle)
    return tuple(args.at)


class Phantom(Node):
    def __init__(self, args):
        super().__init__("sim2real_phantom_opponent")
        self.args, self.t0 = args, time.monotonic()
        self.pub = self.create_publisher(PointStamped, "/opponent/centroid", 10)
        self.create_timer(1.0 / RATE_HZ, self._tick)

    def _tick(self):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "arena"
        msg.point.x, msg.point.y = (float(v) for v in place(self.args, time.monotonic() - self.t0))
        self.pub.publish(msg)


def main():
    ap = argparse.ArgumentParser(description="Publish a made-up /opponent/centroid (see SIM2REAL/README.md).")
    path = ap.add_mutually_exclusive_group()
    path.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"), default=[34.0, 27.0], help="stand at this point (default)")
    path.add_argument("--line", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"), help="go back and forth between two points")
    path.add_argument("--circle", nargs=3, type=float, metavar=("X", "Y", "R"), help="go round a circle, starting at (X + R, Y)")
    ap.add_argument("--speed", type=float, default=4.0, help="ft/s along the line or circle (default 4)")
    args = ap.parse_args()

    rclpy.init()
    node = Phantom(args)
    where = (f"line {args.line}" if args.line else f"circle {args.circle}" if args.circle else f"point {args.at}")
    print(f"[phantom] /opponent/centroid at {RATE_HZ:g} Hz: {where}" + ("" if not (args.line or args.circle) else f", {args.speed:g} ft/s"),
          flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
