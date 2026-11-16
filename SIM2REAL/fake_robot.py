#!/usr/bin/env python3
"""
Fake robot: stands in for the robot, the PPM output, the camera and the IMU,
so a session can be run with nothing connected (run.sh --dry).

    python SIM2REAL/fake_robot.py                                   # the simulator's nominal robot
    python SIM2REAL/fake_robot.py --plant v_max=0.7 tau=0.15 --act-lag 0.05 --obs-lag 0.12
    python SIM2REAL/fake_robot.py --no-imu                          # the IMU link is down
    python SIM2REAL/fake_robot.py --fault override:6:4 --fault no_position:20:2

In
    /control/channels   [left, right, blade, arm]; the treads reach the drive model after --act-lag,
                        and go to zero after 0.5 s of silence, as the PPM output does
    /sim2real/events    the test pilot's events (only for the stand-in driver)
Out
    /nemesis/centroid   where the robot was --obs-lag ago, plus noise, at --camera-hz, stamped when sent
    /nemesis/imu        yaw (from the IMU's own zero, not the arena's), yaw rate and acceleration
    /nemesis/override   false at 20 Hz; true while a scripted override or the stand-in driver has the robot
    /control/channels   the stand-in driver's sticks while it has the robot (as channel_mux would send them)
    /sim2real/truth     every number used here (latched); the recorder puts it in meta.json as "truth"

The motion is plant.Plant, the training simulator's own drive model, run in
its 10 ms steps: a command takes effect at the first step after --act-lag, so
5 ms later on average. There are no walls in it: nothing stops the fake robot
at the edge of the arena.

--fault KIND:START:SECONDS (as often as needed) breaks something for a while,
START seconds after the first command arrives:
    override      the switch goes to the driver, and no sticks arrive (transmitter not plugged in)
    no_position   the camera sends nothing
    coasting      the camera only sends extrapolated positions (point.z = 1)
    no_imu        the IMU sends nothing

--driver adds a stand-in for the human driver of README rule 5: when the pilot
says a maneuver has no room, it takes the robot (override on), drives it to
where the maneuver has room, ends with a short straight run, and gives it back.
"""
import argparse
import json
import math
import random
import time
from collections import deque

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32MultiArray, String

from maneuvers import CTRL_DT, mix, tread_k
from plant import NOMINAL, NOMINAL_LAG, SILENCE_S, Plant
from test_pilot import Floor, latched, load_settings

DT = 0.01                # physics step, s (the simulator's)
FT_TO_M = 0.3048
FAULTS = ("override", "no_position", "coasting", "no_imu")
RUN_FT = 3.0             # the straight run the stand-in driver ends with


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class StandInDriver:
    """Drives the fake robot to a spot where the pilot's next maneuver has room.

    The spot is on the floor's east-west middle line, facing +x: at the west end if the maneuver
    only needs room ahead, at the east end if only behind, in the middle otherwise. The last
    RUN_FT are driven straight from a standstill (forward, or backing into the west end), the
    way a driver has to finish when the pilot has no IMU to tell it which way the robot faces."""

    def __init__(self, floor):
        self.floor, self.phase, self.spot, self.until = floor, None, None, 0.0
        self.asked = {}                      # maneuver name: times the pilot asked for room

    def ask(self, maneuver, needs):
        """The pilot waits for room. Returns True if the driver takes the robot."""
        self.asked[maneuver] = self.asked.get(maneuver, 0) + 1
        if self.phase or self.asked[maneuver] > 2:      # busy, or tried twice already: no better spot to offer
            return False
        f = self.floor
        cx, cy = f.width / 2, f.height / 2
        cy += (f.free(cx, cy, math.pi / 2) - f.free(cx, cy, -math.pi / 2)) / 2
        west, east = cx - f.free(cx, cy, math.pi), cx + f.free(cx, cy, 0.0)
        ahead, behind = needs["ahead_ft"] > 0, needs["behind_ft"] > 0
        if needs["all_round"] or ahead == behind:
            self.spot = ((west + east) / 2, cy, 0.0, 1.0)
        elif ahead:
            self.spot = (west + 2.0, cy, 0.0, -1.0)     # back into the west end
        else:
            self.spot = (east - 2.0, cy, 0.0, 1.0)
        self.phase = "go"
        return True

    def sticks(self, x, y, a, omega, now):
        """(throttle, turn) for this step, or None when the robot is where it should be."""
        tx, ty, heading, way = self.spot
        turn_to = lambda off: max(-0.4, min(0.4, 0.6 * off))
        if self.phase == "go":                          # to the start of the last straight run
            px, py = tx - way * RUN_FT * math.cos(heading), ty - way * RUN_FT * math.sin(heading)
            dist = math.hypot(px - x, py - y)
            if dist > 0.5:
                off = _wrap(math.atan2(py - y, px - x) - a)
                return (min(0.35, 0.08 * dist + 0.08) if abs(off) < 0.3 else 0.0), turn_to(off)
            self.phase = "face"
        if self.phase == "face":
            off = _wrap(heading - a)
            if abs(off) > 0.03 or abs(omega) > 0.1:
                return 0.0, turn_to(off)
            self.phase, self.until = "rest", now + 1.6
        if self.phase == "rest":                        # the pilot must see it standing before the run
            if now < self.until:
                return 0.0, 0.0
            self.phase = "run"
        if self.phase == "run":
            to_go = way * ((tx - x) * math.cos(heading) + (ty - y) * math.sin(heading))
            if to_go > 0.3:
                return 0.25 * way, 0.0
            self.phase, self.until = "stop", now + 1.6
        if now < self.until:
            return 0.0, 0.0
        self.phase = None
        return None


class FakeRobot(Node):
    def __init__(self, args, truth, settings):
        super().__init__("sim2real_fake_robot")
        self.truth, self.rng = truth, random.Random(args.seed)
        self.plant = Plant(**{key: truth[key] for key in NOMINAL})
        x, y, heading = truth["start"]
        self.plant.reset(x, y, heading)
        self.sim_t = time.time()                         # the drive model has been run up to this time
        self.states = deque([(self.sim_t, x, y, heading, 0.0, 0.0, 0.0)], maxlen=400)   # t, x, y, heading, v, omega, dv/dt
        self.commands = deque([(0.0, 0.0, 0.0)])         # (arrival time, left, right)
        self.t_first = None                              # when the first command arrived: faults count from here
        self.faults = truth["faults"]
        self.k = tread_k(float(settings["half_track_ft"]))
        self.driver = StandInDriver(Floor(settings)) if args.driver else None

        self.pos_pub = self.create_publisher(PointStamped, "/nemesis/centroid", 10)
        self.imu_pub = self.create_publisher(Imu, "/nemesis/imu", 10) if truth["imu"] else None
        self.override_pub = self.create_publisher(Bool, "/nemesis/override", 10)
        self.sticks_pub = self.create_publisher(Float32MultiArray, "/control/channels", 10)
        self.truth_pub = self.create_publisher(String, "/sim2real/truth", latched())
        self.truth_pub.publish(String(data=json.dumps(truth)))
        self.create_subscription(Float32MultiArray, "/control/channels", self._on_command, 50)
        if self.driver:
            self.create_subscription(String, "/sim2real/events", self._on_event, 50)
        self.create_timer(DT, self._physics)
        self.create_timer(1.0 / truth["camera_hz"], self._camera)
        if truth["imu"]:
            self.create_timer(1.0 / truth["imu_hz"], self._imu)
        self.create_timer(0.05, self._override)
        self.create_timer(CTRL_DT, self._hands)

    # ── The robot ─────────────────────────────────────────────────────────

    def _on_command(self, msg):
        if len(msg.data) >= 2:
            now = time.time()
            self.t_first = self.t_first or now
            self.commands.append((now, float(msg.data[0]), float(msg.data[1])))

    def _treads_at(self, t):
        """The command in force at time t: the last one that had arrived, zero once it is SILENCE_S old."""
        while len(self.commands) > 1 and self.commands[1][0] <= t:
            self.commands.popleft()
        arrived, left, right = self.commands[0]
        return (left, right) if arrived <= t and t - arrived <= SILENCE_S else (0.0, 0.0)

    def _physics(self):
        now = time.time()
        while self.sim_t + DT <= now:
            left, right = self._treads_at(self.sim_t - self.truth["act_lag"])
            s = self.plant.step(left, right, DT)
            self.sim_t += DT
            self.states.append((self.sim_t, s["x_ft"], s["y_ft"], s["heading_rad"], s["v_fts"], s["omega_rps"],
                                (s["v_fts"] - self.states[-1][4]) / DT))

    def _state_at(self, t):
        """The robot's state at time t, between the two drive-model steps either side of it."""
        for i in range(len(self.states) - 1, 0, -1):
            a, b = self.states[i - 1], self.states[i]
            if a[0] <= t:
                w = min(1.0, (t - a[0]) / (b[0] - a[0]))
                state = [p + w * (q - p) for p, q in zip(a, b)]
                state[3] = a[3] + w * _wrap(b[3] - a[3])          # heading: the short way round
                return state
        return self.states[0]

    def _fault(self, kind):
        since = time.time() - self.t_first if self.t_first else -1.0
        return any(k == kind and start <= since < start + seconds for k, start, seconds in self.faults)

    # ── Camera and IMU ────────────────────────────────────────────────────

    def _camera(self):
        if self._fault("no_position") or self.rng.random() < self.truth["dropout"]:
            return
        _, x, y, *_ = self._state_at(time.time() - self.truth["obs_lag"])
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()        # stamped when sent, as the dashboard does
        msg.header.frame_id = "arena"
        msg.point.x = x + self.rng.gauss(0.0, self.truth["pos_noise"])
        msg.point.y = y + self.rng.gauss(0.0, self.truth["pos_noise"])
        msg.point.z = 1.0 if self._fault("coasting") else 0.0
        self.pos_pub.publish(msg)

    def _imu(self):
        if self._fault("no_imu"):
            return
        _, _, _, heading, v, omega, accel = self._state_at(time.time() - self.truth["imu_lag"])
        yaw = heading + self.truth["imu_yaw_zero"] + self.rng.gauss(0.0, self.truth["hdg_noise"])
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"
        msg.orientation.w, msg.orientation.z = math.cos(yaw / 2), math.sin(yaw / 2)
        msg.angular_velocity.z = omega + self.truth["gyro_bias"] + self.rng.gauss(0.0, self.truth["gyro_noise"])
        msg.linear_acceleration.x = accel * FT_TO_M               # along the body, m/s^2
        msg.linear_acceleration.y = v * omega * FT_TO_M           # towards the centre of the turn
        self.imu_pub.publish(msg)

    # ── The driver's side ─────────────────────────────────────────────────

    def _on_event(self, msg):
        e = json.loads(msg.data)
        if e["event"] == "waiting" and e.get("code") == "no_room" and e.get("needs"):
            if self.driver.ask(e.get("maneuver", ""), e["needs"]):
                x, y, heading, _ = self.driver.spot
                print(f"[fake driver] taking the robot to ({x:.1f}, {y:.1f}) facing {math.degrees(heading):+.0f} deg "
                      f"for {e.get('maneuver', '')}", flush=True)

    def _override(self):
        driving = self.driver is not None and self.driver.phase is not None
        self.override_pub.publish(Bool(data=driving or self._fault("override")))

    def _hands(self):
        """The driver's sticks, sent while the driver has the robot (channel_mux does this on the robot)."""
        if self.driver is not None and self.driver.phase is not None:
            _, x, y, heading, _, omega, _ = self.states[-1]
            sticks = self.driver.sticks(x, y, heading, omega, time.monotonic())
            if sticks is None:
                print("[fake driver] robot in place, giving it back", flush=True)
                sticks = (0.0, 0.0)
        else:
            return
        left, right = mix(*sticks, self.k)
        self.sticks_pub.publish(Float32MultiArray(data=[left, right, -1.0, 0.0]))


def main():
    ap = argparse.ArgumentParser(description="Stand in for robot, camera and IMU (see SIM2REAL/README.md).")
    ap.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "DEG"), default=[12.0, 27.0, 0.0],
                    help="where the robot stands at the start and which way it faces (default 12 27 0)")
    ap.add_argument("--plant", nargs="*", default=[], metavar="NAME=VALUE",
                    help=f"drive model numbers; the rest stay nominal. Names: {', '.join(NOMINAL)}")
    ap.add_argument("--act-lag", type=float, default=NOMINAL_LAG["act_lag"], help="s from a command to the treads")
    ap.add_argument("--obs-lag", type=float, default=NOMINAL_LAG["obs_lag"], help="s from the robot being somewhere to the position message")
    ap.add_argument("--imu-lag", type=float, default=0.01, help="s the IMU message is late")
    ap.add_argument("--pos-noise", type=float, default=0.1, help="ft, 1 sigma, each axis")
    ap.add_argument("--dropout", type=float, default=0.02, help="share of camera frames with no position")
    ap.add_argument("--hdg-noise", type=float, default=0.01, help="rad, 1 sigma, on the IMU's yaw")
    ap.add_argument("--gyro-noise", type=float, default=0.05, help="rad/s, 1 sigma")
    ap.add_argument("--gyro-bias", type=float, default=0.0, help="rad/s")
    ap.add_argument("--imu-yaw-zero", type=float, default=40.0, metavar="DEG", help="the IMU's yaw when the robot faces +x")
    ap.add_argument("--camera-hz", type=float, default=60.0)
    ap.add_argument("--imu-hz", type=float, default=50.0)
    ap.add_argument("--no-imu", action="store_true", help="publish no /nemesis/imu at all")
    ap.add_argument("--fault", action="append", default=[], metavar="KIND:START:SECONDS", help=f"KIND: {', '.join(FAULTS)}")
    ap.add_argument("--driver", action=argparse.BooleanOptionalAction, default=False,
                    help="the stand-in driver makes room when the pilot asks")
    ap.add_argument("--settings", help="the pilot's settings file, for the floor and the mixer (default settings.yaml)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    numbers = dict(NOMINAL)
    for item in args.plant:
        name, _, value = item.partition("=")
        if name not in NOMINAL:
            ap.error(f"--plant {item}: unknown number {name!r}; known: {', '.join(NOMINAL)}")
        numbers[name] = float(value)
    faults = []
    for item in args.fault:
        kind, start, seconds = item.split(":")
        if kind not in FAULTS:
            ap.error(f"--fault {item}: unknown kind {kind!r}; known: {', '.join(FAULTS)}")
        faults.append([kind, float(start), float(seconds)])
    settings = load_settings(args.settings)
    numbers["half_track_ft"] = float(settings["half_track_ft"])     # what the mixer used: needed to read the treads back
    x, y, deg = args.start
    truth = dict(numbers, act_lag=args.act_lag, obs_lag=args.obs_lag, imu_lag=args.imu_lag, pos_noise=args.pos_noise,
                 dropout=args.dropout, hdg_noise=args.hdg_noise, gyro_noise=args.gyro_noise, gyro_bias=args.gyro_bias,
                 imu=not args.no_imu, imu_yaw_zero=math.radians(args.imu_yaw_zero), camera_hz=args.camera_hz,
                 imu_hz=args.imu_hz, silence_s=SILENCE_S, step_s=DT, start=[x, y, math.radians(deg)], faults=faults,
                 driver=args.driver, seed=args.seed)

    rclpy.init()
    node = FakeRobot(args, truth, settings)
    print(f"[fake robot] at ({x:g}, {y:g}) facing {deg:+g} deg; " + ", ".join(
        f"{name} {truth[name]:g}" for name in (*NOMINAL, "act_lag", "obs_lag", "pos_noise")) +
        ("" if truth["imu"] else "; NO IMU") + (f"; faults {faults}" if faults else ""), flush=True)
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
