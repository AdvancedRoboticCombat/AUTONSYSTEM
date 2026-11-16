#!/usr/bin/env python3
"""
Test pilot: plays a list of test drives on /control/channels, under the safety
rules of the README.

    python SIM2REAL/test_pilot.py                               # the standard session
    python SIM2REAL/test_pilot.py --maneuvers quick
    python SIM2REAL/test_pilot.py --maneuvers still,step_fwd_0.6,step_rev_0.6
    python SIM2REAL/test_pilot.py --no-imu --heading 90         # the IMU link is down
    python SIM2REAL/test_pilot.py --settings my_floor.yaml

In
    /nemesis/centroid   geometry_msgs/PointStamped  our position, arena feet (point.z = 1: coasted, not a fix)
    /nemesis/imu        sensor_msgs/Imu             yaw rate and orientation (not with --no-imu)
    /nemesis/override   std_msgs/Bool               true = the driver has the robot
    /control/channels   (its own output and, under override, the driver's sticks: used by --no-imu)
Out
    /control/channels   std_msgs/Float32MultiArray  [left, right, blade, arm], every 0.03 s while it drives.
                                                    Blade is always -1 (stopped) and arm 0.
    /sim2real/events    std_msgs/String             one JSON object per event, for the recorder
    /sim2real/session   std_msgs/String             the floor, mixer and limits in use (latched), for meta.json

The floor, the limits and the mixer are in settings.yaml next to this file.
What the pilot is doing or waiting for is printed, one line per event.

UNTESTED ON THE ROBOT. Everything here has only been run against fake_robot.py.
"""
import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32MultiArray, String

from maneuvers import (CTRL_DT, MANEUVERS, YAW_TOP, command, describe, duration, fit, mildest, mix, reach, session,
                       tread_k, unmix)

HERE = Path(__file__).resolve().parent
SILENCE_S = 0.5          # the PPM output centres the sticks after this long without a command

# How far off the robot's line the pre-check looks, rad either side of the heading: the doubt in
# the heading plus the most the simulator lets a robot pull to one side on a straight drive.
SLACK_IMU = 0.4          # heading from the IMU
SLACK_TRAVEL = 0.5       # no IMU: heading from the last straight drive
SLACK_GIVEN = 0.7        # no IMU: heading is only what the operator said
UNKNOWN = math.pi / 2    # no IMU: with more doubt than this the heading counts as unknown
# Heading from the direction of travel (no IMU)
COURSE_FT = 1.5          # a straight run this long sets the heading
DRIVE_MIN = 0.1          # throttle command that counts as driving
BEND_MAX = 0.2           # rad of possible turning within a run before it stops counting as straight


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def latched(depth=1):
    """QoS for a topic whose past messages a late subscriber still gets."""
    return QoSProfile(depth=depth, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


# ─── Settings and the floor ───────────────────────────────────────────────

def battlebox():
    """The 48 ft BattleBox as the simulator has it (auton_training/saw_walls.py, the walls the
    policy trains inside): ([width, height], its wall blocks as keep-out boxes [x0, y0, x1, y1])."""
    import saw_walls
    from config import ARENA_FT
    size, band = float(ARENA_FT), saw_walls.BAND_FT
    boxes = [[0.0, 0.0, size, band]]                                    # the band along the y = 0 wall
    c, hw, d = saw_walls.BULGE_Y0
    boxes.append([c - hw, band, c + hw, band + d])                      # the block on top of it
    c, hw, d = saw_walls.BULGE_X0
    boxes.append([0.0, c - hw, d, c + hw])                              # the block on the x = 0 wall
    c, hw, d = saw_walls.BULGE_X1
    boxes.append([size - d, c - hw, size, c + hw])                      # and on the x = 48 wall
    return [size, size], boxes


def load_settings(path=None):
    """settings.yaml (with the lines of ``path`` on top, if given) as a dict. Adds ``arena_ft``
    [width, height] and puts the BattleBox's blocks in front of ``keep_out``."""
    s = yaml.safe_load((HERE / "settings.yaml").read_text())
    if path:
        s.update(yaml.safe_load(Path(path).read_text()) or {})
    boxes = [[float(v) for v in box] for box in s.get("keep_out") or []]
    if s["arena"] == "battlebox":
        s["arena_ft"], sim_boxes = battlebox()
        boxes = sim_boxes + boxes
    else:
        s["arena_ft"] = [float(v) for v in s["arena"]]
    s["keep_out"] = boxes
    return s


class Floor:
    """Where the robot's centre may be: inside the arena and outside every keep-out box, by margin_ft."""

    def __init__(self, s):
        self.width, self.height = s["arena_ft"]
        self.boxes, self.margin = s["keep_out"], float(s["margin_ft"])

    def clearance(self, x, y):
        """(ft, what): how far (x, y) is from the nearest limit (a wall or a keep-out box, each
        grown by the margin) and which one that is. Negative: past the limit."""
        near = min((x, "the x = 0 wall"), (self.width - x, f"the x = {self.width:g} wall"),
                   (y, "the y = 0 wall"), (self.height - y, f"the y = {self.height:g} wall"))
        for x0, y0, x1, y1 in self.boxes:
            d = math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))
            near = min(near, (d, f"keep-out box [{x0:g}, {y0:g}, {x1:g}, {y1:g}]"))
        return near[0] - self.margin, near[1]

    def free(self, x, y, direction, slack=0.0):
        """Feet the centre can go from (x, y) before it reaches a limit, on the worst of a fan of
        lines up to ``slack`` rad either side of ``direction``."""
        n = math.ceil(slack / 0.1)
        return min(self._line(x, y, direction + slack * i / max(n, 1)) for i in range(-n, n + 1))

    def _line(self, x, y, a):
        # Each step is as long as the clearance where it starts, so it cannot pass a limit
        c, s, far = math.cos(a), math.sin(a), 0.0
        for _ in range(100):
            room = self.clearance(x + far * c, y + far * s)[0]
            if room < 0.05:
                break
            far += room
        return far


# ─── Heading ──────────────────────────────────────────────────────────────

class ImuHeading:
    """Heading the way saw_policy_node gets it: it starts from a given angle, follows the IMU's
    yaw (its fused orientation, or the gyro integrated) and is nudged by the direction the robot
    travels (auton_training/deploy.py HeadingEstimator)."""
    slack = SLACK_IMU

    def __init__(self, s):
        from deploy import HeadingEstimator
        self.est = HeadingEstimator()
        self.sign = float(s["gyro_sign"])
        if s["yaw_source"] not in ("orientation", "gyro"):
            raise ValueError(f"yaw_source must be orientation or gyro, not {s['yaw_source']!r}")
        self.fused = s["yaw_source"] == "orientation"
        self.gyro = 0.0
        self.recent = deque(maxlen=12)   # the last few yaw rates, for "is it turning"
        self.seen = None                 # time.monotonic() of the last IMU message
        self._dyaw, self._yaw, self._stamp = 0.0, None, None

    def on_imu(self, msg):
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self.gyro = self.sign * float(msg.angular_velocity.z)
        self.recent.append(self.gyro)
        q = msg.orientation
        norm = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
        if self.fused and 0.9 < norm < 1.1:
            yaw = self.sign * math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            if self._yaw is not None:
                self._dyaw += _wrap(yaw - self._yaw)
            self._yaw = yaw
        elif self._stamp is not None:
            self._dyaw += self.gyro * min(max(stamp - self._stamp, 0.0), 0.2)
        self._stamp = stamp
        self.seen = time.monotonic()

    @property
    def turning(self):
        """Yaw rate averaged over the last few IMU messages (one noisy sample is not a turn), rad/s."""
        return sum(self.recent) / len(self.recent) if self.recent else 0.0

    def start(self, heading):
        self._dyaw = 0.0
        self.est.reset(heading)

    def maneuver(self, m):
        """Maneuver m is about to start. The estimator's "launch" check puts a badly wrong heading
        right from the way the robot leaves, taking the throttle's sign for which end leads. That
        holds for a straight pull away from rest, so it is armed for those and for nothing else:
        in a doublet the robot is still rolling forward when the throttle already says reverse."""
        if m["kind"] in ("step", "straight"):
            self.est.reset(self.est.heading)
        else:
            self.est.launched = True

    def tick(self, x, y, fresh, throttle, turn, at_rest, driver):
        """One control step. As in saw_policy_node, the throttle counts as zero while the driver has the robot."""
        dyaw, self._dyaw = self._dyaw, 0.0
        self.est.update(x, y, fresh, self.gyro, dyaw, 0.0 if driver else throttle)

    @property
    def value(self):
        return self.est.heading


class TravelHeading:
    """Heading without an IMU: what the operator said, then the direction of travel.

    Whenever the robot pulls away from rest and covers COURSE_FT in a straight line, the heading
    becomes that line, turned round if the treads were told to go backwards (whoever told them:
    the pilot, or the driver under override). Between such runs the doubt grows by the most the
    robot could have turned on the turn commands it was given (the simulator's top yaw rate).
    Past a quarter turn of doubt the heading is unknown (``slack`` is None): after any spin, any
    arc, and any time the driver turned the robot round without then driving it straight."""
    seen = None

    def __init__(self):
        self.value, self.doubt, self._run = 0.0, math.pi, None

    @property
    def slack(self):
        return self.doubt if self.doubt < UNKNOWN else None

    def start(self, heading):
        self.value, self.doubt, self._run = _wrap(heading), SLACK_GIVEN, None

    def maneuver(self, m):
        pass

    def tick(self, x, y, fresh, throttle, turn, at_rest, driver=False):
        """One control step. ``throttle``, ``turn``: the command on /control/channels now, the
        pilot's or the driver's; ``at_rest``: the camera has not seen the robot move for a second."""
        turned = abs(turn) * YAW_TOP * CTRL_DT
        self.doubt = min(math.pi, self.doubt + turned)
        sign = (throttle > DRIVE_MIN) - (throttle < -DRIVE_MIN)
        if self._run is None:
            if at_rest and sign:                    # pulling away from rest: the command says which end leads
                self._run = [x, y, sign, 0.0]
            return
        x0, y0, way, bent = self._run
        bent += turned
        if (sign and sign != way) or bent > BEND_MAX or (at_rest and not sign):
            self._run = None                        # reversed on the move, not straight, or it has stopped
            return
        self._run[3] = bent
        if fresh and math.hypot(x - x0, y - y0) >= COURSE_FT:
            self.value = _wrap(math.atan2(y - y0, x - x0) + (math.pi if way < 0 else 0.0))
            self.doubt = SLACK_TRAVEL
            self._run = [x, y, way, 0.0]


# ─── The pilot ────────────────────────────────────────────────────────────

class TestPilot(Node):
    def __init__(self, settings, plan, use_imu=True, heading_deg=None, give_up_s=None):
        super().__init__("sim2real_test_pilot")
        self.s, self.plan, self.use_imu = settings, plan, use_imu
        self.floor = Floor(settings)
        self.k = tread_k(float(settings["half_track_ft"]))
        self.heading = ImuHeading(settings) if use_imu else TravelHeading()
        self.first_heading = math.radians(settings["initial_heading_deg"] if heading_deg is None else heading_deg)
        self.give_up_s = float(settings["give_up_s"] if give_up_s is None else give_up_s)

        # Sensors
        self.pos, self.pos_stamp, self._fresh = None, None, False   # last real fix, its stamp (Unix s), new this step
        self.track = deque()                                        # (monotonic time, x, y) of the last second or so
        self.override, self.override_t = None, None                 # the switch and when it last spoke
        self.sticks = (0.0, 0.0, None)                              # throttle, turn on /control/channels, and when
        # Where the session is
        self.i = 0                    # index in the plan of the maneuver to play next
        self.m = None                 # the maneuver being played, as fitted to the floor
        self.step = 0                 # control steps of it sent so far
        self.tried = 0                # times plan[i] was started and aborted
        self.probes = 0               # heading probes played for plan[i]
        self.waited = 0.0             # s plan[i] could not start with the driver's hands off
        self.ready_at = None          # when plan[i] starts, once everything is right
        self.brake_until = 0.0        # after an abort: zero treads until then
        self.heading_on = False       # the heading estimate has been started
        self.said = None              # what was last announced: (reason code, maneuver)
        self.said_at = {}             # and when each such thing was last announced
        self.played = self.aborted = self.skipped = 0
        self.done = False

        self.pub = self.create_publisher(Float32MultiArray, "/control/channels", 10)
        self.events = self.create_publisher(String, "/sim2real/events", latched(1000))
        self.session = self.create_publisher(String, "/sim2real/session", latched())
        limits = {key: settings[key] for key in ("margin_ft", "stop_s", "keep_out", "stale_s", "still_ft", "still_rps",
                                                 "settle_s", "min_share", "brake_s", "tries")}
        limits.update(give_up_s=self.give_up_s, imu=use_imu)
        self.session.publish(String(data=json.dumps(dict(
            arena_ft=settings["arena_ft"], mixer=dict(half_track_ft=float(settings["half_track_ft"]), tread_k=self.k),
            limits=limits))))
        self.create_subscription(PointStamped, "/nemesis/centroid", self._on_position, 10)
        self.create_subscription(Bool, "/nemesis/override", self._on_override, 10)
        self.create_subscription(Float32MultiArray, "/control/channels", self._on_channels, 50)
        if use_imu:
            self.create_subscription(Imu, "/nemesis/imu", self.heading.on_imu, 50)
        self.create_timer(CTRL_DT, self._tick)

        width, height = settings["arena_ft"]
        self._print(f"Floor {width:g} x {height:g} ft, {len(self.floor.boxes)} keep-out boxes, the robot's centre stays "
                    f"{self.floor.margin:g} ft from all of it. Mixer tread_k {self.k:.3f}.")
        for box in self.floor.boxes:
            self._print(f"    keep-out [x0, y0, x1, y1] = {box}")
        self._print(("Heading: starts at " if use_imu else "NO IMU. Heading: the operator's ")
                    + f"{math.degrees(self.first_heading):+.0f} deg when the pilot first gets the robot, "
                    + ("then follows the IMU." if use_imu else "then the direction of travel only."))
        self._print(f"{len(plan)} maneuvers: {', '.join(m['name'] for m in plan)}")

    # ── Sensors ───────────────────────────────────────────────────────────

    def _on_position(self, msg):
        if msg.point.z > 0.5:                 # the tracker is coasting: not a fix (as saw_policy_node)
            return
        self.pos = (float(msg.point.x), float(msg.point.y))
        self.pos_stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self._fresh = True
        self.track.append((time.monotonic(), *self.pos))

    def _on_override(self, msg):
        self.override, self.override_t = bool(msg.data), time.monotonic()

    def _on_channels(self, msg):
        if len(msg.data) >= 2:
            self.sticks = (*unmix(float(msg.data[0]), float(msg.data[1]), self.k), time.monotonic())

    def _moved(self, now):
        """Why the robot does not count as standing still, or "" if it does: the mean of the newer
        half of the last second's fixes against the older half, and the yaw rate."""
        while self.track and now - self.track[0][0] > 1.0:
            self.track.popleft()
        if len(self.track) < 6 or now - self.track[0][0] < 0.7:
            return "not a full second of position fixes yet"
        half = len(self.track) // 2
        old, new = list(self.track)[:half], list(self.track)[half:]
        mean = lambda fixes, i: sum(f[i] for f in fixes) / len(fixes)
        moved = math.hypot(mean(new, 1) - mean(old, 1), mean(new, 2) - mean(old, 2))
        if moved > self.s["still_ft"]:
            return f"it moved {moved:.1f} ft in the last second"
        if self.use_imu and abs(self.heading.turning) > self.s["still_rps"]:
            return f"it is turning at {self.heading.turning:+.2f} rad/s"
        return ""

    def _velocity(self, now):
        """(vx, vy) of the robot, ft/s, from the camera: the newer half of the last 0.3 s of fixes
        against the older half. (0, 0) with too few fixes to tell."""
        fixes = [f for f in self.track if now - f[0] <= 0.3]
        if len(fixes) < 4:
            return 0.0, 0.0
        half = len(fixes) // 2
        mean = lambda part, i: sum(f[i] for f in part) / len(part)
        old, new = fixes[:half], fixes[half:]
        dt = mean(new, 0) - mean(old, 0)
        return ((mean(new, 1) - mean(old, 1)) / dt, (mean(new, 2) - mean(old, 2)) / dt) if dt > 0.02 else (0.0, 0.0)

    def _not_driving(self, now):
        """Why the pilot may not drive this step: (code, text, silent), or None if it may.
        ``silent``: somebody else may be driving, so not even a zero command is sent."""
        stale = float(self.s["stale_s"])
        # Rule 1: the driver
        if self.override is None:
            return "override", "nothing on /nemesis/override yet (is the dashboard running?)", True
        if self.override:
            return "override", "the driver has the robot (override on)", True
        if now - self.override_t > stale:
            return "stale", f"/nemesis/override has been silent for {now - self.override_t:.1f} s", False
        # Rule 2: the sensors
        if self.pos is None:
            return "stale", "no position on /nemesis/centroid yet", False
        age = time.time() - self.pos_stamp
        if age > stale:
            return "stale", f"our position is {age:.1f} s old (vision lost?)", False
        if self.use_imu:
            if self.heading.seen is None:
                return "stale", "nothing on /nemesis/imu yet (start with --no-imu to run without it)", False
            if now - self.heading.seen > stale:
                return "stale", f"the IMU is {now - self.heading.seen:.1f} s old (telemetry lost?)", False
        return None

    # ── Saying and sending ────────────────────────────────────────────────

    def _print(self, text):
        print(f"[pilot {time.strftime('%H:%M:%S')}] {text}", flush=True)

    def _event(self, event, maneuver="", detail="", **more):
        self.events.publish(String(data=json.dumps(dict(t=time.time(), event=event, maneuver=maneuver, detail=detail, **more))))
        self._print(f"{event.upper():15s}{maneuver + ': ' if maneuver else ''}{detail}")

    def _wait(self, code, text, needs=None):
        """Not driving, and why: a ``waiting`` event when the reason changes, a reminder now and then."""
        name = self.plan[self.i]["name"] if self.i < len(self.plan) else ""
        self.ready_at = None
        if self._new((code, name)):
            self._event("waiting", name, text, code=code, needs=needs)
        elif self._new((code, name), every=10.0):
            self._print(f"{'still waiting':15s}{name}: {text}")

    def _new(self, what, every=None):
        """True if ``what`` (a reason for waiting, or being ready) should be announced now: it is
        not what was said last, and was not said in the last 2 s (a robot sitting right on a limit
        flips between two reasons at every noisy fix). With ``every``: a reminder is due."""
        now = time.monotonic()
        since = now - self.said_at.get(what, -math.inf)
        if since < (2.0 if every is None else every) or (every is None and what == self.said):
            return False
        self.said, self.said_at[what] = what, now
        return True

    def _send(self, throttle, turn):
        left, right = mix(throttle, turn, self.k)
        self.pub.publish(Float32MultiArray(data=[left, right, -1.0, 0.0]))    # rule 6: blade stopped, arm idle

    def _next(self):
        self.i += 1
        self.tried = self.probes = 0
        self.waited = 0.0

    # ── One control step ──────────────────────────────────────────────────

    def _tick(self):
        now = time.monotonic()
        if self.m is None and self.i >= len(self.plan) and now >= self.brake_until:
            self.done = True                      # the session is over, whoever has the robot
            return
        fresh, self._fresh = self._fresh, False
        at_rest = not self._moved(now)
        if self.heading_on:
            throttle, turn, when = self.sticks
            live = when is not None and now - when <= SILENCE_S
            self.heading.tick(*self.pos, fresh, throttle if live else 0.0, turn if live else 0.0, at_rest,
                              driver=bool(self.override))

        held = self._not_driving(now)
        if held:
            code, text, silent = held
            if self.m is not None:
                self._abort(text, brake=not silent)
            if silent:
                self.brake_until, self.waited = 0.0, 0.0
            self._wait(code, text)
            if silent:
                return
        if now < self.brake_until:                # stopping after an abort
            self._send(0.0, 0.0)
            return
        if held:
            return

        x, y = self.pos
        if not self.heading_on:                   # the first time the pilot has the robot
            self.heading.start(self.first_heading)
            self.heading_on = True
        if self.m is not None:
            self._play(x, y)
        else:
            self._prepare(x, y, now)

    def _play(self, x, y):
        """A maneuver is running: rule 4, then its next command."""
        m = self.m
        # Where it is, and where it will be in stop_s at the speed it has now (and half-way there)
        vx, vy = self._velocity(time.monotonic())
        stop_s = float(self.s["stop_s"])
        clear, what, ahead = min((*self.floor.clearance(x + vx * t, y + vy * t), t) for t in (0.0, stop_s / 2, stop_s))
        if clear < 0:
            self._abort(f"at ({x:.1f}, {y:.1f}) doing {math.hypot(vx, vy):.1f} ft/s: "
                        + ("" if ahead == 0.0 else f"{ahead:g} s from being ") + f"within {self.floor.margin:g} ft of {what}",
                        brake=True)
            self._send(0.0, 0.0)
            return
        t = self.step * CTRL_DT
        if t >= duration(m) - 1e-9:
            self.m, self.said = None, None
            self._event("maneuver_end", m["name"])
            if m["name"] != "heading_probe":
                self.played += 1
                self._next()
            return
        self._send(*command(m, t))
        self.step += 1

    def _abort(self, why, brake):
        """Stop the maneuver that is running. ``brake``: hold the treads at zero for brake_s
        (nobody else is driving); otherwise say nothing more at all."""
        m, self.m, self.said = self.m, None, None
        self.aborted += 1
        again = True                              # a heading probe is not one of the plan's maneuvers
        if m["name"] != "heading_probe":
            self.tried += 1
            again = self.tried < int(self.s["tries"])
        self._event("abort", m["name"], why + ("" if again else " (not tried again)"))
        if not again:
            self._next()
        if brake:
            self.brake_until = time.monotonic() + float(self.s["brake_s"])

    def _prepare(self, x, y, now):
        """Between maneuvers: rule 3. Start the next one once the robot has stood still and the
        maneuver has fitted for settle_s."""
        want = self.plan[self.i]
        moved = self._moved(now)
        m, why, needs = (None, f"waiting for the robot to stand still: {moved}", None) if moved else self._fit(want, x, y)
        if m is None:
            self.waited += CTRL_DT
            if not moved and 0 < self.give_up_s < self.waited:
                self.skipped += 1
                self._event("abort", want["name"], f"skipped, it could not start for {self.give_up_s:g} s: {why}")
                self._next()
                return
            self._wait("moving" if moved else "no_room", why, needs)
            return
        if self.ready_at is None:
            self.ready_at = now + float(self.s["settle_s"])
            if self._new(("ready", m["name"])):
                self._print(f"{'ready':15s}{m['name']} fits here: it starts in {self.s['settle_s']:g} s "
                            "unless the driver takes the robot (override)")
        elif now >= self.ready_at:
            self._start(m, x, y)
            self._play(x, y)                      # its first command goes out with the start event

    def _fit(self, want, x, y):
        """Rule 3 for one maneuver at this spot: (the maneuver to play, "", None), or
        (None, why it cannot start, the room it needs)."""
        k, share = self.k, float(self.s["min_share"])
        clear, what = self.floor.clearance(x, y)
        if clear < 0:
            return None, (f"the robot is at ({x:.1f}, {y:.1f}), {-clear:.1f} ft inside the {self.floor.margin:g} ft "
                          f"margin of {what}: drive it clear"), dict(ahead_ft=0.0, behind_ft=0.0, all_round=True)
        ahead, behind, turned = reach(want, k)
        least = reach(mildest(want, k, share), k)             # what it needs at its shortest and mildest
        needs = " and ".join(f"{n:.1f} ft {where}" for n, where in zip(least, ("ahead", "behind")) if n)
        slack = self.heading.slack
        straight = not turned and bool(ahead or behind)
        if straight and slack is not None:
            # A straight drive with a known heading: room along its line, both ways
            a = self.heading.value
            free_ahead, free_behind = self.floor.free(x, y, a, slack), self.floor.free(x, y, a + math.pi, slack)
            m = fit(want, free_ahead, free_behind, k, share)
            if m is not None:
                return m, "", None
            return None, (f"no room: it needs {needs}, and has {free_ahead:.1f} ft ahead and {free_behind:.1f} ft behind "
                          f"(heading {math.degrees(a):+.0f} deg, give or take {math.degrees(slack):.0f})"), \
                dict(ahead_ft=ahead, behind_ft=behind, all_round=False)
        # One that turns can end up anywhere within its reach. So can a straight one when nobody
        # knows which way the robot faces.
        m = fit(want, clear, clear, k, share)
        if m is not None:
            return m, "", None
        if not straight:
            return None, f"no room: it needs {least[0]:.1f} ft all round, and has {clear:.1f}", \
                dict(ahead_ft=ahead, behind_ft=behind, all_round=True)
        # Heading unknown (no IMU) and no room to find it with the maneuver itself: a short hop first
        probe = fit(MANEUVERS["heading_probe"], clear, clear, k) if self.probes < int(self.s["tries"]) else None
        if probe is not None:
            return probe, "", None
        return None, (f"heading unknown (no IMU) and no room here for a hop to find it ({clear:.1f} ft all round): "
                      f"drive the robot straight for {COURSE_FT + 0.5:g} ft or more, either way, to where it has {needs}"), \
            dict(ahead_ft=ahead, behind_ft=behind, all_round=False)

    def _start(self, m, x, y):
        self.m, self.step, self.said, self.ready_at = m, 0, None, None
        self.waited = 0.0
        if m["name"] == "heading_probe":
            self.probes += 1
        self.heading.maneuver(m)
        want = self.plan[self.i]
        lowered = (f", level lowered from {want['throttle']:+.2f} to fit" if m["name"] == want["name"]
                   and abs(m["throttle"]) < abs(want["throttle"]) - 1e-6 else "")
        slack = self.heading.slack
        facing = "heading unknown" if slack is None else f"heading {math.degrees(self.heading.value):+.0f} deg"
        self._event("maneuver_start", m["name"],
                    f"{describe(m)}{lowered}; {duration(m):.2f} s in all; from ({x:.1f}, {y:.1f}), {facing}",
                    settings=dict(m, t_start=time.time(), x_ft=x, y_ft=y,
                                  heading_rad=None if slack is None else self.heading.value))


def main():
    ap = argparse.ArgumentParser(description="Play test drives on /control/channels (see SIM2REAL/README.md).")
    ap.add_argument("--maneuvers", default="standard",
                    help="a session (standard, quick) or maneuver names separated by commas; "
                         "python SIM2REAL/maneuvers.py lists them")
    ap.add_argument("--settings", help="a YAML whose lines replace those of settings.yaml")
    ap.add_argument("--no-imu", action="store_true",
                    help="run without /nemesis/imu: no stale-IMU rule, heading from --heading and the direction of travel")
    ap.add_argument("--heading", type=float, metavar="DEG",
                    help="the way the robot faces when the pilot first gets it, counter-clockwise from +x "
                         "(default: initial_heading_deg in the settings)")
    ap.add_argument("--give-up", type=float, metavar="S",
                    help="skip a maneuver that could not start for this long (default: give_up_s in the settings)")
    args = ap.parse_args()

    rclpy.init()
    node = TestPilot(load_settings(args.settings), session(args.maneuvers), use_imu=not args.no_imu,
                     heading_deg=args.heading, give_up_s=args.give_up)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.done:
            node._print(f"Session complete: {node.played} maneuvers played, {node.aborted} aborts, {node.skipped} skipped.")
            end = time.monotonic() + 0.5          # let the last events out
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node._print("Stopped from outside: nothing more is sent, the PPM output centres the sticks.")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
