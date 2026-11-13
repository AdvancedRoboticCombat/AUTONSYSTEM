"""
saw_policy_node: flies a hammer-saw policy trained in auton_training (a SAW_*
checkpoint) on the real robot.

All of the policy-side logic (the 44-column observation, the arm gate, the
fight clock, the tread mixer, the heading estimate) lives in
auton_training/deploy.py, where it is tested against the simulator. This file
only connects it to ROS.

In
    /nemesis/centroid, /opponent/centroid   geometry_msgs/PointStamped, arena coordinates from the
                                            dashboard backend (point.z = 1 marks a coasted position)
    /nemesis/imu                            sensor_msgs/Imu: yaw rate and orientation (required)
    /nemesis/override                       std_msgs/Bool: true = the driver has the robot
    /nemesis/policy/reset                   std_msgs/Empty: the next release of override starts a new fight
    /nemesis/weapon/erpm                    std_msgs/Float64: blade speed, used only if blade_full_erpm is set
Out
    /control/channels                       std_msgs/Float32MultiArray [left, right, blade, arm]
                                            left, right  tread speed, -1..1 of top speed
                                            blade        -1 stopped .. +1 full   (held at -1 unless weapon_enabled)
                                            arm          motor torque: +1 forward for one swing, -1 back
                                                         for one retract, 0 otherwise. Never held on
                                                         after the arm has arrived: the clutch would
                                                         slip. (0 always unless weapon_enabled)

The fight (swing budget, the policy's memory, and the clock the policy reads
as "time left") starts the first time the policy has the robot: the first
control step with override off and both robots and the IMU in view. Nothing
is published while override is on or while our own position or the IMU is
stale. The clock is only an input to the policy; when it reaches zero the
policy keeps driving unless stop_at_bell is set.

Parameters are documented in config/saw_policy.yaml.
"""
import math
import os
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Empty, Float32MultiArray, Float64


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _find_auton_dir(given=""):
    """auton_training/, from the parameter, NEMESIS_ROOT, or this file's place in the repository."""
    tried = []
    for base in (given, os.path.join(os.environ.get("NEMESIS_ROOT", ""), "auton_training") if os.environ.get("NEMESIS_ROOT") else "",
                 str(Path(os.path.realpath(__file__)).parents[2] / "auton_training")):
        if base:
            tried.append(base)
            if (Path(base) / "deploy.py").is_file():
                return str(Path(base).resolve())
    raise FileNotFoundError("auton_training/deploy.py not found (looked in " + ", ".join(tried)
                            + "). Set the auton_dir parameter or NEMESIS_ROOT.")


class SawPolicyNode(Node):
    def __init__(self):
        super().__init__("saw_policy_node")
        P = lambda name, default: self.declare_parameter(name, default).value

        # The simulator's code is plain scripts, not a package: put its folder on the path.
        # It runs on the GPU when there is one (DODGE_DEVICE=cpu forces the CPU).
        auton = _find_auton_dir(str(P("auton_dir", "")))
        if auton not in sys.path:
            sys.path.insert(0, auton)
        import torch
        torch.set_num_threads(2)
        from config import CTRL_DT
        from deploy import Arena, HeadingEstimator, PilotConfig, SawPilot

        # checkpoint_override is the launch file's `checkpoint:=` argument; empty leaves the settings file's choice
        checkpoint = (str(P("checkpoint_override", "")) or str(P("checkpoint", ""))
                      or os.path.join(auton, "ppo_models", "saw_v8a_traves_hold", "SAW_upd375.pt"))
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(auton, checkpoint)

        # Where the camera's arena coordinates sit in the policy's frame
        units = str(P("units", "ft"))
        if units not in ("ft", "m"):
            raise ValueError(f"units must be ft or m, not {units!r}")
        width, height = float(P("arena_width_ft", 0.0)), float(P("arena_height_ft", 0.0))
        self.arena = Arena(width or None, height or None, units_ft=1.0 if units == "ft" else 1.0 / 0.3048,
                           origin=str(P("origin", "corner")), y_down=bool(P("y_down", False)))

        cfg = PilotConfig(
            half_len=float(P("half_len_ft", 0.9)), half_wid=float(P("half_wid_ft", 1.8)),
            half_track=float(P("half_track_ft", 1.48)),
            reach=float(P("reach_ft", 0.8)), tip=float(P("tip_ft", 0.6)),
            latency=float(P("latency_s", 0.15)), cmd_lat=float(P("cmd_latency_s", 0.07)),
            fire_s=float(P("fire_s", 0.35)), retract_s=float(P("retract_s", 0.50)),
            # 0 = drive the arm for exactly its swing time
            arm_drive_s=float(P("arm_drive_s", 0.0)) or None, arm_return_s=float(P("arm_return_s", 0.0)) or None,
            swings=int(P("swings", 10)), fight_s=float(P("fight_seconds", 180.0)),
            opp_kind=str(P("opponent_kind", "horizontal")),
            opp_half_len=float(P("opponent_half_len_ft", 2.5)), opp_half_wid=float(P("opponent_half_wid_ft", 1.0)),
            opp_fore=float(P("opponent_weapon_fore_ft", 2.5)), opp_rad=float(P("opponent_weapon_rad_ft", 1.5)),
            opp_wide=float(P("opponent_weapon_wide_ft", 0.0)))
        self.pilot = SawPilot(checkpoint, cfg, self.arena)
        self.heading = HeadingEstimator(lag_s=max(0.0, cfg.latency - cfg.cmd_lat))

        self.weapon_enabled = bool(P("weapon_enabled", False))
        self.stop_at_bell = bool(P("stop_at_bell", False))
        self.stale_s = float(P("stale_s", 1.0))
        self.gyro_sign = float(P("gyro_sign", 1.0))
        self.yaw_source = str(P("yaw_source", "orientation"))
        if self.yaw_source not in ("orientation", "gyro"):
            raise ValueError(f"yaw_source must be orientation or gyro, not {self.yaw_source!r}")
        self.initial_heading = str(P("initial_heading", "opponent"))
        if self.initial_heading != "opponent":
            float(self.initial_heading)                     # degrees; fail now, not at the bell
        self.blade_full_erpm = float(P("blade_full_erpm", 0.0))

        # Latest sensor values
        self._own = self._opp = None          # (x, y) simulator feet, last real detection
        self._own_t = self._opp_t = None      # rclpy Time of it
        self._own_new = False                 # a new own fix since the last control step
        self._gyro = 0.0
        self._imu_t = None
        self._imu_stamp = None                # header time of the last IMU message, s
        self._yaw_last = None                 # IMU orientation yaw at the last message
        self._dyaw = 0.0                      # yaw turned through since the last control step
        self._erpm = None
        self._override = False
        self._fight_on = False                # the policy has had the robot in this fight
        self._paused_at = None                # when the policy stopped driving in this fight (None: it is driving)
        self._tracking = False                # the heading estimate is running
        self._throttle = 0.0
        self._bell_said = False

        self.create_subscription(PointStamped, "/nemesis/centroid", self._on_own, 10)
        self.create_subscription(PointStamped, "/opponent/centroid", self._on_opp, 10)
        self.create_subscription(Imu, "/nemesis/imu", self._on_imu, 50)
        self.create_subscription(Bool, "/nemesis/override", self._on_override, 10)
        self.create_subscription(Empty, "/nemesis/policy/reset", self._on_reset, 10)
        self.create_subscription(Float64, "/nemesis/weapon/erpm", self._on_erpm, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/control/channels", 10)
        self.dt = CTRL_DT
        self.create_timer(CTRL_DT, self._tick)

        meta = self.pilot.meta
        self.get_logger().info(
            f"Saw policy {os.path.relpath(checkpoint, auton)} (update {meta.get('update')}, {self.pilot.agent.obs_dim} inputs, "
            f"gate {'on, rule ' + self.pilot.gate_rule if self.pilot.gated else 'off'}). "
            f"Arena {self.arena.width:.0f} x {self.arena.height:.0f} ft"
            f"{' (BattleBox walls)' if self.arena.wall is not None else ' (plain)'}, positions in {units} from a "
            f"{self.arena.origin}. Weapon output {'ENABLED' if self.weapon_enabled else 'off (drive only)'}.")

    # ── Sensors ───────────────────────────────────────────────────────────

    def _on_own(self, msg):
        if msg.point.z > 0.5:                 # the tracker is coasting: not a fix, keep the last real one
            return
        self._own = self.arena.to_sim(msg.point.x, msg.point.y)
        self._own_t = Time.from_msg(msg.header.stamp)
        self._own_new = True

    def _on_opp(self, msg):
        if msg.point.z > 0.5:
            return
        self._opp = self.arena.to_sim(msg.point.x, msg.point.y)
        self._opp_t = Time.from_msg(msg.header.stamp)

    def _on_imu(self, msg):
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self._gyro = self.gyro_sign * float(msg.angular_velocity.z)
        q = msg.orientation
        norm = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
        if self.yaw_source == "orientation" and 0.9 < norm < 1.1:
            # The IMU's own fused orientation: no drift from integrating a rate here, and
            # nothing lost when a packet is
            yaw = self.gyro_sign * math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            if self._yaw_last is not None:
                self._dyaw += _wrap(yaw - self._yaw_last)
            self._yaw_last = yaw
        elif self._imu_stamp is not None:
            self._dyaw += self._gyro * min(max(stamp - self._imu_stamp, 0.0), 0.2)
        self._imu_stamp = stamp
        self._imu_t = self.get_clock().now()

    def _on_erpm(self, msg):
        self._erpm = float(msg.data)

    def _on_override(self, msg):
        self._override = bool(msg.data)

    def _on_reset(self, _msg):
        self._fight_on = False
        self._paused_at = None
        self._bell_said = False
        self.get_logger().info("Reset: the next time the policy has the robot starts a new fight.")

    # ── Control step ──────────────────────────────────────────────────────

    def _age(self, t):
        return (self.get_clock().now() - t).nanoseconds / 1e9

    def _tick(self):
        log = self.get_logger()
        dyaw, self._dyaw = self._dyaw, 0.0
        fresh, self._own_new = self._own_new, False
        if self._own is None or self._opp is None or self._imu_t is None:
            log.info("Waiting for " + ", ".join(name for name, have in (
                ("our position (/nemesis/centroid)", self._own is not None),
                ("their position (/opponent/centroid)", self._opp is not None),
                ("the IMU (/nemesis/imu)", self._imu_t is not None)) if not have) + ".", throttle_duration_sec=5)
            return
        x, y = self._own
        ex, ey = self._opp
        dist = math.hypot(ex - x, ey - y)

        # The heading estimate runs from the moment there is data, whoever is driving
        if not self._tracking:
            self.heading.reset(self._first_heading(x, y, ex, ey))
            self._tracking = True
        a = self.heading.update(x, y, fresh, self._gyro, dyaw, 0.0 if self._override else self._throttle, dist)

        if self._override:
            self._pause()
            return
        own_age, imu_age = self._age(self._own_t), self._age(self._imu_t)
        if own_age > self.stale_s or imu_age > self.stale_s:
            what = (f"our position is {own_age:.1f} s old (vision lost?)" if own_age > self.stale_s
                    else f"the IMU is {imu_age:.1f} s old (telemetry lost?)")
            log.warn(f"Not driving: {what}.", throttle_duration_sec=1)
            self._pause()
            return

        if not self._fight_on:
            # The bell: both robots in their squares. The heading starts from where the robot was set down.
            self.pilot.reset()
            self.heading.reset(self._first_heading(x, y, ex, ey))
            a = self.heading.heading
            self._fight_on, self._paused_at = True, None
            log.info(f"Fight started: {self.pilot.cfg.fight_s:.0f} s, {self.pilot.cfg.swings} swings, "
                     f"heading {math.degrees(a):+.0f} deg, opponent {dist:.1f} ft away.")
        elif self._paused_at is not None:
            self.pilot.resume(self._age(self._paused_at))    # the fight clock kept running
            self._paused_at = None

        if self.stop_at_bell and self.pilot.time_left <= 0.0:
            if not self._bell_said:
                log.info("The fight clock has run out: the policy has stopped. "
                         "Publish std_msgs/Empty on /nemesis/policy/reset for another fight.")
                self._bell_said = True
            return

        blade_read = -1.0
        if self.blade_full_erpm > 0.0 and self._erpm is not None:
            blade_read = min(1.0, abs(self._erpm) / self.blade_full_erpm)
        cmd = self.pilot.step(x, y, ex, ey, a, self._gyro, blade_read=blade_read)
        self._throttle = cmd.throttle

        msg = Float32MultiArray()
        msg.data = [cmd.left, cmd.right,
                    cmd.blade if self.weapon_enabled else -1.0,
                    cmd.arm_motor if self.weapon_enabled else 0.0]
        self.pub.publish(msg)
        log.info(f"t-{cmd.time_left:5.1f}s  heading {math.degrees(a):+4.0f} deg  L {cmd.left:+.2f} R {cmd.right:+.2f}  "
                 f"blade {cmd.blade:+.2f}  arm {'SWING' if cmd.arm_motor > 0 else 'back' if cmd.arm_motor < 0 else 'idle'}  "
                 f"gate {'open' if cmd.gate_open else 'shut'}  swings left {cmd.swings_left}  "
                 f"opponent {dist:4.1f} ft" + ("" if self.weapon_enabled else "  [weapon output off]"),
                 throttle_duration_sec=1)
        if cmd.fired:
            log.info(f"Swing {self.pilot.cfg.swings - cmd.swings_left} of {self.pilot.cfg.swings}"
                     + ("" if self.weapon_enabled else " (not sent: weapon output off)"))

    def _pause(self):
        """The policy is not driving this step (the driver has the robot, or a sensor is lost)."""
        self._throttle = 0.0
        if self._fight_on and self._paused_at is None:
            self._paused_at = self.get_clock().now()

    def _first_heading(self, x, y, ex, ey):
        if self.initial_heading == "opponent":
            return math.atan2(ey - y, ex - x)
        return self.arena.heading_to_sim(math.radians(float(self.initial_heading)))


def main(args=None):
    rclpy.init(args=args)
    node = SawPolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # On Ctrl-C under `ros2 launch` the context is already shut down.
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
