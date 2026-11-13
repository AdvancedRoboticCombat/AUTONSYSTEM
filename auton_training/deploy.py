"""
Running a saw policy on the real robot.

The policy only works if it is handed the observation it was trained on. This
file builds that observation, runs the arm gate and mixes the drive exactly as
the simulator does, from what the robot has: a camera fix of both robots, a
gyro, and the numbers the team measures before a fight. eval/deploy_check.py
plays the same fights through this file and on the simulator's own observation.

    arena = Arena()                                  # the 48 ft BattleBox
    pilot = SawPilot("ppo_models/saw_v8a_traves_hold/SAW_upd375.pt", PilotConfig(), arena)
    heading = HeadingEstimator()

    pilot.reset(); heading.reset(first_heading)
    every CTRL_DT (0.03 s):
        x, y   = arena.to_sim(our fix)               # ft, simulator frame
        ex, ey = arena.to_sim(their fix)
        a = heading.update(x, y, fresh, gyro, gyro * CTRL_DT, last_throttle)
        cmd = pilot.step(x, y, ex, ey, a, gyro)
        send cmd.left, cmd.right, cmd.blade, cmd.arm_motor

Frame and units (the simulator's): feet, seconds, radians. The origin is an
arena corner, x and y run along two walls, headings are atan2(y, x) and grow
counter-clockwise seen from above, and the gyro is positive the same way.

nemesis_policy_engine/saw_policy.py is the ROS node around this file.
"""
import math
from collections import deque
from dataclasses import dataclass

import torch

import saw_walls
from agent import load_agent, gate_from_meta
from config import (
    ARENA_FT, ARM_SWITCH, CTRL_DT, DEVICE, DT, ENEMY_VEL_WINDOW, N_RAYS, OUR_WEAPON_TIP_RANGE, PHYS_PER_CTRL,
    RAY_RANGE_FT, RAY_STEP_FT, SAW_ACT_DIM, V_MAX_FT, WALL_CLAMP_IN, YAW_RATE_MAX, angle_wrap,
)
from saw_gate import ArmGate, CAL_KEYS, G_FIRING, G_RETRACT, W_NONE, W_VERT, W_DRUM, W_HORIZ

WEAPON_KINDS = {"none": W_NONE, "vert": W_VERT, "drum": W_DRUM, "horizontal": W_HORIZ}


# ─── What the team measures or writes down before a fight ─────────────────

@dataclass
class PilotConfig:
    """Our robot, the opponent's card and the fight. Feet and seconds.

    The defaults are the simulator's nominal robot and a Traves-like bar
    spinner. None of the robot numbers has been measured yet; the policy was
    trained across a range around each (config.SAW_DR, DR_*), so close is
    enough, but the gate's timing numbers decide where the hammer lands.
    """
    # Our body, centre to edge
    half_len: float = 0.9           # CAD body: 1.78 ft front to back
    half_wid: float = 1.8           # and 3.63 ft side to side
    half_track: float = 1.48        # centre to tread centreline (0.90 m track in the CAD); sets the drive mixer
    # The hammer saw, for the arm gate
    reach: float = 0.8              # where the saw comes down, beyond our front edge (CAD: 1.67 ft ahead of centre)
    tip: float = OUR_WEAPON_TIP_RANGE   # contact radius of the saw
    latency: float = 0.15           # camera capture to the arm starting to move (sensing + command delay)
    cmd_lat: float = 0.07           # command sent to the arm starting to move
    fire_s: float = 0.35            # full swing, back to front
    retract_s: float = 0.50         # and front to back
    # How long the arm motor is driven each way. None: for the swing time itself, so the torque
    # stops as the arm arrives. The motor goes through a slip clutch: holding it against the
    # target or against its back stop grinds the clutch.
    arm_drive_s: float = None       # forward, per swing
    arm_return_s: float = None      # back, per swing
    # The fight
    swings: int = 10                # swings the arm has in it
    fight_s: float = 180.0
    # The opponent's card
    opp_kind: str = "horizontal"    # none, vert, drum or horizontal
    opp_half_len: float = 2.5
    opp_half_wid: float = 1.0
    opp_fore: float = 2.5           # weapon centre ahead of their centre (a horizontal's hub)
    opp_rad: float = 1.5            # weapon contact radius (a horizontal's bar tip radius)
    opp_wide: float = 0.0           # half width of a vert's or drum's front; 0 for a horizontal

    def __post_init__(self):
        if self.opp_kind not in WEAPON_KINDS:
            raise ValueError(f"opp_kind must be one of {sorted(WEAPON_KINDS)}, not {self.opp_kind!r}")


# ─── The arena ────────────────────────────────────────────────────────────

class Arena:
    """Where the walls are, and how the camera's arena coordinates map onto
    the simulator's 48 ft frame.

    Arena()                     the BattleBox the policy trained in: 48 ft
                                square with the simulator's wall bulges
                                (saw_walls). Its y = 0 wall is the one with
                                the band; orient the arena points to match.
    Arena(width_ft, height_ft)  a plain rectangular test floor, centred in
                                the 48 ft frame. The rays see only its walls.

    ``units_ft`` is feet per unit of the incoming coordinates (0.3048 for
    metres), ``origin`` "corner" or "center", and ``y_down`` True when the
    incoming y axis runs the other way (image-style coordinates), which would
    otherwise mirror every turn.
    """

    def __init__(self, width_ft=None, height_ft=None, units_ft=1.0, origin="corner", y_down=False):
        if origin not in ("corner", "center"):
            raise ValueError(f"origin must be corner or center, not {origin!r}")
        plain = width_ft is not None or height_ft is not None
        self.width = float(width_ft if width_ft is not None else ARENA_FT)
        self.height = float(height_ft if height_ft is not None else ARENA_FT)
        if not (0 < self.width <= ARENA_FT and 0 < self.height <= ARENA_FT):
            raise ValueError(f"the arena must fit the policy's {ARENA_FT} ft frame, not {self.width} x {self.height} ft")
        self.units_ft, self.origin, self.y_down = float(units_ft), origin, bool(y_down)
        self.off_x, self.off_y = (ARENA_FT - self.width) / 2, (ARENA_FT - self.height) / 2
        self.lo_x, self.hi_x = self.off_x + WALL_CLAMP_IN, self.off_x + self.width - WALL_CLAMP_IN
        self.lo_y, self.hi_y = self.off_y + WALL_CLAMP_IN, self.off_y + self.height - WALL_CLAMP_IN
        self.wall = None if plain else saw_walls.raster()[0].flatten()
        self.ray_angles = torch.arange(N_RAYS, device=DEVICE) * (2 * math.pi / N_RAYS)
        self.ray_t = torch.arange(RAY_STEP_FT, RAY_RANGE_FT + 1e-6, RAY_STEP_FT, device=DEVICE)

    def to_sim(self, x, y):
        """A point in the camera's arena coordinates, as simulator feet."""
        x, y = x * self.units_ft, y * self.units_ft
        if self.origin == "center":
            x, y = x + self.width / 2, y + self.height / 2
        if self.y_down:
            y = self.height - y
        return x + self.off_x, y + self.off_y

    def heading_to_sim(self, angle):
        """A direction given as atan2(dy, dx) in the camera's arena coordinates, rad, as a simulator heading."""
        return -angle if self.y_down else angle

    def rays(self, x, y, a):
        """environment.Observer._rays: distance along each body-frame ray to
        the wall or the first wall cell, as a share of RAY_RANGE_FT."""
        ang = a.unsqueeze(1) + self.ray_angles
        cx, sy = torch.cos(ang), torch.sin(ang)
        x1, y1 = x.unsqueeze(1), y.unsqueeze(1)
        tiny = 1e-6
        tx = torch.where(cx > 0, (self.hi_x - x1) / cx.clamp(min=tiny), (self.lo_x - x1) / cx.clamp(max=-tiny))
        ty = torch.where(sy > 0, (self.hi_y - y1) / sy.clamp(min=tiny), (self.lo_y - y1) / sy.clamp(max=-tiny))
        reach = torch.minimum(tx, ty).clamp(0.0, RAY_RANGE_FT)
        if self.wall is not None:
            px = (x1.unsqueeze(2) + cx.unsqueeze(2) * self.ray_t).long().clamp(0, ARENA_FT - 1)
            py = (y1.unsqueeze(2) + sy.unsqueeze(2) * self.ray_t).long().clamp(0, ARENA_FT - 1)
            blocked = self.wall[py * ARENA_FT + px]
            first = torch.where(blocked, self.ray_t.expand_as(blocked),
                                torch.full_like(blocked, RAY_RANGE_FT, dtype=torch.float32)).amin(dim=2)
            reach = torch.minimum(reach, first)
        return reach / RAY_RANGE_FT


# ─── Our heading ──────────────────────────────────────────────────────────

class HeadingEstimator:
    """Our heading from the gyro, kept honest by the camera track.

    The camera gives no heading. A gyro alone drifts without bound, and the
    policy was trained on a heading that wanders around the truth by a
    fraction of a radian at most (config.SAW_DR hdg_bias). So whenever the
    robot covers ground, the line it actually travelled along (from its own
    position fixes) pulls the estimate back. Which end led is taken to be
    the one the estimate already says, so this only works while the estimate
    is within a quarter turn of the truth; a larger disagreement than
    ``max_step`` is left alone as a slide, a shove or a hit.

    The start is the exception. For the first ``launch_s`` seconds, while the
    opponent is still further than ``launch_clear_ft``, the robot pulls away
    from rest under its own throttle, so the throttle's sign says which end
    leads. If the first such run disagrees with the estimate by more than
    ``max_step``, the estimate is replaced: a robot set down facing the wrong
    way is put right as it leaves its square. After that nothing recovers a
    heading that is more than a quarter turn out.

    On six recorded simulator fights: 0.11 rad
    rms with a gyro drifting 0.002 rad/s, 0.16 with 0.01 rad/s. Without a
    gyro, turning rate guessed from the commands, it was over 1 rad: the
    policy cannot fly without the IMU.

    Plain floats; one call per control step.
    """

    def __init__(self, gain=0.08, window=8, lag_s=0.08, min_travel_ft=1.2, max_gyro=3.0, max_step=0.6,
                 launch_s=4.0, launch_clear_ft=7.0, min_throttle=0.3):
        self.gain, self.window, self.min_travel = gain, int(window), min_travel_ft
        self.max_gyro, self.max_step, self.min_throttle = max_gyro, max_step, min_throttle
        self.launch_steps, self.launch_clear = int(round(launch_s / CTRL_DT)), launch_clear_ft
        self.lag = max(0, int(round(lag_s / CTRL_DT)))       # control steps a fix is older than the gyro
        self.heading = 0.0
        self.reset(0.0)

    def reset(self, heading):
        """A new fight, with the heading the robot is believed to start at."""
        self.heading = float(angle_wrap(heading))
        self._fix = deque(maxlen=self.window + 1)            # (x, y, fresh)
        self._drive = deque(maxlen=self.window + 1)          # (gyro, throttle)
        self._hdg = deque(maxlen=self.window + self.lag + 1) # heading after each update
        self._age = 0
        self.launched = False
        self.corrections = 0                                 # nudges from the track so far
        self.relaunched = False                              # the start heading was replaced

    def update(self, x, y, fresh, gyro, dyaw, throttle, enemy_dist=None):
        """One control step. ``x, y`` our position fix (ft) and whether it is
        new this step, ``gyro`` the yaw rate now (rad/s), ``dyaw`` the yaw
        turned through since the last call (rad; from the IMU's own
        orientation where it gives one, else gyro x time), ``throttle`` the
        throttle sent last step (-1..1), ``enemy_dist`` ft to their fix.
        Returns the heading (rad)."""
        self.heading = float(angle_wrap(self.heading + dyaw))
        self._fix.append((float(x), float(y), bool(fresh)))
        self._drive.append((float(gyro), float(throttle)))
        self._hdg.append(self.heading)
        self._age += 1
        if len(self._fix) <= self.window or len(self._hdg) < self._hdg.maxlen:
            return self.heading
        (x0, y0, f0), (x1, y1, f1) = self._fix[0], self._fix[-1]
        dx, dy = x1 - x0, y1 - y0
        if (not (f0 and f1) or max(abs(d[0]) for d in self._drive) > self.max_gyro
                or math.hypot(dx, dy) < self.min_travel):
            return self.heading
        course = math.atan2(dy, dx)
        # The fixes span the window that ended ``lag`` steps ago. On an arc the chord points the
        # way the robot faced at the window's middle, so that is the heading to compare with.
        then = self._hdg[len(self._hdg) - 1 - self.lag - self.window // 2]
        fwd = float(angle_wrap(course - then))               # if the nose led
        back = float(angle_wrap(course + math.pi - then))    # if the tail did
        near = fwd if abs(fwd) <= abs(back) else back
        if not self.launched and self._age <= self.launch_steps:
            thr = [d[1] for d in self._drive]
            pulling = all(t > self.min_throttle for t in thr) or all(t < -self.min_throttle for t in thr)
            if pulling and (enemy_dist is None or enemy_dist > self.launch_clear):
                self.launched = True
                by_throttle = fwd if thr[0] > 0 else back
                if abs(by_throttle) > self.max_step:
                    self._shift(by_throttle)
                    self.relaunched = True
                    return self.heading
        if abs(near) <= self.max_step:
            self._shift(self.gain * near)
            self.corrections += 1
        return self.heading

    def _shift(self, d):
        self.heading = float(angle_wrap(self.heading + d))
        self._hdg = deque((float(angle_wrap(h + d)) for h in self._hdg), maxlen=self._hdg.maxlen)


# ─── The pilot ────────────────────────────────────────────────────────────

@dataclass
class Command:
    """One control step's output.

    left, right   tread speeds as a share of top speed, -1..1
    blade         blade speed asked for: -1 stopped .. +1 full
    arm_motor     what to send the arm motor: +1 drive forward (a swing), -1 drive back, 0 no
                  torque. Only non-zero while the arm is travelling: never held down on the
                  target after a swing, never held against the back stop while parked.
    arm           the gate's arm switch, as the simulator has it: +1 down (swing and hold),
                  -1 parked. Not for a motor: it stays on for as long as the state lasts.
    throttle, turn, arm_request   the policy's own outputs
    gate_open     the gate would carry out a swing asked for now
    fired         the gate started a swing this step
    swings_left, time_left (s)
    """
    left: float
    right: float
    blade: float
    arm_motor: float
    arm: float
    throttle: float
    turn: float
    arm_request: float
    gate_open: bool
    fired: bool
    swings_left: int
    time_left: float


class SawPilot:
    """A saw checkpoint with everything the simulator wraps round it: the
    44-column observation, the arm gate, the fight clock and the tread mixer.
    One robot. ``step`` is called once per CTRL_DT."""

    def __init__(self, checkpoint, cfg=None, arena=None):
        self.agent, self.meta = load_agent(str(checkpoint), DEVICE)
        if self.meta.get("rules") != "saw" or self.agent.act_dim != SAW_ACT_DIM:
            raise ValueError(f"{checkpoint} is not a saw policy ({self.meta.get('rules')} rules, "
                             f"{self.agent.act_dim} actions); this pilot runs SAW_* checkpoints")
        self.cfg = cfg or PilotConfig()
        self.arena = arena or Arena()
        self.gated = gate_from_meta(self.meta)
        self.gate_rule = self.meta.get("gate_rule") or "heading"
        self.gate_persist = int(self.meta.get("gate_persist") or 1)
        self.gate = ArmGate(1)
        # Treads: full throttle and full turn together ask one side for more than its top speed
        self.tread_k = self.cfg.half_track * YAW_RATE_MAX / V_MAX_FT
        self.reset()

    def reset(self):
        """A new fight, from the bell: memory cleared, arm parked, blade
        stopped, a full swing budget and a full clock."""
        cfg = self.cfg
        row = lambda v, dtype=torch.float32: torch.tensor([v], dtype=dtype, device=DEVICE)
        self.state = self.agent.initial_state(1, DEVICE)
        self.fresh = True
        self.prev_action = torch.zeros(1, SAW_ACT_DIM, device=DEVICE)
        self.own_hist = self.other_hist = None
        self.hist_ptr = 0
        self.steps = 0                                       # physics steps since the bell
        self.horizon = int(round(cfg.fight_s / DT))
        self.cal = {k: row(float(getattr(cfg, k))) for k in CAL_KEYS}
        self.gate.reset(torch.zeros(1, dtype=torch.long, device=DEVICE), row(int(cfg.swings), torch.long), self.cal)
        self.gate_run = 0
        self.gate_open = torch.zeros(1, dtype=torch.bool, device=DEVICE)
        self.blade_sent, self.thr_sent = row(-1.0), row(0.0)
        self.card = dict(kind=row(WEAPON_KINDS[cfg.opp_kind], torch.long), len=row(cfg.opp_half_len),
                         wid=row(cfg.opp_half_wid), fore=row(cfg.opp_fore), rad=row(cfg.opp_rad),
                         wide=row(cfg.opp_wide))
        self.obs = None

    def resume(self, paused_s=0.0):
        """The policy takes the robot back after ``paused_s`` seconds in
        which it was not driving (the driver had it, or the sensors were
        lost): same fight, same swings and memory, and the clock has kept
        running. The position histories start over (so the pause does not
        read as one huge jump) and the last drive command is taken as none.
        Nobody knows what the weapon did meanwhile: a swing in progress is
        taken back, and the blade has to be spun up again before the gate
        opens."""
        self.steps += int(round(max(0.0, paused_s) / DT))
        self.own_hist = self.other_hist = None
        self.prev_action[:, 0:2] = 0.0
        self.thr_sent = torch.zeros_like(self.thr_sent)
        self.gate.force_retract(torch.ones(1, dtype=torch.bool, device=DEVICE))
        self.gate.blade_est = torch.zeros_like(self.gate.blade_est)
        self.gate_run = 0

    @property
    def time_left(self):
        return max(0, self.horizon - self.steps) * DT

    def _observe(self, x, y, ex, ey, a, gyro, arm_read, blade_read):
        """saw_env.SawEnv._observe for one robot, on its real sensors."""
        cfg = self.cfg
        first = self.fresh                       # the first observation of the fight
        seed = self.own_hist is None             # the position histories start here (fight start, or resume)
        if seed:
            self.own_hist = torch.stack([x, y], dim=1).unsqueeze(1).repeat(1, ENEMY_VEL_WINDOW, 1)
            self.other_hist = torch.stack([ex, ey], dim=1).unsqueeze(1).repeat(1, ENEMY_VEL_WINDOW, 1)
            self.hist_ptr = 0

        # environment.Observer.observe, with our speed taken from our own track (speed_from_track)
        oldest = self.other_hist[:, self.hist_ptr]
        evx_w = (ex - oldest[:, 0]) / (ENEMY_VEL_WINDOW * CTRL_DT)
        evy_w = (ey - oldest[:, 1]) / (ENEMY_VEL_WINDOW * CTRL_DT)
        was = self.own_hist[:, self.hist_ptr]
        speed = (((x - was[:, 0]) * torch.cos(a) + (y - was[:, 1]) * torch.sin(a))
                 / (ENEMY_VEL_WINDOW * CTRL_DT)).clamp(-1.5 * V_MAX_FT, 1.5 * V_MAX_FT)
        if not seed:
            self.other_hist[:, self.hist_ptr, 0] = ex
            self.other_hist[:, self.hist_ptr, 1] = ey
            self.own_hist[:, self.hist_ptr, 0] = x
            self.own_hist[:, self.hist_ptr, 1] = y
            self.hist_ptr = (self.hist_ptr + 1) % ENEMY_VEL_WINDOW
        cos_a, sin_a = torch.cos(a), torch.sin(a)
        dx, dy = ex - x, ey - y
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        fwd, left = (dx * cos_a + dy * sin_a) / dist, (-dx * sin_a + dy * cos_a) / dist
        half = ARENA_FT / 2
        prev = self.prev_action
        base = torch.cat([torch.stack([
            (x - half) / half, (y - half) / half,
            sin_a, cos_a,
            speed / V_MAX_FT,
            gyro / YAW_RATE_MAX,
            prev[:, 0], prev[:, 1],
            fwd, left,
            (dist / ARENA_FT).clamp(max=1.0),
            torch.exp(-dist / 8.0),
            (( evx_w * cos_a + evy_w * sin_a) / V_MAX_FT).clamp(-1.5, 1.5),
            ((-evx_w * sin_a + evy_w * cos_a) / V_MAX_FT).clamp(-1.5, 1.5),
            (ex - half) / half, (ey - half) / half,
        ], dim=1), self.arena.rays(x, y, a),
            # SawEnv._weapon_obs
            torch.stack([prev[:, 2], prev[:, 3], arm_read, blade_read,
                         torch.full_like(x, cfg.half_len - 2.0), torch.full_like(x, cfg.half_wid - 2.0)], dim=1),
        ], dim=1)

        if not first:
            self.gate.advance(CTRL_DT, base, self.blade_sent, self.thr_sent, self.gated)
        hdg = torch.zeros(1, 6, device=DEVICE)               # the camera gives no opponent heading
        open_ = self.gate.open(base, self.card, hdg, rule=self.gate_rule)
        if first:
            self.gate_run = 0
            open_ = open_ & (self.gate_persist <= 1)
        else:
            self.gate_run = self.gate_run + 1 if bool(open_[0]) else 0
            open_ = torch.full_like(open_, self.gate_run >= self.gate_persist)
        self.gate_open = open_
        kind = self.card["kind"]
        left_steps = torch.tensor([max(0, self.horizon - self.steps)], dtype=torch.float32, device=DEVICE)
        extra = torch.stack([(self.gate.swings_left().float() / 10.0).clamp(0.0, 3.0),
                             left_steps * DT / 180.0, open_.float(),
                             (kind == W_VERT).float(), (kind == W_DRUM).float(), (kind == W_HORIZ).float(),
                             self.card["len"] - 2.0, self.card["wid"] - 2.0], dim=1)
        return torch.cat([base, extra, hdg], dim=1)

    @torch.no_grad()
    def step(self, x, y, ex, ey, heading, gyro, arm_read=-1.0, blade_read=-1.0):
        """One control step.

        x, y, ex, ey   our and their position fix, simulator feet (Arena.to_sim).
                       A robot the camera has lost keeps its last fix.
        heading        our heading, rad (HeadingEstimator)
        gyro           our yaw rate, rad/s, counter-clockwise positive
        arm_read       arm position 0 (parked) .. 1 (down), or -1 with no sensor
        blade_read     blade speed as a share of full, or -1 with no reading
        """
        t = lambda v: torch.tensor([float(v)], dtype=torch.float32, device=DEVICE)
        obs = self._observe(t(x), t(y), t(ex), t(ey), t(heading), t(gyro), t(arm_read), t(blade_read))
        self.obs = obs
        done = torch.full((1,), 1.0 if self.fresh else 0.0, device=DEVICE)
        action, self.state = self.agent.act_deterministic(obs, self.state, done)
        self.fresh = False

        # SawEnv.step: the policy asks, the gate decides what is sent
        request = action[:, 2] > ARM_SWITCH
        arm_out, fire, _ = self.gate.command(request, self.gate_open, action[:, 2], self.gated)
        self.blade_sent, self.thr_sent = action[:, 3].clone(), action[:, 0].clone()
        self.prev_action = action
        self.steps += PHYS_PER_CTRL

        # The arm motor: torque only while the arm travels. The gate's switch stays "down" through
        # its hold and "back" for as long as the arm is parked; sent to a motor that would hold it
        # against the target and then against its back stop, slipping the clutch the whole time.
        cfg = self.cfg
        phase, t = int(self.gate.phase[0]), float(self.gate.t_phase[0])
        drive_s = cfg.fire_s if cfg.arm_drive_s is None else cfg.arm_drive_s
        return_s = cfg.retract_s if cfg.arm_return_s is None else cfg.arm_return_s
        arm_motor = 0.0
        if phase == G_FIRING and t < drive_s:
            arm_motor = 1.0
        elif phase == G_RETRACT and t < return_s:
            arm_motor = -1.0

        # SawEnv._physics_step: throttle and turn beyond a tread's top speed are scaled back together
        thr, turn = float(action[0, 0]), float(action[0, 1])
        over = max(1.0, abs(thr) + self.tread_k * abs(turn))
        thr, turn = thr / over, turn / over
        return Command(left=thr - self.tread_k * turn, right=thr + self.tread_k * turn,
                       blade=float(action[0, 3]), arm_motor=arm_motor, arm=float(arm_out[0]),
                       throttle=float(action[0, 0]), turn=float(action[0, 1]), arm_request=float(action[0, 2]),
                       gate_open=bool(self.gate_open[0]), fired=bool(fire[0]),
                       swings_left=int(self.gate.swings_left()[0]), time_left=self.time_left)
