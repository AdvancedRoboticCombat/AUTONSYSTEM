"""
Batched combat simulator.

    integrate_robot   one 100 Hz physics step; both robots go through it
    Observer          one robot's sensor stack: latency, noise, dropout, and
                      the observation vector built only from what the real
                      robot measures
    BattleEnv         the fight: resets, rules, rewards, control-rate stepping

BattleEnv.step() takes one policy action per env and advances PHYS_PER_CTRL
physics steps. Each physics step is a short pipeline:

    1. commands      our lagged action, the opponent's held command
    2. physics       integrate_robot for both robots
    3. rules         _detect_events: who hit whom, hazards. Detection only.
    4. contact       _resolve_body_contact: non-lethal bumps
    5. outcome       terminal rewards, done flags, end reasons

Dense rewards and the observation are produced once per control step.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from config import (
    DEVICE, ARENA_FT, ROBOT_FT, WALL_CLAMP_IN, DT, CTRL_DT, PHYS_PER_CTRL, MAX_STEPS, HAZ_PATH,
    V_MAX_FT, A_LAT_MAX_FT, YAW_RATE_MAX, A_LONG_MAX_FT, ALPHA_MAX_RPS2,
    GRIP_STATIC_FT, GRIP_KINETIC_FT, DRIFT_RECOVERY_TAU, WALL_SCRUB, BODY_RESTITUTION,
    DR_V_MAX, DR_GRIP, DR_YAW, DR_ALPHA, DR_BLEED, DR_TAU_S, DR_ACT_LAG_S, DR_OBS_LAG_S,
    DR_YAW_GAIN, DR_PULL, DR_DEADBAND, WEAR_P, DR_WEAR_SPEED, DR_WEAR_PULL,
    DR_POS_NOISE_FT, DR_HDG_NOISE, DR_EN_POS_NOISE_FT, DR_DROPOUT_P, DR_SPEED_GAIN,
    DR_POS_BIAS_FT, DR_EN_POS_BIAS_FT, DR_HDG_BIAS, BIAS_TAU_S, DR_OUTLIER_P, OUTLIER_FT,
    LAG_JITTER_STEPS,
    SPEED_NOISE_FT, GYRO_NOISE, ENEMY_VEL_WINDOW, N_RAYS, RAY_RANGE_FT, RAY_STEP_FT,
    OUR_WEAPON_REACH, OUR_WEAPON_TIP_RANGE, WIN_HIT_MIN_FWD, REAR_HIT_MIN_REVERSE,
    OUR_WALL_CONTACT_ENDS_EPISODE, GAMMA, R_WIN, R_LOSS, R_CLASH, R_MUTUAL, R_HAZARD,
    TIME_COST_PER_S, R_WALL_HIT, R_WALL_TOUCH_PER_S, R_ACTION_RATE,
    K_FLANK, FLANK_FAR, FLANK_NEAR, K_APPROACH, angle_wrap,
    OBS_DIM, ACT_DIM, WALL_FRICTION,
)
from opponent_library import OpponentMixture, SELFPLAY

# Bumped whenever the motion model, rules or observation change, so
# evaluate.py reports are only compared like for like. This one names the
# classic rules; the saw rules carry their own (saw_env.SawEnv).
PHYSICS_VERSION = "sensor-obs-v2"

# Stress testing: evaluate.py --stress widens every DR_* range about its
# midpoint by this factor, to see how the policy copes with physics and
# sensors outside what it trained on. Training always runs at 1.0.
DR_STRETCH = 1.0

# Every per-episode randomization of our robot and its sensors, by name. These
# are the classic ranges; a ruleset can widen any of them (BattleEnv.ranges,
# config.SAW_DR).
RANGES = dict(
    v_max=DR_V_MAX, grip=DR_GRIP, yaw=DR_YAW, alpha=DR_ALPHA, bleed=DR_BLEED, tau=DR_TAU_S,
    act_lag=DR_ACT_LAG_S, obs_lag=DR_OBS_LAG_S, yaw_gain=DR_YAW_GAIN, pull=DR_PULL, deadband=DR_DEADBAND,
    pos_noise=DR_POS_NOISE_FT, hdg_noise=DR_HDG_NOISE, en_pos_noise=DR_EN_POS_NOISE_FT,
    dropout=DR_DROPOUT_P, outlier=DR_OUTLIER_P, speed_gain=DR_SPEED_GAIN,
    pos_bias=DR_POS_BIAS_FT, en_pos_bias=DR_EN_POS_BIAS_FT, hdg_bias=DR_HDG_BIAS,
)

_DRIFT_DECAY  = 1.0 - DT / (DRIFT_RECOVERY_TAU + DT)
_ACT_RING     = 2 * int(round(DR_ACT_LAG_S[1] / DT)) + 1      # latency buffers, with room for a stress test
_OBS_RING     = 2 * int(round(DR_OBS_LAG_S[1] / DT)) + 1
_ARENA_DIAG   = ARENA_FT * math.sqrt(2.0)

# ─── Reward terms (per-episode totals in BattleEnv.rw, for logging) ───────
RW_OUTCOME, RW_TIME, RW_SHAPE, RW_WALL, RW_SMOOTH = range(5)
RW_NAMES = ["outcome", "time", "shape", "wall", "smooth"]
RW_N = len(RW_NAMES)

# ─── Episode end reasons, from our robot's side ───────────────────────────
END_NONE, END_OUR_HIT, END_ENEMY_HIT, END_TIP_CLASH, END_MUTUAL, END_OUR_WALL, END_OUR_HAZARD, END_TIMEOUT = range(8)
END_NAMES = {
    END_OUR_HIT:    "our_hit",      # our weapon landed cleanly        -> win
    END_ENEMY_HIT:  "enemy_hit",    # their weapon landed cleanly      -> loss
    END_TIP_CLASH:  "tip_clash",    # weapon met weapon                -> loss
    END_MUTUAL:     "mutual",       # both weapons landed at once      -> loss
    END_OUR_WALL:   "our_wall",     # only if OUR_WALL_CONTACT_ENDS_EPISODE
    END_OUR_HAZARD: "our_hazard",   # we drove into a hazard           -> loss
    END_TIMEOUT:    "timeout",      # nobody landed anything in 20 s   -> draw
}
# Saw rules: only the clock ends a fight, which is then scored on damage.
END_PTS_WIN, END_PTS_LOSS, END_PTS_DRAW = 8, 9, 10
END_NAMES.update({
    END_PTS_WIN:  "ahead",          # we dealt more damage than we took -> win
    END_PTS_LOSS: "behind",         # we took more than we dealt        -> loss
    END_PTS_DRAW: "level",          # nothing in it                     -> draw
})
WIN_ENDS  = (END_OUR_HIT, END_PTS_WIN)
LOSS_ENDS = (END_ENEMY_HIT, END_TIP_CLASH, END_MUTUAL, END_OUR_WALL, END_OUR_HAZARD, END_PTS_LOSS)
DRAW_ENDS = (END_TIMEOUT, END_PTS_DRAW)


def load_arena():
    """Returns (hazard mask (48, 48) bool, distance to nearest hazard in ft),
    both indexed [y, x]."""
    if not HAZ_PATH.exists():
        raise FileNotFoundError(f"arena hazard map not found at {HAZ_PATH}; "
                                "refusing to run in an arena with no hazards")
    hazards = np.load(HAZ_PATH)
    dist = (distance_transform_edt(~hazards) if hazards.any()
            else np.full(hazards.shape, float(ARENA_FT))).astype(np.float32)
    return torch.from_numpy(hazards).to(DEVICE), torch.from_numpy(dist).to(DEVICE)


def _U(k, lo, hi):
    return torch.empty(k, device=DEVICE).uniform_(lo, hi)


def _dr(k, rng):
    """k draws from a DR_* range (widened by DR_STRETCH). Ranges that start
    at or above zero never go negative."""
    lo, hi = rng
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * DR_STRETCH
    x = _U(k, mid - half, mid + half)
    return x.clamp(min=0.0) if lo >= 0 else x


def _lag_steps(k, rng, ring):
    """k latencies from a DR range in seconds, as whole physics steps."""
    return (_dr(k, rng) / DT).round().long().clamp(1, ring - 2)


# ─── Robot dynamics ───────────────────────────────────────────────────────

class RobotParams:
    """Per-env physical limits of one robot. Our robot's are redrawn every
    episode from the DR_* ranges; the opponent's are set relative to ours."""
    FIELDS = ("v_max", "yaw_cap", "a_lat_cap", "dv_max", "dalpha", "ctrl_alpha",
              "bleed", "grip_s", "grip_k", "yaw_gain", "pull", "deadband")

    def __init__(self, n):
        for name in self.FIELDS:
            setattr(self, name, torch.zeros(n, device=DEVICE))
        self.yaw_gain.fill_(1.0)

    def randomize(self, idx, v_max, yaw_cap, alpha, grip, yaw_gain=1.0, pull=0.0, deadband=0.0, ranges=RANGES):
        """Fill slots ``idx`` given the headline limits and the grip
        multiplier; actuator lag and turn bleed are drawn here. The drive
        imperfections default to none: the scripted opponents carry theirs in
        the driver model."""
        k = idx.numel()
        tau = _dr(k, ranges["tau"])
        self.yaw_gain[idx], self.pull[idx], self.deadband[idx] = yaw_gain, pull, deadband
        self.v_max[idx]      = v_max
        self.yaw_cap[idx]    = yaw_cap
        self.a_lat_cap[idx]  = A_LAT_MAX_FT            # controller limit, known exactly
        self.dv_max[idx]     = A_LONG_MAX_FT * grip * DT
        self.dalpha[idx]     = alpha * DT
        self.ctrl_alpha[idx] = tau / (tau + DT)
        self.bleed[idx]      = _dr(k, ranges["bleed"])
        self.grip_s[idx]     = GRIP_STATIC_FT * grip
        self.grip_k[idx]     = GRIP_KINETIC_FT * grip


def integrate_robot(x, y, a, v, omega, drifting, drift_vel,
                    throttle, target_omega, par, alive,
                    traction=None, yaw_scale=None, scrape_walls=False):
    """One 100 Hz step of the differential-drive model. Both robots go through
    this exact function; only the state tensors and ``par`` differ.

    State, all (n,):
        x, y        position, ft
        a           heading, rad
        v           signed body-forward speed, ft/s
        omega       yaw rate, rad/s
        drifting    bool, tyres have let go laterally
        drift_vel   body-lateral slide speed (+ = to the left), ft/s
    Command:
        throttle      fraction of top speed, [-1, 1]
        target_omega  requested yaw rate, rad/s
    par    RobotParams
    alive  (n,) float mask, 0 freezes the robot
    The saw rules pass three extras; left out, the step is the classic one:
        traction   (n,) share of drive and steering grip available (a robot
                   that has just been thrown by a hit has little)
        yaw_scale  (n,) share of the yaw-rate cap available
        scrape_walls  True: a wall takes only the motion into it, so a robot
                   can scrape along one. False: any wall contact halves its
                   speed every step, at any angle.

    Returns the new state and the speed (ft/s) at which the robot ran into a
    wall this step, 0 where it did not touch one.
    """
    dalpha  = par.dalpha  if traction  is None else par.dalpha * traction
    dv_max  = par.dv_max  if traction  is None else par.dv_max * traction
    yaw_cap = par.yaw_cap if yaw_scale is None else par.yaw_cap * yaw_scale

    # 0. What the drive makes of the command: a deadband around zero throttle,
    #    a yaw gain that is not exactly one, and a pull to one side under power
    throttle = torch.sign(throttle) * (throttle.abs() - par.deadband).clamp(min=0.0) / (1.0 - par.deadband)
    target_omega = target_omega * par.yaw_gain + par.pull * throttle * par.yaw_cap

    # 1. Yaw: acceleration limit, then the centripetal and mechanical rate caps
    omega = omega + torch.maximum(torch.minimum(target_omega - omega, dalpha), -dalpha)
    cap   = torch.minimum(par.a_lat_cap / v.abs().clamp(min=0.5), yaw_cap)
    omega = torch.maximum(torch.minimum(omega, cap), -cap)

    # 2. Forward speed: turning lowers the reachable speed, then actuator lag
    #    and the traction limit
    target_v = throttle * par.v_max * (1.0 - par.bleed * omega.abs() / par.yaw_cap)
    smoothed = v * par.ctrl_alpha + target_v * (1.0 - par.ctrl_alpha)
    v = v + torch.maximum(torch.minimum(smoothed - v, dv_max), -dv_max) * alive

    # 3. Heading
    a = angle_wrap(a + omega * DT * alive)

    # 4. Lateral grip. v*omega is the centripetal acceleration the turn needs
    #    (signed, toward the left). Past the static limit the tyres let go and
    #    only kinetic friction is left, so the robot slides to the outside of
    #    the turn and scrubs forward speed until the demand drops again.
    demand   = v * omega
    start    = (demand.abs() > par.grip_s) & ~drifting & (alive > 0)
    stop     = (demand.abs() < par.grip_k) & drifting
    drifting = (drifting | start) & ~stop
    deficit  = (demand.abs() - par.grip_k).clamp(min=0.0) * drifting
    slide    = drift_vel - torch.sign(demand) * deficit * DT
    v         = v + omega * slide * DT * drifting * alive          # sliding costs forward speed
    drift_vel = torch.where(drifting, slide, drift_vel * _DRIFT_DECAY)

    # 5. Position
    cos_a, sin_a = torch.cos(a), torch.sin(a)
    vx = cos_a * v - sin_a * drift_vel
    vy = sin_a * v + cos_a * drift_vel
    new_x = x + vx * DT * alive
    new_y = y + vy * DT * alive

    # 6. Walls are solid
    lo, hi = WALL_CLAMP_IN, ARENA_FT - WALL_CLAMP_IN
    hit_x = (new_x < lo) | (new_x > hi)
    hit_y = (new_y < lo) | (new_y > hi)
    wall_impact = (hit_x * vx.abs() + hit_y * vy.abs()) * alive
    touching  = hit_x | hit_y
    if scrape_walls:
        # The wall takes the motion into it and, through friction, a share of
        # the motion along it. The rest carries on, so the robot scrapes
        # along the wall or drives off it at an angle.
        rub = lambda along, into: torch.sign(along) * (along.abs() - WALL_FRICTION * into.abs()).clamp(min=0.0)
        wx = torch.where(hit_x, torch.zeros_like(vx), torch.where(hit_y, rub(vx, vy), vx))
        wy = torch.where(hit_y, torch.zeros_like(vy), torch.where(hit_x, rub(vy, vx), vy))
        v         = torch.where(touching,  cos_a * wx + sin_a * wy, v)
        drift_vel = torch.where(touching, -sin_a * wx + cos_a * wy, drift_vel)
    else:
        # Classic rules: the robot stops against the wall and has to turn out
        v         = torch.where(touching, v * WALL_SCRUB, v)
        drift_vel = torch.where(touching, torch.zeros_like(drift_vel), drift_vel)
    return (new_x.clamp(lo, hi), new_y.clamp(lo, hi), a, v, omega, drifting, drift_vel, wall_impact)


# ─── Sensors ──────────────────────────────────────────────────────────────

class Observer:
    """What one robot's sensors report about the fight.

    The real robot has its own pose, the other robot's position, a gyro and
    wheel speed. It does not have the other robot's heading, top speed or
    intentions, so none of that is in here. Everything arrives late (with
    jitter) and noisy; position fixes also carry slowly wandering offsets, are
    occasionally missed, and the enemy's sometimes jumps.

    Observation layout (OBS_DIM = 24):
         0, 1   own position, centred, / half arena
         2, 3   sin, cos of own heading
         4      own forward speed / V_MAX          (wheel odometry)
         5      own yaw rate / YAW_RATE_MAX        (gyro)
         6, 7   previous action (throttle, turn)
         8, 9   cos, sin of bearing to the enemy in our body frame
        10      distance to the enemy / arena size
        11      proximity exp(-distance / 8 ft)
        12, 13  enemy velocity in our body frame / V_MAX (forward, left),
                differenced from its last ENEMY_VEL_WINDOW position fixes
        14, 15  enemy position, centred, / half arena
        16..23  distance to the nearest wall or hazard along 8 body-frame
                rays (0, 45, ... 315 deg) / RAY_RANGE_FT
    Conventions: positions in ft with the origin at an arena corner, headings
    are atan2(y, x) in that frame, and "left" is +90 deg from the heading in
    the same sense. To run the policy on the robot, build exactly this vector
    from the localisation and tracking outputs, once per CTRL_DT.

    ``ring`` is the length of the truth history in physics steps and ``cols``
    its width: 7, or 8 when the ruleset's truth carries the other robot's
    heading in column 7 (kept in ``other_hdg`` from the same fixes as its
    position, for rulesets that sense it).

    ``camera`` None: a position fix is the delayed truth of each control step
    (the classic sensors). Otherwise the camera takes a frame every ``period``
    physics steps, ``camera`` fixed or (lo, hi) drawn per episode with a random
    phase, and the fixes of a frame, noise included, are held until the next
    one: the fix is as old as the delay plus the frame's age. Setting
    ``camera`` to None on a live Observer goes back to the classic sensors.
    """

    def __init__(self, n, hazards, ranges=RANGES, ring=_OBS_RING, cols=7, camera=None):
        self.n = n
        self.ranges = ranges
        self.ring_n = ring
        self.camera = camera
        # Saw rules: (n,) bool each, the tracker has lost the other robot, or
        # ours, and goes on reporting the last fix. None: it never does.
        self.lost = None
        self.lost_own = None
        self._ar = torch.arange(n, device=DEVICE)
        self.hazards_flat = hazards.flatten()
        self.has_hazards  = bool(hazards.any())
        self.ray_angles = torch.arange(N_RAYS, device=DEVICE) * (2 * math.pi / N_RAYS)
        self.ray_t = torch.arange(RAY_STEP_FT, RAY_RANGE_FT + 1e-6, RAY_STEP_FT, device=DEVICE)

        f = lambda *shape: torch.zeros(*shape, device=DEVICE)
        # Truth history at physics rate: x, y, heading, speed, yaw rate, other x, other y (, other heading)
        self.ring = f(n, ring, cols)
        self.ptr  = 0
        self.t    = 0                                        # physics steps recorded so far
        self.lag  = torch.ones(n, dtype=torch.long, device=DEVICE)
        # Camera frames: physics steps per frame, its phase, the capture step of the last frame used
        self.period = torch.ones(n, dtype=torch.long, device=DEVICE)
        self.phase  = torch.zeros(n, dtype=torch.long, device=DEVICE)
        self.frame_seen = torch.full((n,), -10 ** 9, dtype=torch.long, device=DEVICE)
        self.e_fresh = torch.ones(n, dtype=torch.bool, device=DEVICE)   # the other robot's fix was renewed this control step
        self.other_hdg = f(n)                                # the other robot's true heading at the capture of its fix
        self.last_delay = torch.zeros(n, dtype=torch.long, device=DEVICE)   # physics steps from that capture to now
        # This episode's sensor quality
        self.pos_sigma, self.hdg_sigma, self.other_sigma = f(n), f(n), f(n)
        self.drop_p, self.outlier_p, self.speed_gain = f(n), f(n), f(n)
        self.bias_sigma = f(n, 5)                            # x, y, other x, other y, heading
        self.bias = f(n, 5)                                  # slowly wandering offsets on those five
        self.fix = f(n, 4)                                   # last position fix: x, y, other x, other y
        self.other_hist = f(n, ENEMY_VEL_WINDOW, 2)          # enemy fixes, control rate
        self.own_hist = f(n, ENEMY_VEL_WINDOW, 2)            # and our own
        self.hist_ptr = 0
        # True (set by a ruleset): the robot has no wheel odometry, and its
        # speed is its own position track differenced like the enemy's
        self.speed_from_track = False

    def reset(self, idx, truth):
        """New episode in slots ``idx``: redraw this episode's sensor quality
        and fill every buffer with the starting state."""
        k, R = idx.numel(), self.ranges
        self.lag[idx]         = _lag_steps(k, R["obs_lag"], self.ring_n - LAG_JITTER_STEPS)
        self.pos_sigma[idx]   = _dr(k, R["pos_noise"])
        self.hdg_sigma[idx]   = _dr(k, R["hdg_noise"])
        self.other_sigma[idx] = _dr(k, R["en_pos_noise"])
        self.drop_p[idx]      = _dr(k, R["dropout"])
        self.outlier_p[idx]   = _dr(k, R["outlier"])
        self.speed_gain[idx]  = _dr(k, R["speed_gain"])
        own, other, hdg = _dr(k, R["pos_bias"]), _dr(k, R["en_pos_bias"]), _dr(k, R["hdg_bias"])
        self.bias_sigma[idx] = torch.stack([own, own, other, other, hdg], dim=1)
        self.bias[idx] = torch.randn(k, 5, device=DEVICE) * self.bias_sigma[idx]

        start = truth[idx]
        self.ring[idx] = start.unsqueeze(1)
        white = torch.stack([self.pos_sigma[idx], self.pos_sigma[idx],
                             self.other_sigma[idx], self.other_sigma[idx]], dim=1)
        self.fix[idx] = start[:, [0, 1, 5, 6]] + self.bias[idx, :4] + torch.randn(k, 4, device=DEVICE) * white
        self.other_hist[idx] = self.fix[idx][:, 2:4].unsqueeze(1)
        self.own_hist[idx] = self.fix[idx][:, 0:2].unsqueeze(1)
        if start.shape[1] > 7:
            self.other_hdg[idx] = start[:, 7]
        self.e_fresh[idx] = True
        if self.camera is not None:
            if isinstance(self.camera, (tuple, list)):
                lo, hi = self.camera
                self.period[idx] = torch.randint(int(lo), int(hi) + 1, (k,), device=DEVICE)
            else:
                self.period[idx] = int(self.camera)
            self.phase[idx] = (torch.rand(k, device=DEVICE) * self.period[idx]).long()
            self.frame_seen[idx] = -10 ** 9

    def record(self, truth):
        """Store the true state after a physics step."""
        self.ptr = (self.ptr + 1) % self.ring_n
        self.ring[:, self.ptr] = truth
        self.t += 1

    def observe(self, prev_action, idx=None):
        """Build the observation for this control step. With ``idx`` it
        returns only those rows and consumes nothing: that is how freshly
        reset envs get their first observation."""
        first = idx is not None
        sel = idx if first else self._ar
        k = sel.numel()
        noise = lambda: torch.randn(k, device=DEVICE)
        jitter = torch.randint(-LAG_JITTER_STEPS, LAG_JITTER_STEPS + 1, (k,), device=DEVICE)
        d = (self.lag[sel] + jitter).clamp(min=1)
        s = self.ring[sel, (self.ptr - d) % self.ring_n]                                    # delayed truth

        if not first:
            camera = self.camera is not None
            src = s
            if camera:
                # The newest camera frame captured at or before the delayed step
                # c, and the truth at its capture: positions come from there
                c = self.t - d
                off = torch.remainder(c + self.phase, self.period)
                frame = c - off
                fd = (d + off).clamp(max=self.ring_n - 1)
                src = self.ring[sel, (self.ptr - fd) % self.ring_n]
                new = frame != self.frame_seen
            # Offsets wander slowly (an Ornstein-Uhlenbeck walk around zero)
            self.bias = (self.bias * (1.0 - CTRL_DT / BIAS_TAU_S)
                         + torch.randn(k, 5, device=DEVICE) * self.bias_sigma * math.sqrt(2.0 * CTRL_DT / BIAS_TAU_S))
            fresh = torch.stack([src[:, 0] + noise() * self.pos_sigma,
                                 src[:, 1] + noise() * self.pos_sigma,
                                 src[:, 5] + noise() * self.other_sigma,
                                 src[:, 6] + noise() * self.other_sigma], dim=1) + self.bias[:, :4]
            # Now and then the tracker throws the enemy somewhere it is not
            wild = (torch.rand(k, device=DEVICE) < self.outlier_p).unsqueeze(1)
            fresh[:, 2:4] += wild * torch.empty(k, 2, device=DEVICE).uniform_(-OUTLIER_FT, OUTLIER_FT)
            # Position fixes (ours and the enemy's) can be missed; the last one is held
            missed = (torch.rand(k, device=DEVICE) < self.drop_p).unsqueeze(1)
            if self.lost is not None:
                fresh[:, 2:4] = torch.where(self.lost.unsqueeze(1), self.fix[:, 2:4], fresh[:, 2:4])
            if self.lost_own is not None:
                fresh[:, 0:2] = torch.where(self.lost_own.unsqueeze(1), self.fix[:, 0:2], fresh[:, 0:2])
            if camera:
                # A frame's fixes, noise and all, stand until the next frame
                upd = new & ~missed[:, 0]
                self.fix = torch.where(upd.unsqueeze(1), fresh, self.fix)
                self.frame_seen = torch.where(new, frame, self.frame_seen)
                self.last_delay = fd
            else:
                self.fix = torch.where(missed, self.fix, fresh)
                upd = ~missed[:, 0]
                self.last_delay = d
            if self.lost is not None:
                upd = upd & ~self.lost
            self.e_fresh = upd
            if self.ring.shape[2] > 7:
                self.other_hdg = torch.where(upd, src[:, 7], self.other_hdg)
        x, y, ex, ey = self.fix[sel].unbind(1)
        a     = s[:, 2] + self.bias[sel, 4] + noise() * self.hdg_sigma[sel]
        speed = s[:, 3] * self.speed_gain[sel] + noise() * SPEED_NOISE_FT
        gyro  = s[:, 4] + noise() * GYRO_NOISE
        prev_action = prev_action[sel]

        # Enemy velocity from its own position track
        oldest = self.other_hist[sel, self.hist_ptr]
        evx_w  = (ex - oldest[:, 0]) / (ENEMY_VEL_WINDOW * CTRL_DT)
        evy_w  = (ey - oldest[:, 1]) / (ENEMY_VEL_WINDOW * CTRL_DT)
        if self.speed_from_track:
            # No wheel odometry: our own speed is our own track differenced
            # the same way and taken along our heading. It is as noisy and as
            # late as the track, reads zero while the camera has lost us, and
            # jumps when it finds us again.
            was = self.own_hist[sel, self.hist_ptr]
            speed = (((x - was[:, 0]) * torch.cos(a) + (y - was[:, 1]) * torch.sin(a))
                     / (ENEMY_VEL_WINDOW * CTRL_DT)).clamp(-1.5 * V_MAX_FT, 1.5 * V_MAX_FT)
        if not first:
            self.other_hist[:, self.hist_ptr, 0] = ex
            self.other_hist[:, self.hist_ptr, 1] = ey
            self.own_hist[:, self.hist_ptr, 0] = x
            self.own_hist[:, self.hist_ptr, 1] = y
            self.hist_ptr = (self.hist_ptr + 1) % ENEMY_VEL_WINDOW

        cos_a, sin_a = torch.cos(a), torch.sin(a)
        dx, dy = ex - x, ey - y
        dist   = torch.sqrt(dx * dx + dy * dy + 1e-6)
        fwd, left = (dx * cos_a + dy * sin_a) / dist, (-dx * sin_a + dy * cos_a) / dist
        half = ARENA_FT / 2
        return torch.cat([torch.stack([
            (x - half) / half, (y - half) / half,
            sin_a, cos_a,
            speed / V_MAX_FT,
            gyro / YAW_RATE_MAX,
            prev_action[:, 0], prev_action[:, 1],
            fwd, left,
            (dist / ARENA_FT).clamp(max=1.0),
            torch.exp(-dist / 8.0),
            (( evx_w * cos_a + evy_w * sin_a) / V_MAX_FT).clamp(-1.5, 1.5),
            ((-evx_w * sin_a + evy_w * cos_a) / V_MAX_FT).clamp(-1.5, 1.5),
            (ex - half) / half, (ey - half) / half,
        ], dim=1), self._rays(x, y, a)], dim=1)

    def _rays(self, x, y, a):
        """Distance along each body-frame ray to the wall (exact) or the first
        hazard cell (marched), as a fraction of RAY_RANGE_FT."""
        lo, hi = WALL_CLAMP_IN, ARENA_FT - WALL_CLAMP_IN
        ang = a.unsqueeze(1) + self.ray_angles
        cx, sy = torch.cos(ang), torch.sin(ang)
        x1, y1 = x.unsqueeze(1), y.unsqueeze(1)
        tiny = 1e-6
        tx = torch.where(cx > 0, (hi - x1) / cx.clamp(min=tiny), (lo - x1) / cx.clamp(max=-tiny))
        ty = torch.where(sy > 0, (hi - y1) / sy.clamp(min=tiny), (lo - y1) / sy.clamp(max=-tiny))
        reach = torch.minimum(tx, ty).clamp(0.0, RAY_RANGE_FT)
        if self.has_hazards:
            px = (x1.unsqueeze(2) + cx.unsqueeze(2) * self.ray_t).long().clamp(0, ARENA_FT - 1)
            py = (y1.unsqueeze(2) + sy.unsqueeze(2) * self.ray_t).long().clamp(0, ARENA_FT - 1)
            blocked = self.hazards_flat[py * ARENA_FT + px]
            first = torch.where(blocked, self.ray_t.expand_as(blocked),
                                torch.full_like(blocked, RAY_RANGE_FT, dtype=torch.float32)).amin(dim=2)
            reach = torch.minimum(reach, first)
        return reach / RAY_RANGE_FT


# ─── Fight ────────────────────────────────────────────────────────────────

def bodies_overlap(dx, dy, cos_o, sin_o, cos_e, sin_e):
    """Whether the two square bodies overlap: separating-axis test on both
    squares' axes. (dx, dy) is enemy minus us."""
    half = ROBOT_FT * 0.5

    def overlap_on(ax, ay):
        ours   = half * ((cos_o * ax + sin_o * ay).abs() + (-sin_o * ax + cos_o * ay).abs())
        theirs = half * ((cos_e * ax + sin_e * ay).abs() + (-sin_e * ax + cos_e * ay).abs())
        return (dx * ax + dy * ay).abs() <= ours + theirs

    return (overlap_on(cos_o, sin_o) & overlap_on(-sin_o, cos_o) &
            overlap_on(cos_e, sin_e) & overlap_on(-sin_e, cos_e))


@dataclass
class StepEvents:
    """What the rules found in one physics step, as per-env bool masks."""
    win:     torch.Tensor   # our weapon landed cleanly
    loss:    torch.Tensor   # their weapon landed cleanly
    clash:   torch.Tensor   # weapon tips met
    mutual:  torch.Tensor   # both weapons landed at once
    hazard:  torch.Tensor   # we are in a hazard cell
    wall:    torch.Tensor   # we touched a wall (only when that ends the episode)
    bump:    torch.Tensor   # bodies overlap without weapon contact
    dx: torch.Tensor        # enemy - us
    dy: torch.Tensor
    dist: torch.Tensor


@dataclass
class StepInfo:
    """Extra outputs of BattleEnv.step for training and logging. All refer to
    the episode that was running before any auto-reset."""
    end_reason: torch.Tensor   # (n,) END_* code, END_NONE where the episode goes on
    opp_type:   torch.Tensor   # (n,) opponent type id
    timeout:    torch.Tensor   # (n,) bool, ended on the clock. Bootstrap only if env.timeout_bootstrap
                               # (the classic rules); under the saw rules the bell is terminal
    final_obs:  torch.Tensor   # (n, obs_dim) observation after the step, before any reset
    rw:         torch.Tensor   # (n, RW_N) per-term reward totals of the episode so far
    tally:      torch.Tensor = None   # saw rules: (n, TALLY_N) hits, damage and time in trouble this episode
    fight_s:    torch.Tensor = None   # saw rules: (n,) float, seconds the fight was set to last
    hdg_mode:   torch.Tensor = None   # saw rules: (n,) long, what the camera says of their heading: 0 none, 1 axis, 2 axis and nose


class BattleEnv:
    """n fights in parallel, under the classic rules. saw_env.SawEnv builds
    the saw rules on top of this class; make_env() picks between them.

    stage   "combat"    armed opponents from the curriculum mix at level p
            "pretrain"  drills against unarmed targets: same robot, physics,
                        sensors, rules and trainer, only the opponent differs
    p       curriculum level in [0, 1]; scales opponent skill and the mix
    auto_reset  True (training): finished envs restart inside step().
                False (evaluation, visualizer): they stay frozen until reset.
    """
    rules = "classic"
    physics_version = PHYSICS_VERSION
    obs_dim, act_dim = OBS_DIM, ACT_DIM
    tally_names = ()      # columns of StepInfo.tally; the classic rules keep none
    spawn_clear = 2.0     # ft that start positions keep from any hazard
    ranges = RANGES       # what is randomized per episode, and over what
    sharp_opponents = False   # True: the harder opponent field (opponent_library.SHARP)
    speed_from_track = False  # True: no wheel odometry, the speed input is the camera track differenced
    field = None              # saw_field.SawField of a ruleset that draws its own opponent field; None: the classic draws
    gamma = GAMMA             # discount of the shaping (the trainer may set it)
    timeout_bootstrap = True  # the trainer bootstraps the value of a fight that ended on the clock
    obs_ring = _OBS_RING      # truth history our observer keeps, physics steps
    obs_cols = 7              # and its width (see Observer)
    p_range = None            # (lo, hi): a ruleset with a field draws p per fight from it

    def _camera_spec(self):
        """Our observer's camera (see Observer); None: the classic sensors."""
        return None

    def __init__(self, n, stage="combat", p=0.0, auto_reset=True):
        assert stage in ("combat", "pretrain")
        self.n, self.stage, self.p, self.auto_reset = n, stage, p, auto_reset
        self._ar = torch.arange(n, device=DEVICE)
        self.hazards, self.hazard_dist = self._arena()
        self.opponents = OpponentMixture(n, self.hazard_dist)
        self.opponents.sharp = self.sharp_opponents
        self.selfplay = None          # SelfPlayController, or None
        self.selfplay_frac = 0.0
        self.manual_enemy = False     # True: a human drives the enemy through manual_cmd

        f = lambda *shape: torch.zeros(*shape, device=DEVICE)
        b = lambda: torch.zeros(n, dtype=torch.bool, device=DEVICE)
        i = lambda: torch.zeros(n, dtype=torch.long, device=DEVICE)

        # Our robot
        self.ox, self.oy, self.oa, self.ovx, self.o_omega = f(n), f(n), f(n), f(n), f(n)
        self.o_drift, self.o_drifting = f(n), b()
        self.our = RobotParams(n)
        # Enemy robot
        self.ex, self.ey, self.ea, self.evx, self.e_omega = f(n), f(n), f(n), f(n), f(n)
        self.e_drift, self.e_drifting = f(n), b()
        self.enemy = RobotParams(n)
        self.en_cmd     = f(n, 2)     # (throttle, yaw-rate request) held between control steps
        self.manual_cmd = f(n, 2)     # (throttle, turn) from the keyboard

        # Our command path: latency ring at physics rate
        self.cmd_ring = f(n, _ACT_RING, self.act_dim)
        self.cmd_lag  = i()
        self.cmd_ptr  = 0
        self.prev_action = f(n, self.act_dim)

        # Sensors: ours, and the enemy's own (used only by self-play opponents, which have no camera model)
        self.observer    = Observer(n, self.hazards, self.ranges, ring=self.obs_ring, cols=self.obs_cols,
                                    camera=self._camera_spec())
        self.en_observer = Observer(n, self.hazards, self.ranges, ring=_OBS_RING, cols=self.obs_cols, camera=None)
        self.observer.speed_from_track = self.en_observer.speed_from_track = self.speed_from_track

        # Episode bookkeeping
        self.done, self.won, self.timeout = b(), b(), b()
        self.end_reason = i()
        self.steps      = i()
        self.prev_dist  = f(n)
        self.phi        = f(n)        # shaping potential at the last control step
        self.reward     = f(n)        # reward accumulated during the current control step
        self.rw         = f(n, RW_N)  # per-term totals for the running episode
        self.fitness    = f(n)        # total return of the running episode
        self.wall_touch = b()
        # Mid-fight wear: the physics step it strikes at (past the clock = never) and what it does
        self.wear_at, self.wear_speed, self.wear_pull = i(), f(n), f(n)

        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────────

    def _arena(self):
        """(hazard cells, ft to the nearest one). The saw rules have walls of their own."""
        return load_arena()

    def reset(self):
        """Start a fresh episode in every env and return the observation."""
        self.reset_at(self._ar)
        return self._observe(self._ar)

    def reset_at(self, idx):
        """Start a fresh episode in the envs listed in ``idx``. Every piece of
        per-episode state is re-initialised here so nothing leaks between
        episodes."""
        k = idx.numel()
        if k == 0:
            return
        lo, hi = WALL_CLAMP_IN + 2.0, ARENA_FT - WALL_CLAMP_IN - 2.0

        # Poses: anywhere clear of walls and hazards, 10 to 34 ft apart. Half
        # the episodes open from rest, like the start of a match; the rest
        # open mid-fight, already moving.
        ox, oy = self._free_points(k, lo, hi)
        ex, ey = self._free_points(k, lo, hi)
        for _ in range(8):
            d = torch.sqrt((ex - ox) ** 2 + (ey - oy) ** 2)
            bad = (d < 10.0) | (d > 34.0)
            nx, ny = self._free_points(k, lo, hi)
            ex, ey = torch.where(bad, nx, ex), torch.where(bad, ny, ey)
        self.ox[idx], self.oy[idx], self.ex[idx], self.ey[idx] = ox, oy, ex, ey
        self.oa[idx] = _U(k, -math.pi, math.pi)
        self.ea[idx] = _U(k, -math.pi, math.pi)

        # Physics: ours from the DR ranges, the opponent's relative to ours.
        # Both robots are on the same floor, so their grip moves together.
        R = self.ranges
        grip = _dr(k, R["grip"])
        self.our.randomize(idx,
                           v_max=V_MAX_FT * _dr(k, R["v_max"]),
                           yaw_cap=YAW_RATE_MAX * _dr(k, R["yaw"]),
                           alpha=ALPHA_MAX_RPS2 * _dr(k, R["alpha"]),
                           grip=grip,
                           yaw_gain=_dr(k, R["yaw_gain"]), pull=_dr(k, R["pull"]),
                           deadband=_dr(k, R["deadband"]), ranges=R)
        worn = torch.rand(k, device=DEVICE) < WEAR_P
        self.wear_at[idx] = torch.where(worn, torch.randint(0, MAX_STEPS, (k,), device=DEVICE), MAX_STEPS + 1)
        self.wear_speed[idx] = _dr(k, DR_WEAR_SPEED)
        self.wear_pull[idx]  = _dr(k, DR_WEAR_PULL)
        if self.field is None:
            self.opponents.reset(idx, self.p, self.stage == "pretrain", self.ox, self.oy,
                                 selfplay_frac=self.selfplay_frac if self.selfplay is not None else 0.0)
            opp = self.opponents
            self.enemy.randomize(idx,
                                 v_max=self.our.v_max[idx] * opp.speed_ratio[idx],
                                 yaw_cap=self.our.yaw_cap[idx] * opp.agility_ratio[idx],
                                 alpha=self.our.dalpha[idx] / DT * opp.agility_ratio[idx],
                                 grip=grip * _U(k, 0.85, 1.15), ranges=R)
            self.enemy.a_lat_cap[idx] = A_LAT_MAX_FT * _U(k, 0.8, 1.1)
        else:
            # The ruleset's own field: p per fight, profiles, the opponent's physique apart from ours
            p_k, override = self.field.before_opponents(idx, self)
            self.opponents.reset(idx, p_k, self.stage == "pretrain", self.ox, self.oy,
                                 selfplay_frac=0.0, types_override=override)
            opp = self.opponents
            self.field.enemy_physique(idx, self, grip)

        rolling = torch.rand(k, device=DEVICE) < 0.5
        self.ovx[idx] = rolling * _U(k, -0.3, 1.0) * self.our.v_max[idx]
        self.evx[idx] = rolling * _U(k, 0.0, 1.0) * self.enemy.v_max[idx] * opp.throttle[idx] * opp.armed[idx]
        for t in (self.o_omega, self.o_drift, self.e_omega, self.e_drift):
            t[idx] = 0.0
        self.o_drifting[idx] = False
        self.e_drifting[idx] = False
        self.en_cmd[idx] = 0.0
        self.manual_cmd[idx] = 0.0
        self._adjust_start(idx)

        self.cmd_lag[idx] = _lag_steps(k, R["act_lag"], _ACT_RING)
        self.cmd_ring[idx] = 0.0
        self.prev_action[idx] = 0.0

        self.done[idx], self.won[idx], self.timeout[idx] = False, False, False
        self.end_reason[idx] = END_NONE
        self.steps[idx] = 0
        self.rw[idx] = 0.0
        self.fitness[idx] = 0.0
        self.wall_touch[idx] = False
        self.prev_dist[idx] = torch.sqrt((self.ex[idx] - self.ox[idx]) ** 2
                                         + (self.ey[idx] - self.oy[idx]) ** 2 + 1e-6)
        self.phi[idx] = self._potential()[idx]

        self.observer.reset(idx, self._truth())
        self.en_observer.reset(idx, self._truth(mirror=True))
        if self.selfplay is not None:
            self.selfplay.reset(idx)

    def _adjust_start(self, idx):
        """Hook for rulesets that change how an episode opens. Called once
        both robots have their poses, physics and speeds, and before the
        sensors and bookkeeping are initialised from them."""

    def _free_points(self, k, lo, hi):
        """k random points at least ``spawn_clear`` ft from any hazard."""
        x, y = _U(k, lo, hi), _U(k, lo, hi)
        for _ in range(8):
            bad = self.hazard_dist[y.long(), x.long()] < self.spawn_clear
            x, y = torch.where(bad, _U(k, lo, hi), x), torch.where(bad, _U(k, lo, hi), y)
        return x, y

    def _truth(self, mirror=False):
        """True state as an Observer wants it, from our side or the enemy's."""
        if mirror:
            return torch.stack([self.ex, self.ey, self.ea, self.evx, self.e_omega, self.ox, self.oy], dim=1)
        return torch.stack([self.ox, self.oy, self.oa, self.ovx, self.o_omega, self.ex, self.ey], dim=1)

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self, action):
        """Advance one control step. ``action`` is (n, act_dim), each entry
        clamped to [-1, 1]: throttle and turn, plus arm and blade under the
        saw rules. Returns (obs, reward, done, StepInfo)."""
        action = action.clamp(-1.0, 1.0)
        was_alive = ~self.done
        self.reward.zero_()
        self.wall_touch.zero_()
        self._apply_wear()
        self._enemy_decides()

        for _ in range(PHYS_PER_CTRL):
            self._physics_step(action)
            self.observer.record(self._truth())
            if self.selfplay is not None:
                self.en_observer.record(self._truth(mirror=True))
            self.steps += (~self.done).long()

        self._clock()
        self._dense_rewards(action, was_alive)
        self.prev_action = torch.where(was_alive.unsqueeze(1), action, self.prev_action)
        obs = self._observe()

        done = self.done.clone()
        info = self._info(obs)
        reward = self.reward.clone()
        if self.auto_reset:
            idx = done.nonzero(as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                self.reset_at(idx)
                obs = obs.clone()
                obs[idx] = self._observe(idx)
        return obs, reward, done, info

    def _physics_step(self, action):
        """One 100 Hz step: commands, motion, rules, contact, outcome."""
        alive = (~self.done).float()
        throttle, target_omega = self._lagged_command(action)
        (self.ox, self.oy, self.oa, self.ovx, self.o_omega,
         self.o_drifting, self.o_drift, our_impact) = integrate_robot(
            self.ox, self.oy, self.oa, self.ovx, self.o_omega, self.o_drifting, self.o_drift,
            throttle, target_omega, self.our, alive)
        (self.ex, self.ey, self.ea, self.evx, self.e_omega,
         self.e_drifting, self.e_drift, _) = integrate_robot(
            self.ex, self.ey, self.ea, self.evx, self.e_omega, self.e_drifting, self.e_drift,
            self.en_cmd[:, 0], self.en_cmd[:, 1], self.enemy, alive)

        ev = self._detect_events(our_impact)
        self._resolve_body_contact(ev)
        self._outcome(ev, our_impact)

    def _clock(self):
        """A timeout ends the episode without being a terminal state."""
        self.timeout = ~self.done & (self.steps >= MAX_STEPS)
        self.done |= self.timeout
        self.end_reason.masked_fill_(self.timeout, END_TIMEOUT)

    def _observe(self, idx=None):
        """The policy's observation for this control step; with ``idx``, the
        first observation of the episodes that just started in those envs."""
        return self.observer.observe(self.prev_action, idx=idx)

    def _info(self, obs):
        return StepInfo(end_reason=self.end_reason.clone(), opp_type=self.opponents.types.clone(),
                        timeout=self.timeout.clone(), final_obs=obs, rw=self.rw.clone())

    def fight_stats(self):
        """Per-env numbers about the running fight beyond its outcome, as
        {name: (n,) tensor}, for evaluate.py and audit.py. The classic rules
        have none."""
        return {}

    def _apply_wear(self):
        """Where this episode's wear moment has come, our robot loses some top
        speed and starts pulling to one side. It happens once."""
        hit = self.steps >= self.wear_at
        self.our.v_max = torch.where(hit, self.our.v_max * self.wear_speed, self.our.v_max)
        self.our.pull  = torch.where(hit, self.our.pull + self.wear_pull, self.our.pull)
        self.wear_at   = torch.where(hit, MAX_STEPS + 1, self.wear_at)

    def _lag(self, action):
        """Push the policy command through the latency ring and return the
        (n, act_dim) command that reaches the robot this physics step."""
        size = self.cmd_ring.shape[1]
        self.cmd_ptr = (self.cmd_ptr + 1) % size
        self.cmd_ring[:, self.cmd_ptr] = action
        return self.cmd_ring[self._ar, (self.cmd_ptr - self.cmd_lag) % size]

    def _lagged_command(self, action):
        """What reaches the drive this physics step: throttle and the yaw
        rate asked for."""
        cmd = self._lag(action)
        return cmd[:, 0], cmd[:, 1] * YAW_RATE_MAX

    def _enemy_decides(self):
        """Once per control step the enemy's driver (script, frozen policy or
        human) picks a throttle and a yaw-rate request."""
        if self.manual_enemy:
            self.en_cmd[:, 0] = self.manual_cmd[:, 0]
            self.en_cmd[:, 1] = self.manual_cmd[:, 1] * YAW_RATE_MAX
            return
        throttle, omega = self.opponents.act(self.ex, self.ey, self.ea, self.evx,
                                             self.ox, self.oy, self.enemy)
        if self.selfplay is not None:
            sp = self.opponents.types == SELFPLAY
            act = self.selfplay.act(self.en_observer, sp)
            throttle = torch.where(sp, act[:, 0], throttle)
            omega    = torch.where(sp, act[:, 1] * YAW_RATE_MAX, omega)
        self.en_cmd[:, 0], self.en_cmd[:, 1] = throttle, omega

    # ── Rules ─────────────────────────────────────────────────────────────

    def _detect_events(self, our_impact):
        """Apply the fight rules to the current poses. A weapon tip inside the
        other robot's body, while its owner drives it in at speed, is a hit.
        Unarmed drill targets cannot hit or clash."""
        ended = self.done.clone()

        gy = self.oy.long().clamp(0, ARENA_FT - 1)
        gx = self.ox.long().clamp(0, ARENA_FT - 1)
        hazard = self.hazards[gy, gx] & ~ended
        ended |= hazard
        wall = (our_impact > 0) & ~ended if OUR_WALL_CONTACT_ENDS_EPISODE else torch.zeros_like(ended)
        ended |= wall

        dx, dy = self.ex - self.ox, self.ey - self.oy
        dist   = torch.sqrt(dx * dx + dy * dy + 1e-6)
        in_range = (dist < ROBOT_FT + OUR_WEAPON_REACH + OUR_WEAPON_TIP_RANGE) & ~ended
        closing  = dist < self.prev_dist
        self.prev_dist = dist
        armed = self.opponents.armed

        cos_o, sin_o = torch.cos(self.oa), torch.sin(self.oa)
        cos_e, sin_e = torch.cos(self.ea), torch.sin(self.ea)
        reach = ROBOT_FT * 0.5 + OUR_WEAPON_REACH
        of_x, of_y = self.ox + cos_o * reach, self.oy + sin_o * reach     # our front tip
        or_x, or_y = self.ox - cos_o * reach, self.oy - sin_o * reach     # our rear tip
        ef_x, ef_y = self.ex + cos_e * reach, self.ey + sin_e * reach
        er_x, er_y = self.ex - cos_e * reach, self.ey - sin_e * reach

        strike_half = ROBOT_FT * 0.5 + OUR_WEAPON_TIP_RANGE

        def tip_in_body(tx, ty, cx, cy, c, s):
            lx =  (tx - cx) * c + (ty - cy) * s
            ly = -(tx - cx) * s + (ty - cy) * c
            return (lx.abs() < strike_half) & (ly.abs() < strike_half)

        we_hit = in_range & closing & (
            (tip_in_body(of_x, of_y, self.ex, self.ey, cos_e, sin_e) & (self.ovx > WIN_HIT_MIN_FWD)) |
            (tip_in_body(or_x, or_y, self.ex, self.ey, cos_e, sin_e) & (self.ovx < -REAR_HIT_MIN_REVERSE)))

        en_approach = -(cos_e * dx + sin_e * dy) * self.evx     # > 0: enemy moving toward us
        they_hit = in_range & armed & (en_approach > 0) & (
            (tip_in_body(ef_x, ef_y, self.ox, self.oy, cos_o, sin_o) & (self.evx > WIN_HIT_MIN_FWD)) |
            (tip_in_body(er_x, er_y, self.ox, self.oy, cos_o, sin_o) & (self.evx < -REAR_HIT_MIN_REVERSE)))

        clash_sq = (OUR_WEAPON_TIP_RANGE * 2.0) ** 2
        near = lambda ax, ay, bx, by: (ax - bx) ** 2 + (ay - by) ** 2 < clash_sq
        clash = in_range & armed & (near(of_x, of_y, ef_x, ef_y) | near(of_x, of_y, er_x, er_y) |
                                    near(or_x, or_y, ef_x, ef_y) | near(or_x, or_y, er_x, er_y))

        touching = bodies_overlap(dx, dy, cos_o, sin_o, cos_e, sin_e)

        return StepEvents(
            win=we_hit & ~they_hit & ~clash,
            loss=they_hit & ~we_hit & ~clash,
            clash=clash,
            mutual=we_hit & they_hit & ~clash,
            hazard=hazard, wall=wall,
            bump=touching & in_range & ~we_hit & ~they_hit & ~clash,
            dx=dx, dy=dy, dist=dist)

    def _resolve_body_contact(self, ev):
        """Bodies collide without weapon contact: equal masses exchange
        momentum along the line between them and are pushed apart. A shove
        from the side shows up as lateral slide, which the tyres then kill."""
        m = ev.bump
        nx, ny = ev.dx / ev.dist, ev.dy / ev.dist            # us -> enemy
        cos_o, sin_o = torch.cos(self.oa), torch.sin(self.oa)
        cos_e, sin_e = torch.cos(self.ea), torch.sin(self.ea)
        # Closing speed along the normal, from full world velocities
        o_n = (cos_o * self.ovx - sin_o * self.o_drift) * nx + (sin_o * self.ovx + cos_o * self.o_drift) * ny
        e_n = (cos_e * self.evx - sin_e * self.e_drift) * nx + (sin_e * self.evx + cos_e * self.e_drift) * ny
        impulse = torch.where(m, (0.5 * (1.0 + BODY_RESTITUTION) * (o_n - e_n)).clamp(min=0.0), 0.0)
        # -impulse*n on us, +impulse*n on them, resolved onto each body's axes
        self.ovx     = self.ovx     - impulse * (cos_o * nx + sin_o * ny)
        self.o_drift = self.o_drift - impulse * (-sin_o * nx + cos_o * ny)
        self.evx     = self.evx     + impulse * (cos_e * nx + sin_e * ny)
        self.e_drift = self.e_drift + impulse * (-sin_e * nx + cos_e * ny)

        push = torch.where(m, (ROBOT_FT - ev.dist).clamp(min=0.0) * 0.5, 0.0)
        lo, hi = WALL_CLAMP_IN, ARENA_FT - WALL_CLAMP_IN
        self.ox = (self.ox - nx * push).clamp(lo, hi)
        self.oy = (self.oy - ny * push).clamp(lo, hi)
        self.ex = (self.ex + nx * push).clamp(lo, hi)
        self.ey = (self.ey + ny * push).clamp(lo, hi)

    # ── Rewards and termination ───────────────────────────────────────────

    def _add_reward(self, term, delta):
        self.reward += delta
        self.fitness += delta
        self.rw[:, term] += delta

    def _outcome(self, ev, our_impact):
        """Terminal rewards and done flags for this physics step."""
        self._add_reward(RW_OUTCOME,
                         ev.win.float() * R_WIN
                         - ev.loss.float() * R_LOSS
                         - ev.clash.float() * R_CLASH
                         - ev.mutual.float() * R_MUTUAL
                         - (ev.hazard | ev.wall).float() * R_HAZARD)
        # Running into a wall costs in proportion to how hard
        hit = (our_impact > 0) & ~self.done
        self._add_reward(RW_WALL, -R_WALL_HIT * hit.float() * our_impact / V_MAX_FT)
        self.wall_touch |= hit

        for mask, reason in ((ev.hazard, END_OUR_HAZARD), (ev.wall, END_OUR_WALL),
                             (ev.win, END_OUR_HIT), (ev.loss, END_ENEMY_HIT),
                             (ev.clash, END_TIP_CLASH), (ev.mutual, END_MUTUAL)):
            self.done |= mask
            self.end_reason.masked_fill_(mask, reason)
        self.won |= ev.win

    def _potential(self):
        """Shaping potential phi(s), from the true state (the reward may know
        the enemy's heading even though the policy cannot see it).

        Combat: near the enemy, standing in front of its weapon is bad and
        standing on its flank with our weapon pointed at it is good. This is
        the dodge-then-hit signal: get off the line of the charge, then turn in.
        Drills: close in on the target with the weapon pointed at it.
        """
        dx, dy = self.ex - self.ox, self.ey - self.oy
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        rx, ry = dx / dist, dy / dist                                      # us -> enemy
        aimed = (torch.cos(self.oa) * rx + torch.sin(self.oa) * ry).abs()   # our weapon axis on target
        facing_us = -(torch.cos(self.ea) * rx + torch.sin(self.ea) * ry)    # 1: its front points at us
        threat  = facing_us.clamp(min=0.0) ** 2
        opening = 1.0 - facing_us ** 2                                      # 1: we are square on its side
        near = ((FLANK_FAR - dist) / (FLANK_FAR - FLANK_NEAR)).clamp(0.0, 1.0)
        combat = K_FLANK * near * (opening * aimed - threat)
        drill  = K_APPROACH * (1.0 - dist / _ARENA_DIAG) * (0.5 + 0.5 * aimed)
        return torch.where(self.opponents.armed, combat, drill)

    def _dense_rewards(self, action, was_alive):
        """Per-control-step rewards for envs that were running this step."""
        alive = was_alive.float()
        self._add_reward(RW_TIME, -TIME_COST_PER_S * CTRL_DT * alive)
        self._add_reward(RW_WALL, -R_WALL_TOUCH_PER_S * CTRL_DT * self.wall_touch.float())
        self._add_reward(RW_SMOOTH, -R_ACTION_RATE * ((action - self.prev_action) ** 2).sum(1) * alive)

        # Potential-based shaping. A true terminal state has potential 0; a
        # timeout does not, the fight simply carries on past the clock.
        phi = self._potential()
        terminal = self.done & ~self.timeout
        self._add_reward(RW_SHAPE, (torch.where(terminal, 0.0, self.gamma * phi) - self.phi) * alive)
        self.phi = phi


def env_class(rules):
    """The environment class of a ruleset: "classic" (this file) or "saw"
    (saw_env.py). See config.RULES. Its obs_dim and act_dim size the policy."""
    if rules == "classic":
        return BattleEnv
    if rules == "saw":
        from saw_env import SawEnv
        return SawEnv
    raise ValueError(f"unknown ruleset {rules!r}; choose one of classic, saw")


def env_for_checkpoint(meta, n, **kw):
    """The environment a checkpoint was trained in, from its meta: its
    ruleset and, under the saw rules, whether the arm gate was on and which
    rule it used. A saw-v7 checkpoint (no gate in its meta) drives the arm
    itself. ``arm_gate`` and ``gate_rule`` in ``kw`` override the meta."""
    rules = meta.get("rules", "classic")
    rule = kw.pop("gate_rule", None) or meta.get("gate_rule") or "heading"
    if rules != "saw":
        kw.pop("arm_gate", None)
        return make_env(rules, n, **kw)
    if kw.get("arm_gate") is None:
        kw["arm_gate"] = bool(meta.get("arm_gate", False))
    env = make_env(rules, n, **kw)
    env.gate_rule = rule
    if meta.get("gate_persist"):
        env.gate_persist = int(meta["gate_persist"])
    return env


def make_env(rules, n, stage="combat", p=0.0, auto_reset=True, scenario=None, arm_gate=None, p_range=None):
    """n parallel fights under a ruleset. ``scenario`` (saw_field.Scenario),
    ``arm_gate`` and ``p_range`` exist only under the saw rules; the classic
    rules accept arm_gate=False (no gate) and refuse anything else."""
    cls = env_class(rules)
    if rules == "classic":
        if scenario is not None or p_range is not None or arm_gate:
            raise ValueError("scenario, arm_gate and p_range exist only under the saw rules")
        return cls(n, stage=stage, p=p, auto_reset=auto_reset)
    return cls(n, stage=stage, p=p, auto_reset=auto_reset, scenario=scenario, arm_gate=arm_gate, p_range=p_range)
