"""
The arm gate of the saw rules (saw-v8a).

Under ``arm_gate`` the policy's arm output is a request. The arm fires only
when the policy asks and the gate was open in the observation the policy
just saw: arm parked, a whole swing left, blade at GATE_BLADE_READY or more,
a camera fix of both robots that is fresh and old enough to give a velocity,
and our hammer predicted to land legally on them. The gate then runs the
whole cycle itself: fire, land, hold GATE_HOLD_STEPS control steps, retract,
parked again.

The gate sees only what the robot has: the policy's own observation (columns
0..29 and 38..43, see decode_obs and their_heading), an opponent card the
team fills in before the fight, a per-fight calibration of our own robot
(GateCal), and its own timers, swing count and blade estimate. It never reads
the simulator's true state. Everything here is tensors in, tensors out, with
the state in one plain object (ArmGate), using only + - * /, sin cos atan2
sqrt abs, minimum maximum clamp where, comparisons, & | ~, any and indexing,
so the robot can run a numpy copy of it. tests/test_gate.py holds a scalar
reference of the prediction that is the template for that copy.

Units: ft, s, rad, world frame (origin at an arena corner, headings
atan2(y, x)). All per-row tensors are (n,) unless said otherwise.
"""
import math

import torch

from config import (
    ARM_DOWN, ARM_SWITCH, V_MAX_FT, YAW_RATE_MAX, ARENA_FT, EN_FRONT_DISCS, ENEMY_VEL_WINDOW, DEVICE, angle_wrap,
)

# Weapon kinds, as saw_env numbers them (saw_env asserts they agree)
W_NONE, W_VERT, W_DRUM, W_HORIZ = 0, 1, 2, 3

# Gate phases
G_PARKED, G_FIRING, G_HOLD, G_RETRACT = 0, 1, 2, 3

GATE_HOLD_STEPS      = 4      # control steps the arm stays commanded down once the landing is detected
GATE_SHRINK_FT       = 0.2    # the landing must be legal with our tip this much smaller and their keep-out this much wider
GATE_FRONT_MARGIN_FT = 0.4    # their keep-out margin as the gate assumes it (the judges' 0.2..0.6 is not measured)
GATE_MOVING_FT_S     = 3.0    # over this tracked speed they face the way they travel
GATE_BLADE_READY     = 0.9    # blade share of full speed needed to fire
GATE_BLADE_TAU_S     = 0.95   # blade estimate: slowest spin-up time constant
GATE_BLADE_SAG       = 0.6    # ... and most spin-up lost at full drive throttle
GATE_BLADE_DROP      = 0.2    # ... and least speed kept through a bite or a stall
GATE_BLADE_DOWN_TAU_S = 0.2   # ... and fastest slow-down: DR_BLADE_TAU_S low end x the spin-up speed-up at the lowest top
GATE_TIME_MARGIN     = 1.15   # calibrated arm times may be 10% short: wait this much longer
GATE_LAT_PAD_S       = 0.04   # 2 sigma of the command-latency calibration error
GATE_FIRE_TIMEOUT_S  = 0.2    # with an arm reading, FIRING ends this long after the timer landing even without one
GATE_PARKED_READ     = 0.05   # arm reading at or under this, twice in a row, is parked
GATE_READ_PRESENT    = -0.5   # a reading is present when its column is above this (absent reads exactly -1)
GATE_STALE_STEPS     = 3      # the gate closes after more control steps than this with a bit-identical fix
GATE_WARM_STEPS      = ENEMY_VEL_WINDOW + 1   # control steps of fresh fixes before the tracked velocity is whole again
GATE_EST_START_SIGMA = 0.10   # error of the blade estimate a fight that opens mid-fight starts with
GATE_CONTACT_SAMPLES = 4      # stop_at_contact: times sampled for a first contact before the landing,
GATE_CONTACT_ITERS   = 4      # ... then halvings between the last clear and the first touching one (T / 64)
GATE_CAL_LEN_FT      = 0.05   # calibration errors, U(-x, x): our half length and half width,
GATE_CAL_REACH_FT    = 0.05   # ... the hammer reach beyond our front edge,
GATE_CAL_TIP_FT      = 0.05   # ... the tip radius,
GATE_CAL_LAT_SIGMA_S = 0.020  # ... both latencies (1 sigma),
GATE_CAL_TIME_SHARE  = 0.10   # ... fall and retract times, x U(1 - x, 1 + x)

CAL_KEYS = ("half_len", "half_wid", "reach", "tip", "latency", "cmd_lat", "fire_s", "retract_s")


def _sel(t, idx):
    return t if idx is None else t[idx]


# ── Geometry ──────────────────────────────────────────────────────────────

def body_gap1(px, py, cx, cy, c, s, half_len, half_wid):
    """(n,) distance in ft from one point per row to that row's rectangle,
    0 inside it. The rectangle has centre (cx, cy), heading (c, s) = (cos,
    sin) and half sizes. saw_env.body_gap without the m axis, same order of
    operations."""
    rx, ry = px - cx, py - cy
    out_x = ((rx * c + ry * s).abs() - half_len).clamp(min=0.0)
    out_y = ((-rx * s + ry * c).abs() - half_wid).clamp(min=0.0)
    return torch.sqrt(out_x * out_x + out_y * out_y)


def front_discs(ex, ey, c, s, fore, wide):
    """(n, EN_FRONT_DISCS) x and y of a spinner's front-disc centres, as
    saw_env.Weapon.front_discs: ``fore`` ahead of the centre, spread ``wide``
    to either side."""
    lat = wide.unsqueeze(1) * torch.linspace(-1.0, 1.0, EN_FRONT_DISCS, device=wide.device, dtype=wide.dtype)
    return ((ex + c * fore).unsqueeze(1) - s.unsqueeze(1) * lat,
            (ey + s * fore).unsqueeze(1) + c.unsqueeze(1) * lat)


def landing_class(fx, fy, tip, ex, ey, ea, e_len, e_wid, kind, fore, wide, rad, guard, front_margin, shrink=0.0):
    """Where a hammer landing at (fx, fy) with tip radius ``tip`` comes down
    on the opponent body (ex, ey, ea, e_len, e_wid) carrying weapon (kind,
    fore, wide, rad). ``guard`` (bool) marks rows whose weapon guards its
    surroundings (in SawEnv._rules: spinner & (~horizontal | spinning)).
    ``front_margin`` is (n,) or a float, ``shrink`` a float >= 0.

    Returns (legal, side, floor), bool each: on their body off the weapon
    side, on their weapon side (or their blade), on the floor. Exactly one
    is True per row. At shrink 0 this is SawEnv._rules bit for bit. A
    positive shrink makes our contact radius smaller and widens the keep-out
    round their front and round their discs, so it only ever removes legal
    landings."""
    c, s = torch.cos(ea), torch.sin(ea)
    spinner = (kind == W_VERT) | (kind == W_DRUM) | (kind == W_HORIZ)
    m = front_margin + shrink
    t_in = tip - shrink
    f_in = body_gap1(fx, fy, ex, ey, c, s, e_len, e_wid) < t_in
    dx, dy = front_discs(ex, ey, c, s, fore, wide)
    d2 = (fx.unsqueeze(1) - dx) ** 2 + (fy.unsqueeze(1) - dy) ** 2
    meet_r = (d2 < ((t_in + rad) ** 2).unsqueeze(1)).any(dim=1)                 # touching one of their discs
    near = guard & (d2 < ((tip + m + rad) ** 2).unsqueeze(1)).any(dim=1)       # within the widened margin of one
    ahead = c * (fx - ex) + s * (fy - ey) - e_len                               # ft past their front edge
    zone = (spinner & (ahead > -m)) | near
    on = f_in | (guard & meet_r)
    return f_in & ~zone, on & zone, ~on


# ── What the gate reads from the observation ─────────────────────────────

def decode_obs(obs):
    """Columns 0..29 of the policy observation, (n, >= 30), back into world
    units: dict of (n,) tensors x, y (our position fix, ft), a (our heading,
    rad), v (our forward speed, ft/s), w (our yaw rate, rad/s), ex, ey (their
    position fix), evx, evy (their tracked velocity, world frame, ft/s),
    arm_read, blade_read (columns 26, 27: -1 absent, else in [0, 1]).
    Column 28 (our true half length) is not read: the gate uses its own
    calibration."""
    half = ARENA_FT / 2
    sa, ca = obs[:, 2], obs[:, 3]                         # sin and cos of our heading
    f, l = obs[:, 12] * V_MAX_FT, obs[:, 13] * V_MAX_FT   # their velocity in our body frame: forward, left
    return dict(x=obs[:, 0] * half + half, y=obs[:, 1] * half + half, a=torch.atan2(sa, ca),
                v=obs[:, 4] * V_MAX_FT, w=obs[:, 5] * YAW_RATE_MAX,
                ex=obs[:, 14] * half + half, ey=obs[:, 15] * half + half,
                evx=f * ca - l * sa, evy=f * sa + l * ca,
                arm_read=obs[:, 26], blade_read=obs[:, 27])


def their_heading(d, hdg=None):
    """Their heading as the gate assumes it, (n,) rad. ``d`` is a decode_obs
    dict; ``hdg`` is observation columns 38..43, (n, 6), or None.

    With no reading: the way they travel when their tracked speed is over
    GATE_MOVING_FT_S, else facing us. An axis reading (columns 38..40, the
    angle delta = bearing(them -> us) - their heading, modulo pi) picks the
    end of the axis nearest that guess. A nose reading (41..43, delta itself)
    beats both."""
    bearing = torch.atan2(d["y"] - d["ey"], d["x"] - d["ex"])          # from them to us
    speed = torch.sqrt(d["evx"] * d["evx"] + d["evy"] * d["evy"])
    travel = torch.atan2(d["evy"], d["evx"])
    guess = torch.where(speed > GATE_MOVING_FT_S, travel, bearing)
    if hdg is None:
        return guess
    axis_ok, nose_ok = hdg[:, 0] > 0.5, hdg[:, 3] > 0.5
    da = 0.5 * torch.atan2(hdg[:, 2], hdg[:, 1])                       # delta modulo pi
    c1 = angle_wrap(bearing - da)
    c2 = angle_wrap(c1 + math.pi)
    axis = torch.where(torch.cos(c1 - guess) >= torch.cos(c2 - guess), c1, c2)
    nose = angle_wrap(bearing - torch.atan2(hdg[:, 5], hdg[:, 4]))
    return torch.where(nose_ok, nose, torch.where(axis_ok, axis, guess))


def _poses_at(d, T):
    """Our centre and heading (x, y, a) and their centre (ex, ey) T seconds
    after the observation: we follow a constant speed and yaw rate arc, they
    go straight at their tracked velocity. T is (n,), or (n, m) for m times
    per row."""
    g = (lambda v: v.unsqueeze(1)) if T.dim() == 2 else (lambda v: v)
    x, y, a, v, w = g(d["x"]), g(d["y"]), g(d["a"]), g(d["v"]), g(d["w"])
    small = w.abs() < 1e-3
    w_safe = torch.where(small, torch.ones_like(w), w)
    a1 = a + w * T
    dx = torch.where(small, v * T * torch.cos(a), v / w_safe * (torch.sin(a1) - torch.sin(a)))
    dy = torch.where(small, v * T * torch.sin(a), v / w_safe * (torch.cos(a) - torch.cos(a1)))
    return x + dx, y + dy, a1, g(d["ex"]) + g(d["evx"]) * T, g(d["ey"]) + g(d["evy"]) * T


def landing_horizon(cal):
    """Seconds from the capture of an observation to the landing of a swing
    fired on the next control step: the observation's age plus the command
    latency (cal.latency), plus the fall from parked."""
    return cal.latency + ARM_DOWN * cal.fire_s


def predict_landing(d, h_their, cal):
    """Where our hammer comes down and where they are then, if the arm is
    fired on the control step after this observation, over the horizon
    landing_horizon(cal), with nothing in the way. They keep the heading
    ``h_their``. Returns (fx, fy, ex, ey, ea), (n,) each."""
    x, y, a, ex, ey = _poses_at(d, landing_horizon(cal))
    reach = cal.half_len + cal.reach                                   # hammer point ahead of our centre
    return x + reach * torch.cos(a), y + reach * torch.sin(a), ex, ey, h_their


def bodies_touch(dx, dy, co, so, ce, se, o_len, o_wid, e_len, e_wid):
    """Separating-axis test of two rectangles, ours and theirs: (dx, dy) is
    their centre minus ours, (co, so) and (ce, se) the headings as cos and
    sin, sizes are half lengths and widths. True where they touch or
    overlap (saw_env.body_contact's first output)."""
    touch = torch.ones_like(dx, dtype=torch.bool)
    for ax, ay in ((co, so), (-so, co), (ce, se), (-se, ce)):
        ours = o_len * (co * ax + so * ay).abs() + o_wid * (-so * ax + co * ay).abs()
        theirs = e_len * (ce * ax + se * ay).abs() + e_wid * (-se * ax + ce * ay).abs()
        touch = touch & (ours + theirs - (dx * ax + dy * ay).abs() >= 0)
    return touch


def stop_at_contact(d, landing, cal, e_len, e_wid):
    """Bodies do not pass through each other. Where the predicted paths
    bring the bodies into contact before the landing, both are taken as
    stopped where they first touch, and the hammer comes down from there.
    The first contact is found on GATE_CONTACT_SAMPLES evenly spaced times
    up to the landing, then GATE_CONTACT_ITERS halvings between the last
    clear time and the first touching one. Rows touching already and closing
    land from where they are now; rows touching and drawing apart, and rows
    that never touch, keep predict_landing's answer. ``landing`` is
    predict_landing's output, ``e_len``, ``e_wid`` their half sizes from the
    card. Returns (fx, fy, ex, ey, ea)."""
    fx, fy, ex, ey, ea = landing
    T = landing_horizon(cal)
    reach = cal.half_len + cal.reach
    ce, se = torch.cos(ea), torch.sin(ea)

    def touch_at(t):
        g = (lambda v: v.unsqueeze(1)) if t.dim() == 2 else (lambda v: v)
        x, y, a, ex_t, ey_t = _poses_at(d, t)
        return bodies_touch(ex_t - x, ey_t - y, torch.cos(a), torch.sin(a), g(ce), g(se),
                            g(cal.half_len), g(cal.half_wid), g(e_len), g(e_wid))

    k = GATE_CONTACT_SAMPLES
    ts = T.unsqueeze(1) * (torch.arange(1, k + 1, device=T.device, dtype=T.dtype) / k)
    hit_s = touch_at(ts)                                                      # (n, k)
    later = hit_s.any(dim=1)
    hi = torch.where(hit_s, ts, 2.0 * T.unsqueeze(1)).amin(dim=1)             # first touching sample
    lo = (hi - T / k).clamp(min=0.0)                                          # the clear one before it
    for _ in range(GATE_CONTACT_ITERS):
        mid = 0.5 * (lo + hi)
        t = touch_at(mid)
        lo, hi = torch.where(t, lo, mid), torch.where(t, mid, hi)
    now = touch_at(torch.zeros_like(T))
    closing = ((d["evx"] - d["v"] * torch.cos(d["a"])) * (d["ex"] - d["x"])
               + (d["evy"] - d["v"] * torch.sin(d["a"])) * (d["ey"] - d["y"])) < 0
    stop = torch.where(now, closing, later)
    x, y, a, ex_c, ey_c = _poses_at(d, torch.where(now, torch.zeros_like(T), lo))
    return (torch.where(stop, x + reach * torch.cos(a), fx), torch.where(stop, y + reach * torch.sin(a), fy),
            torch.where(stop, ex_c, ex), torch.where(stop, ey_c, ey), ea)


# ── State ─────────────────────────────────────────────────────────────────

class GateCal:
    """Per-fight calibration of our own robot, what the team measures, each
    (n,) float: half_len, half_wid (ft, our body), reach (ft, beyond the front edge), tip
    (ft, contact radius of the saw), latency (s, age of an observation plus
    the command latency: the prediction horizon), cmd_lat (s, command to arm
    motion: the timers), fire_s and retract_s (s, full travel each way)."""

    def __init__(self, n=0, **rows):
        for k in CAL_KEYS:
            setattr(self, k, rows[k] if k in rows else torch.zeros(n, device=DEVICE))

    def rows(self, idx=None):
        if idx is None:
            return self
        return GateCal(**{k: getattr(self, k)[idx] for k in CAL_KEYS})


class ArmGate:
    """The gate's state for n fights, all (n,) tensors.

    phase      G_PARKED, G_FIRING, G_HOLD or G_RETRACT
    t_phase    s since the command that started this phase went out
    hold_n     control steps of HOLD left
    latch      the arm switch as the robot sees it (on above +ARM_SWITCH, off below -ARM_SWITCH)
    fires      rising edges of that switch this fight; budget: swings this fight
    blade_est  blade speed the gate believes, share of full, from what it sent
    prev_arm_read  last arm reading (1.0 when absent)
    stale_n, stale_o  control steps their / our position fix has been bit-identical
    last_e, last_o    (n, 2) the last fixes, observation columns 14, 15 and 0, 1
    warm_n     control steps since the fix was last stale or the fight began
    cal        GateCal
    """

    def __init__(self, n):
        f = lambda v=0.0: torch.full((n,), v, device=DEVICE)
        i = lambda: torch.zeros(n, dtype=torch.long, device=DEVICE)
        self.n = n
        self.phase, self.hold_n, self.fires, self.budget = i(), i(), i(), i()
        self.stale_n, self.stale_o, self.warm_n = i(), i(), i()
        self.t_phase, self.blade_est, self.prev_arm_read = f(), f(), f(1.0)
        self.latch = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        self.had_hdg = torch.zeros(n, dtype=torch.bool, device=DEVICE)   # this fight's camera has given a heading at least once
        self.last_e = torch.full((n, 2), -1e9, device=DEVICE)
        self.last_o = torch.full((n, 2), -1e9, device=DEVICE)
        self.cal = GateCal(n)

    def reset(self, idx, budget, cal_rows, blade_est0=None):
        """A new fight in rows ``idx``: parked, nothing fired, ``budget``
        swings, calibration ``cal_rows`` ({key: (k,) float}). ``blade_est0``
        (k,) is where the blade estimate starts (0 when None: a fight from
        the bell)."""
        self.phase[idx], self.hold_n[idx], self.fires[idx] = G_PARKED, 0, 0
        self.budget[idx] = torch.as_tensor(budget, device=DEVICE).long()
        self.stale_n[idx], self.stale_o[idx], self.warm_n[idx] = 0, 0, 0
        self.had_hdg[idx] = False
        self.t_phase[idx], self.prev_arm_read[idx] = 0.0, 1.0
        self.blade_est[idx] = 0.0 if blade_est0 is None else blade_est0
        self.latch[idx] = False
        self.last_e[idx], self.last_o[idx] = -1e9, -1e9
        for k in CAL_KEYS:
            getattr(self.cal, k)[idx] = cal_rows[k]

    def swings_left(self, idx=None):
        return (_sel(self.budget, idx) - _sel(self.fires, idx)).clamp(min=0)

    def _en(self, enabled):
        return torch.as_tensor(enabled, dtype=torch.bool, device=DEVICE).expand(self.n)

    def command(self, request, open_prev, raw_arm, enabled):
        """Once per control step, before the command goes out. ``request``
        (bool): the policy asks to fire; ``open_prev`` (bool): the gate was
        open in the observation the policy acted on; ``raw_arm`` (float): the
        policy's arm output; ``enabled``: Python bool or (n,) bool, the gate
        drives the arm. Returns (arm_out float, fire bool, rise bool): the arm
        command to send, rows the gate fired this step, rows whose arm switch
        turned on. With the gate off the arm follows ``raw_arm`` and the
        phases only follow the switch. No random draws."""
        en = self._en(enabled)
        fire = en & request & open_prev & (self.phase == G_PARKED) & (self.fires < self.budget)
        self.phase = torch.where(fire, G_FIRING, self.phase)
        self.t_phase = torch.where(fire, 0.0, self.t_phase)
        gate_out = torch.where((self.phase == G_FIRING) | (self.phase == G_HOLD), 1.0, -1.0)
        arm_out = torch.where(en, gate_out, raw_arm)
        rise = ~self.latch & (arm_out > ARM_SWITCH)
        fall = self.latch & (arm_out < -ARM_SWITCH)
        self.latch = (self.latch | rise) & ~fall
        self.fires = self.fires + rise.long()
        off = ~en
        self.phase = torch.where(off & rise, G_FIRING, self.phase)
        self.t_phase = torch.where(off & rise, 0.0, self.t_phase)
        self.phase = torch.where(off & fall, G_RETRACT, self.phase)
        self.t_phase = torch.where(off & fall, 0.0, self.t_phase)
        return arm_out, fire, rise

    def advance(self, dt, base_obs, blade_sent, thr_sent, enabled):
        """Once per control step, after the step's physics, on the new
        observation's columns 0..29 (``base_obs``, (n, >= 30)), before
        ready/open. ``blade_sent``, ``thr_sent``: the blade and throttle
        commands that went out this step (what reached the radio, not the
        policy's raw output). Moves the timers and phases, the blade estimate
        and the staleness counters."""
        en = self._en(enabled)
        cal = self.cal
        arm_read = base_obs[:, 26]
        self.t_phase = self.t_phase + dt
        has_arm = arm_read > GATE_READ_PRESENT
        # Down: the reading says so, or (no reading) the slowest the arm can be has passed
        t_land = cal.cmd_lat + GATE_LAT_PAD_S + ARM_DOWN * cal.fire_s * GATE_TIME_MARGIN
        t_back = cal.cmd_lat + GATE_LAT_PAD_S + cal.retract_s * GATE_TIME_MARGIN
        landed = torch.where(has_arm, (arm_read >= ARM_DOWN) | (self.t_phase >= t_land + GATE_FIRE_TIMEOUT_S),
                             self.t_phase >= t_land)
        to_hold = (self.phase == G_FIRING) & landed
        self.hold_n = torch.where((self.phase == G_HOLD) & en, self.hold_n - 1, self.hold_n)
        self.phase = torch.where(to_hold, G_HOLD, self.phase)
        self.hold_n = torch.where(to_hold, GATE_HOLD_STEPS, self.hold_n)
        self.t_phase = torch.where(to_hold, 0.0, self.t_phase)
        self.blade_est = torch.where(to_hold, self.blade_est * GATE_BLADE_DROP, self.blade_est)
        to_back = en & (self.phase == G_HOLD) & (self.hold_n <= 0)
        self.phase = torch.where(to_back, G_RETRACT, self.phase)
        self.t_phase = torch.where(to_back, 0.0, self.t_phase)
        # Parked: two parked readings and at least the quickest retract, or (no reading) the slowest one
        parked_read = (has_arm & (arm_read <= GATE_PARKED_READ) & (self.prev_arm_read <= GATE_PARKED_READ)
                       & (self.t_phase >= cal.retract_s / GATE_TIME_MARGIN))
        retracted = torch.where(has_arm, parked_read, self.t_phase >= t_back)
        self.phase = torch.where((self.phase == G_RETRACT) & retracted & ~self.latch, G_PARKED, self.phase)
        self.prev_arm_read = torch.where(has_arm, arm_read, 1.0)
        # Blade: first-order toward what was sent. Up as slowly as the slowest blade, and not at all
        # while the arm may still be down on them (a blade on a robot gets no faster); down as fast
        # as the fastest one.
        target = (blade_sent.clamp(-1.0, 1.0) + 1.0) * 0.5
        lifted = cal.cmd_lat + GATE_LAT_PAD_S + (1.0 - ARM_DOWN) * cal.retract_s * GATE_TIME_MARGIN
        held = (self.phase == G_HOLD) | ((self.phase == G_RETRACT) & (self.t_phase < lifted))
        rate = torch.where(target > self.blade_est,
                           (1.0 - GATE_BLADE_SAG * thr_sent.abs().clamp(max=1.0)) * ~held / GATE_BLADE_TAU_S,
                           1.0 / GATE_BLADE_DOWN_TAU_S)
        self.blade_est = self.blade_est + (target - self.blade_est) * (dt * rate).clamp(max=1.0)
        # Staleness of both position fixes, and how long they have been fresh
        e, o = base_obs[:, 14:16], base_obs[:, 0:2]
        same_e = (e[:, 0] == self.last_e[:, 0]) & (e[:, 1] == self.last_e[:, 1])
        same_o = (o[:, 0] == self.last_o[:, 0]) & (o[:, 1] == self.last_o[:, 1])
        self.stale_n = torch.where(same_e, self.stale_n + 1, 0)
        self.stale_o = torch.where(same_o, self.stale_o + 1, 0)
        stale = (self.stale_n > GATE_STALE_STEPS) | (self.stale_o > GATE_STALE_STEPS)
        self.warm_n = torch.where(stale, 0, self.warm_n + 1)
        self.last_e, self.last_o = e.clone(), o.clone()

    def ready(self, blade_read, idx=None):
        """Arm parked, a whole swing left, blade up to speed (the reading
        when there is one, else the estimate). (k,) bool."""
        blade_ok = torch.where(blade_read > GATE_READ_PRESENT, blade_read >= GATE_BLADE_READY,
                               _sel(self.blade_est, idx) >= GATE_BLADE_READY)
        return (_sel(self.phase, idx) == G_PARKED) & (self.swings_left(idx) >= 1) & blade_ok

    def open(self, obs, card, hdg_cols, idx=None, front_margin=GATE_FRONT_MARGIN_FT, shrink=GATE_SHRINK_FT,
             rule="heading"):
        """Whether a fire on the next control step is allowed, (k,) bool.
        ``obs``: the observation's columns 0..29 (k, >= 30); ``card``: dict of
        (k,) tensors kind, len, wid, fore, rad, wide (the opponent card);
        ``hdg_cols``: columns 38..43 (k, 6) or None. Their weapon is always
        assumed live and guarding.

        ``rule`` says what to do in rows with no heading reading, where which
        way they point is only a guess (their_heading): "heading" trusts the
        guess and keeps the hammer off the weapon side it implies; "body"
        asks only that the hammer comes down on their body, and leaves the
        side it lands on to the policy."""
        d = decode_obs(obs)
        h = their_heading(d, hdg_cols)
        cal = self.cal.rows(idx)
        fx, fy, ex, ey, ea = stop_at_contact(d, predict_landing(d, h, cal), cal, card["len"], card["wid"])
        kind = card["kind"]
        if rule == "body":
            read = torch.zeros_like(kind, dtype=torch.bool) if hdg_cols is None else (
                (hdg_cols[:, 0] > 0.5) | (hdg_cols[:, 3] > 0.5))
            kind = torch.where(read, kind, W_NONE)
            # A camera that gives headings and has dropped this frame is not a camera that gives none: the
            # arm waits for the next reading instead of falling back to the looser rule
            if idx is None:
                self.had_hdg |= read
            else:
                self.had_hdg[idx] |= read
            blind = _sel(self.had_hdg, idx) & ~read
        elif rule != "heading":
            raise ValueError(f"unknown gate rule {rule!r}; choose heading or body")
        else:
            blind = torch.zeros_like(kind, dtype=torch.bool)
        spinner = (kind == W_VERT) | (kind == W_DRUM) | (kind == W_HORIZ)
        legal, _, _ = landing_class(fx, fy, cal.tip, ex, ey, ea, card["len"], card["wid"], kind, card["fore"],
                                    card["wide"], card["rad"], guard=spinner, front_margin=front_margin, shrink=shrink)
        tracked = ((_sel(self.stale_n, idx) <= GATE_STALE_STEPS) & (_sel(self.stale_o, idx) <= GATE_STALE_STEPS)
                   & (_sel(self.warm_n, idx) >= GATE_WARM_STEPS))
        return self.ready(d["blade_read"], idx) & legal & tracked & ~blind

    def force_retract(self, mask):
        """Rows in ``mask`` that are firing or holding start retracting now."""
        go = mask & ((self.phase == G_FIRING) | (self.phase == G_HOLD))
        self.phase = torch.where(go, G_RETRACT, self.phase)
        self.t_phase = torch.where(go, 0.0, self.t_phase)
