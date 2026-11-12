"""
Scripted opponents that drive like people.

Every opponent is the same driver model with a different tactic on top:

    perception   sees our position late (reaction time) and extrapolates it
                 with imperfect lead
    tactic       picks a heading to steer for and a throttle
    hands        aim wobble and bias, attention lapses, pauses, wall avoidance
    control      turns the heading error into a yaw-rate request with a
                 braking curve, and eases the throttle while pointed away

The driver only produces (throttle, yaw-rate request). The robot itself is
moved by environment.integrate_robot, the same function that moves ours.

Tactics (type id), with the opening each one leaves:

    DUMMY     parked target, no weapon                   (drills only)
    DRIFTER   wanders slowly, no weapon                  (drills; armed, in the
              first combat levels of the sharp field)
    TURRET    holds ground with its weapon on us, lunges when we come close,
              goes hunting when it runs out of patience.  Slow to re-aim.
    PURSUER   drives straight at where it thinks we will be, never lets up.
              Wide turning circle at speed, overshoots a late sidestep.
    CHARGER   lines up, commits to a full-speed charge, then has to recover.
              Telegraphs, cannot steer much mid-charge, exposed afterwards.
    FEINTER   a charger that breaks off some of its charges early.
              Stopped and close after a feint.
    FLANKER   swings wide (or orbits) to come in from the side.
              Shows us its own side while it does.
    WEAVER    zigzags in. Slow to close, converges predictably.
    EVADER    backs away facing us and counter-charges when crowded.
              Passive, slower in reverse, can be cornered.
    PINNER    gets on the arena side of us and drives us toward the wall.
              Takes the long way round and comes from a known direction.
    SELFPLAY  slot driven by a frozen copy of our policy (see opponent_selfplay).

Skill scales with the curriculum level p. Independently of p, every
individual is dealt exactly one weakness (slow reactions, poor turning,
sloppy aim, low top speed, or inattention), and no opponent is both faster
and more agile than we are, so every matchup has a way to win.

That is the classic field. With ``OpponentMixture.sharp`` set (the saw rules
set it) the expert end is harder and the beginner end easier, and the first
combat levels include drifters that carry a weapon: see SHARP below.
"""
import math

import torch

from config import (
    DEVICE, DT, CTRL_DT, ARENA_FT, WALL_CLAMP_IN, angle_wrap,
)

# ─── Types ────────────────────────────────────────────────────────────────
(DUMMY, DRIFTER, TURRET, PURSUER, CHARGER, FEINTER,
 FLANKER, WEAVER, EVADER, PINNER, SELFPLAY) = range(11)
TYPE_NAMES = ["Dummy", "Drifter", "Turret", "Pursuer", "Charger", "Feinter",
              "Flanker", "Weaver", "Evader", "Pinner", "SelfPlay"]
N_TYPES      = len(TYPE_NAMES)
DRILL_TYPES  = (DUMMY, DRIFTER)
COMBAT_TYPES = (TURRET, PURSUER, CHARGER, FEINTER, FLANKER, WEAVER, EVADER, PINNER)
# Passive styles: reported, but not allowed to block the curriculum.
SOFT_TYPES   = (TURRET, EVADER)

# Share of each tactic. Early training sees the simple ones; the full mix is
# reached at p = MIX_FULL_AT.
MIX_EARLY = {TURRET: .20, PURSUER: .35, CHARGER: .20, WEAVER: .10, EVADER: .15}
MIX_FULL  = {TURRET: .08, PURSUER: .20, CHARGER: .20, FEINTER: .10,
             FLANKER: .14, WEAVER: .08, EVADER: .08, PINNER: .12}
MIX_FULL_AT = 0.4

# ─── Driver skill: (range at p=0, range at p=1) ───────────────────────────
REACT_S       = ((0.50, 0.80), (0.22, 0.38))   # how stale their view of us is
LEAD          = ((0.00, 0.10), (0.20, 0.70))   # fraction of our motion they anticipate
AIM_SIGMA     = ((0.15, 0.30), (0.04, 0.10))   # rad, slow wobble in where they point
THROTTLE      = ((0.50, 0.70), (0.85, 1.00))   # how much of their top speed they use
SPEED_RATIO   = ((0.40, 0.60), (0.75, 1.05))   # their top speed / ours
AGILITY_RATIO = ((0.55, 0.80), (0.70, 1.05))   # their yaw rate and yaw accel / ours
STEER_GAIN    = ((0.40, 0.70), (0.60, 1.00))   # how briskly they wind the steering on (1 = as fast as the robot allows)
AIM_BIAS      = (0.15, 0.05)                   # rad, fixed per-driver offset, p=0 -> p=1
LAPSE_HZ      = (0.40, 0.10)                   # attention lapses per second
MAX_EDGE      = 1.05    # cap on speed ratio x agility ratio: never better at both
REACT_FLOOR_S = 0.20    # nobody reacts faster than this

# The one weakness each individual is dealt.
W_SLOW_REACT, W_POOR_TURN, W_SLOPPY_AIM, W_SLOW, W_INATTENTIVE = range(5)
N_WEAKNESSES = 5
W_NO_WEAKNESS = 5       # sharp field only
WEAKNESS_NAMES = ["slow_react", "poor_turn", "sloppy_aim", "slow", "inattentive", "none"]

CLASSIC = dict(REACT_S=REACT_S, LEAD=LEAD, AIM_SIGMA=AIM_SIGMA, THROTTLE=THROTTLE, SPEED_RATIO=SPEED_RATIO,
               AGILITY_RATIO=AGILITY_RATIO, STEER_GAIN=STEER_GAIN, AIM_BIAS=AIM_BIAS, LAPSE_HZ=LAPSE_HZ,
               MAX_EDGE=MAX_EDGE, REACT_FLOOR_S=REACT_FLOOR_S)

# ─── The sharp field ──────────────────────────────────────────────────────
# Experts react faster, lead better, wander less, use the whole throttle, and
# their robot can be quicker than ours: real opponents are sometimes slower
# and sometimes faster. The higher the level, the more of them are dealt no
# weakness at all, and the more of them guard: with us close and off their
# nose they drop what they were doing and come about, backing off as they
# turn, instead of leaving a side open.
#
# Beginners are slower to react and to come about than the classic ones, and
# none of them guard. A spinner that keeps its weapon on us beats our saw
# weapon to weapon, so the only way in is round the side, and the first level
# has to leave that way open for it to be found.
SHARP = dict(
    REACT_S       = ((0.75, 1.20), (0.15, 0.30)),
    LEAD          = ((0.00, 0.10), (0.35, 0.85)),
    AIM_SIGMA     = ((0.15, 0.30), (0.03, 0.08)),
    THROTTLE      = ((0.50, 0.70), (0.90, 1.00)),
    SPEED_RATIO   = ((0.40, 0.60), (0.70, 1.50)),   # ours is 15.4 mph; heavyweights list 9 to 12 (slow), 15 to 20, a few 22+
    AGILITY_RATIO = ((0.30, 0.50), (0.75, 1.15)),
    STEER_GAIN    = ((0.40, 0.70), (0.75, 1.00)),
    AIM_BIAS      = (0.15, 0.03),
    LAPSE_HZ      = (0.40, 0.05),
    MAX_EDGE      = 1.25,
    REACT_FLOOR_S = 0.15,
)
# How good the best of them are. Level p of the sharp field draws its drivers
# at p x SHARP_TOP of the scale above, so at 0.7 the top level is what 0.7
# used to be: 63% guard instead of 90%, they react in 0.33 to 0.57 s instead
# of 0.15 to 0.30, and are up to 1.23x our speed instead of 1.5x. The team
# asked for drivers "a little worse" (2026-11-10). The tactic mix still
# follows p itself.
SHARP_TOP = 0.70
NO_WEAKNESS_AT_P1 = 0.50          # share dealt no weakness at p = 1 (none at p = 0)
GUARD_SHARE       = (0.00, 0.90)  # share that guard, p = 0 -> p = 1
GUARD_FT          = 9.0           # they guard when we are this close ...
GUARD_RAD         = 1.0           # ... and this far off their nose (57 deg)
# A blade that is not up to speed does no damage (config.DR_BITE_FLOOR), so a
# driver who knows their robot gets clear after a hit and lets it spin up.
RESPIN_SHARE      = (0.00, 0.85)  # share that do, p = 0 -> p = 1. The rest come straight back in with a slow blade.
RESPIN_AT         = (0.65, 0.90)  # they wait until the blade is back to this share of full speed
RESPIN_FT         = 10.0          # backing away while we are within this, standing with their nose on us beyond it
# The first sharp levels also meet drifters that carry a weapon. They wander
# without regard for us, so their nose is where they are going: the plainest
# case of "come in from where the weapon is not". Gone by p = MIX_FULL_AT.
SHARP_MIX_EARLY = {DRIFTER: .30, TURRET: .14, PURSUER: .25, CHARGER: .14, WEAVER: .07, EVADER: .10}

# ─── Fixed driver constants ───────────────────────────────────────────────
OU_AIM_TAU   = 0.30     # s, how long an aim error persists
PAUSE_HZ     = 0.03     # per second, chance of stopping to reassess
KILL_RADIUS  = 5.0      # ft, inside this they steer at us, not at the lead point
AVOID_MARGIN = 2.5      # ft, start steering off a wall or hazard this far out
STEER_KP     = 8.0      # 1/s, proportional steering gain near zero error
VEL_SLOTS    = 4        # control steps their estimate of our velocity spans
_RING        = int(1.2 / CTRL_DT) + VEL_SLOTS + 2   # perception history slots


def _lerp(a, b, t):
    return a + (b - a) * t


def _U(k, lo, hi):
    if torch.is_tensor(lo) or torch.is_tensor(hi):
        # per-row bounds (a per-fight p under the saw rules)
        return lo + (hi - lo) * torch.rand(k, device=DEVICE)
    return torch.empty(k, device=DEVICE).uniform_(lo, hi)


def _skill(k, table, p):
    """Uniform draw from a (novice range, expert range) table at level p."""
    (n_lo, n_hi), (e_lo, e_hi) = table
    return _U(k, _lerp(n_lo, e_lo, p), _lerp(n_hi, e_hi, p))


def type_mix(p, sharp=False):
    """Probability of each type id at curriculum level p."""
    t = min(1.0, max(0.0, p / MIX_FULL_AT))
    early = SHARP_MIX_EARLY if sharp else MIX_EARLY
    probs = torch.zeros(N_TYPES, device=DEVICE)
    for k in (DRIFTER,) + COMBAT_TYPES:
        probs[k] = _lerp(early.get(k, 0.0), MIX_FULL.get(k, 0.0), t)
    return probs / probs.sum()


def type_mix_rows(p, sharp=False):
    """type_mix for a (k,) tensor of levels: (k, N_TYPES), each row sums to 1."""
    t = (p / MIX_FULL_AT).clamp(0.0, 1.0).unsqueeze(1)
    early = SHARP_MIX_EARLY if sharp else MIX_EARLY
    lo = torch.zeros(N_TYPES, device=DEVICE)
    hi = torch.zeros(N_TYPES, device=DEVICE)
    for k in (DRIFTER,) + COMBAT_TYPES:
        lo[k], hi[k] = early.get(k, 0.0), MIX_FULL.get(k, 0.0)
    probs = _lerp(lo, hi, t)
    return probs / probs.sum(1, keepdim=True)


class OpponentMixture:
    def __init__(self, n, hazard_dist):
        self.n = n
        self.hazard_dist = hazard_dist      # (48, 48) ft to the nearest hazard cell
        self._ar = torch.arange(n, device=DEVICE)
        # Evaluation hook: a (n,) LongTensor pins each slot's type on every reset.
        self.forced_types = None

        f = lambda fill=0.0: torch.full((n,), fill, device=DEVICE)
        self.types    = torch.zeros(n, dtype=torch.long, device=DEVICE)
        self.weakness = torch.zeros(n, dtype=torch.long, device=DEVICE)
        self.armed    = torch.zeros(n, dtype=torch.bool, device=DEVICE)

        # Skill and physique (read by the env to build the enemy's physics)
        self.react, self.lead, self.aim_sigma, self.aim_bias = f(), f(), f(), f()
        self.throttle, self.lapse_hz, self.turn_floor        = f(), f(), f()
        self.steer = f(1.0)
        self.speed_ratio, self.agility_ratio                 = f(1.0), f(1.0)
        self.delay = torch.ones(n, dtype=torch.long, device=DEVICE)

        # Tactic parameters
        self.aim_time, self.track, self.recover_t, self.feint_p = f(), f(), f(), f()
        self.lunge_range, self.standoff, self.orbit_r, self.strike_range = f(), f(), f(), f()
        self.weave_amp, self.weave_period = f(), f()
        self.orbiter = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        # The sharp field (see SHARP). False: the classic field, exactly as it always was.
        self.sharp  = False
        self.guards = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        self.respins = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        self.respin_at = f()

        # Behaviour state
        self.phase   = torch.zeros(n, dtype=torch.long, device=DEVICE)   # 0 setup, 1 attack, 2 recover
        self.t_phase, self.t_aux, self.t_feint = f(), f(), f()
        self.pause_t, self.lapse_t             = f(), f()
        self.side, self.d_min                  = f(1.0), f(99.0)
        self.feint = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        self.ou_aim, self.wander_hdg           = f(), f()
        self.held_thr, self.held_omega         = f(), f()

        # What they have seen of us: (x, y) per control step
        self.seen = torch.zeros(n, _RING, 2, device=DEVICE)
        self.ptr  = 0

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, idx, p, drills, ox, oy, selfplay_frac=0.0, types_override=None):
        """Deal a new individual into each slot in ``idx``: type, skill, the
        one weakness, tactic parameters, and a clean behaviour state.

        ``p`` is a float, or a (k,) tensor with one level per slot (the saw
        rules). ``types_override`` (k,) long pins the type where it is >= 0
        (saw_field's Traves-like tactics); ``forced_types`` still wins."""
        k = idx.numel()
        if k == 0:
            return
        r = torch.rand(k, device=DEVICE)
        if drills:
            types = torch.where(r < 0.5, DUMMY, DRIFTER)
        else:
            if torch.is_tensor(p):
                cum   = torch.cumsum(type_mix_rows(p, self.sharp), 1)
                types = (r.unsqueeze(1) > cum).sum(1).clamp(max=N_TYPES - 1)
            else:
                cum   = torch.cumsum(type_mix(p, self.sharp), 0)
                types = (r.unsqueeze(1) > cum.unsqueeze(0)).sum(1).clamp(max=N_TYPES - 1)
            if selfplay_frac > 0:
                types = torch.where(torch.rand(k, device=DEVICE) < selfplay_frac, SELFPLAY, types)
        if types_override is not None:
            types = torch.where(types_override >= 0, types_override, types)
        if self.forced_types is not None:
            types = self.forced_types[idx]
        armed = types >= TURRET
        if self.sharp and not drills:
            armed = armed | (types == DRIFTER)       # in sharp combat the drifters carry a weapon
        is_charger = (types == CHARGER) | (types == FEINTER)

        # Skill at this curriculum level
        S = SHARP if self.sharp else CLASSIC
        if self.sharp:
            p = p * SHARP_TOP        # the tactic mix above went by the level; how good they are goes by this
        react    = _skill(k, S["REACT_S"], p)
        lead     = _skill(k, S["LEAD"], p)
        aim_sig  = _skill(k, S["AIM_SIGMA"], p)
        aim_bias = _U(k, -1.0, 1.0) * _lerp(*S["AIM_BIAS"], p)
        throttle = _skill(k, S["THROTTLE"], p)
        lapse_hz = _lerp(*S["LAPSE_HZ"], p) * _U(k, 0.5, 1.5)
        speed    = _skill(k, S["SPEED_RATIO"], p)
        agility  = _skill(k, S["AGILITY_RATIO"], p)
        recover  = _U(k, 0.4, 1.0)

        # Exactly one weakness each, at any level (the sharp field spares some experts)
        weak = torch.randint(0, N_WEAKNESSES, (k,), device=DEVICE)
        if self.sharp:
            weak = torch.where(torch.rand(k, device=DEVICE) < NO_WEAKNESS_AT_P1 * p, W_NO_WEAKNESS, weak)
            guards = armed & (types != SELFPLAY) & (torch.rand(k, device=DEVICE) < _lerp(*GUARD_SHARE, p))
            recover = recover * torch.where(guards, 0.6, 1.0)
            scripted = armed & (types != SELFPLAY) & (types != DRIFTER)
            self.respins[idx] = scripted & (torch.rand(k, device=DEVICE) < _lerp(*RESPIN_SHARE, p))
            self.respin_at[idx] = _U(k, *RESPIN_AT)
        else:
            guards = torch.zeros(k, dtype=torch.bool, device=DEVICE)
        self.guards[idx] = guards
        react    = react + (weak == W_SLOW_REACT) * _U(k, 0.12, 0.25)
        agility  = agility * torch.where(weak == W_POOR_TURN, _U(k, 0.60, 0.75), 1.0)
        aim_sig  = aim_sig + (weak == W_SLOPPY_AIM) * _U(k, 0.08, 0.15)
        lead     = lead * torch.where(weak == W_SLOPPY_AIM, 0.3, 1.0)
        speed    = speed * torch.where(weak == W_SLOW, _U(k, 0.75, 0.85), 1.0)
        lapse_hz = lapse_hz * torch.where(weak == W_INATTENTIVE, 4.0, 1.0)
        recover  = recover * torch.where(weak == W_INATTENTIVE, 1.8, 1.0)

        # Winnable matchups: never faster and more agile than us at once
        agility = torch.minimum(agility, S["MAX_EDGE"] / speed)
        react   = react.clamp(min=S["REACT_FLOOR_S"])

        # Drill targets just trundle; a self-play copy gets our own physique
        is_drill = ~armed
        is_sp    = types == SELFPLAY
        throttle = torch.where(is_drill, _U(k, 0.12, 0.30), throttle)
        speed    = torch.where(is_drill, 1.0, torch.where(is_sp, _U(k, 0.90, 1.00), speed))
        agility  = torch.where(is_drill | is_sp, 1.0, agility)

        self.types[idx], self.weakness[idx], self.armed[idx] = types, weak, armed
        self.react[idx], self.lead[idx] = react, lead
        self.aim_sigma[idx], self.aim_bias[idx] = aim_sig, aim_bias
        self.throttle[idx], self.lapse_hz[idx] = throttle, lapse_hz
        self.speed_ratio[idx], self.agility_ratio[idx] = speed, agility
        self.delay[idx] = (react / CTRL_DT).round().long().clamp(1, _RING - VEL_SLOTS - 1)
        self.turn_floor[idx] = _U(k, 0.0, 0.5)   # 0 stops to pivot, 0.5 swings a wide arc
        self.steer[idx] = _skill(k, S["STEER_GAIN"], p)

        # Tactic parameters (drawn for every slot; each tactic reads its own)
        snap = torch.rand(k, device=DEVICE) < 0.2            # some chargers barely telegraph
        self.aim_time[idx]     = _U(k, 0.35, 1.2) * torch.where(snap, 0.15, 1.0)
        self.track[idx]        = _U(k, 0.2, 1.0)             # steering authority mid-charge
        self.recover_t[idx]    = recover
        self.feint_p[idx]      = torch.where(types == FEINTER, _U(k, 0.3, 0.6), 0.0)
        self.lunge_range[idx]  = _U(k, 6.5, 10.0)
        self.standoff[idx]     = _U(k, 10.0, 16.0)
        self.orbiter[idx]      = torch.rand(k, device=DEVICE) < 0.4
        self.orbit_r[idx]      = _U(k, 7.5, 11.0)
        self.strike_range[idx] = _U(k, 7.0, 10.0)
        self.weave_amp[idx]    = _U(k, 0.3, 0.8)
        self.weave_period[idx] = _U(k, 0.3, 0.8)

        # Behaviour state
        self.phase[idx]   = 0
        self.t_phase[idx] = torch.where(is_charger, self.aim_time[idx],
                                        torch.where(types == FLANKER, _U(k, 1.0, 3.0), 0.0))
        self.t_aux[idx]   = torch.where(types == TURRET, _U(k, 4.0, 9.0),
                                        torch.where(types == EVADER, _U(k, 3.0, 7.0), 0.0))
        self.t_feint[idx], self.pause_t[idx], self.lapse_t[idx] = 0.0, 0.0, 0.0
        self.side[idx]    = torch.where(torch.rand(k, device=DEVICE) < 0.5, 1.0, -1.0)
        self.d_min[idx]   = 99.0
        self.feint[idx]   = False
        self.ou_aim[idx], self.held_thr[idx], self.held_omega[idx] = 0.0, 0.0, 0.0
        self.wander_hdg[idx] = _U(k, -math.pi, math.pi)

        # Until they have watched us for a while, they know where we started
        self.seen[idx] = torch.stack([ox[idx], oy[idx]], dim=1).unsqueeze(1)

    # ── Drive ─────────────────────────────────────────────────────────────

    def act(self, ex, ey, ea, evx, ox, oy, par, blade=None):
        """One control step for every opponent. Arguments are the enemy's own
        state, our true position, the enemy's physical limits ``par`` and,
        under the saw rules, how fast its blade is turning (0..1).
        Returns (throttle in [-1, 1], yaw-rate request in rad/s)."""
        n, dt = self.n, CTRL_DT
        rand = lambda: torch.rand(n, device=DEVICE)
        U    = lambda lo, hi: _U(n, lo, hi)
        lo, hi = WALL_CLAMP_IN, ARENA_FT - WALL_CLAMP_IN
        T = self.types

        # ── Perception: where we were `react` seconds ago, and our velocity then
        self.ptr = (self.ptr + 1) % _RING
        self.seen[:, self.ptr, 0] = ox
        self.seen[:, self.ptr, 1] = oy
        now = self.seen[self._ar, (self.ptr - self.delay) % _RING]
        old = self.seen[self._ar, (self.ptr - self.delay - VEL_SLOTS) % _RING]
        tx, ty   = now[:, 0], now[:, 1]
        tvx, tvy = (now[:, 0] - old[:, 0]) / (VEL_SLOTS * dt), (now[:, 1] - old[:, 1]) / (VEL_SLOTS * dt)

        dx, dy  = tx - ex, ty - ey
        d       = torch.sqrt(dx * dx + dy * dy + 1e-6)
        bearing = torch.atan2(dy, dx)
        # Lead point: extrapolate over their own delay plus the time to reach us
        horizon = self.lead * (self.react + (d / (evx.abs() + 8.0)).clamp(max=1.0))
        ax  = (tx + tvx * horizon).clamp(lo, hi)
        ay  = (ty + tvy * horizon).clamp(lo, hi)
        aim = torch.where(d < KILL_RADIUS, bearing, torch.atan2(ay - ey, ax - ex))
        off_nose = angle_wrap(bearing - ea).abs()

        is_turret  = T == TURRET
        is_charger = (T == CHARGER) | (T == FEINTER)
        is_flanker = T == FLANKER
        is_evader  = T == EVADER
        phased     = is_turret | is_charger | is_flanker | is_evader

        # ── Phase machine: setup -> attack -> recover -> setup
        self.t_phase -= dt
        self.t_aux   -= dt
        self.t_feint -= dt
        self.pause_t -= dt
        self.lapse_t -= dt
        setup, attack, recover = self.phase == 0, self.phase == 1, self.phase == 2

        # An attack is over once we got close and the gap is opening again
        self.d_min = torch.where(attack, torch.minimum(self.d_min, d), self.d_min)
        passed      = (d > self.d_min + 2.5) & (self.d_min < 9.0)
        feint_abort = attack & self.feint & (self.t_feint <= 0)
        end_attack  = phased & attack & (passed | (self.t_phase <= 0) | feint_abort)
        end_recover = phased & recover & (self.t_phase <= 0)

        wall_clear = torch.minimum(torch.minimum(ex - lo, hi - ex), torch.minimum(ey - lo, hi - ey))
        lunge   = d < self.lunge_range
        trigger = (
            (is_turret  & ((lunge & (off_nose < 0.35)) | (self.t_aux <= 0)))        # in reach, or out of patience
            | (is_charger & (self.t_phase <= 0) & (off_nose < 0.25))                # lined up
            | (is_flanker & torch.where(self.orbiter, self.t_phase <= 0, d < self.strike_range))
            | (is_evader  & ((d < 7.0) | (wall_clear < 3.5) | (self.t_aux <= 0)))   # crowded, cornered or bored
        )
        start_attack = phased & setup & trigger & (self.pause_t <= 0)

        attack_dur = torch.where(is_turret, torch.where(lunge, U(0.5, 0.9), U(2.0, 4.0)),
                     torch.where(is_evader, U(0.8, 1.6),
                     torch.where(is_flanker, U(1.2, 2.0), 3.0)))
        recover_dur = torch.where(feint_abort, U(0.2, 0.4),
                      torch.where(is_charger, self.recover_t, U(0.3, 0.6)))
        setup_dur = torch.where(is_charger, self.aim_time * U(0.6, 1.4),
                    torch.where(is_flanker, U(1.0, 3.0), 0.0))

        self.phase = torch.where(start_attack, 1,
                     torch.where(end_attack, 2,
                     torch.where(end_recover, 0, self.phase)))
        self.t_phase = torch.where(start_attack, attack_dur,
                       torch.where(end_attack, recover_dur,
                       torch.where(end_recover, setup_dur, self.t_phase)))
        self.feint   = torch.where(start_attack, rand() < self.feint_p, self.feint)
        self.t_feint = torch.where(start_attack, U(0.15, 0.5), self.t_feint)
        self.d_min   = torch.where(start_attack, d, self.d_min)
        self.t_aux   = torch.where(end_recover & is_turret, U(4.0, 9.0),
                       torch.where(end_recover & is_evader, U(3.0, 7.0), self.t_aux))
        self.side    = torch.where(end_attack & is_flanker & (rand() < 0.5), -self.side, self.side)
        setup   = phased & (self.phase == 0)
        attack  = phased & (self.phase == 1)
        recover = phased & (self.phase == 2)

        # ── Tactics: a heading to steer for (g) and a throttle (u)
        g, u = aim, self.throttle                                   # PURSUER, SELFPLAY fallback

        hold = setup & (is_turret | is_charger)                     # face us; chargers creep while aiming
        g = torch.where(hold, bearing, g)
        u = torch.where(hold, torch.where(is_turret, 0.0, 0.08), u)

        # FLANKER setup: swing to a point beside us, or orbit at a fixed radius
        beside  = bearing + self.side * (math.pi / 2)
        swing   = (0.7 * d).clamp(3.0, 12.0)
        g_arc   = torch.atan2((ty + torch.sin(beside) * swing).clamp(lo, hi) - ey,
                              (tx + torch.cos(beside) * swing).clamp(lo, hi) - ex)
        tangent = bearing + math.pi + self.side * (math.pi / 2)
        g_orbit = tangent + ((d - self.orbit_r) / self.orbit_r).clamp(-0.7, 0.7) * angle_wrap(bearing - tangent)
        g = torch.where(setup & is_flanker, torch.where(self.orbiter, g_orbit, g_arc), g)

        # EVADER setup: keep the weapon on us and hold the standoff ring
        kite = setup & is_evader
        g = torch.where(kite, bearing, g)
        u = torch.where(kite, torch.where(d < self.standoff - 2.0, -0.7 * self.throttle,
                              torch.where(d > self.standoff + 4.0, 0.5 * self.throttle, 0.0)), u)

        g = torch.where(attack, aim, g)                             # attack: flat out at the lead point
        u = torch.where(attack, 1.0, u)
        g = torch.where(recover, bearing, g)                        # recover: off the throttle, come about
        u = torch.where(recover, 0.0, u)

        # WEAVER: pursue with a heading offset that flips side and shrinks up close
        is_weaver = T == WEAVER
        flip = is_weaver & (self.t_aux <= 0)
        self.side  = torch.where(flip, -self.side, self.side)
        self.t_aux = torch.where(flip, self.weave_period * U(0.6, 1.4), self.t_aux)
        g = torch.where(is_weaver, aim + self.side * self.weave_amp * (d / 14.0).clamp(0.2, 1.0), g)

        # PINNER: get on the arena side of us, then drive us at the nearest wall
        is_pinner = T == PINNER
        to_l, to_r, to_b, to_t = tx, ARENA_FT - tx, ty, ARENA_FT - ty
        nearest = torch.minimum(torch.minimum(to_l, to_r), torch.minimum(to_b, to_t))
        nx = torch.where(nearest == to_l, 1.0, torch.where(nearest == to_r, -1.0, 0.0))   # wall -> arena
        ny = torch.where(nx != 0, 0.0, torch.where(nearest == to_b, 1.0, -1.0))
        stand    = (0.6 * d).clamp(max=8.0)
        lined_up = (-(dx * nx + dy * ny) / d > 0.75) | (d < 8.0)   # we are between it and the wall
        g_herd   = torch.atan2((ty + ny * stand).clamp(lo, hi) - ey, (tx + nx * stand).clamp(lo, hi) - ex)
        g = torch.where(is_pinner, torch.where(lined_up, aim, g_herd), g)
        u = torch.where(is_pinner & lined_up, 1.0, u)

        # DRIFTER: amble on a heading that changes every couple of seconds
        is_drifter = T == DRIFTER
        turn = is_drifter & (self.t_aux <= 0)
        self.wander_hdg = torch.where(turn, U(-math.pi, math.pi), self.wander_hdg)
        self.t_aux      = torch.where(turn, U(1.0, 3.0), self.t_aux)
        g = torch.where(is_drifter, self.wander_hdg, g)

        # ── Guard (sharp field): never leave a side open. With us close and
        # off their nose they come about at once, backing away from a robot
        # on their front quarter and pulling away from one behind them, and
        # whatever attack they were on is over.
        exposed = None
        if self.sharp:
            exposed = self.guards & (d < GUARD_FT) & (off_nose > GUARD_RAD)
            g = torch.where(exposed, bearing, g)
            u = torch.where(exposed, torch.where(off_nose > 1.6, 0.5, -0.3) * self.throttle, u)
            broken = exposed & attack
            self.phase   = torch.where(broken, 2, self.phase)
            self.t_phase = torch.where(broken, U(0.15, 0.35), self.t_phase)

        # ── Spin back up (sharp field): with a slow blade they keep their
        # nose on us and get clear, backing away from a robot in front of
        # them and pulling away from one behind, until the blade is back.
        winding = None
        if self.sharp and blade is not None:
            winding = self.respins & (blade < self.respin_at)
            clear = torch.where(off_nose > 1.6, 1.0, -1.0) * self.throttle
            g = torch.where(winding, bearing, g)
            u = torch.where(winding, torch.where(d < RESPIN_FT, clear, 0.0), u)
            broken = winding & attack
            self.phase   = torch.where(broken, 2, self.phase)
            self.t_phase = torch.where(broken, U(0.15, 0.35), self.t_phase)

        # ── Hands
        # Pause: now and then a driver stops to reassess (never mid-attack or up close)
        start_pause = self.armed & ~attack & (d > 10.0) & (self.pause_t <= 0) & (rand() < PAUSE_HZ * dt)
        self.pause_t = torch.where(start_pause, U(0.7, 2.0), self.pause_t)
        paused = self.armed & ~attack & (self.pause_t > 0)
        g = torch.where(paused, bearing, g)
        u = torch.where(paused, 0.0, u)

        # Aim error: slow wobble plus a fixed per-driver bias, smaller up close
        self.ou_aim = (self.ou_aim * (1.0 - dt / OU_AIM_TAU)
                       + torch.randn(n, device=DEVICE) * self.aim_sigma * math.sqrt(2.0 * dt / OU_AIM_TAU))
        g = g + self.armed * (self.ou_aim * (d / 15.0).clamp(0.2, 1.0) + self.aim_bias)

        # Wall and hazard avoidance while driving forward, unless they are on top of us
        reach = 2.0 + 0.35 * evx.abs()
        lx, ly = ex + torch.cos(g) * reach, ey + torch.sin(g) * reach
        clear = torch.minimum(torch.minimum(lx - lo, hi - lx), torch.minimum(ly - lo, hi - ly))
        hz = self.hazard_dist[ly.long().clamp(0, ARENA_FT - 1), lx.long().clamp(0, ARENA_FT - 1)] - 1.0
        swerve = ((AVOID_MARGIN - torch.minimum(clear, hz)) / AVOID_MARGIN).clamp(0.0, 1.0) * 0.9
        avoid  = (u > 0.05) & ((d > 7.0) | ~self.armed)
        to_centre = torch.atan2(ARENA_FT / 2 - ey, ARENA_FT / 2 - ex)
        g = torch.where(avoid, g + angle_wrap(to_centre - g) * swerve, g)

        # ── Control: heading error -> yaw-rate request on a braking curve
        err   = angle_wrap(g - ea)
        alpha = par.dalpha / DT
        authority = torch.where(attack & is_charger, self.track, torch.where(recover, 0.6, 1.0))
        if exposed is not None:
            authority = torch.where(exposed, 1.0, authority)
        if winding is not None:
            authority = torch.where(winding, 1.0, authority)
        rate  = torch.minimum(torch.sqrt(1.4 * alpha * err.abs()), STEER_KP * err.abs()) * self.steer
        omega = torch.sign(err) * torch.minimum(rate, par.yaw_cap * authority)
        # Ease off the throttle while pointed the wrong way
        aligned  = torch.cos(err).clamp(min=0.0) ** 1.5
        throttle = torch.where(u > 0, u * (self.turn_floor + (1.0 - self.turn_floor) * aligned), u)

        parked   = T == DUMMY
        throttle = torch.where(parked, 0.0, throttle)
        omega    = torch.where(parked, 0.0, omega)

        # Attention lapse: the sticks stay where they were for a moment
        start_lapse = self.armed & (self.lapse_t <= 0) & (rand() < self.lapse_hz * dt)
        self.lapse_t = torch.where(start_lapse, U(0.15, 0.45), self.lapse_t)
        lapsed   = self.lapse_t > 0
        throttle = torch.where(lapsed, self.held_thr, throttle)
        omega    = torch.where(lapsed, self.held_omega, omega)
        self.held_thr, self.held_omega = throttle, omega

        return throttle.clamp(-1.0, 1.0), omega
