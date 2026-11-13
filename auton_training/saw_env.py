"""
The saw rules: our robot's real weapon, and fights that run to the clock.

Our weapon is one saw blade on an arm that swings over the top of the robot.
The robot is set up for a fight as one of two things, never both:

    hammer saw    The arm swings over and brings the blade down just beyond
                  the nose, onto whatever is there. Parked, the blade sits
                  behind the robot and touches nothing. This is how the team
                  runs it (config.VERT_SETUP_P is 0), so the hammer is the
                  whole weapon: spin up, get the nose on them, strike, back
                  off, spin up again.
    vert          Arm locked. The parked blade works like a vertical
                  spinner: back into them to hit. Kept in the rules, dealt
                  to no fight.

The policy commands four things: throttle, turn, the arm (fire or park) and
the blade speed. Movement and the saw affect each other: hard driving starves
the blade of current while it spins up, a fast blade makes the robot slower
to turn, and every bite takes speed off the blade that it has to win back.

What is different from the classic rules (environment.BattleEnv):

    hits        do not end the fight. A hit scores its damage, throws the
                robot that took it and leaves it without drive for a moment;
                a weapon held in contact keeps biting. Only the clock ends an
                episode, and the fight is then scored on damage.
    opponents   scripted opponents carry a vert, a drum or a horizontal
                spinner (a bar that reaches past the nose and often past the
                front corners, and throws what it hits sideways). Each bites
                whenever its front touches our body. They drive sharper
                (opponent_library.SHARP) and can be quicker than we are.
    their front is off limits. Our blade scores nothing on the front of a
                spinner, on its front corners or near what its blade sweeps,
                and takes the hit if that blade is cutting. Damage is only
                ever dealt to a side or the back. Their blade does a full
                hit wherever it lands on us: our forks are not on the hammer
                end.
    a hit has to be charged. A bite is worth the energy the blade carries
                over what armour shrugs off (DR_BITE_FLOOR): nothing much
                below two thirds of full speed, and every bite takes the
                blade's speed. A blade that is on a robot cannot spin up:
                its motor is working against that robot. Tapping at a target
                or sitting on it does nothing; the way to hurt is to spin
                up, hit, get clear and do it again.
    sizes       both robots are rectangles whose length and width are drawn
                per fight. The policy is told its own.
    our robot   wider randomization (config.SAW_DR), a reverse gear that can
                be slower than forward, commands that are sometimes lost on
                the radio, a camera that sometimes loses either robot (more
                often while they touch), and a heading estimate that wanders
                and jumps when we are hit: the camera gives positions only.
    hazards     count under any part of our body, cost damage for as long
                as we are on them, and do not end the fight.
    walls       take only the motion into them, so a robot can scrape along
                one and drive off it. Jammed between a wall and the other
                robot it can barely turn.
    starts      a share of fights open in a bad spot: pinned on a wall,
                cornered, or with the opponent already on us.
    fights      evaluation fights last 20 s. Training mixes 20 s, 60 s and
                3 minute fights so the policy's memory has run a whole real
                one. Keeping away from the opponent is charged by the second,
                in the reward and on the scorecard: a real fight is decided
                on aggression as much as on damage.
    the arm     has about ten full swings in it over three minutes.

Every number is an estimate (config.py, "Saw rules"; docs/assumptions.md).
The ones named DR_ are redrawn per episode so the policy cannot lean on any
single value.
"""
import math

import torch

from config import (
    DEVICE, ARENA_FT, WALL_CLAMP_IN, DT, CTRL_DT, MAX_STEPS, V_MAX_FT, YAW_RATE_MAX,
    OUR_WEAPON_REACH, OUR_WEAPON_TIP_RANGE, GAMMA_SAW, MPH2FT, R_SAW_WALL_HIT, R_SAW_WALL_PER_S, R_ACTION_RATE,
    K_FLANK, FLANK_FAR, FLANK_NEAR, K_APPROACH, BODY_RESTITUTION,
    SAW_OBS_DIM, SAW_ACT_DIM, SAW_DR, DR_REVERSE, DR_CMD_DROP_P, DR_TRACK_LOSS_HZ, TRACK_LOSS_S,
    NO_ARM_READING_P, NO_BLADE_READING_P, DR_TREAD_SHARE, TRACK_LOSS_WHO, TRACK_LOSS_TOUCHING, DR_HDG_KICK,
    DR_OUR_HALF_LEN, DR_OUR_HALF_WID, DR_EN_HALF_LEN, DR_EN_HALF_WID,
    ARM_SWITCH, ARM_PARKED, ARM_DOWN, HAMMER_REACH, ARM_JAM_P, ARM_JAM_P_DRILLS, ARM_SWINGS_3MIN, VERT_SETUP_P,
    DR_ARM_FIRE_S, DR_ARM_RETRACT_S, DR_BLADE_TAU_S, DR_BLADE_KEEP, DR_BLADE_SAG, DR_BLADE_GYRO, DR_BLADE_TOP,
    BLADE_MIN_BITE, BLADE_FLOOR_KEEP, BLADE_NOISE, ARM_NOISE,
    EN_WEAPON_SHARE, DR_EN_POWER, DR_EN_TAU_S, DR_EN_GYRO, DR_EN_KEEP, EN_RECOIL, DR_EN_NOSE_FT,
    DR_VERT_WIDE, DR_DRUM_WIDE, DR_HORIZ_HUB, EN_BLADE_FT, EN_FRONT_DISCS, DR_FRONT_MARGIN_FT, EN_BLADE_START, DR_BITE_FLOOR,
    DMG_VERT, DMG_HAMMER_SAW, DMG_HAMMER_BLOW, DMG_CLASH, DMG_SPINNER,
    RETREAT_S, RETREAT_FT, R_RETREAT_PER_FT, R_RAM_PER_S, R_SLOW_PER_S, WALL_FRICTION,
    REBITE_S, CLASH_KEEP, KNOCK_VERT_FT_S, KNOCK_SPIN_FT_S, KNOCK_CLASH_FT_S, KNOCK_RECOIL, KNOCK_YAW, HORIZ_SIDEWAYS,
    STUN_S, AIR_S, BLOW_STUN_S, STUN_TRACTION, DR_SQUEEZE_YAW, SQUEEZE_GAP_FT, HAZARD_DMG_PER_S, HAZARD_DMG_ENTER, R_DAMAGE, R_TAKEN_SHARE, ATTACK_IN_DMG, PTS_MARGIN, STRIKE_S,
    FIGHT_LEN_S, R_PASSIVE_PER_S, PASSIVE_FT, PASSIVE_CLOSE_FT_S, PASSIVE_PTS_PER_S,
    SPAWN_CLEAR_FT, SPAWN_PINNED, SPAWN_CORNERED, SPAWN_CONTACT, SPAWN_PINNING, angle_wrap,
    SAW_OBS_RING, CAMERA_PERIOD_STEPS, BELL_SHORT_P, BELL_FACE_RAD, PRESS_V_CAP_MPH, R_FOUL, DR_HAMMER_REACH,
    HDG_SENSE_P, HDG_SENSE_SIGMA_DEG, HDG_SENSE_DROP_P, PRIV_DIM,
)
from environment import (
    BattleEnv, RANGES, integrate_robot, _U, _dr, _lag_steps, _ACT_RING, _ARENA_DIAG,
    RW_OUTCOME, RW_TIME, RW_SHAPE, RW_WALL, RW_SMOOTH, END_PTS_WIN, END_PTS_LOSS, END_PTS_DRAW,
)
from opponent_library import SELFPLAY, N_TYPES
import saw_gate
import saw_field
from saw_gate import (
    ArmGate, landing_class, GATE_CAL_LEN_FT, GATE_CAL_REACH_FT, GATE_CAL_TIP_FT, GATE_CAL_LAT_SIGMA_S,
    GATE_CAL_TIME_SHARE, GATE_EST_START_SIGMA,
)
import saw_walls
from saw_field import SawField, Scenario, PROFILE_TRAVES, EN_ACT_LAG_S, HEADING_MODES

PHYSICS_VERSION = "saw-v10"

# Weapon kinds: nothing (drill targets), the three a scripted opponent can
# carry, and our hammer saw (also carried by a self-play copy of us)
W_NONE, W_VERT, W_DRUM, W_HORIZ, W_SAW = range(5)
W_NAMES = ["none", "vert", "drum", "horizontal", "saw"]
assert (W_NONE, W_VERT, W_DRUM, W_HORIZ) == (saw_gate.W_NONE, saw_gate.W_VERT, saw_gate.W_DRUM, saw_gate.W_HORIZ)
assert (W_NONE, W_VERT, W_DRUM, W_HORIZ) == (saw_field.W_NONE, saw_field.W_VERT, saw_field.W_DRUM, saw_field.W_HORIZ)
SPINNERS = (W_VERT, W_DRUM, W_HORIZ)
_NEVER = saw_field.NEVER  # a physics step no fight reaches
NO_SWING_LIMIT = 1e9    # Weapon.swings of an arm nobody is counting

# What happened in the fight so far (SawEnv.tally columns), from our side.
# Bites and clashes are counts; a clash is our blade meeting another saw's, or
# landing on a spinner's weapon side. Every swing of the arm lands in exactly
# one of three places and counts there, a full swing as 1 and a short one as
# less: on their body off the weapon side (hammer_blows), on their weapon side
# (weapon_side_landings), or on the floor (hammer_floor). Solid strikes are
# the blade bites worth ATTACK_IN_DMG or more on their own. The gate columns
# count control steps (requests, open, ready) and arm firings; swing_budget
# is the fight's budget, set once.
(T_VERT, T_HAMMER, T_BLOW, T_MISS, T_CLASH, T_TAKEN, T_DEALT, T_HURT,
 T_PINNED_S, T_HAZARD_S, T_STUNNED_S, T_PASSIVE_S,
 T_SOLID, T_FOUL, T_REQ, T_OPEN, T_FIRE, T_READY, T_REQ_OPEN, T_BUDGET) = range(20)
TALLY_NAMES = ["vert_hits", "hammer_bites", "hammer_blows", "hammer_floor", "clashes", "hits_taken",
               "damage_dealt", "damage_taken", "pinned_s", "hazard_s", "stunned_s", "passive_s",
               "solid_strikes", "weapon_side_landings", "gate_requests", "gate_open_steps", "gate_fires",
               "gate_ready_steps", "gate_req_open", "swing_budget"]
TALLY_N = len(TALLY_NAMES)


def attack_in(tally):
    """Which fights we got an attack in, from their tallies (torch or numpy,
    (..., TALLY_N)): our blade did ATTACK_IN_DMG or more. The arm's own blow
    is left out, so thumping them with a blade that is not turning is not
    one, and neither is any number of taps."""
    return tally[..., T_DEALT] - DMG_HAMMER_BLOW * tally[..., T_BLOW] >= ATTACK_IN_DMG

# How a fight can open (SawEnv._bad_spots, SawEnv.spot)
PINNED, CORNERED, CONTACT, PINNING, FREE = range(5)
SPOT_NAMES = ["pinned", "cornered", "contact", "pinning", "free"]


def body_gap(px, py, cx, cy, c, s, half_len, half_wid):
    """Distance in ft from points to a rectangular body, 0 inside it. The
    points are (n, m); the body's centre, heading (cos, sin) and half sizes
    are (n,)."""
    rx, ry = px - cx.unsqueeze(1), py - cy.unsqueeze(1)
    c, s = c.unsqueeze(1), s.unsqueeze(1)
    out_x = ((rx * c + ry * s).abs() - half_len.unsqueeze(1)).clamp(min=0.0)
    out_y = ((-rx * s + ry * c).abs() - half_wid.unsqueeze(1)).clamp(min=0.0)
    return torch.sqrt(out_x * out_x + out_y * out_y)


def body_contact(dx, dy, cos_o, sin_o, cos_e, sin_e, o_len, o_wid, e_len, e_wid):
    """Two rectangular bodies, ours and theirs: (dx, dy) is theirs minus
    ours, sizes are half lengths and half widths. Separating-axis test on
    both bodies' axes. Returns (touching, depth, nx, ny): whether they
    overlap, by how much along the axis of least overlap, and that axis as a
    unit vector pointing from us to them. Moving them ``depth`` apart along it
    is the shortest way to separate them."""
    depth = nx = ny = None
    for ax, ay in ((cos_o, sin_o), (-sin_o, cos_o), (cos_e, sin_e), (-sin_e, cos_e)):
        ours   = o_len * (cos_o * ax + sin_o * ay).abs() + o_wid * (-sin_o * ax + cos_o * ay).abs()
        theirs = e_len * (cos_e * ax + sin_e * ay).abs() + e_wid * (-sin_e * ax + cos_e * ay).abs()
        along  = dx * ax + dy * ay
        over   = ours + theirs - along.abs()
        sign   = torch.where(along >= 0, 1.0, -1.0)
        if depth is None:
            depth, nx, ny = over, ax * sign, ay * sign
        else:
            less = over < depth
            depth = torch.where(less, over, depth)
            nx, ny = torch.where(less, ax * sign, nx), torch.where(less, ay * sign, ny)
    return depth >= 0, depth.clamp(min=0.0), nx, ny


def _by_kind(k, kind, table):
    """One draw per slot from the (vert, drum, horizontal) entry of a config
    table that its weapon kind picks; 0 for any other kind."""
    out = torch.zeros(k, device=DEVICE)
    for w, rng in zip(SPINNERS, table):
        out = torch.where(kind == w, _dr(k, rng), out)
    return out


class Weapon:
    """One robot's weapon, as (n,) tensors.

    A hammer saw (W_SAW) has an arm and a blade. ``arm`` runs from 0, parked
    with the blade behind the robot, to 1, down over the nose. The arm command
    is a switch: fire or park. A spinner (W_VERT, W_DRUM, W_HORIZ) is a blade
    at the front that is always asked for full speed. W_NONE is no weapon.

    Where a blade can touch is a disc on the floor plan. A saw has one behind
    the robot (``aft`` from the centre) and one ahead of it (``fore``). A
    spinner has a row of them at ``fore``, spread ``wide`` to either side:
    across the blade and the forks that feed it for a vert, across the whole
    front for a drum, and all on the hub, as one big disc, for a horizontal.
    ``rad`` is the radius of the front discs.
    """

    def __init__(self, n):
        f = lambda v=0.0: torch.full((n,), v, device=DEVICE)
        z = lambda dtype: torch.zeros(n, dtype=dtype, device=DEVICE)
        self.kind = z(torch.long)
        self.arm, self.blade = f(), f()
        self.fire   = z(torch.bool)   # the arm is being swung forward, or held down
        self.jammed = z(torch.bool)   # this episode the arm is dead: it stays parked whatever is asked
        self.vert   = z(torch.bool)   # this fight the robot is set up as a vert: arm locked, the parked blade is the weapon
        self.swings = f(NO_SWING_LIMIT)   # full swings the arm still has in it this fight (SawEnv deals them out)
        self.lowest = f()             # furthest back the arm has been since it last came down
        self.blow   = f()             # this physics step the arm came down, this hard (0 otherwise, 1 a full swing)
        self.strike = f()             # how hard the swing that has just landed was; 0 once STRIKE_S has passed.
                                      # The saw cuts from above only while this is up
        self.strike_left = f()        # seconds of that left
        self.rebite = f()             # seconds until the blade can bite again
        self.dead   = z(torch.bool)   # a spinner whose weapon has failed: the blade stops and stays stopped (SawEnv sets this)
        self.bound  = z(torch.bool)   # the blade is on a robot: it gets no faster until it is clear (SawEnv sets this)
        self.power  = f(1.0)          # damage per bite at its top speed, against ours at 7,696 rpm
        # This episode's hardware
        self.t_fire, self.t_retract, self.tau = f(0.2), f(0.5), f(1.0)
        self.keep, self.sag, self.gyro = f(0.5), f(), f()
        self.recoil = f(KNOCK_RECOIL)   # share of a hit's throw that comes back at the robot that landed it
        self.turn   = f(1.0)            # which way a horizontal spins: +1 throws what it hits to its own left
        # Where it can touch, ft in the robot's own frame
        self.aft, self.fore, self.wide = f(2.5), f(2.5), f()
        self.rad = f(OUR_WEAPON_TIP_RANGE)

    def reset(self, idx, kind, half_len, half_wid, jam_p=ARM_JAM_P, vert_p=VERT_SETUP_P):
        k = idx.numel()
        saw, vert, drum, horiz = kind == W_SAW, kind == W_VERT, kind == W_DRUM, kind == W_HORIZ
        spinner = vert | drum | horiz
        self.kind[idx] = kind
        for t in (self.arm, self.lowest, self.blow, self.rebite, self.strike, self.strike_left):
            t[idx] = 0.0
        self.fire[idx] = False
        self.bound[idx] = False
        self.dead[idx] = False
        self.swings[idx] = NO_SWING_LIMIT
        # A fight's setup: a hammer saw, or (config.VERT_SETUP_P) a vert, whose arm is locked
        self.vert[idx] = saw & (torch.rand(k, device=DEVICE) < vert_p)
        self.jammed[idx] = self.vert[idx] | (saw & (torch.rand(k, device=DEVICE) < jam_p))
        # Ours starts anywhere from stopped to full speed; theirs is mostly up to speed already
        self.blade[idx]     = torch.where(saw, _U(k, 0.0, 1.0), torch.where(spinner, _U(k, *EN_BLADE_START), 0.0))
        # A saw may be held to the tip-speed rule: `top` is the share of its 7,696 rpm it is allowed, and its
        # speed here is a share of that. A held blade hits with top^2 of the energy, carries top of the
        # angular momentum, and climbing the same curve is at 90% of a lower top sooner.
        top = torch.where(saw, _dr(k, DR_BLADE_TOP).clamp(max=1.0), 1.0)
        sooner = torch.log(1.0 / (1.0 - 0.9 * top)) / math.log(10.0)
        self.power[idx]     = torch.where(spinner, _by_kind(k, kind, DR_EN_POWER), top ** 2)
        self.t_fire[idx]    = _dr(k, DR_ARM_FIRE_S)
        self.t_retract[idx] = _dr(k, DR_ARM_RETRACT_S)
        self.tau[idx]       = torch.where(spinner, _by_kind(k, kind, DR_EN_TAU_S), _dr(k, DR_BLADE_TAU_S) * sooner)
        self.keep[idx]      = torch.where(spinner, _by_kind(k, kind, DR_EN_KEEP), _dr(k, DR_BLADE_KEEP))
        self.sag[idx]       = torch.where(saw, _dr(k, DR_BLADE_SAG), 0.0)
        self.gyro[idx]      = torch.where(saw, _dr(k, DR_BLADE_GYRO) * top, _by_kind(k, kind, DR_EN_GYRO))
        recoil = torch.full((k,), KNOCK_RECOIL, device=DEVICE)
        for w, share in zip(SPINNERS, EN_RECOIL):
            recoil = torch.where(kind == w, share, recoil)
        self.recoil[idx] = recoil
        self.turn[idx]   = torch.where(torch.rand(k, device=DEVICE) < 0.5, 1.0, -1.0)

        # Geometry. ``nose`` is how far a spinner's blade reaches past its front edge.
        nose = _by_kind(k, kind, DR_EN_NOSE_FT)
        hub  = half_len * _dr(k, DR_HORIZ_HUB)
        self.aft[idx]  = half_len + OUR_WEAPON_REACH
        self.fore[idx] = torch.where(saw, half_len + HAMMER_REACH,
                                     torch.where(horiz, hub, half_len + nose - EN_BLADE_FT))
        self.wide[idx] = torch.where(vert, half_wid * _dr(k, DR_VERT_WIDE),
                                     torch.where(drum, half_wid * _dr(k, DR_DRUM_WIDE), 0.0))
        self.rad[idx]  = torch.where(horiz, half_len - hub + nose,
                                     torch.where(saw, OUR_WEAPON_TIP_RANGE, EN_BLADE_FT))

    def step(self, arm_cmd, blade_cmd, load, alive):
        """One physics step. ``arm_cmd`` and ``blade_cmd`` are in [-1, 1],
        ``load`` in [0, 1] is how hard the drive is pulling (|throttle|)."""
        saw = self.kind == W_SAW
        # Arm: a switch with a dead zone in the middle, so a noisy command does not chatter it.
        # An arm without a whole swing left does not start another; one that is down can still be held there.
        can_fire = saw & ~self.jammed & (self.swings >= 1.0)
        self.fire = (self.fire | (can_fire & (arm_cmd > ARM_SWITCH))) & ~(arm_cmd < -ARM_SWITCH)
        was_down = self.arm >= ARM_DOWN
        rate = torch.where(self.fire, DT / self.t_fire, -DT / self.t_retract)
        self.arm = (self.arm + rate * alive).clamp(0.0, 1.0)
        self.lowest = torch.minimum(self.lowest, self.arm)
        # It lands as hard as the swing was long: wagging the arm near the bottom does nothing,
        # and takes that much less out of what the arm has left
        landed = saw & (self.arm >= ARM_DOWN) & ~was_down
        self.blow = torch.where(landed, ((ARM_DOWN - self.lowest) / ARM_DOWN).clamp(0.0, 1.0) ** 2, 0.0)
        self.swings = self.swings - self.blow
        self.lowest = torch.where(landed, self.arm, self.lowest)
        # The saw cuts on its way down, as hard as the swing was long, and not after
        self.strike_left = torch.where(landed, torch.full_like(self.arm, STRIKE_S),
                                       (self.strike_left - DT * alive).clamp(min=0.0))
        self.strike = torch.where(landed, self.blow, self.strike * (self.strike_left > 0))
        # Blade: first-order approach to the asked speed. Spinning up shares the battery with the
        # drive, and does not happen at all while the blade is on a robot.
        target = torch.where(saw, (blade_cmd + 1.0) * 0.5, (self.spinner & ~self.dead).float())
        gain = DT / self.tau * torch.where(target > self.blade, (1.0 - self.sag * load) * ~self.bound,
                                           torch.ones_like(load))
        self.blade = (self.blade + (target - self.blade) * gain * alive).clamp(0.0, 1.0)
        self.blade = torch.where(self.dead, 0.0, self.blade)
        self.rebite = (self.rebite - DT * alive).clamp(min=0.0)

    @property
    def spinner(self):
        """A scripted opponent's weapon: vert, drum or horizontal."""
        return (self.kind == W_VERT) | (self.kind == W_DRUM) | (self.kind == W_HORIZ)

    @property
    def spinning(self):
        """Fast enough to cut."""
        return self.blade >= BLADE_MIN_BITE

    @property
    def rear_low(self):
        """The blade is down behind the robot as a weapon: the parked saw of
        a robot set up as a vert. A hammer saw's parked blade touches nothing."""
        return (self.kind == W_SAW) & self.vert & (self.arm <= ARM_PARKED)

    @property
    def front_low(self):
        """The blade is down in front of the robot (a fired saw, or any spinner)."""
        return torch.where(self.kind == W_SAW, self.arm >= ARM_DOWN, self.spinner)

    def front_discs(self, x, y, c, s):
        """World positions of the front discs, as two (n, EN_FRONT_DISCS) tensors."""
        lat = self.wide.unsqueeze(1) * torch.linspace(-1.0, 1.0, EN_FRONT_DISCS, device=DEVICE)
        return ((x + c * self.fore).unsqueeze(1) - s.unsqueeze(1) * lat,
                (y + s * self.fore).unsqueeze(1) + c.unsqueeze(1) * lat)


class SawEnv(BattleEnv):
    """n fights in parallel under the saw rules. Same motion model, sensors
    and trainer as BattleEnv; see the module docstring for what differs.

    Observation (SAW_OBS_DIM = 44): the 24 numbers documented on
    environment.Observer, then
        24, 25  previous arm and blade outputs of the policy
        26      arm position (0 parked behind, 1 down over the nose)
        27      blade speed as a share of full speed
                (26 and 27 each read -1 if the robot does not report it)
        28, 29  our own half length and half width, in ft, minus 2
        30      swings the arm has left, / 10
        31      time left on the clock, / 180 s
        32      1 if the arm gate is open: a fire asked for now is carried out
        33..35  their weapon on the card: vert, drum, horizontal
        36, 37  their half length and half width on the card, in ft, minus 2
        38..40  which way they point, where the camera gives their long
                axis: valid, cos and sin of twice the angle between the line
                from them to us and their heading
        41..43  the same with the nose told from the tail: valid, cos, sin
                (38..43 are all 0 in a fight with no such reading, which is
                every training fight: config.HDG_SENSE_P)
    Action (SAW_ACT_DIM = 4), each in [-1, 1]:
        0, 1    throttle, turn
        2       arm. With the gate (arm_gate, the default) above +ARM_SWITCH
                asks for a swing, and the gate (saw_gate.ArmGate) carries it
                out if it is open: fire, hold, retract, park, on its own.
                Without the gate it is the switch itself: above +ARM_SWITCH
                fire, below -ARM_SWITCH park, between keep doing what it was
        3       blade speed asked for: -1 stopped, +1 full
    """
    rules = "saw"
    physics_version = PHYSICS_VERSION
    obs_dim, act_dim = SAW_OBS_DIM, SAW_ACT_DIM
    tally_names = tuple(TALLY_NAMES)
    spawn_clear = SPAWN_CLEAR_FT
    ranges = {**RANGES, **SAW_DR}
    sharp_opponents = True
    speed_from_track = True      # the real robot has no wheel odometry
    gamma = GAMMA_SAW
    timeout_bootstrap = False    # the bell ends the fight: nothing comes after it
    obs_ring, obs_cols = SAW_OBS_RING, 8

    def _camera_spec(self):
        period = self.scenario.camera_period
        return CAMERA_PERIOD_STEPS if period is None else int(period)

    def __init__(self, n, stage="combat", p=0.0, auto_reset=True, scenario=None, arm_gate=None, p_range=None,
                 gate_rule="heading"):
        f = lambda *shape: torch.zeros(*shape, device=DEVICE)
        b = lambda: torch.zeros(n, dtype=torch.bool, device=DEVICE)
        full = lambda v: torch.full((n,), v, device=DEVICE)
        # The field, the gate and what they need are in place before
        # BattleEnv.__init__, because that resets every fight
        self._selfplay = None
        self.scenario = scenario or Scenario()
        self.swings_3min = None      # a fixed swing budget per three minutes in place of ARM_SWINGS_3MIN (the trainer's --swings-3min)
        self.gate_shrink = None      # ft: the gate's safety margin on a landing in place of saw_gate.GATE_SHRINK_FT
        self.gate_persist = 1        # control steps in a row the landing must look legal before the gate opens
        self.gate_run = torch.zeros(n, dtype=torch.long, device=DEVICE)
        self.r_retreat, self.r_ram, self.r_slow = R_RETREAT_PER_FT, R_RAM_PER_S, R_SLOW_PER_S   # saw-v9 tempo charges
        self.r_foul = R_FOUL         # what a swing on the floor or on their weapon side costs (the trainer's --foul)
        self.arm_gate = True if arm_gate is None else bool(arm_gate)
        self.gate_rule = gate_rule                       # what the gate asks with no heading reading (ArmGate.open)
        self.p_range = p_range
        self.field = SawField(n, self.scenario)
        self.gate = ArmGate(n)
        self.gate_open, self.gate_ready = b(), b()       # as of the last observation
        self.from_bell = b()                             # this fight started from the bell
        self.reach = full(HAMMER_REACH)                  # ft beyond our front edge where the saw comes down
        self.hdg_mode = torch.zeros(n, dtype=torch.long, device=DEVICE)   # heading reading: 0 none, 1 axis, 2 axis and nose
        self.hdg_sigma, self.hdg_err = f(n), f(n)        # its noise (rad, 1 sigma), and the error on the current camera frame
        self.hdg_drop = torch.ones(n, dtype=torch.bool, device=DEVICE)    # the current frame carries no reading
        self._arm_out, self._prev_arm_out = full(-1.0), full(-1.0)        # the arm command sent this control step, and the last
        self._blade_sent, self._thr_sent = full(-1.0), f(n)              # blade and throttle as they reached the radio
        self.saw_o, self.saw_e = Weapon(n), Weapon(n)
        # Bodies: half length (along the heading) and half width, ft
        self.o_len, self.o_wid, self.e_len, self.e_wid = full(2.0), full(2.0), full(2.0), full(2.0)
        self.o_stun, self.e_stun = f(n), f(n)            # seconds of drive still lost to the last hit
        self.o_air, self.e_air = f(n), f(n)              # seconds still off the floor after being thrown
        self.front_margin = full(0.7)                    # ft: how far a spinner's weapon side reaches back and out
        self.armour_o = full(0.4)                        # share of a blade's full energy our armour shrugs off
        self.armour_e = full(0.4)                        # ... and theirs
        self.forced_weapon = None                        # W_VERT, W_DRUM or W_HORIZ: the tools pin what scripted opponents carry
        self.o_squeezed, self.e_squeezed = b(), b()      # jammed between a wall and the other robot
        self.squeeze_yaw = full(1.0)                     # share of yaw authority left when that happens
        self.o_on_wall, self.o_on_hazard = b(), b()
        self.retreat, self.last_dist = f(n), f(n)        # seconds of backing-off time left after a swing; distance last step
        self.spot = torch.full((n,), FREE, dtype=torch.long, device=DEVICE)   # how this fight opened
        self.tally = f(n, TALLY_N)
        self.horizon = torch.full((n,), MAX_STEPS, dtype=torch.long, device=DEVICE)   # physics steps this fight lasts
        # Our robot's extra faults
        self.rev, self.e_rev = full(1.0), full(1.0)      # top speed in reverse / forward
        self.cmd_drop = f(n)                             # chance per control step that the radio loses a command
        self.sent = f(n, SAW_ACT_DIM)                    # the last command that did get through
        self.track_loss_hz, self.lost_t = f(n), f(n)     # how often the camera loses a robot; seconds until it has it back
        self.lost_who = torch.zeros(n, dtype=torch.long, device=DEVICE)   # whom it lost: 0 them, 1 us, 2 both
        self.touching = b()                              # the bodies were in contact at the last physics step
        self.hdg_kick = f(n)                             # rad (1 sigma) a full-strength hit throws our heading estimate off by
        self.no_arm_reading, self.no_blade_reading = b(), b()   # this fight the robot reports no arm position / no blade speed
        # Each tread has a top speed. A full turn command takes this share of
        # it, and throttle + turn beyond 1 are scaled back together, as the
        # deploy mixer does (nemesis_policy/drive.py, "ratio").
        self.tread_k, self.e_tread_k = f(n), f(n)
        # The enemy's command path. Scripted drivers carry their own delays; a
        # self-play copy gets our latency, and its weapon commands ride along.
        self.en_ring = f(n, _ACT_RING, SAW_ACT_DIM)
        self.en_lag  = torch.zeros(n, dtype=torch.long, device=DEVICE)
        self.en_ptr  = 0
        self.en_wcmd = f(n, 2)                           # (arm, blade) command of self-play copies
        # Sample points for the hazard footprint: a 5 x 5 grid over the body, in half sizes
        g = torch.linspace(-1.0, 1.0, 5, device=DEVICE)
        self._body_u, self._body_v = (t.flatten() for t in torch.meshgrid(g, g, indexing="ij"))
        # ... and for the walls: the corners and the middles of the sides
        self._rim_u = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 0.0, 0.0], device=DEVICE)
        self._rim_v = torch.tensor([1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0], device=DEVICE)

        super().__init__(n, stage=stage, p=p, auto_reset=auto_reset)

    # Self-play copies of our robot would need a gate, a card and a camera
    # of their own. saw-v8a has scripted opponents only.
    @property
    def selfplay(self):
        return self._selfplay

    @selfplay.setter
    def selfplay(self, value):
        if value is not None:
            raise ValueError("self-play opponents are not supported under saw-v8a")

    @property
    def selfplay_frac(self):
        return 0.0

    @selfplay_frac.setter
    def selfplay_frac(self, value):
        if value > 0:
            raise ValueError("self-play opponents are not supported under saw-v8a")

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self):
        obs = super().reset()
        if self.auto_reset:
            # Only the clock ends a fight, so training starts the clocks
            # staggered: otherwise every env would restart in the same step
            # for ever, and most updates would see no finished fight at all.
            self.steps = (torch.rand(self.n, device=DEVICE) * self.horizon).long()
            # A fight joined part-way has that part of its swings left, and
            # a weapon failure that was still to come falls in what remains
            left = (self.horizon - self.steps).clamp(min=1).float() / self.horizon.float()
            part = torch.floor(self.saw_o.swings * left + torch.rand(self.n, device=DEVICE)).clamp(min=1.0)
            keep = (self.saw_o.swings >= NO_SWING_LIMIT) | torch.full_like(self.done, self.scenario.swings is not None)
            self.saw_o.swings = torch.where(keep, self.saw_o.swings, part)
            self.gate.budget = torch.where(keep, self.gate.budget, part.long())
            self.tally[:, T_BUDGET] = self.gate.budget.float()
            at = self.field.dead_at
            timed = (at > 0) & (at < _NEVER)
            later = self.steps + (at.float() / self.horizon.float() * (self.horizon - self.steps).float()).long()
            self.field.dead_at = torch.where(timed, later, at)
            obs = self._observe(self._ar)
        return obs

    def reset_at(self, idx):
        if idx.numel() == 0:
            return
        super().reset_at(idx)
        k = idx.numel()
        # The gate starts parked, with what the team would have measured of
        # our own robot. Its blade estimate starts at nothing from the bell;
        # a fight that opens mid-way is a moment in one where the gate has
        # been following its own blade commands all along.
        budget = self.saw_o.swings[idx].clamp(max=1e9).round().long()
        est = (self.saw_o.blade[idx] + torch.randn(k, device=DEVICE) * GATE_EST_START_SIGMA).clamp(0.0, 1.0)
        self.gate.reset(idx, budget, self._gate_cal(idx), blade_est0=torch.where(self.from_bell[idx], 0.0, est))
        self.gate_open[idx], self.gate_ready[idx] = False, False
        self.tally[idx, T_BUDGET] = budget.float()
        self._arm_out[idx], self._prev_arm_out[idx] = -1.0, -1.0
        self._blade_sent[idx], self._thr_sent[idx] = -1.0, 0.0
        # What the camera says about which way they point, this fight
        r = torch.rand(k, device=DEVICE)
        if self.scenario.heading_mode is not None:
            self.hdg_mode[idx] = HEADING_MODES.index(self.scenario.heading_mode)
        else:
            edges = torch.tensor(HDG_SENSE_P, device=DEVICE).cumsum(0)
            self.hdg_mode[idx] = (r.unsqueeze(1) >= edges).sum(1).clamp(max=len(HDG_SENSE_P) - 1)
        self.hdg_sigma[idx] = torch.deg2rad(_U(k, *HDG_SENSE_SIGMA_DEG))
        self.hdg_err[idx] = 0.0
        self.hdg_drop[idx] = True

    def _gate_cal(self, idx):
        """The gate's calibration for the fights ``idx``, from the robot as
        it is now: each number is the truth with the error a bench
        measurement would have."""
        k = idx.numel()
        U = lambda x: _U(k, -x, x)
        share = lambda: _U(k, 1.0 - GATE_CAL_TIME_SHARE, 1.0 + GATE_CAL_TIME_SHARE)
        lat = lambda steps: (steps.float() * DT + torch.randn(k, device=DEVICE) * GATE_CAL_LAT_SIGMA_S).clamp(min=0.0)
        ob, cmd = self.observer, self.cmd_lag[idx]
        seen = (ob.lag[idx] + cmd).float() + 0.5 * (ob.period[idx] - 1).float()     # physics steps, capture to arm motion
        return dict(half_len=self.o_len[idx] + U(GATE_CAL_LEN_FT),
                    half_wid=self.o_wid[idx] + U(GATE_CAL_LEN_FT),
                    reach=self.reach[idx] + U(GATE_CAL_REACH_FT),
                    tip=self.saw_o.rad[idx] + U(GATE_CAL_TIP_FT),
                    latency=lat(seen), cmd_lat=lat(cmd),
                    fire_s=self.saw_o.t_fire[idx] * share(),
                    retract_s=self.saw_o.t_retract[idx] * share())

    def recalibrate_gate(self, idx=None):
        """Redraw the gate's calibration from the robot as it is now. The
        tools call this after they change a swing time, a latency or the
        reach of fights that are already set up. The reach is taken from
        where the saw comes down now. Budget and phase are left alone."""
        idx = self._ar if idx is None else idx
        self.reach[idx] = self.saw_o.fore[idx] - self.o_len[idx]
        for key, rows in self._gate_cal(idx).items():
            getattr(self.gate.cal, key)[idx] = rows

    def _adjust_start(self, idx):
        k = idx.numel()
        opp, R, sc = self.opponents, self.ranges, self.scenario
        selfplay = opp.types[idx] == SELFPLAY
        rand = lambda: torch.rand(k, device=DEVICE)

        # Bodies. A self-play copy is a robot like ours.
        self.o_len[idx], self.o_wid[idx] = _dr(k, DR_OUR_HALF_LEN), _dr(k, DR_OUR_HALF_WID)
        self.e_len[idx] = torch.where(selfplay, _dr(k, DR_OUR_HALF_LEN), _dr(k, DR_EN_HALF_LEN))
        self.e_wid[idx] = torch.where(selfplay, _dr(k, DR_OUR_HALF_WID), _dr(k, DR_EN_HALF_WID))
        self.field.bodies(idx, self.e_len, self.e_wid, opp.armed[idx])

        # How long this fight lasts, and whether it starts from the bell:
        # every three minute fight does, and a share of the shorter ones
        if sc.fight_s is not None:
            self.horizon[idx] = round(sc.fight_s / DT)
        elif self.auto_reset:
            lengths = torch.tensor([round(sec / DT) for sec, _ in FIGHT_LEN_S], device=DEVICE)
            shares  = torch.tensor([share for _, share in FIGHT_LEN_S], device=DEVICE).cumsum(0)
            self.horizon[idx] = lengths[(rand().unsqueeze(1) >= shares).sum(1).clamp(max=len(FIGHT_LEN_S) - 1)]
        else:
            self.horizon[idx] = MAX_STEPS
        bell = (self.horizon[idx] >= round(180.0 / DT)) | (rand() < BELL_SHORT_P)
        if sc.from_bell is not None:
            bell = torch.full_like(bell, bool(sc.from_bell))
        self.from_bell[idx] = bell

        # Weapons: ours is the saw; a scripted opponent carries a vert, a drum or a horizontal
        shares = torch.tensor(EN_WEAPON_SHARE if sc.weapon_shares is None else sc.weapon_shares,
                              dtype=torch.float32, device=DEVICE)
        edges = (shares / shares.sum()).cumsum(0)
        carried = 1 + (rand().unsqueeze(1) >= edges).sum(1).clamp(max=len(SPINNERS) - 1)     # W_VERT .. W_HORIZ
        carried = self.field.force_kind(idx, carried)
        if self.forced_weapon is not None:
            carried = torch.full_like(carried, self.forced_weapon)
        self.saw_o.reset(idx, torch.full((k,), W_SAW, dtype=torch.long, device=DEVICE), self.o_len[idx], self.o_wid[idx],
                         jam_p=ARM_JAM_P_DRILLS if self.stage == "pretrain" else ARM_JAM_P)
        self.saw_e.reset(idx, torch.where(selfplay, W_SAW, torch.where(opp.armed[idx], carried, W_NONE)),
                         self.e_len[idx], self.e_wid[idx])
        # Where our saw comes down: nobody has measured it, so it is drawn
        reach = _dr(k, DR_HAMMER_REACH)
        self.reach[idx] = reach if sc.reach_ft is None else float(sc.reach_ft)
        self.saw_o.fore[idx] = self.o_len[idx] + self.reach[idx]
        self.field.weapon(idx, self.saw_e, self.e_len, self.e_wid, self.horizon[idx])
        self.field.blade_starts(idx, bell, self.saw_o, self.saw_e)

        # A self-play copy drives a robot as imperfect as ours, through the
        # same command delay. A scripted driver's commands take a moment too.
        self.enemy.yaw_gain[idx] = torch.where(selfplay, _dr(k, R["yaw_gain"]), self.enemy.yaw_gain[idx])
        self.enemy.pull[idx]     = torch.where(selfplay, _dr(k, R["pull"]), self.enemy.pull[idx])
        self.enemy.deadband[idx] = torch.where(selfplay, _dr(k, R["deadband"]), self.enemy.deadband[idx])
        like_ours, scripted = _lag_steps(k, R["act_lag"], _ACT_RING), _lag_steps(k, EN_ACT_LAG_S, _ACT_RING)
        self.en_lag[idx]  = torch.where(selfplay, like_ours, torch.zeros_like(scripted) if sc.legacy_field else scripted)
        self.e_rev[idx]   = torch.where(selfplay, _dr(k, DR_REVERSE), 1.0)
        self.en_ring[idx] = 0.0
        self.en_wcmd[idx] = 0.0

        # Our robot's extra faults
        self.rev[idx] = _dr(k, DR_REVERSE)
        self.cmd_drop[idx] = _dr(k, DR_CMD_DROP_P)
        self.track_loss_hz[idx] = _dr(k, DR_TRACK_LOSS_HZ)
        self.hdg_kick[idx] = _dr(k, DR_HDG_KICK)
        no_arm, no_blade = rand() < NO_ARM_READING_P, rand() < NO_BLADE_READING_P
        if sc.arm_reading is not None:
            no_arm = torch.full_like(no_arm, not sc.arm_reading)
        if sc.blade_reading is not None:
            no_blade = torch.full_like(no_blade, not sc.blade_reading)
        self.no_arm_reading[idx], self.no_blade_reading[idx] = no_arm, no_blade
        self.tread_k[idx] = _dr(k, DR_TREAD_SHARE) * self.o_wid[idx] * YAW_RATE_MAX / V_MAX_FT
        self.e_tread_k[idx] = torch.where(selfplay, _dr(k, DR_TREAD_SHARE) * self.e_wid[idx] * YAW_RATE_MAX / V_MAX_FT, 0.0)
        self.sent[idx] = 0.0
        self.lost_t[idx] = 0.0
        self.touching[idx] = False

        # Wear, if this fight has any, strikes somewhere inside it
        worn = self.wear_at[idx] <= MAX_STEPS
        self.wear_at[idx] = torch.where(worn, (rand() * self.horizon[idx]).long(), _NEVER)
        # The arm has so many full swings in it over three minutes
        # (ARM_SWINGS_3MIN: most often about ten, sometimes many more). A
        # shorter fight gets its share of them, and never less than one. The
        # drills do not count: the policy first has to find out what the arm
        # is for. A scripted opponent has no arm.
        seconds = self.horizon[idx] * DT
        parts = torch.tensor([share for share, _ in ARM_SWINGS_3MIN], device=DEVICE).cumsum(0)
        lo = torch.tensor([rng[0] for _, rng in ARM_SWINGS_3MIN], device=DEVICE)
        hi = torch.tensor([rng[1] for _, rng in ARM_SWINGS_3MIN], device=DEVICE)
        part = (rand().unsqueeze(1) >= parts).sum(1).clamp(max=len(ARM_SWINGS_3MIN) - 1)
        in_3min = lo[part] + (hi[part] - lo[part]) * rand()
        if self.swings_3min is not None:
            in_3min = torch.full_like(in_3min, float(self.swings_3min))
        counted = torch.floor(in_3min * seconds / 180.0 + rand()).clamp(min=1.0)
        if sc.swings is not None:
            counted = torch.full_like(counted, float(sc.swings))
        self.saw_o.swings[idx] = NO_SWING_LIMIT if self.stage == "pretrain" else counted
        self.saw_e.swings[idx] = NO_SWING_LIMIT
        self.front_margin[idx] = _dr(k, DR_FRONT_MARGIN_FT)
        self.armour_o[idx] = _dr(k, DR_BITE_FLOOR)
        self.armour_e[idx] = _dr(k, DR_BITE_FLOOR)

        for t in (self.o_stun, self.e_stun, self.o_air, self.e_air, self.retreat):
            t[idx] = 0.0
        for t in (self.o_squeezed, self.e_squeezed, self.o_on_wall, self.o_on_hazard):
            t[idx] = False
        self.squeeze_yaw[idx] = _dr(k, DR_SQUEEZE_YAW)
        self.tally[idx] = 0.0
        self.spot[idx] = FREE

        if self.stage == "combat":
            self._bad_spots(idx)
        # From the bell the robots stand still and point at each other, give or take
        to_them = torch.atan2(self.ey[idx] - self.oy[idx], self.ex[idx] - self.ox[idx])
        off_o, off_e = _U(k, -BELL_FACE_RAD, BELL_FACE_RAD), _U(k, -BELL_FACE_RAD, BELL_FACE_RAD)
        self.oa[idx]  = torch.where(bell, angle_wrap(to_them + off_o), self.oa[idx])
        self.ea[idx]  = torch.where(bell, angle_wrap(to_them + math.pi + off_e), self.ea[idx])
        self.ovx[idx] = torch.where(bell, 0.0, self.ovx[idx])
        self.evx[idx] = torch.where(bell, 0.0, self.evx[idx])
        # A fight that opens on a wall or a hazard is not charged for arriving there
        ox, oy = self.ox[idx], self.oy[idx]
        lo, hi = WALL_CLAMP_IN + 1e-3, ARENA_FT - WALL_CLAMP_IN - 1e-3
        self.o_on_wall[idx] = (ox <= lo) | (ox >= hi) | (oy <= lo) | (oy >= hi)
        self.o_on_hazard[idx] = self._over_hazard(ox, oy, self.oa[idx], self.o_len[idx], self.o_wid[idx])
        # What the opponents think they know about us has to match where we now are
        opp.seen[idx] = torch.stack([ox, oy], dim=1).unsqueeze(1)

    def _bad_spots(self, idx):
        """Move a share of the new fights so they open in trouble, or with
        the upper hand. Nothing ends a fight early, so the policy will meet
        these states anyway; starting in them makes sure it meets them from
        the first update and practises getting out."""
        k = idx.numel()
        lo, hi = WALL_CLAMP_IN, ARENA_FT - WALL_CLAMP_IN
        U = lambda a, b: _U(k, a, b)
        coin = lambda: torch.rand(k, device=DEVICE) < 0.5
        edges = torch.tensor([SPAWN_PINNED, SPAWN_CORNERED, SPAWN_CONTACT, SPAWN_PINNING], device=DEVICE).cumsum(0)
        spot = (torch.rand(k, device=DEVICE).unsqueeze(1) >= edges).sum(1)     # PINNED .. FREE
        ox, oy = self.ox[idx], self.oy[idx]
        o_len, o_wid, e_len, e_wid = self.o_len[idx], self.o_wid[idx], self.e_len[idx], self.e_wid[idx]

        def across(half_len, half_wid, heading, line):
            """How far a body at this heading reaches along the direction ``line``."""
            return half_len * torch.cos(heading - line).abs() + half_wid * torch.sin(heading - line).abs()

        # A robot that is pointed at the other one (give or take a quarter
        # radian) reaches about this far toward it
        e_nose, o_nose = e_len + 0.25 * e_wid, o_len + 0.3 * o_wid

        # A place on a wall, the unit normal from it into the arena, and the
        # places one body out from it: for them with us on the wall, and for
        # us with them on it
        side = torch.randint(0, 4, (k,), device=DEVICE)
        along = U(lo + 3.0, hi - 3.0)
        wx = torch.where(side == 0, lo, torch.where(side == 1, hi, along))
        wy = torch.where(side == 2, lo, torch.where(side == 3, hi, along))
        nx = (side == 0).float() - (side == 1).float()
        ny = (side == 2).float() - (side == 3).float()
        inward = torch.atan2(ny, nx)
        slip = U(-1.0, 1.0)
        out_e = across(o_len, o_wid, self.oa[idx], inward) + e_nose + U(0.1, 0.8)
        out_o = across(e_len, e_wid, self.ea[idx], inward) + o_nose + U(0.1, 0.8)
        ax, ay = wx + nx * out_e - ny * slip, wy + ny * out_e + nx * slip      # them, with us on the wall
        px, py = wx + nx * out_o - ny * slip, wy + ny * out_o + nx * slip      # us, with them on the wall

        # A place tucked into a corner, and one further out toward the middle
        cx, cy = torch.where(coin(), lo, hi), torch.where(coin(), lo, hi)
        sx, sy = torch.sign(ARENA_FT / 2 - cx), torch.sign(ARENA_FT / 2 - cy)
        kx, ky = cx + sx * U(0.0, 2.5), cy + sy * U(0.0, 2.5)
        out, reach = torch.atan2(sy, sx) + U(-0.5, 0.5), U(6.0, 11.0)
        fx, fy = kx + torch.cos(out) * reach, ky + torch.sin(out) * reach

        # A place right beside where we already are
        around = U(-math.pi, math.pi)
        near = across(o_len, o_wid, self.oa[idx], around) + e_nose + U(0.1, 0.8)
        bx, by = ox + torch.cos(around) * near, oy + torch.sin(around) * near

        def pick(pinned, cornered, contact, pinning, free):
            return torch.where(spot == PINNED, pinned, torch.where(spot == CORNERED, cornered,
                   torch.where(spot == CONTACT, contact, torch.where(spot == PINNING, pinning, free))))

        n_ox, n_oy = pick(wx, kx, ox, px, ox), pick(wy, ky, oy, py, oy)
        n_ex = pick(ax, fx, bx, wx, self.ex[idx]).clamp(lo, hi)
        n_ey = pick(ay, fy, by, wy, self.ey[idx]).clamp(lo, hi)
        at_us = torch.atan2(n_oy - n_ey, n_ox - n_ex)             # the heading that points them at us
        # With the upper hand we open with the weapon on them: the nose of a hammer saw, the tail of a vert
        ours_on = torch.where(self.saw_o.vert[idx], 0.0, math.pi)
        n_oa = torch.where(spot == PINNING, angle_wrap(at_us + ours_on + U(-0.3, 0.3)), self.oa[idx])
        pressing = spot != PINNING                                  # (and not FREE, below)
        n_ea = torch.where(pressing, angle_wrap(at_us + U(-0.25, 0.25)), self.ea[idx])

        # Keep the ordinary start wherever the new one would put the robots
        # inside each other. Three of the four walls carry hazards, so being
        # pinned or cornered on one is part of the drill; the other two starts
        # keep our body clear of them.
        inside, _, _, _ = body_contact(n_ex - n_ox, n_ey - n_oy, torch.cos(n_oa), torch.sin(n_oa),
                                       torch.cos(n_ea), torch.sin(n_ea), o_len, o_wid, e_len, e_wid)
        trapped = (spot == PINNED) | (spot == CORNERED)
        ok = ((spot != FREE) & ~inside & (trapped | ~self._over_hazard(n_ox, n_oy, n_oa, o_len, o_wid))
              & ~self.from_bell[idx])
        pressing = ok & pressing
        self.spot[idx] = torch.where(ok, spot, FREE)

        opp = self.opponents
        keep = lambda new, old: torch.where(ok, new, old)
        self.ox[idx], self.oy[idx], self.oa[idx] = keep(n_ox, ox), keep(n_oy, oy), keep(n_oa, self.oa[idx])
        self.ex[idx], self.ey[idx] = keep(n_ex, self.ex[idx]), keep(n_ey, self.ey[idx])
        self.ea[idx]  = torch.where(pressing, n_ea, self.ea[idx])
        self.ovx[idx] = keep(torch.zeros(k, device=DEVICE), self.ovx[idx])
        # They come in at a share of their top speed, and not at more than a
        # robot that is about to hit something is still doing
        top = self.enemy.v_max[idx]
        if not self.scenario.legacy_field:
            top = top.clamp(max=PRESS_V_CAP_MPH * MPH2FT)
        self.evx[idx] = torch.where(pressing, U(0.2, 0.8) * top * opp.armed[idx],
                                    keep(torch.zeros(k, device=DEVICE), self.evx[idx]))

        # Where they are pressing, they are mid-attack and their weapon has
        # only just bitten: there is a moment to react, not a free hit.
        p_idx = idx[pressing]
        opp.phase[p_idx]   = 1
        opp.t_phase[p_idx] = _U(p_idx.numel(), 0.8, 2.0)
        opp.d_min[p_idx]   = torch.sqrt((n_ex - n_ox) ** 2 + (n_ey - n_oy) ** 2)[pressing]
        self.saw_e.rebite[p_idx] = _U(p_idx.numel(), 0.2, 0.6)

    def _arena(self):
        return saw_walls.raster()

    def _wall(self, ours):
        """Walls are solid for the whole body: its corners and the middles of its sides are kept
        out of them, the wall takes the motion into it and, through friction, a share of the
        motion along it. Returns (ft of clear floor before the push, negative if it was in the
        wall; speed into the wall that was taken)."""
        if ours:
            x, y, a, L, W, v, d = self.ox, self.oy, self.oa, self.o_len, self.o_wid, self.ovx, self.o_drift
        else:
            x, y, a, L, W, v, d = self.ex, self.ey, self.ea, self.e_len, self.e_wid, self.evx, self.e_drift
        c, s = torch.cos(a), torch.sin(a)
        u, w = L.unsqueeze(1) * self._rim_u, W.unsqueeze(1) * self._rim_v
        px = x.unsqueeze(1) + c.unsqueeze(1) * u - s.unsqueeze(1) * w
        py = y.unsqueeze(1) + s.unsqueeze(1) * u + c.unsqueeze(1) * w
        x0, x1, y0, y1 = (p.max(dim=1).values for p in saw_walls.penetration(px, py))
        gap = -torch.maximum(torch.maximum(x0, x1), torch.maximum(y0, y1))
        live = ~self.done
        x0, x1, y0, y1 = (p.clamp(min=0.0) * live for p in (x0, x1, y0, y1))
        vx, vy = c * v - s * d, s * v + c * d
        hit_x = ((x0 > 0) & (vx < 0)) | ((x1 > 0) & (vx > 0))
        hit_y = ((y0 > 0) & (vy < 0)) | ((y1 > 0) & (vy > 0))
        impact = hit_x * vx.abs() + hit_y * vy.abs()
        rub = lambda along, into: torch.sign(along) * (along.abs() - WALL_FRICTION * into.abs()).clamp(min=0.0)
        wx = torch.where(hit_x, torch.zeros_like(vx), torch.where(hit_y, rub(vx, vy), vx))
        wy = torch.where(hit_y, torch.zeros_like(vy), torch.where(hit_x, rub(vy, vx), vy))
        nv, nd = c * wx + s * wy, -s * wx + c * wy
        nx, ny = x + x0 - x1, y + y0 - y1
        if ours:
            self.ox, self.oy, self.ovx, self.o_drift = nx, ny, nv, nd
        else:
            self.ex, self.ey, self.evx, self.e_drift = nx, ny, nv, nd
        return gap, impact

    def _over_hazard(self, x, y, a, half_len, half_wid):
        """True where any part of a robot body at this pose is over a hazard."""
        c, s = torch.cos(a).unsqueeze(1), torch.sin(a).unsqueeze(1)
        u, v = half_len.unsqueeze(1) * self._body_u, half_wid.unsqueeze(1) * self._body_v
        px = x.unsqueeze(1) + c * u - s * v
        py = y.unsqueeze(1) + s * u + c * v
        ix = px.floor().long().clamp(0, ARENA_FT - 1)
        iy = py.floor().long().clamp(0, ARENA_FT - 1)
        return self.hazards[iy, ix].any(dim=1)

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self, action):
        action = action.clamp(-1.0, 1.0)
        alive = ~self.done
        # The arm: the policy asks, and the gate decides what is sent (with
        # the gate off, the policy's own value is). The gate was open or
        # shut in the observation the policy has just acted on.
        request = action[:, 2] > ARM_SWITCH
        arm_out, _, rise = self.gate.command(request, self.gate_open, action[:, 2], self.arm_gate)
        self.tally[:, T_REQ]      += request & alive
        self.tally[:, T_OPEN]     += self.gate_open & alive
        self.tally[:, T_FIRE]     += rise & alive
        self.tally[:, T_READY]    += self.gate_ready & alive
        self.tally[:, T_REQ_OPEN] += request & self.gate_open & alive
        cmd = action.clone()
        cmd[:, 2] = arm_out
        self._arm_out = arm_out
        # Radio: now and then a command does not arrive, and the last one stands
        dropped = torch.rand(self.n, device=DEVICE) < self.cmd_drop
        self.sent = torch.where(dropped.unsqueeze(1), self.sent, cmd)
        self._blade_sent, self._thr_sent = self.sent[:, 3].clone(), self.sent[:, 0].clone()
        # Camera: now and then it loses a robot for a moment, more often
        # while the two touch, and goes on reporting the last place it saw it
        self.lost_t = (self.lost_t - CTRL_DT).clamp(min=0.0)
        rate = self.track_loss_hz * torch.where(self.touching, TRACK_LOSS_TOUCHING, 1.0)
        losing = (self.lost_t <= 0) & (torch.rand(self.n, device=DEVICE) < rate * CTRL_DT)
        self.lost_t = torch.where(losing, _U(self.n, *TRACK_LOSS_S), self.lost_t)
        who = torch.rand(self.n, device=DEVICE)
        them, us = TRACK_LOSS_WHO
        self.lost_who = torch.where(losing, (who >= them).long() + (who >= them + us).long(), self.lost_who)
        lost = self.lost_t > 0
        self.observer.lost = lost & (self.lost_who != 1)
        self.observer.lost_own = lost & (self.lost_who != 0)
        return super().step(action)

    def _apply_wear(self):
        """As in the classic rules, but a fight here can outlast MAX_STEPS,
        so "never" has to be later than that."""
        hit = self.steps >= self.wear_at
        self.our.v_max = torch.where(hit, self.our.v_max * self.wear_speed, self.our.v_max)
        self.our.pull  = torch.where(hit, self.our.pull + self.wear_pull, self.our.pull)
        self.wear_at   = torch.where(hit, _NEVER, self.wear_at)

    def _enemy_decides(self):
        self.en_wcmd.zero_()
        if self.manual_enemy:
            return super()._enemy_decides()
        throttle, omega = self.opponents.act(self.ex, self.ey, self.ea, self.evx,
                                             self.ox, self.oy, self.enemy,
                                             blade=torch.where(self.saw_e.dead, 1.0, self.saw_e.blade))
        if self.selfplay is None:
            self.en_cmd[:, 0], self.en_cmd[:, 1] = throttle, omega
            return
        sp = self.opponents.types == SELFPLAY
        act = self.selfplay.act(self.en_observer, sp,
                                extra=self._weapon_obs(self.saw_e, self.selfplay.prev_action, self.e_len, self.e_wid))
        self.en_cmd[:, 0] = torch.where(sp, act[:, 0], throttle)
        self.en_cmd[:, 1] = torch.where(sp, act[:, 1] * YAW_RATE_MAX, omega)
        self.en_wcmd = act[:, 2:4].clone()               # zero outside the self-play slots

    def _enemy_lag(self):
        """The enemy's command as it reaches its robot this physics step:
        throttle, yaw rate asked for (rad/s), arm, blade."""
        size = self.en_ring.shape[1]
        self.en_ptr = (self.en_ptr + 1) % size
        self.en_ring[:, self.en_ptr] = torch.cat([self.en_cmd, self.en_wcmd], dim=1)
        return self.en_ring[self._ar, (self.en_ptr - self.en_lag) % size]

    def _physics_step(self, action):
        alive = (~self.done).float()
        so, se = self.saw_o, self.saw_e
        cmd, en = self._lag(self.sent), self._enemy_lag()

        # Treads: full throttle and full turn together ask one side for more
        # than its top speed. Both are scaled back by the same factor, which
        # keeps the turn radius. A scripted opponent's driver model has its
        # own limits; a self-play copy has treads like ours.
        over = (cmd[:, 0].abs() + self.tread_k * cmd[:, 1].abs()).clamp(min=1.0)
        cmd = torch.cat([cmd[:, 0:2] / over.unsqueeze(1), cmd[:, 2:4]], dim=1)
        e_over = torch.where(self.e_tread_k > 0,
                             (en[:, 0].abs() + self.e_tread_k * en[:, 1].abs() / YAW_RATE_MAX).clamp(min=1.0), 1.0)
        en = torch.cat([en[:, 0:2] / e_over.unsqueeze(1), en[:, 2:4]], dim=1)

        se.dead = se.dead | (se.spinner & (self.steps >= self.field.dead_at))
        so.step(cmd[:, 2], cmd[:, 3], cmd[:, 0].abs(), alive)
        se.step(en[:, 2], en[:, 3], en[:, 0].abs(), alive)

        # A robot that has just been thrown is off the floor for a moment
        # (nothing slows or turns it), then slides without drive and with
        # little grip. A spinning blade resists the turn. Jammed on a wall, a
        # robot can barely turn at all. Reverse can be a slower gear than forward.
        o_out, e_out = self.o_stun > 0, self.e_stun > 0
        o_fly, e_fly = self.o_air > 0, self.e_air > 0
        one = torch.ones_like(alive)
        o_thr = torch.where(cmd[:, 0] < 0, cmd[:, 0] * self.rev, cmd[:, 0])
        e_thr = torch.where(en[:, 0] < 0, en[:, 0] * self.e_rev, en[:, 0])
        (self.ox, self.oy, self.oa, self.ovx, self.o_omega,
         self.o_drifting, self.o_drift, our_impact) = integrate_robot(
            self.ox, self.oy, self.oa, self.ovx, self.o_omega, self.o_drifting, self.o_drift,
            o_thr * ~o_out, cmd[:, 1] * YAW_RATE_MAX * ~o_out, self.our, alive,
            traction=torch.where(o_out, STUN_TRACTION, 1.0),
            yaw_scale=(1.0 - so.gyro * so.blade) * torch.where(self.o_squeezed, self.squeeze_yaw, one),
            scrape_walls=True, free=o_fly, sliding=o_out & ~o_fly, wall_in=torch.minimum(self.o_len, self.o_wid))
        (self.ex, self.ey, self.ea, self.evx, self.e_omega,
         self.e_drifting, self.e_drift, _) = integrate_robot(
            self.ex, self.ey, self.ea, self.evx, self.e_omega, self.e_drifting, self.e_drift,
            e_thr * ~e_out, en[:, 1] * ~e_out, self.enemy, alive,
            traction=torch.where(e_out, STUN_TRACTION, 1.0),
            yaw_scale=(1.0 - se.gyro * se.blade) * torch.where(self.e_squeezed, self.squeeze_yaw, one),
            scrape_walls=True, free=e_fly, sliding=e_out & ~e_fly, wall_in=torch.minimum(self.e_len, self.e_wid))
        self.o_stun = (self.o_stun - DT * alive).clamp(min=0.0)
        self.e_stun = (self.e_stun - DT * alive).clamp(min=0.0)
        self.o_air = (self.o_air - DT * alive).clamp(min=0.0)
        self.e_air = (self.e_air - DT * alive).clamp(min=0.0)

        self._rules(our_impact)

    # ── Rules ─────────────────────────────────────────────────────────────

    def _rules(self, our_impact):
        """Everything that happens between the robots in one physics step:
        weapon contact, body contact, hazards and walls, and what each does."""
        so, se = self.saw_o, self.saw_e
        live = ~self.done
        dx, dy = self.ex - self.ox, self.ey - self.oy
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        cos_o, sin_o = torch.cos(self.oa), torch.sin(self.oa)
        cos_e, sin_e = torch.cos(self.ea), torch.sin(self.ea)
        e_saw, e_horiz = se.kind == W_SAW, se.kind == W_HORIZ
        our_body   = (self.ox, self.oy, cos_o, sin_o, self.o_len, self.o_wid)
        their_body = (self.ex, self.ey, cos_e, sin_e, self.e_len, self.e_wid)
        col = lambda t: t.unsqueeze(1)

        # Where each blade can touch (see Weapon). A saw has two places:
        # behind the robot while the arm is parked, and beyond its nose once
        # the arm is down. A spinner has its row of front discs.
        tip = torch.full_like(so.rad, OUR_WEAPON_TIP_RANGE)
        o_rx, o_ry = self.ox - cos_o * so.aft,  self.oy - sin_o * so.aft
        o_fx, o_fy = self.ox + cos_o * so.fore, self.oy + sin_o * so.fore
        e_rx, e_ry = self.ex - cos_e * se.aft,  self.ey - sin_e * se.aft
        e_fx, e_fy = se.front_discs(self.ex, self.ey, cos_e, sin_e)            # (n, discs)

        o_r_in = body_gap(col(o_rx), col(o_ry), *their_body).squeeze(1) < tip
        o_f_in = body_gap(col(o_fx), col(o_fy), *their_body).squeeze(1) < so.rad
        e_r_in = body_gap(col(e_rx), col(e_ry), *our_body).squeeze(1) < tip
        e_f_gap = body_gap(e_fx, e_fy, *our_body)                              # (n, discs)
        e_f_in = (e_f_gap < col(se.rad)).any(dim=1)

        o_ready, e_ready = live & (so.rebite <= 0), live & (se.rebite <= 0)
        o_blade, e_blade = so.blade, se.blade                  # speeds going into this step's contact
        # What a bite is worth: the blade's energy over what the armour it lands on shrugs off
        worth = lambda blade, floor: ((blade ** 2 - floor) / (1.0 - floor)).clamp(min=0.0)
        o_energy, e_energy = worth(o_blade, self.armour_e) * so.power, worth(e_blade, self.armour_o) * se.power

        def meet(px, py, pr, qx, qy, qr):
            """Disc p, (n,), against discs q, (n, m): any pair closer than their radii."""
            return ((col(px) - qx) ** 2 + (col(py) - qy) ** 2 < col(pr + qr) ** 2).any(dim=1)

        # Saw on saw (a self-play copy): both blades are down at the same
        # place, and each takes a share of the other's
        clash = e_saw & o_ready & e_ready & (so.spinning | se.spinning) & (
            (so.rear_low  & se.front_low & meet(o_rx, o_ry, tip, e_fx, e_fy, se.rad)) |
            (so.rear_low  & se.rear_low  & meet(o_rx, o_ry, tip, col(e_rx), col(e_ry), tip)) |
            (so.front_low & se.front_low & meet(o_fx, o_fy, so.rad, e_fx, e_fy, se.rad)) |
            (so.front_low & se.rear_low  & meet(o_fx, o_fy, so.rad, col(e_rx), col(e_ry), tip)))

        # A spinner's weapon side is off limits: everything from a margin
        # behind its front edge forwards, and everything within that margin
        # of its blade. A vert's or a drum's blade is its front. A
        # horizontal's bar guards whatever it sweeps, round the corners and
        # down the sides, for as long as it is turning fast enough to cut.
        # Our blade scores nothing there and loses its speed, and within
        # that margin of a blade that is cutting it is our blade that gets hit.
        margin = self.front_margin
        guards = se.spinner & (~e_horiz | se.spinning)
        ahead = lambda px, py: cos_e * (px - self.ex) + sin_e * (py - self.ey) - self.e_len   # ft past their front edge
        o_r_near = guards & meet(o_rx, o_ry, tip + margin, e_fx, e_fy, se.rad)       # our parked blade is by theirs
        o_f_near = guards & meet(o_fx, o_fy, so.rad + margin, e_fx, e_fy, se.rad)    # ... our hammer saw is
        o_r_zone = (se.spinner & (ahead(o_rx, o_ry) > -margin)) | o_r_near
        o_f_zone = (se.spinner & (ahead(o_fx, o_fy) > -margin)) | o_f_near
        o_r_on = so.rear_low  & (o_r_in | (guards & meet(o_rx, o_ry, tip, e_fx, e_fy, se.rad)))      # touching them at all
        o_f_on = so.front_low & (o_f_in | (guards & meet(o_fx, o_fy, so.rad, e_fx, e_fy, se.rad)))
        fouled = live & ((o_r_on & o_r_zone) | (o_f_on & o_f_zone))                  # our blade is on their weapon side
        caught = live & ((o_r_on & o_r_near) | (o_f_on & o_f_near))                  # ... and their blade has it
        stalled = fouled & o_ready & so.spinning

        # Where their blade touched us. Nothing of ours is armoured against it.
        nearest = e_f_gap.argmin(dim=1, keepdim=True)                           # the disc that touched
        hit_x, hit_y = e_fx.gather(1, nearest).squeeze(1), e_fy.gather(1, nearest).squeeze(1)

        # Weapon on body. A blade bites when it is down, cutting, and ready
        # again after its last bite. The hammer arm also lands a blow of its
        # own when it comes down, blade or no blade.
        o_vert   = o_r_in & so.rear_low  & so.spinning & o_ready & ~clash & ~o_r_zone
        o_hammer = o_f_in & so.front_low & so.spinning & o_ready & ~clash & ~o_f_zone & (so.strike > 0)
        e_vert   = e_r_in & se.rear_low  & se.spinning & e_ready & ~clash
        e_front  = (e_f_in | caught) & se.front_low & se.spinning & e_ready & ~clash & (~e_saw | (se.strike > 0))
        o_blow = so.blow * (o_f_in & live & ~o_f_zone)         # 0..1, how hard
        e_blow = se.blow * (e_f_in & live)
        o_miss = so.blow * (~o_f_in & live)                    # the arm came down clear of their body
        e_miss = se.blow * (~e_f_in & live)
        o_floor = so.blow * (~o_f_on & live)                   # ... on the floor
        o_side  = so.blow * (o_f_on & o_f_zone & live)         # ... on their weapon side
        solid = o_hammer & (DMG_HAMMER_SAW * so.strike * o_energy >= ATTACK_IN_DMG)

        dealt = ((o_vert * DMG_VERT + o_hammer * DMG_HAMMER_SAW * so.strike) * o_energy + o_blow * DMG_HAMMER_BLOW
                 + clash * DMG_CLASH * o_energy)
        hurt = ((e_vert * DMG_VERT + e_front * torch.where(e_saw, DMG_HAMMER_SAW * se.strike, DMG_SPINNER)) * e_energy
                + e_blow * DMG_HAMMER_BLOW + clash * DMG_CLASH * e_energy)

        # Bodies are solid whatever the weapons are doing
        touching, depth, nx, ny = body_contact(dx, dy, cos_o, sin_o, cos_e, sin_e,
                                               self.o_len, self.o_wid, self.e_len, self.e_wid)
        touching = touching & live
        self.touching = touching
        self._shove(touching, depth, nx, ny)

        # What the hits do to the robots. Off-centre hits also spin the
        # target, and the robot that hit gets a share of the throw back.
        o_size, e_size = 0.5 * (self.o_len + self.o_wid), 0.5 * (self.e_len + self.e_wid)

        def spin(px, py, cx, cy, ux, uy, size):
            """Yaw from a push along the unit vector (ux, uy) at (px, py) on a body centred at (cx, cy)."""
            return (((px - cx) * uy - (py - cy) * ux) / size).clamp(-1.0, 1.0) * KNOCK_YAW

        # Our vert throws them away from the tail that hit them. Our hammer
        # comes straight down and throws nobody, but like any other hit it
        # costs them their drive for a moment. The throw goes with the
        # blade's tip speed; losing the drive goes with the damage.
        v = o_vert * o_blade * so.power.sqrt()
        self._throw(False, -cos_o * v * KNOCK_VERT_FT_S, -sin_o * v * KNOCK_VERT_FT_S,
                    v * spin(o_rx, o_ry, self.ex, self.ey, -cos_o, -sin_o, e_size), o_vert * o_energy * STUN_S,
                    air=o_vert * o_energy * AIR_S)
        self._throw(True, cos_o * v * KNOCK_VERT_FT_S * so.recoil, sin_o * v * KNOCK_VERT_FT_S * so.recoil)
        self.e_stun = torch.maximum(self.e_stun, o_hammer * so.strike * o_energy * BLOW_STUN_S)

        # Their spinner on us. A vert or a drum throws us the way it points.
        # A horizontal throws us mostly along its blade's travel where it
        # touched: sideways, and round.
        f = e_front * ~e_saw * e_blade * se.power
        hub_x, hub_y = self.ex + cos_e * se.fore, self.ey + sin_e * se.fore
        ux, uy = self.ox - hub_x, self.oy - hub_y
        away = torch.sqrt(ux * ux + uy * uy + 1e-6)
        ux, uy = ux / away, uy / away                                           # hub -> us
        tx = torch.where(e_horiz, -uy * se.turn * HORIZ_SIDEWAYS + ux * (1.0 - HORIZ_SIDEWAYS), cos_e)
        ty = torch.where(e_horiz,  ux * se.turn * HORIZ_SIDEWAYS + uy * (1.0 - HORIZ_SIDEWAYS), sin_e)
        r = e_vert * e_blade * se.power.sqrt()                                  # a self-play copy's vert on us
        wx = tx * f * KNOCK_SPIN_FT_S - cos_e * r * KNOCK_VERT_FT_S
        wy = ty * f * KNOCK_SPIN_FT_S - sin_e * r * KNOCK_VERT_FT_S
        # The hit's strength sets how long we are in the air and without drive. The spinner
        # that landed it goes the other way with its recoil share of all three.
        strength = (e_front.float() + e_vert) * e_energy
        self._throw(True, wx, wy,
                    f * spin(hit_x, hit_y, self.ox, self.oy, tx, ty, o_size)
                    + r * spin(e_rx, e_ry, self.ox, self.oy, -cos_e, -sin_e, o_size),
                    strength * STUN_S, air=strength * AIR_S)
        self._throw(False, -wx * se.recoil, -wy * se.recoil,
                    stun=strength * se.recoil * STUN_S, air=strength * se.recoil * AIR_S)

        c = clash.float()                                      # weapon on weapon: both go backwards
        ax, ay = dx / dist, dy / dist
        self._throw(True,  -ax * c * KNOCK_CLASH_FT_S, -ay * c * KNOCK_CLASH_FT_S, stun=c * BLOW_STUN_S * 0.5)
        self._throw(False,  ax * c * KNOCK_CLASH_FT_S,  ay * c * KNOCK_CLASH_FT_S, stun=c * BLOW_STUN_S * 0.5)

        # A hit also shakes our heading estimate: the gyro it is built on
        # saturates in the impact. It settles again as the estimate does
        # everywhere else (the observer's wandering offset).
        shock = f + r + 0.5 * c
        self.observer.bias[:, 4] += torch.randn_like(shock) * self.hdg_kick * shock

        # ... and to the blades
        o_bit, e_bit = o_vert | o_hammer, e_vert | e_front
        so.blade = torch.where(clash | stalled, o_blade * CLASH_KEEP, torch.where(
            o_bit, o_blade * so.keep, o_blade * (1.0 - (1.0 - BLADE_FLOOR_KEEP) * o_miss)))
        se.blade = torch.where(clash, e_blade * CLASH_KEEP, torch.where(
            e_bit, e_blade * se.keep, e_blade * (1.0 - (1.0 - BLADE_FLOOR_KEEP) * e_miss)))
        so.rebite = torch.where(o_bit | clash | stalled, REBITE_S, so.rebite)
        se.rebite = torch.where(e_bit | clash, REBITE_S, se.rebite)
        # A blade that is on a robot gets no faster: what its motor has goes
        # into that robot a little at a time, and none of it is a hit. The
        # next hit takes getting clear first.
        so.bound = live & (o_r_on | o_f_on)
        se.bound = live & ((e_r_in & se.rear_low) | ((e_f_in | caught) & se.front_low))

        # Hazards: any part of our body over one costs damage for as long as it stays
        # (saw-v9: the bulges are wall now, nothing stands over them, so this no longer happens)
        on_hazard = torch.zeros_like(live)
        burnt = on_hazard * (HAZARD_DMG_PER_S * DT) + (on_hazard & ~self.o_on_hazard) * HAZARD_DMG_ENTER
        self.o_on_hazard = on_hazard

        # Walls: running into one costs once, by how hard. Staying on one is
        # charged per control step (_dense_rewards).
        (o_gap, o_hit), (e_gap, _) = self._wall(True), self._wall(False)
        our_impact = torch.maximum(our_impact, o_hit)
        o_wall = live & (o_gap <= 1e-3)
        self._add_reward(RW_WALL, -R_SAW_WALL_HIT * (o_wall & ~self.o_on_wall) * our_impact / V_MAX_FT)
        self.wall_touch |= o_wall
        self.o_on_wall = o_wall
        self.o_squeezed = touching & (o_gap < SQUEEZE_GAP_FT)
        self.e_squeezed = touching & (e_gap < SQUEEZE_GAP_FT)

        # Aggression: out of reach of an armed opponent and not driving at
        # it. Charged by the second, here and on the scorecard (_clock).
        closing = ((cos_o * self.ovx - sin_o * self.o_drift) * dx + (sin_o * self.ovx + cos_o * self.o_drift) * dy) / dist
        passive = live & self.opponents.armed & (dist > PASSIVE_FT) & (closing < PASSIVE_CLOSE_FT_S)
        self._add_reward(RW_TIME, -R_PASSIVE_PER_S * DT * passive)

        # Hit, back off, come again. For a while after a swing lands, on
        # anything, distance gained is paid and staying on them is charged.
        # And all fight long, standing about is charged: a robot that keeps
        # moving is hard to hit and looks like the aggressor.
        backing = live & (self.retreat > 0)
        gained = (dist.clamp(max=RETREAT_FT) - self.last_dist.clamp(max=RETREAT_FT)) * backing
        self._add_reward(RW_TIME, self.r_retreat * gained - self.r_ram * DT * (backing & touching))
        self.retreat = torch.where(so.blow > 0, RETREAT_S, (self.retreat - DT * live).clamp(min=0.0))
        self.last_dist = dist
        speed = torch.sqrt(self.ovx ** 2 + self.o_drift ** 2)
        self._add_reward(RW_TIME, -self.r_slow * DT * (1.0 - speed / V_MAX_FT).clamp(0.0, 1.0) * (live & (self.o_stun <= 0)))

        # Score. On the scorecard a unit taken is a unit taken. In the reward
        # what their weapon does to us counts R_TAKEN_SHARE of that: the team
        # wants attacks landed more than fights won. A hazard is nobody's
        # attack and costs in full. A swing that lands on the floor or on
        # their weapon side is one of few, gone for nothing: a small charge.
        self._add_reward(RW_OUTCOME, R_DAMAGE * (dealt - R_TAKEN_SHARE * hurt - burnt) - self.r_foul * (o_floor + o_side))
        self.tally[:, :T_REQ] += torch.stack([
            o_vert.float(), o_hammer.float(), o_blow, o_floor, (clash | stalled).float(), e_bit.float() + e_blow,
            dealt, hurt + burnt, self.o_squeezed * DT, on_hazard * DT, (live & (self.o_stun > 0)) * DT, passive * DT,
            solid.float(), o_side,
        ], dim=1)

    def _shove(self, touching, depth, nx, ny):
        """Bodies in contact: equal masses exchange momentum along the
        contact normal and are moved apart along it, half the overlap each. A
        shove from the side shows up as lateral slide, which the tyres then
        kill. A robot against a wall cannot give way, so the other one is
        moved again on the following steps until they are clear."""
        cos_o, sin_o = torch.cos(self.oa), torch.sin(self.oa)
        cos_e, sin_e = torch.cos(self.ea), torch.sin(self.ea)
        # Closing speed along the normal, from full world velocities
        o_n = (cos_o * self.ovx - sin_o * self.o_drift) * nx + (sin_o * self.ovx + cos_o * self.o_drift) * ny
        e_n = (cos_e * self.evx - sin_e * self.e_drift) * nx + (sin_e * self.evx + cos_e * self.e_drift) * ny
        impulse = torch.where(touching, (0.5 * (1.0 + BODY_RESTITUTION) * (o_n - e_n)).clamp(min=0.0), 0.0)
        # -impulse*n on us, +impulse*n on them, resolved onto each body's axes
        self.ovx     = self.ovx     - impulse * (cos_o * nx + sin_o * ny)
        self.o_drift = self.o_drift - impulse * (-sin_o * nx + cos_o * ny)
        self.evx     = self.evx     + impulse * (cos_e * nx + sin_e * ny)
        self.e_drift = self.e_drift + impulse * (-sin_e * nx + cos_e * ny)

        push = torch.where(touching, depth * 0.5, 0.0)
        # each centre stays as far from the walls as its body's short half size (_wall keeps the corners out)
        o_lo, e_lo = torch.minimum(self.o_len, self.o_wid), torch.minimum(self.e_len, self.e_wid)
        keep = lambda v, lo: torch.maximum(torch.minimum(v, ARENA_FT - lo), lo)
        self.ox = keep(self.ox - nx * push, o_lo)
        self.oy = keep(self.oy - ny * push, o_lo)
        self.ex = keep(self.ex + nx * push, e_lo)
        self.ey = keep(self.ey + ny * push, e_lo)

    def _throw(self, ours, wx, wy, spin=0.0, stun=None, air=None):
        """Add a world-frame velocity (wx, wy), a yaw rate, a loss of drive
        (``stun``, s) and time off the floor (``air``, s) to our robot or
        theirs."""
        if ours:
            c, s = torch.cos(self.oa), torch.sin(self.oa)
            self.ovx, self.o_drift = self.ovx + wx * c + wy * s, self.o_drift - wx * s + wy * c
            self.o_omega = self.o_omega + spin
            if stun is not None:
                self.o_stun = torch.maximum(self.o_stun, stun)
            if air is not None:
                self.o_air = torch.maximum(self.o_air, air)
        else:
            c, s = torch.cos(self.ea), torch.sin(self.ea)
            self.evx, self.e_drift = self.evx + wx * c + wy * s, self.e_drift - wx * s + wy * c
            self.e_omega = self.e_omega + spin
            if stun is not None:
                self.e_stun = torch.maximum(self.e_stun, stun)
            if air is not None:
                self.e_air = torch.maximum(self.e_air, air)

    def _clock(self):
        """The clock is the only thing that ends a fight. It is then scored
        on damage, less what the time we spent keeping away cost us with the
        judges. The bell is the end: nothing after it is worth anything
        (timeout_bootstrap is False), so a swing saved past it is wasted."""
        self.timeout = ~self.done & (self.steps >= self.horizon)
        self.done |= self.timeout
        margin = self.tally[:, T_DEALT] - self.tally[:, T_HURT] - PASSIVE_PTS_PER_S * self.tally[:, T_PASSIVE_S]
        result = torch.where(margin >= PTS_MARGIN, END_PTS_WIN,
                             torch.where(margin <= -PTS_MARGIN, END_PTS_LOSS, END_PTS_DRAW))
        self.end_reason = torch.where(self.timeout, result, self.end_reason)
        self.won |= self.timeout & (margin >= PTS_MARGIN)

    # ── Observation ───────────────────────────────────────────────────────

    def _weapon_obs(self, weapon, prev_action, half_len, half_wid, no_arm=None, no_blade=None, idx=None):
        """The six columns the saw rules add to the observation, for one
        robot. ``no_arm`` and ``no_blade`` mark the robots that do not report
        that reading; it is -1 for them."""
        sel = self._ar if idx is None else idx
        noise = lambda sigma: torch.randn(sel.numel(), device=DEVICE) * sigma
        arm = (weapon.arm[sel] + noise(ARM_NOISE)).clamp(0.0, 1.0)
        blade = (weapon.blade[sel] + noise(BLADE_NOISE)).clamp(0.0, 1.0)
        if no_arm is not None:
            arm = torch.where(no_arm[sel], -1.0, arm)
        if no_blade is not None:
            blade = torch.where(no_blade[sel], -1.0, blade)
        return torch.stack([prev_action[sel, 2], prev_action[sel, 3], arm, blade,
                            half_len[sel] - 2.0, half_wid[sel] - 2.0], dim=1)

    def _truth(self, mirror=False):
        """As BattleEnv's, with the other robot's heading as an eighth
        column: the camera frame a position fix comes from also carries it."""
        return torch.cat([super()._truth(mirror), (self.oa if mirror else self.ea).unsqueeze(1)], dim=1)

    def _heading_cols(self, idx=None):
        """Columns 38..43: which way they point, as the camera gives it. The
        reading belongs to the camera frame their position fix came from: it
        is as old, it is held as long, and its noise and whether it is
        missing are redrawn only when that frame is."""
        if idx is None:
            noise = torch.randn(self.n, device=DEVICE) * self.hdg_sigma
            drop = torch.rand(self.n, device=DEVICE) < HDG_SENSE_DROP_P
            fresh = self.observer.e_fresh
            self.hdg_err = torch.where(fresh, noise, self.hdg_err)
            self.hdg_drop = torch.where(fresh, drop, self.hdg_drop)
            fix, hdg, err = self.observer.fix, self.observer.other_hdg, self.hdg_err
            mode, drop = self.hdg_mode, self.hdg_drop
        else:
            fix, hdg, err = self.observer.fix[idx], self.observer.other_hdg[idx], self.hdg_err[idx]
            mode, drop = self.hdg_mode[idx], self.hdg_drop[idx]
        d = angle_wrap(torch.atan2(fix[:, 1] - fix[:, 3], fix[:, 0] - fix[:, 2]) - (hdg + err))
        axis, nose = ((mode >= 1) & ~drop).float(), ((mode == 2) & ~drop).float()
        return torch.stack([axis, torch.cos(2 * d) * axis, torch.sin(2 * d) * axis,
                            nose, torch.cos(d) * nose, torch.sin(d) * nose], dim=1)

    def _card(self, idx=None):
        """What the team wrote down about the opponent before the fight."""
        F = self.field
        card = dict(kind=F.card_kind, len=F.card_len, wid=F.card_wid, fore=F.card_fore, rad=F.card_rad, wide=F.card_wide)
        return card if idx is None else {key: v[idx] for key, v in card.items()}

    def _observe(self, idx=None):
        base = torch.cat([self.observer.observe(self.prev_action, idx=idx),
                          self._weapon_obs(self.saw_o, self.prev_action, self.o_len, self.o_wid,
                                           no_arm=self.no_arm_reading, no_blade=self.no_blade_reading, idx=idx)], dim=1)
        if idx is None:
            self.gate.advance(CTRL_DT, base, self._blade_sent, self._thr_sent, self.arm_gate)
        hdg, card = self._heading_cols(idx), self._card(idx)
        ready = self.gate.ready(base[:, 27], idx)
        more = {} if self.gate_shrink is None else {"shrink": float(self.gate_shrink)}
        open_ = self.gate.open(base, card, hdg, idx, rule=self.gate_rule, **more)
        if idx is None:
            self.gate_run = torch.where(open_, self.gate_run + 1, 0)
            open_ = self.gate_run >= self.gate_persist
        else:
            self.gate_run[idx] = 0
            open_ = open_ & (self.gate_persist <= 1)
        if idx is None:
            self.gate_open, self.gate_ready = open_, ready
            left = self.horizon - self.steps
        else:
            self.gate_open[idx], self.gate_ready[idx] = open_, ready
            left = self.horizon[idx] - self.steps[idx]
        kind = card["kind"]
        extra = torch.stack([(self.gate.swings_left(idx).float() / 10.0).clamp(0.0, 3.0),
                             left.clamp(min=0).float() * DT / 180.0, open_.float(),
                             (kind == W_VERT).float(), (kind == W_DRUM).float(), (kind == W_HORIZ).float(),
                             card["len"] - 2.0, card["wid"] - 2.0], dim=1)
        return torch.cat([base, extra, hdg], dim=1)

    def privileged(self):
        """(n, PRIV_DIM) of the true state as it is now, for a critic that is
        used in training only: what the policy cannot see and the value of
        a state depends on."""
        so, se, F = self.saw_o, self.saw_e, self.field
        f = lambda t: t.float()
        to_us = angle_wrap(torch.atan2(self.oy - self.ey, self.ox - self.ex) - self.ea)
        left = (self.horizon - self.steps).clamp(min=0).float() * DT / 180.0
        cos_o, sin_o = torch.cos(self.oa), torch.sin(self.oa)
        legal, _, _ = landing_class(self.ox + cos_o * so.fore, self.oy + sin_o * so.fore, so.rad,
                                    self.ex, self.ey, self.ea, self.e_len, self.e_wid, se.kind, se.fore, se.wide, se.rad,
                                    guard=se.spinner & ((se.kind != W_HORIZ) | se.spinning),
                                    front_margin=self.front_margin)
        types = torch.nn.functional.one_hot(self.opponents.types.clamp(0, N_TYPES - 1), N_TYPES).float()
        kinds = torch.nn.functional.one_hot(se.kind.clamp(max=W_HORIZ), W_HORIZ + 1).float()
        cols = [torch.cos(to_us), torch.sin(to_us), self.e_omega / YAW_RATE_MAX, se.blade, f(se.dead),
                se.fore / 3.0, se.rad / 2.0, se.wide / 2.0, self.e_len - 2.0, self.e_wid - 2.0, self.front_margin,
                so.arm, so.blade, so.swings.clamp(0.0, 30.0) / 10.0, F.p]
        tail = [f(F.profile == PROFILE_TRAVES), self.enemy.v_max / V_MAX_FT]
        last = [left, self.horizon.float() * DT / 180.0, (f(self.gate.swings_left()) / 10.0).clamp(max=3.0),
                (f(self.gate.budget) / 10.0).clamp(max=3.0), f(self.gate_open), f(self.from_bell),
                f(self.hdg_mode) / 2.0, f(legal)]
        out = torch.cat([torch.stack(cols, 1), types, torch.stack(tail, 1), kinds, torch.stack(last, 1)], dim=1)
        assert out.shape[1] == PRIV_DIM
        return out

    def _info(self, obs):
        info = super()._info(obs)
        info.tally = self.tally.clone()
        info.fight_s = self.horizon.float() * DT
        info.hdg_mode = self.hdg_mode.clone()
        return info

    def fight_stats(self):
        return {name: self.tally[:, k] for k, name in enumerate(TALLY_NAMES)}

    # ── Rewards ───────────────────────────────────────────────────────────

    def _potential(self):
        """Shaping potential, from the true state. Combat: near the enemy it
        is bad to stand in front of its weapon, and good to be anywhere else
        around it with our weapon pointed at it: the nose of a hammer saw,
        the tail of a robot set up as a vert. Drills: close in with that end
        on target."""
        dx, dy = self.ex - self.ox, self.ey - self.oy
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        rx, ry = dx / dist, dy / dist                                      # us -> enemy
        nose_on = torch.cos(self.oa) * rx + torch.sin(self.oa) * ry         # 1: our nose points at it, -1: our tail
        aimed = torch.where(self.saw_o.vert, (-nose_on).clamp(min=0.0), nose_on.clamp(min=0.0))
        facing_us = -(torch.cos(self.ea) * rx + torch.sin(self.ea) * ry)    # 1: its nose points at us
        threat = facing_us.clamp(min=0.0) ** 2
        near = ((FLANK_FAR - dist) / (FLANK_FAR - FLANK_NEAR)).clamp(0.0, 1.0)
        combat = K_FLANK * near * ((1.0 - threat) * aimed - threat)
        drill  = K_APPROACH * (1.0 - dist / _ARENA_DIAG) * (0.5 + 0.5 * aimed)
        return torch.where(self.opponents.armed, combat, drill)

    def _dense_rewards(self, action, was_alive):
        """Per-control-step rewards. The bell is the only end and it is
        terminal, so the potential there is 0. Time itself is free (every
        fight lasts its full clock); time spent keeping away from an armed
        opponent is charged in _rules, with the rest of the score. With the
        gate on, the smoothness charge on the arm is on what the gate sent:
        the arm the robot moves, not the request."""
        alive = was_alive.float()
        self._add_reward(RW_WALL, -R_SAW_WALL_PER_S * CTRL_DT * self.wall_touch.float())
        now, before = action, self.prev_action
        if self.arm_gate:
            now, before = action.clone(), self.prev_action.clone()
            now[:, 2], before[:, 2] = self._arm_out, self._prev_arm_out
        self._add_reward(RW_SMOOTH, -R_ACTION_RATE * ((now - before) ** 2).sum(1) * alive)
        self._prev_arm_out = torch.where(was_alive, self._arm_out, self._prev_arm_out)
        phi = self._potential()
        self._add_reward(RW_SHAPE, (torch.where(self.done, 0.0, self.gamma * phi) - self.phi) * alive)
        self.phi = phi
