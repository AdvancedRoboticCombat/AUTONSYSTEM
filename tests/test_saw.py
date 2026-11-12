"""
The saw rules: the hammer saw itself, what hits do, the opponents' weapons,
body sizes, the clock, hazards, walls, starts, what goes wrong with our own
robot, and what the policy sees.

Run with pytest if you have it, or directly:
    uv run python tests/test_saw.py
"""
import math

import _runner  # noqa: F401  (puts the repo root on sys.path)
import torch

from config import (
    HAMMER_REACH,
    DEVICE, DT, CTRL_DT, GAMMA, GAMMA_SAW, ARENA_FT, WALL_CLAMP_IN, MAX_STEPS, PHYS_PER_CTRL, YAW_RATE_MAX, ALPHA_MAX_RPS2,
    OBS_DIM, ACT_DIM, SAW_OBS_DIM, SAW_ACT_DIM, SAW_DR, DR_GRIP, ARM_DOWN, ARM_NOISE, BLADE_NOISE,
    OUR_WEAPON_TIP_RANGE, EN_BLADE_FT, KNOCK_RECOIL, EN_WEAPON_SHARE, ARM_JAM_P, ARM_JAM_P_DRILLS, VERT_SETUP_P,
    NO_ARM_READING_P, NO_BLADE_READING_P, DR_TREAD_SHARE, V_MAX_FT, ENEMY_VEL_WINDOW,
    DR_OUR_HALF_LEN, DR_OUR_HALF_WID, DR_EN_HALF_LEN, DR_EN_HALF_WID, FIGHT_LEN_S, R_PASSIVE_PER_S, PASSIVE_FT,
    PASSIVE_CLOSE_FT_S, PASSIVE_PTS_PER_S, PTS_MARGIN, ARM_SWINGS_3MIN, DR_ARM_FIRE_S, DR_BLADE_TAU_S, DR_BLADE_GYRO,
    DMG_VERT, DMG_HAMMER_SAW, DMG_HAMMER_BLOW, DMG_CLASH, R_DAMAGE, R_TAKEN_SHARE, ATTACK_IN_DMG, STRIKE_S,
    DR_FRONT_MARGIN_FT, EN_FRONT_DISCS, DR_VERT_WIDE, DR_DRUM_WIDE, DR_EN_POWER, DR_EN_TAU_S, DR_BITE_FLOOR,
    DR_BLADE_KEEP, STUN_S, angle_wrap, DR_BLADE_TOP, BLADE_ON_BIAS, DR_EN_GYRO,
    HAZARD_DMG_PER_S, HAZARD_DMG_ENTER, SPAWN_PINNED, SPAWN_CORNERED, SPAWN_CONTACT, SPAWN_PINNING,
)
from environment import (
    BattleEnv, RobotParams, RANGES, integrate_robot, make_env, RW_OUTCOME, RW_TIME, RW_SHAPE, RW_WALL,
    END_PTS_WIN, END_PTS_LOSS, END_PTS_DRAW,
)
from opponent_library import (
    COMBAT_TYPES, TYPE_NAMES, DRIFTER, PURSUER, SELFPLAY, SHARP, SHARP_MIX_EARLY, MIX_FULL_AT, AGILITY_RATIO, REACT_S,
    RESPIN_SHARE, RESPIN_AT, SHARP_TOP, GUARD_SHARE,
)
from environment import RW_SMOOTH
from config import (
    R_FOUL, HDG_SENSE_P, HDG_SENSE_DROP_P, BELL_SHORT_P, BELL_FACE_RAD, DR_HAMMER_REACH, ARM_SWITCH, angle_wrap,
)
from saw_field import (
    Scenario, PROFILE_FIELD, PROFILE_TRAVES, EN_HALF_WID, TRAVES_HALF_LEN, TRAVES_HALF_WID, TRAVES_TAU_S, TRAVES_POWER,
    HORIZ_HUB, HORIZ_BAR_FT, HORIZ_TAU_S, TRAVES_HUB, TRAVES_BAR_FT,
)
from config import DMG_SPINNER, RETREAT_S, RETREAT_FT, R_RETREAT_PER_FT, R_SLOW_PER_S, V_MAX_FT
import saw_walls
from saw_env import T_SOLID, T_FOUL, T_REQ, T_FIRE, T_BUDGET, TALLY_N
from saw_env import (
    SawEnv, Weapon, body_gap, body_contact, W_NONE, W_VERT, W_DRUM, W_HORIZ, W_SAW, SPINNERS, TALLY_NAMES,
    T_VERT, T_HAMMER, T_BLOW, T_MISS, T_CLASH, T_TAKEN, T_DEALT, T_HURT, T_PINNED_S, T_HAZARD_S, T_PASSIVE_S,
    PINNED, CORNERED, CONTACT, PINNING, _NEVER, NO_SWING_LIMIT, T_BLOW, attack_in,
)

_full = lambda n, v: torch.full((n,), float(v), device=DEVICE)


def _nominal(par, v_max=20.0):
    par.v_max[:], par.yaw_cap[:] = v_max, YAW_RATE_MAX
    par.dalpha[:] = ALPHA_MAX_RPS2 * DT
    par.ctrl_alpha[:] = 0.05 / (0.05 + DT)
    par.bleed[:] = 0.35
    par.dv_max[:] = 77.0 * DT
    par.grip_s[:], par.grip_k[:] = 64.0, 19.0
    par.a_lat_cap[:] = 98.4
    par.yaw_gain[:], par.pull[:], par.deadband[:] = 1.0, 0.0, 0.0


def _env(n=4, stage="pretrain", tactic=None):
    """A saw env on clean, nominal hardware: both robots 4 ft square, no
    command delay or loss, an arm that works with a parked blade that is live
    too (the vert setup's blade on a hammer saw's arm, which no fight is
    dealt, so that one env can test either), no limit on its swings, no wear,
    treads without a limit, a camera that loses nobody, every reading there,
    a spinner that feels half of our blade when the weapons meet, and an
    enemy that holds still unless told otherwise. ``tactic`` pins the
    opponent type, which decides whether it is armed: in combat it then
    carries a vert whose blade reaches 0.45 ft past its nose, in the drills
    nothing."""
    env = SawEnv(n, stage=stage, p=1.0, auto_reset=False, arm_gate=False)
    if tactic is not None:
        env.opponents.forced_types = torch.full((n,), tactic, dtype=torch.long, device=DEVICE)
        env.reset()
    # The saw-v7 sensors and field: a new camera frame every step, a scripted driver with no delay,
    # a weapon that never fails, the nominal reach
    env.observer.camera = None
    env.observer.period[:], env.observer.phase[:] = 1, 0
    env.en_lag[:] = 0
    env.en_ring[:] = 0.0
    env.saw_e.dead[:] = False
    env.field.dead_at[:] = _NEVER
    env.reach[:] = HAMMER_REACH
    env.manual_enemy = True
    for par in (env.our, env.enemy):
        _nominal(par)
    env.cmd_lag[:] = 0
    env.wear_at[:] = _NEVER
    env.horizon[:] = MAX_STEPS
    for t in (env.o_len, env.o_wid, env.e_len, env.e_wid):
        t[:] = 2.0
    env.rev[:], env.e_rev[:] = 1.0, 1.0
    env.cmd_drop[:], env.track_loss_hz[:], env.hdg_kick[:] = 0.0, 0.0, 0.0
    env.no_arm_reading[:], env.no_blade_reading[:] = False, False
    env.tread_k[:], env.e_tread_k[:] = 0.0, 0.0
    env.front_margin[:] = 0.5
    env.armour_o[:], env.armour_e[:] = 0.4, 0.4
    env.saw_o.power[:] = 1.0                                  # our blade at its full 7,696 rpm
    env.saw_e.power[env.saw_e.kind == W_SAW] = 1.0
    for w in (env.saw_o, env.saw_e):
        w.jammed[:] = False
        w.vert[:] = True
        w.swings[:] = NO_SWING_LIMIT
        w.t_fire[:], w.t_retract[:], w.tau[:] = 0.2, 0.5, 1.0
        w.keep[:], w.sag[:], w.gyro[:] = 0.5, 0.0, 0.0
        w.aft[:], w.fore[:], w.wide[:], w.rad[:] = 2.5, 2.5, 0.0, OUR_WEAPON_TIP_RANGE
        w.recoil[:] = KNOCK_RECOIL
    se = env.saw_e
    armed = se.spinner
    se.kind[armed] = W_VERT
    se.fore[armed], se.wide[armed], se.rad[armed] = 1.9, 0.2, EN_BLADE_FT
    return env


def _place(env, ox, oy, oa, ex, ey, ea):
    """Both robots at known poses, at rest, saws parked and ready, with a
    clean slate for the fight."""
    env.ox[:], env.oy[:], env.oa[:] = ox, oy, oa
    env.ex[:], env.ey[:], env.ea[:] = ex, ey, ea
    for t in (env.ovx, env.evx, env.o_omega, env.e_omega, env.o_drift, env.e_drift, env.o_stun, env.e_stun):
        t[:] = 0.0
    for t in (env.o_squeezed, env.e_squeezed, env.o_on_wall, env.o_on_hazard, env.done, env.touching):
        t[:] = False
    env.steps[:] = 0
    env.tally[:] = 0.0
    env.rw[:] = 0.0
    for w in (env.saw_o, env.saw_e):
        w.arm[:], w.lowest[:], w.rebite[:], w.blow[:] = 0.0, 0.0, 0.0, 0.0
        w.fire[:], w.bound[:] = False, False
    env.phi = env._potential()


def _act(env, throttle=0.0, turn=0.0, arm=-1.0, blade=1.0):
    a = torch.zeros(env.n, SAW_ACT_DIM, device=DEVICE)
    a[:, 0], a[:, 1], a[:, 2], a[:, 3] = throttle, turn, arm, blade
    return a


def _saw(n=1, **hardware):
    """A hammer saw on its own, on a 4 ft robot, with known swing times and no faults."""
    w = Weapon(n)
    w.reset(torch.arange(n, device=DEVICE), torch.full((n,), W_SAW, dtype=torch.long, device=DEVICE),
            _full(n, 2.0), _full(n, 2.0))
    w.jammed[:] = False
    w.vert[:] = True                                          # as in _env: the arm works and the parked blade is live
    w.t_fire[:], w.t_retract[:], w.tau[:], w.sag[:] = 0.2, 0.5, 1.0, 0.0
    for name, value in hardware.items():
        getattr(w, name)[:] = value
    return w


def _run(w, arm, blade=1.0, seconds=0.1, load=0.0):
    """Hold one command on a weapon for a while. Returns the blows it landed."""
    n = w.arm.numel()
    blows = torch.zeros(n, device=DEVICE)
    load = load if torch.is_tensor(load) else _full(n, load)
    for _ in range(round(seconds / DT)):
        w.step(_full(n, arm), _full(n, blade), load, _full(n, 1.0))
        blows += w.blow
    return blows


# ─── The weapon ───────────────────────────────────────────────────────────

def test_arm_is_a_switch_that_holds_its_last_order():
    w = _saw()
    assert w.rear_low.all() and not w.front_low.any()        # parked: the saw is a vert behind the robot
    _run(w, arm=0.2, seconds=0.5)                             # inside the dead zone: nothing happens
    assert w.arm.item() == 0.0
    blows = _run(w, arm=1.0, seconds=0.25)                    # fire: over the top in the swing time
    assert w.front_low.all() and not w.rear_low.any()
    assert abs(blows.item() - 1.0) < 1e-3, blows
    _run(w, arm=0.0, seconds=0.5)                             # no order: it stays down
    assert w.front_low.all()
    _run(w, arm=-1.0, seconds=0.3)                            # park: on its way back it is neither
    assert not w.front_low.any() and not w.rear_low.any()
    _run(w, arm=-1.0, seconds=0.3)
    assert w.rear_low.all()


def test_a_dead_arm_stays_parked():
    w = _saw(jammed=True)
    blows = _run(w, arm=1.0, seconds=0.5)
    assert w.arm.item() == 0.0 and w.rear_low.all() and blows.item() == 0.0


def test_the_arm_has_only_so_many_swings_in_it():
    """Eight to thirty in three minutes, most often about ten. Once they are spent the arm
    will not start another, though one that is down can be held there; a
    short wag takes little out of what is left."""
    w = _saw(swings=2.0)
    for _ in range(2):
        assert abs(_run(w, arm=1.0, seconds=0.3).item() - 1.0) < 1e-3
        _run(w, arm=0.0, seconds=0.5)
        assert w.front_low.all(), "a landed saw can be held down"
        _run(w, arm=-1.0, seconds=0.6)
    assert w.swings.item() < 1e-3
    assert _run(w, arm=1.0, seconds=0.5).item() == 0.0 and w.arm.item() == 0.0 and w.rear_low.all()

    wag = _saw(swings=2.0)
    _run(wag, arm=1.0, seconds=0.3)
    for _ in range(5):
        _run(wag, arm=-1.0, seconds=0.06)
        _run(wag, arm=1.0, seconds=0.06)
    assert wag.swings.item() > 0.8, wag.swings

    torch.manual_seed(0)
    short = SawEnv(4096, stage="combat", p=1.0, auto_reset=False)            # evaluation: every fight is 20 s
    lo, hi = ARM_SWINGS_3MIN[0][1][0], ARM_SWINGS_3MIN[-1][1][1]
    most = math.floor(hi * 20.0 / 180.0) + 1
    assert short.saw_o.swings.min().item() == 1.0 and 2.0 < short.saw_o.swings.max().item() <= most
    drills = SawEnv(256, stage="pretrain", auto_reset=False)
    assert (drills.saw_o.swings == NO_SWING_LIMIT).all(), "the drills count swings"
    train = SawEnv(8192, stage="combat", p=1.0, auto_reset=True)
    train.reset_at(train._ar)                                 # whole fights: reset() joins each part-way through
    whole = train.horizon == round(180.0 / DT)
    assert whole.any() and lo <= train.saw_o.swings[whole].min().item() and train.saw_o.swings[whole].max().item() <= hi
    mean = sum(share * 0.5 * (a + b) for share, (a, b) in ARM_SWINGS_3MIN)
    assert abs(train.saw_o.swings[whole].mean().item() - mean) < 0.8
    assert (train.saw_o.swings >= 1).all()

    part = _saw(swings=0.6)                                   # less than a whole swing left: the arm does not start one
    assert _run(part, arm=1.0, seconds=0.5).item() == 0.0 and part.arm.item() == 0.0


def test_the_weapon_is_dealt_what_was_measured_on_the_robot():
    """The arm swings over in 0.35 s. The blade (15.5 in, 17 lb, 7,696 rpm)
    is up to speed in 1.68 s and takes a third to two thirds of the turn
    rate away at full speed. Those are for the full 7,696 rpm: a blade held
    to the tip-speed rule (DR_BLADE_TOP) gets there sooner and rolls us less."""
    torch.manual_seed(0)
    so = SawEnv(4096, stage="combat", p=1.0, auto_reset=False).saw_o
    top = so.power.sqrt()
    sooner = torch.log(1.0 / (1.0 - 0.9 * top)) / math.log(10.0)
    assert DR_ARM_FIRE_S[0] < 0.35 < DR_ARM_FIRE_S[1] and 0.28 <= so.t_fire.min().item() <= so.t_fire.max().item() <= 0.45
    up_to_speed = 1.68 / (so.tau / sooner)                    # time constants in the measured spin-up
    assert 1.7 < up_to_speed.min().item() and up_to_speed.max().item() < 3.8
    inertia = 0.5 * (17 * 0.4536) * (0.5 * 15.5 * 0.0254) ** 2            # a solid disc, kg m^2
    momentum = inertia * 7696 * 2 * math.pi / 60                           # N m s
    assert 115 < momentum < 125 and 45e3 < 0.5 * momentum * 7696 * 2 * math.pi / 60 < 52e3
    lifts_at = 113.4 * 9.81 * torch.tensor([0.28, 0.54]) / momentum        # rad/s at which a tread comes off the floor
    lost = 1.0 - lifts_at / YAW_RATE_MAX
    assert abs(DR_BLADE_GYRO[0] - lost.min().item()) < 0.05 and abs(DR_BLADE_GYRO[1] - lost.max().item()) < 0.05
    at_full = so.gyro / top
    assert DR_BLADE_GYRO[0] - 1e-4 <= at_full.min().item() and at_full.max().item() <= DR_BLADE_GYRO[1] + 1e-4
    assert so.gyro.min().item() < DR_BLADE_GYRO[0], "a held blade rolls us less"


def test_a_short_swing_lands_soft():
    """The blow is as hard as the swing was long, so wagging the arm at the
    bottom of its travel cannot be farmed."""
    w = _saw()
    assert abs(_run(w, arm=1.0, seconds=0.3).item() - 1.0) < 1e-3
    _run(w, arm=-1.0, seconds=0.25)                           # lift it halfway back ...
    half = _run(w, arm=1.0, seconds=0.3).item()               # ... and bring it down again
    assert 0.1 < half < 0.5, half
    wag = 0.0
    for _ in range(20):
        _run(w, arm=-1.0, seconds=0.05)
        wag += _run(w, arm=1.0, seconds=0.05).item()
    assert wag < 0.05, wag


def test_hard_driving_slows_the_spin_up():
    """The drive and the saw share a battery: at full throttle the blade gets
    only part of its torque. Spinning down is not affected."""
    w = _saw(n=2, sag=0.6, blade=0.0)
    load = torch.tensor([0.0, 1.0], device=DEVICE)
    _run(w, arm=-1.0, blade=1.0, seconds=1.0, load=load)      # one second asking for full speed
    free, loaded = w.blade.tolist()
    assert abs(free - (1 - math.exp(-1.0))) < 0.01, free       # first order, time constant tau
    assert abs(loaded - (1 - math.exp(-0.4))) < 0.01, loaded   # 60% of the torque went to the drive
    w.blade[:] = 1.0
    _run(w, arm=-1.0, blade=-1.0, seconds=1.0, load=load)
    assert abs(w.blade[0].item() - w.blade[1].item()) < 1e-6 and w.blade[0].item() < 0.4


def test_a_fast_blade_makes_the_robot_slower_to_turn():
    env = _env(n=2)
    _place(env, 24.0, 24.0, 0.0, 40.0, 40.0, 0.0)
    env.saw_o.gyro[:] = 0.5
    env.saw_o.blade[0], env.saw_o.blade[1] = 1.0, 0.0
    act = _act(env, turn=1.0)
    act[0, 3], act[1, 3] = 1.0, -1.0                          # one blade held at full speed, one stopped
    for _ in range(30):
        env.step(act)
    assert abs(env.o_omega[0].item() - 0.5 * YAW_RATE_MAX) < 0.05, env.o_omega
    assert abs(env.o_omega[1].item() - YAW_RATE_MAX) < 0.05, env.o_omega


def test_their_verts_and_drums_turn_slower_with_the_blade_up_and_horizontals_do_not():
    """A blade on a cross-wise axle rolls the robot when it turns, whoever it
    belongs to. A horizontal's axle is upright and it turns as it likes."""
    torch.manual_seed(0)
    se = SawEnv(8192, stage="combat", p=1.0, auto_reset=False).saw_e
    for kind, (lo, hi) in zip(SPINNERS, DR_EN_GYRO):
        mine = se.gyro[se.kind == kind]
        assert lo - 1e-4 <= mine.min().item() and mine.max().item() <= hi + 1e-4
    assert DR_EN_GYRO[2] == (0.0, 0.0) and DR_EN_GYRO[0][0] > DR_EN_GYRO[1][0] and DR_EN_GYRO[0][1] < 0.9
    lo, hi = DR_BLADE_GYRO
    assert DR_EN_GYRO[0][1] >= hi, "a vert carries more angular momentum than our saw"

    env = _env(n=3, tactic=PURSUER, stage="combat")
    _place(env, 8.0, 8.0, 0.0, 36.0, 36.0, 0.0)
    se = env.saw_e
    se.kind[2] = W_HORIZ
    se.gyro[:] = torch.tensor([0.6, 0.6, 0.0], device=DEVICE)
    se.blade[:] = torch.tensor([1.0, 0.0, 1.0], device=DEVICE)
    se.tau[:] = 1e6                                           # and the blades stay as they are
    env.manual_cmd[:, 0], env.manual_cmd[:, 1] = 0.0, 1.0       # the enemy is asked for a full turn on the spot
    for _ in range(30):
        env.step(_act(env, blade=-1.0))
    turn = (env.e_omega / YAW_RATE_MAX).tolist()
    assert abs(turn[0] - 0.4) < 0.02 and abs(turn[1] - 1.0) < 0.02 and abs(turn[2] - 1.0) < 0.02, turn


# ─── Hits ─────────────────────────────────────────────────────────────────

def test_vert_bites_throws_and_stuns_and_the_fight_goes_on():
    env = _env()
    _place(env, 24.0, 24.0, 0.0, 24.0 - 4.3, 24.0, 0.0)       # the target sits on our tail
    env.saw_o.blade[:] = 1.0
    _, _, done, info = env.step(_act(env, arm=-1.0, blade=1.0))
    assert not done.any()
    assert (info.tally[:, T_VERT] == 1).all(), info.tally[:, T_VERT]
    dealt = info.tally[:, T_DEALT]
    assert torch.allclose(dealt, torch.full_like(dealt, DMG_VERT), atol=0.05), dealt
    assert torch.allclose(env.rw[:, RW_OUTCOME], R_DAMAGE * dealt)
    assert (env.evx < -5.0).all(), "the target was not thrown away from our tail"
    assert (env.e_stun > 0.2).all(), "the target kept its drive"
    assert (env.ovx > 0.5).all(), "we felt no recoil"
    assert (env.saw_o.blade < 0.6).all(), "the bite took no speed off the blade"


def test_a_hit_has_to_be_charged():
    """A bite is worth the energy the blade carries over what armour shrugs
    off: at 70% of its speed a sixth of a full hit, at half speed nothing.
    The bite takes the blade's speed either way, so tapping at a target never
    gets the blade back to where it counts."""
    env = _env(n=3)
    _place(env, 24.0, 24.0, 0.0, 24.0 - 4.3, 24.0, 0.0)       # the target sits on our tail
    so = env.saw_o
    so.blade[:] = torch.tensor([1.0, 0.7, 0.5], device=DEVICE)
    _, _, _, info = env.step(_act(env, blade=1.0))
    assert (info.tally[:, T_VERT] == 1).all(), "all three touched and bit"
    floor = 0.4                                               # _env
    worth = torch.tensor([1.0, (0.49 - floor) / (1 - floor), 0.0], device=DEVICE)
    assert torch.allclose(info.tally[:, T_DEALT], DMG_VERT * worth, atol=0.01), info.tally[:, T_DEALT]
    assert (so.blade < torch.tensor([0.6, 0.45, 0.35], device=DEVICE)).all(), "a bite took no speed off the blade"
    assert env.e_stun[0].item() > 0.2 and env.e_stun[2].item() == 0.0, "a tap took their drive away"
    assert env.evx[2].item() < -3.0, "though it still shoves them"

    # Held on a target after a charged hit and biting as often as it can, a blade adds next to nothing
    w = _saw(tau=0.7, keep=0.5)
    w.blade[:] = 1.0
    taps = -1.0                                               # the first bite is the charged hit, worth 1
    for step in range(round(6.0 / DT)):                       # six seconds, a bite every REBITE_S
        if step % round(0.40 / DT) == 0:
            taps += ((w.blade ** 2 - floor) / (1 - floor)).clamp(min=0.0).item()
            w.blade *= 0.5
        w.step(_full(1, -1.0), _full(1, 1.0), _full(1, 0.0), _full(1, 1.0))
    assert taps < 0.3, f"fourteen taps after the hit were worth {taps:.2f} hits"

    torch.manual_seed(0)
    drawn = SawEnv(4096, stage="combat", p=1.0, auto_reset=False)
    lo, hi = DR_BITE_FLOOR
    for armour in (drawn.armour_o, drawn.armour_e):
        assert 0.25 < lo <= armour.min().item() and armour.max().item() <= hi < 0.6
    assert 0.5 * (DR_BLADE_KEEP[0] + DR_BLADE_KEEP[1]) ** 2 / 4 < lo, "a blade is still worth something straight after a bite"


def test_a_blade_on_a_robot_does_not_spin_back_up():
    """Charge up, hit, get clear, charge again. A blade that is held on a
    robot has its motor working against that robot and gets no faster, so
    sitting on a target after the hit is worth nothing more. Lifting the saw
    clear, letting it spin up and bringing it down again is another hit."""
    w = _saw(n=2, tau=0.5)
    w.blade[:] = 0.3
    w.bound[0] = True
    _run(w, arm=-1.0, blade=1.0, seconds=1.5)
    assert abs(w.blade[0].item() - 0.3) < 1e-6 and w.blade[1].item() > 0.9
    w.blade[:] = 0.8
    _run(w, arm=-1.0, blade=-1.0, seconds=1.5)
    assert (w.blade < 0.1).all(), "a blade that is switched off slows down wherever it is"

    env = _env(n=2)
    _place(env, 24.0, 24.0, 0.0, 24.0 + 4.3, 24.0, 0.0)       # the target sits on our nose
    so = env.saw_o
    so.blade[:] = 1.0
    down = _act(env, arm=1.0, blade=1.0)
    for _ in range(8):                                        # 0.24 s: the saw lands, a blow and a full bite
        env.step(down)
    first = env.tally[:, T_DEALT].clone()
    assert torch.allclose(first, _full(2, DMG_HAMMER_BLOW + DMG_HAMMER_SAW), atol=0.03)
    arm = torch.tensor([1.0, -1.0], device=DEVICE)            # one holds the saw on them, the other lifts it clear
    for _ in range(67):                                       # for two seconds
        env.step(_act(env, arm=arm, blade=1.0))
    assert so.bound.tolist() == [True, False]
    assert so.blade[0].item() < 0.51 and so.blade[1].item() > 0.85, so.blade     # what the bite left it, and no more
    assert (env.tally[0, T_DEALT] - first[0]).item() < 0.01, "holding the saw on them went on paying"
    for _ in range(10):                                       # and the second brings it down again
        env.step(down)
    again = (env.tally[:, T_DEALT] - first).tolist()
    assert again[0] < 0.01 and again[1] > 0.7 * (DMG_HAMMER_BLOW + DMG_HAMMER_SAW), again

    # The same goes for a spinner with its nose in our side
    env = _env(n=2, tactic=PURSUER, stage="combat")
    ey = torch.tensor([24.0 - 4.3, 24.0 - 8.0], device=DEVICE)    # one is touching us, the other is 3.7 ft short
    _place(env, 24.0, 24.0, 0.0, 24.0, ey, math.pi / 2)
    env.saw_e.blade[:] = 0.2                                  # too slow to cut, so nothing throws us apart
    for _ in range(40):                                       # 1.2 s
        env.step(_act(env, blade=-1.0))
    assert env.saw_e.bound.tolist() == [True, False]
    assert env.saw_e.blade[0].item() < 0.21 and env.saw_e.blade[1].item() > 0.6, env.saw_e.blade


def test_hammer_lands_a_blow_and_holding_it_down_adds_nothing():
    env = _env()
    _place(env, 24.0, 24.0, 0.0, 24.0 + 4.3, 24.0, 0.0)       # the target sits on our nose
    env.saw_o.blade[:] = 1.0
    fire = _act(env, arm=1.0, blade=1.0)
    for _ in range(4):                                        # 0.12 s: the arm is still on its way over
        env.step(fire)
    assert (env.tally[:, T_DEALT] == 0).all() and (env.saw_o.arm < ARM_DOWN).all()
    for _ in range(4):                                        # 0.24 s: it has landed
        env.step(fire)
    one = torch.ones(env.n, device=DEVICE)
    assert torch.allclose(env.tally[:, T_BLOW], one, atol=0.01), env.tally[:, T_BLOW]
    assert (env.tally[:, T_HAMMER] == 1).all()
    assert torch.allclose(env.tally[:, T_DEALT], (DMG_HAMMER_BLOW + DMG_HAMMER_SAW) * one, atol=0.03)
    assert env.evx.abs().max() < 0.5, "a hammer throws nobody"
    assert (env.e_stun > 0.2).all(), "but a full hit from above costs them their drive for a moment"
    for _ in range(30):                                       # held down on them for another 0.9 s
        _, _, done, _ = env.step(fire)
    assert not done.any()
    assert (env.tally[:, T_HAMMER] == 1).all(), env.tally[:, T_HAMMER]     # a saw that is already down cuts nothing more
    assert torch.allclose(env.tally[:, T_DEALT], (DMG_HAMMER_BLOW + DMG_HAMMER_SAW) * one, atol=0.03)
    assert torch.allclose(env.tally[:, T_BLOW], one, atol=0.01)            # and there was only the one swing


def test_a_saw_that_is_already_down_is_not_a_weapon():
    """Laid on the floor in front and driven into them, the saw cuts nothing
    and gets no faster while it is on them. Lifted and brought down on them,
    it cuts; brought down from half way, it cuts a fifth as hard."""
    env = _env()
    so = env.saw_o
    _place(env, 24.0, 24.0, 0.0, 24.0, 40.0, 0.0)             # nobody near our nose
    down = _act(env, arm=1.0, blade=1.0)
    for _ in range(10):
        env.step(down)
    assert (so.arm >= ARM_DOWN).all() and torch.allclose(env.tally[:, T_MISS], torch.ones(env.n, device=DEVICE), atol=0.01)
    assert (so.strike == 0).all(), "the swing was still live long after it landed"
    env.ex[:], env.ey[:] = 24.0 + 4.3, 24.0                   # now they are under it
    so.blade[:] = 0.6
    for _ in range(30):
        env.step(down)
    assert (env.tally[:, T_HAMMER] == 0).all() and (env.tally[:, T_DEALT] == 0).all(), env.tally[:, T_DEALT]
    assert (so.blade < 0.61).all(), "a blade lying on a robot got faster"       # one physics step before it is held
    lift = _act(env, arm=-1.0, blade=1.0)
    for _ in range(20):                                       # all the way back (0.6 s)
        env.step(lift)
    so.blade[:] = 1.0
    for _ in range(8):
        env.step(down)
    assert (env.tally[:, T_HAMMER] == 1).all(), env.tally[:, T_HAMMER]
    full = env.tally[:, T_DEALT].clone()
    assert torch.allclose(full, torch.full_like(full, DMG_HAMMER_BLOW + DMG_HAMMER_SAW), atol=0.03), full
    for _ in range(8):                                        # half way back (0.24 s of a 0.5 s lift) ...
        env.step(lift)
    assert ((so.arm > 0.4) & (so.arm < 0.65)).all(), so.arm
    so.blade[:], so.rebite[:] = 1.0, 0.0
    for _ in range(6):                                        # ... and down again
        env.step(down)
    assert (env.tally[:, T_HAMMER] == 2).all(), env.tally[:, T_HAMMER]
    short = env.tally[:, T_DEALT] - full
    assert ((short > 0.1 * full) & (short < 0.4 * full)).all(), (short, full)


def test_hammer_on_the_floor_is_a_miss():
    env = _env()
    _place(env, 24.0, 24.0, 0.0, 24.0, 40.0, 0.0)             # nobody near our nose
    env.saw_o.blade[:] = 1.0
    for _ in range(10):
        env.step(_act(env, arm=1.0, blade=1.0))
    assert torch.allclose(env.tally[:, T_MISS], torch.ones(env.n, device=DEVICE), atol=0.01)
    assert (env.tally[:, T_DEALT] == 0).all() and (env.tally[:, T_BLOW] == 0).all()
    assert (env.saw_o.blade < 0.7).all(), "the floor took no speed off the blade"


def test_spinner_bites_throws_and_stuns_us_and_the_fight_goes_on():
    env = _env(tactic=PURSUER, stage="combat")
    assert (env.saw_e.kind == W_VERT).all()
    _place(env, 24.0, 24.0, 0.0, 24.0, 24.0 - 4.3, math.pi / 2)   # its nose is in our right side
    env.saw_e.blade[:], env.saw_e.power[:] = 1.0, 0.8
    _, _, done, info = env.step(_act(env))
    assert not done.any()
    assert (info.tally[:, T_TAKEN] == 1).all(), info.tally[:, T_TAKEN]
    hurt = info.tally[:, T_HURT]
    assert torch.allclose(hurt, torch.full_like(hurt, 0.8 * DMG_SPINNER), atol=0.04), hurt
    # In the reward their hit costs less than ours pays: attacks landed are
    # what the team is after. The scorecard above has it in full.
    assert R_TAKEN_SHARE < 1.0 and torch.allclose(env.rw[:, RW_OUTCOME], -R_DAMAGE * R_TAKEN_SHARE * hurt)
    assert (env.o_drift > 3.0).all(), "we were not thrown the way the spinner points"
    assert (env.o_stun > 0.2).all(), "we kept our drive"


def test_our_blade_on_a_spinners_weapon_scores_nothing_and_takes_the_hit():
    env = _env(tactic=PURSUER, stage="combat")
    _place(env, 24.0, 24.0, 0.0, 24.0 - 4.6, 24.0, 0.0)       # its nose on our parked saw, 0.6 ft short of our body
    env.saw_o.blade[:], env.saw_e.blade[:], env.saw_e.power[:] = 1.0, 1.0, 0.8
    _, _, done, info = env.step(_act(env))
    assert not done.any()
    assert (info.tally[:, T_CLASH] == 1).all() and (info.tally[:, T_VERT] == 0).all()
    assert (info.tally[:, T_DEALT] == 0).all(), "weapon on weapon did them damage"
    assert (info.tally[:, T_TAKEN] == 1).all()
    assert torch.allclose(info.tally[:, T_HURT], _full(env.n, 0.8 * DMG_SPINNER), atol=0.04), "it is a full hit from their weapon"
    assert (env.saw_o.blade < 0.3).all(), "our blade kept its speed"
    assert (env.saw_e.blade > 0.4).all(), "theirs lost more than a bite takes"
    assert (env.ovx > 1.0).all() and (env.evx < -1.0).all(), "the robots were not thrown apart"
    margin = SawEnv(4096, stage="combat", p=1.0, auto_reset=False).front_margin
    lo, hi = DR_FRONT_MARGIN_FT
    assert 0.1 < lo <= margin.min().item() and margin.max().item() <= hi
    assert abs(margin.mean().item() - 0.5 * (lo + hi)) < 0.02


def test_a_spinners_front_is_off_limits_and_its_side_is_not():
    """Our tail goes into a vert's side three times: well behind its front
    edge, on its front corner with its blade cutting, and on its front corner
    with its blade stopped."""
    env = _env(n=3, tactic=PURSUER, stage="combat")
    ox = torch.tensor([23.0, 25.8, 25.8], device=DEVICE)      # 3 ft behind its front edge; 0.2 ft, inside the 0.5 ft margin
    _place(env, ox, 19.7, -math.pi / 2, 24.0, 24.0, 0.0)      # backed into its right side: our blade is 0.2 ft into it
    so, se = env.saw_o, env.saw_e
    se.wide[:] = 1.2                                          # blade and forks reach to 0.8 ft from each side
    so.blade[:], se.power[:] = 1.0, 1.0
    se.blade[:] = torch.tensor([1.0, 1.0, 0.0], device=DEVICE)
    _, _, done, info = env.step(_act(env))
    assert not done.any()
    assert info.tally[:, T_VERT].tolist() == [1.0, 0.0, 0.0], "only the side is there to be hit"
    assert info.tally[:, T_CLASH].tolist() == [0.0, 1.0, 1.0]
    assert info.tally[:, T_TAKEN].tolist() == [0.0, 1.0, 0.0], "a blade that is cutting hits the one that reached for it"
    assert torch.allclose(info.tally[:, T_DEALT], torch.tensor([DMG_VERT, 0.0, 0.0], device=DEVICE), atol=0.05)
    assert torch.allclose(info.tally[:, T_HURT], torch.tensor([0.0, DMG_SPINNER, 0.0], device=DEVICE), atol=0.04)
    assert (so.blade[1:] < 0.3).all(), "our blade kept its speed on their front"


def test_a_narrow_verts_front_corner_stops_our_blade_but_its_blade_is_out_of_reach():
    env = _env(n=2, tactic=PURSUER, stage="combat")
    so, se = env.saw_o, env.saw_e
    for t in (env.e_wid, env.o_wid):
        t[:] = 2.5                                            # a wide robot with a blade 0.2 ft either side of its middle
    _place(env, 25.8, 19.2, -math.pi / 2, 24.0, 24.0, 0.0)    # our blade is 0.2 ft into its side, 0.2 ft behind its front edge
    env.ox[0] = 23.0                                          # the first is 3 ft behind it instead
    so.blade[:], se.blade[:], se.power[:] = 1.0, 1.0, 1.0
    _, _, _, info = env.step(_act(env))
    assert info.tally[:, T_VERT].tolist() == [1.0, 0.0] and info.tally[:, T_CLASH].tolist() == [0.0, 1.0]
    assert (info.tally[:, T_TAKEN] == 0).all(), "its blade is 2 ft away across its front"
    assert so.blade[1].item() < 0.3, "our blade kept its speed on their forks"


def test_their_blade_does_a_full_hit_wherever_it_lands_on_us():
    """A vert drives into our nose, into our side, and into our nose with the
    hammer saw down in front of it. Our forks are not on the hammer end, so
    nothing is taken off any of them."""
    env = _env(n=3, tactic=PURSUER, stage="combat")
    so, se = env.saw_o, env.saw_e
    ea = torch.tensor([math.pi, math.pi / 2, math.pi], device=DEVICE)
    ex = torch.tensor([24.0 + 4.3, 24.0, 24.0 + 4.3], device=DEVICE)
    ey = torch.tensor([24.0, 24.0 - 4.3, 24.0], device=DEVICE)
    _place(env, 24.0, 24.0, 0.0, ex, ey, ea)
    so.arm[2], so.lowest[2], so.fire[2] = 1.0, 1.0, True
    so.blade[:], se.blade[:], se.power[:] = 0.0, 1.0, 1.0
    arm = torch.tensor([-1.0, -1.0, 1.0], device=DEVICE)
    _, _, _, info = env.step(_act(env, arm=arm, blade=-1.0))
    assert (info.tally[:, T_TAKEN] == 1).all()
    assert torch.allclose(info.tally[:, T_HURT], DMG_SPINNER * torch.ones(3, device=DEVICE), atol=0.04), info.tally[:, T_HURT]
    assert env.o_stun[0].item() > 0.1 and abs(env.o_stun[0].item() - env.o_stun[1].item()) < 0.05, env.o_stun
    assert not hasattr(env, "nose_armour")


def test_our_blade_may_be_held_to_the_tip_speed_rule():
    """Nobody knows yet whether the 370 ft/s limit applies to this robot, so
    each fight draws the share of its 7,696 rpm the blade tops out at. A held
    blade hits with the square of that, rolls the robot less in a turn, and
    is back at its own top speed sooner."""
    lo, hi = DR_BLADE_TOP
    assert abs(lo - 370.0 / 520.0) < 0.01 and hi == 1.0
    torch.manual_seed(0)
    so = SawEnv(8192, stage="combat", p=1.0, auto_reset=False).saw_o
    top = so.power.sqrt()
    assert lo - 1e-4 <= top.min().item() < lo + 0.02 and 0.98 < top.max().item() <= 1.0
    held, free = top < lo + 0.03, top > 0.97
    assert so.gyro[held].mean().item() < 0.8 * so.gyro[free].mean().item()
    assert so.tau[held].mean().item() < 0.55 * so.tau[free].mean().item()
    assert so.tau[free].min().item() > 0.85 * DR_BLADE_TAU_S[0]

    # Backed into a vert's side at top blade speed, at the full 7,696 rpm and held to the rule
    env = _env(n=2, tactic=PURSUER, stage="combat")
    _place(env, 23.0, 19.7, -math.pi / 2, 24.0, 24.0, 0.0)
    so, se = env.saw_o, env.saw_e
    se.wide[:] = 1.2
    so.blade[:], se.blade[:], se.power[:] = 1.0, 1.0, 1.0
    so.power[:] = torch.tensor([1.0, lo ** 2], device=DEVICE)
    _, _, _, info = env.step(_act(env))
    assert info.tally[:, T_VERT].tolist() == [1.0, 1.0]
    want = torch.tensor([DMG_VERT, DMG_VERT * lo ** 2], device=DEVICE)
    assert torch.allclose(info.tally[:, T_DEALT], want, atol=0.05), info.tally[:, T_DEALT]
    assert env.e_stun[1].item() < 0.6 * env.e_stun[0].item(), env.e_stun


def test_a_new_saw_policy_starts_with_the_blade_on():
    """Asking for half speed, which is what an untrained head does, never
    lands a bite worth anything, and nothing would teach it to ask for more."""
    from agent import Agent
    saw, classic = Agent(SAW_OBS_DIM, SAW_ACT_DIM), Agent(OBS_DIM, ACT_DIM)
    assert saw.actor_mean.bias.tolist() == [0.0, 0.0, 0.0, BLADE_ON_BIAS] and BLADE_ON_BIAS >= 1.0
    assert classic.actor_mean.bias.abs().sum().item() == 0.0
    w = _saw(n=4096, tau=0.7)
    w.blade[:] = 0.0
    torch.manual_seed(0)
    for _ in range(round(4.0 / DT)):                          # an untrained head's blade command, four seconds of it
        w.step(_full(4096, -1.0), (BLADE_ON_BIAS + torch.randn(4096, device=DEVICE)).clamp(-1, 1), _full(4096, 0.0), _full(4096, 1.0))
    assert (w.blade ** 2).mean().item() > DR_BITE_FLOOR[1] + 0.1, "the blade an untrained policy runs is not worth a bite"


def test_a_horizontals_bar_guards_its_sides_only_while_it_turns():
    env = _env(n=2, tactic=PURSUER, stage="combat")
    so, se = env.saw_o, env.saw_e
    se.kind[:] = W_HORIZ
    se.fore[:], se.rad[:], se.wide[:] = 1.0, 2.4, 0.0         # hub 1 ft ahead of centre; the bar sweeps 0.4 ft past each side
    _place(env, 23.0, 19.7, -math.pi / 2, 24.0, 24.0, 0.0)    # backed into its side, 3 ft behind its front edge
    so.blade[:], se.power[:] = 1.0, 1.0
    se.blade[:] = torch.tensor([1.0, 0.1], device=DEVICE)     # at speed; just hit something and still spinning up
    _, _, done, info = env.step(_act(env))
    assert not done.any()
    assert info.tally[:, T_VERT].tolist() == [0.0, 1.0]
    assert info.tally[:, T_CLASH].tolist() == [1.0, 0.0]
    assert info.tally[:, T_TAKEN].tolist() == [1.0, 0.0]


def test_opponents_weapons_span_the_heavyweight_field():
    """Stored energy over our blade's 49 kJ, and spin-up time: some of every
    kind hit harder than we do, and none is back up to speed as fast."""
    torch.manual_seed(0)
    env = SawEnv(8192, stage="combat", p=1.0, auto_reset=False)
    se, traves = env.saw_e, env.field.profile == PROFILE_TRAVES
    for kind, (lo, hi), (t_lo, t_hi) in zip(SPINNERS, DR_EN_POWER, DR_EN_TAU_S):
        mine = (se.kind == kind) & ~traves
        if kind == W_HORIZ:
            t_lo, t_hi = HORIZ_TAU_S                          # a big bar is slow to spin up
        assert lo - 1e-4 <= se.power[mine].min().item() and se.power[mine].max().item() <= hi + 1e-4
        assert lo < 0.7 and hi > 1.1 and (se.power[mine] > 1.0).any() and (se.power[mine] < 0.7).any()
        assert t_lo - 1e-4 <= se.tau[mine].min().item() and se.tau[mine].max().item() <= t_hi + 1e-4
        assert t_lo > DR_BLADE_TAU_S[0]
    assert traves.any() and (se.kind[traves] == W_HORIZ).all()
    for t, (lo, hi) in ((se.power, TRAVES_POWER), (se.tau, TRAVES_TAU_S)):
        assert lo - 1e-4 <= t[traves].min().item() and t[traves].max().item() <= hi + 1e-4


def test_an_expert_backs_off_to_spin_a_slow_blade_back_up():
    """A blade that is not up to speed does no damage, so most experts get
    clear with their nose on us until it is. Beginners come straight back."""
    torch.manual_seed(0)
    n = 512
    env = SawEnv(n, stage="combat", p=1.0, auto_reset=False, scenario=Scenario(weapon_fail_share=0.0, from_bell=False))
    env.opponents.forced_types = torch.full((n,), PURSUER, dtype=torch.long, device=DEVICE)
    env.reset()
    opp, se = env.opponents, env.saw_e
    assert abs(opp.respins.float().mean().item() - RESPIN_SHARE[1] * SHARP_TOP) < 0.06
    assert RESPIN_AT[0] <= opp.respin_at.min().item() and opp.respin_at.max().item() <= RESPIN_AT[1]
    _place(env, 24.0, 24.0, math.pi, 30.0, 24.0, math.pi)     # they are 6 ft off our nose, facing us; our tail points away
    opp.seen[:] = torch.stack([env.ox, env.oy], dim=1).unsqueeze(1)    # and they have watched us stand there
    se.blade[:] = 0.3
    se.tau[:] = 30.0                                          # and the blade stays slow for the length of the test
    gap = (env.ex - env.ox).clone()
    for _ in range(30):                                       # 0.9 s
        env.step(_act(env, blade=-1.0))
    moved = (env.ex - env.ox) - gap
    assert moved[opp.respins].median().item() > 1.0, "an expert with a slow blade did not back away"
    assert moved[~opp.respins].median().item() < -1.0, "the others did not come on"
    assert (angle_wrap(env.ea - math.pi).abs()[opp.respins] < 0.6).float().mean().item() > 0.8, "they took their nose off us"

    green = SawEnv(n, stage="combat", p=0.0, auto_reset=False)
    assert not green.opponents.respins.any(), "a beginner does not know to"
    classic = make_env("classic", n, stage="combat", p=1.0, auto_reset=False)
    assert not classic.opponents.respins.any()


def test_a_thrown_robot_has_no_drive_until_it_recovers():
    env = _env(n=2)
    _place(env, 12.0, 24.0, 0.0, 36.0, 40.0, 0.0)
    env.o_stun[0] = 0.30
    go = _act(env, throttle=1.0, blade=-1.0)
    for _ in range(5):                                        # 0.15 s at full throttle
        env.step(go)
    assert abs(env.ovx[0].item()) < 0.05 and env.ovx[1].item() > 5.0, env.ovx
    for _ in range(12):                                       # another 0.36 s: the drive is back
        env.step(go)
    assert env.ovx[0].item() > 5.0, env.ovx


def test_drill_targets_carry_no_weapon():
    torch.manual_seed(0)
    env = SawEnv(512, stage="pretrain", p=0.0, auto_reset=False)
    env.reset()
    assert (env.saw_e.kind == W_NONE).all() and (env.saw_o.kind == W_SAW).all()
    for _ in range(300):
        env.step(_act(env))
    assert (env.tally[:, T_TAKEN] == 0).all() and (env.tally[:, T_CLASH] == 0).all()


# ─── The opponents' weapons ───────────────────────────────────────────────

def test_opponents_carry_verts_drums_and_horizontals():
    torch.manual_seed(0)
    env = SawEnv(8192, stage="combat", p=1.0, auto_reset=False, scenario=Scenario(traves_share=0.0))
    se, armed = env.saw_e, env.opponents.armed
    assert armed.float().mean().item() > 0.5
    assert (se.kind[~armed] == W_NONE).all()
    for kind, share in zip(SPINNERS, EN_WEAPON_SHARE):
        got = (se.kind[armed] == kind).float().mean().item()
        assert abs(got - share) < 0.03, (kind, got, share)
    vert, drum, horiz = (se.kind == k for k in SPINNERS)
    past_nose = se.fore + se.rad - env.e_len                  # how far the blade reaches beyond the front edge
    assert 0.29 < past_nose[vert].min().item() and past_nose[vert].max().item() < 0.61
    assert 0.19 < past_nose[drum].min().item() and past_nose[drum].max().item() < 0.41
    hub = se.fore[horiz] / env.e_len[horiz]                   # a horizontal: a hub near the front edge, and a bar
    assert HORIZ_HUB[0] - 1e-4 <= hub.min().item() and hub.max().item() <= HORIZ_HUB[1] + 1e-4
    assert HORIZ_BAR_FT[0] - 1e-4 <= se.rad[horiz].min().item() and se.rad[horiz].max().item() <= HORIZ_BAR_FT[1] + 1e-4
    assert past_nose[horiz].min().item() > 0.2, "a horizontal's bar does not reach past its own front edge"
    share, row = se.wide / env.e_wid, vert | drum             # what the front hits with, as a share of the body's width
    assert DR_VERT_WIDE[0] - 1e-4 <= share[vert].min().item() and share[vert].max().item() <= DR_VERT_WIDE[1] + 1e-4
    assert DR_DRUM_WIDE[0] - 1e-4 <= share[drum].min().item() and share[drum].max().item() <= DR_DRUM_WIDE[1] + 1e-4
    assert share[vert].min().item() > 0.4, "a vert's front is a point again"
    assert share[drum].mean().item() > share[vert].mean().item(), "a drum is no wider than a vert"
    assert (2 * se.wide[row] / (EN_FRONT_DISCS - 1) < 2 * se.rad[row]).all(), "there is a gap between two front discs"
    assert (se.wide[horiz] == 0).all() and se.rad[horiz].min().item() > 0.64     # a hub ahead of the front edge, a short bar
    assert (se.rad[horiz] > env.e_wid[horiz]).any(), "no horizontal's bar reaches past its own corners"


def test_the_first_combat_levels_meet_armed_drifters_and_beginners_who_leave_a_side_open():
    """The way in against a spinner is round its side. The first level leaves
    it open: nobody guards, beginners come about slowly, and some of the
    field wanders about with a weapon and takes no notice of us."""
    torch.manual_seed(0)
    env = SawEnv(8192, stage="combat", p=0.0, auto_reset=False, scenario=Scenario(traves_share=0.0))
    opp, drifting = env.opponents, env.opponents.types == DRIFTER
    assert abs(drifting.float().mean().item() - SHARP_MIX_EARLY[DRIFTER]) < 0.03
    assert opp.armed.all() and (env.saw_e.kind != W_NONE).all(), "every combat opponent carries a weapon"
    assert not opp.guards.any()
    assert opp.agility_ratio.max().item() <= SHARP["AGILITY_RATIO"][0][1] + 1e-6 < AGILITY_RATIO[0][0]
    assert SHARP["REACT_S"][0][0] > REACT_S[0][0] and SHARP["REACT_S"][1][1] < REACT_S[1][1]
    ex, ey = env.ex.clone(), env.ey.clone()
    for _ in range(40):
        env.step(_act(env, blade=-1.0))
    moved = torch.sqrt((env.ex - ex) ** 2 + (env.ey - ey) ** 2)
    assert moved[drifting].median().item() > 1.0, "an armed drifter still wanders"

    late = SawEnv(4096, stage="combat", p=MIX_FULL_AT, auto_reset=False)
    assert not (late.opponents.types == DRIFTER).any(), "drifters are gone once the full mix is in"
    assert late.opponents.guards.any()

    drills = SawEnv(1024, stage="pretrain", p=0.0, auto_reset=False)
    assert not drills.opponents.armed.any() and (drills.saw_e.kind == W_NONE).all(), "drill targets stay unarmed"
    classic = make_env("classic", 4096, stage="combat", p=0.0, auto_reset=False)
    assert set(classic.opponents.types.unique().tolist()) <= set(COMBAT_TYPES), "the classic field has no drifters"


def test_a_horizontal_reaches_round_its_corner_and_throws_sideways():
    env = _env(n=3, tactic=PURSUER, stage="combat")
    se = env.saw_e
    se.kind[:2] = W_HORIZ                                     # env 2 keeps its vert
    se.fore[:2], se.rad[:2], se.wide[:2] = 1.0, 2.4, 0.0      # hub 1 ft ahead of centre; the bar sweeps 0.4 ft past each side
    se.turn[0], se.turn[1] = 1.0, -1.0
    _place(env, 25.0, 28.3, 0.0, 24.0, 24.0, 0.0)             # we sit beside its front corner, 0.3 ft clear of its body
    se.blade[:], se.power[:] = 1.0, 1.0
    _, _, done, info = env.step(_act(env, blade=-1.0))
    assert not done.any()
    assert info.tally[:, T_TAKEN].tolist() == [1.0, 1.0, 0.0], "only the horizontals reach there"
    assert info.tally[:, T_CLASH].sum().item() == 0
    # Thrown along the blade's travel where it touched: opposite ways for opposite spins
    assert env.ovx[0].item() < -4.0 and env.ovx[1].item() > 4.0, env.ovx
    assert (env.o_drift[:2] > 0).all(), "and a little away from the hub"
    assert env.ovx[2].item() == 0.0 and env.o_stun[2].item() == 0.0


def test_the_hammer_scores_nothing_on_a_weapon_side_either():
    """Nose to nose with a horizontal and with a vert, and on the back of a
    vert that is facing away."""
    env = _env(n=3, tactic=PURSUER, stage="combat")
    se, so = env.saw_e, env.saw_o
    se.kind[0] = W_HORIZ
    se.fore[0], se.rad[0], se.wide[0] = 1.0, 2.4, 0.0
    ea = torch.tensor([math.pi, math.pi, 0.0], device=DEVICE)
    _place(env, 24.0, 24.0, 0.0, 24.0 + 4.3, 24.0, ea)
    so.arm[:], so.lowest[:], so.fire[:] = 1.0, 1.0, True       # our saw has this moment come down on them
    so.strike[:], so.strike_left[:] = 1.0, STRIKE_S
    so.blade[:], se.blade[:], se.power[:] = 1.0, 1.0, 1.0
    _, _, _, info = env.step(_act(env, arm=1.0, blade=1.0))
    assert info.tally[:, T_HAMMER].tolist() == [0.0, 0.0, 1.0]
    assert info.tally[:, T_CLASH].tolist() == [1.0, 1.0, 0.0]
    assert info.tally[:, T_TAKEN].tolist() == [1.0, 1.0, 0.0]
    assert torch.allclose(info.tally[:, T_DEALT], torch.tensor([0.0, 0.0, DMG_HAMMER_SAW], device=DEVICE), atol=0.03)


# ─── Bodies ───────────────────────────────────────────────────────────────

def test_bodies_are_rectangles():
    z = lambda v: _full(1, v)
    long, narrow = z(2.5), z(1.25)
    # Two long narrow robots 3 ft apart, side by side: clear of each other
    touching, _, _, _ = body_contact(z(0.0), z(3.0), z(1.0), z(0.0), z(1.0), z(0.0), long, narrow, long, narrow)
    assert not touching.item()
    # Turn theirs across ours and its nose is 0.75 ft into our side
    touching, depth, nx, ny = body_contact(z(0.0), z(3.0), z(1.0), z(0.0), z(0.0), z(1.0), long, narrow, long, narrow)
    assert touching.item() and abs(depth.item() - 0.75) < 1e-5
    assert abs(nx.item()) < 1e-6 and abs(ny.item() - 1.0) < 1e-6, "the way out is straight away from us"
    # Points against a body: ahead of the nose, off the front corner, and inside
    px = torch.tensor([[3.0, 3.0, 0.0]], device=DEVICE)
    py = torch.tensor([[0.0, 1.65, 0.0]], device=DEVICE)
    gap = body_gap(px, py, z(0.0), z(0.0), z(1.0), z(0.0), long, narrow)[0].tolist()
    assert abs(gap[0] - 0.5) < 1e-5 and abs(gap[1] - math.hypot(0.5, 0.4)) < 1e-5 and gap[2] == 0.0


def test_sizes_are_drawn_per_fight_and_the_policy_is_told_its_own():
    torch.manual_seed(0)
    env = SawEnv(4096, stage="combat", p=1.0, auto_reset=False)
    armed, prof = env.opponents.armed, env.field.profile
    field, traves = armed & (prof == PROFILE_FIELD), armed & (prof == PROFILE_TRAVES)
    assert field.any() and traves.any()
    for t, (lo, hi) in ((env.o_len, DR_OUR_HALF_LEN), (env.o_wid, DR_OUR_HALF_WID),
                        (env.e_len[field], DR_EN_HALF_LEN), (env.e_wid[field], EN_HALF_WID),
                        (env.e_len[traves], TRAVES_HALF_LEN), (env.e_wid[traves], TRAVES_HALF_WID)):
        assert t.min().item() >= lo - 1e-5 and t.max().item() <= hi + 1e-5 and t.std().item() > 0.1
    old = SawEnv(4096, stage="combat", p=1.0, auto_reset=False, scenario=Scenario(legacy_field=True))
    drills = SawEnv(1024, stage="pretrain", auto_reset=False)
    for e in (old, drills):                                   # the saw-v7 field and the drill targets keep the v7 body
        for t, (lo, hi) in ((e.e_len, DR_EN_HALF_LEN), (e.e_wid, DR_EN_HALF_WID)):
            assert t.min().item() >= lo - 1e-5 and t.max().item() <= hi + 1e-5
    obs = env._observe(env._ar)
    assert torch.allclose(obs[:, 28], env.o_len - 2.0) and torch.allclose(obs[:, 29], env.o_wid - 2.0)
    # Whatever the sizes, no fight opens with the robots inside each other
    inside, depth, _, _ = body_contact(env.ex - env.ox, env.ey - env.oy, torch.cos(env.oa), torch.sin(env.oa),
                                       torch.cos(env.ea), torch.sin(env.ea),
                                       env.o_len, env.o_wid, env.e_len, env.e_wid)
    assert depth.max().item() < 1e-3, f"a fight opened {depth.max().item():.2f} ft inside the other robot"


# ─── The clock ────────────────────────────────────────────────────────────

def test_nothing_ends_a_fight_but_the_clock():
    """A robot that never moves is hit again and again by every tactic, and
    every one of its fights still runs the full 20 s. The hits are few and
    full: a spinner's blade has to be clear of us to spin back up."""
    torch.manual_seed(0)
    pinned = torch.tensor(COMBAT_TYPES, device=DEVICE).repeat_interleave(48)
    env = SawEnv(pinned.numel(), stage="combat", p=1.0, auto_reset=False)
    env.opponents.forced_types = pinned
    env.reset()
    idle = _act(env, arm=0.0, blade=-1.0)
    last = MAX_STEPS // PHYS_PER_CTRL + 1                     # the control step the clock runs out in
    for t in range(1, last + 1):
        _, _, done, info = env.step(idle)
        if t < last:
            assert not done.any(), f"a fight ended at control step {t} of {last}"
    assert done.all() and info.timeout.all()
    assert set(info.end_reason.unique().tolist()) <= {END_PTS_WIN, END_PTS_LOSS, END_PTS_DRAW}
    assert not env.won.any()
    for t in COMBAT_TYPES:
        m = pinned == t
        assert env.tally[m, T_TAKEN].mean().item() > 2.0, f"{TYPE_NAMES[t]} barely touched a parked robot"
        assert env.tally[m, T_HURT].mean().item() > 1.0, f"{TYPE_NAMES[t]} barely hurt a parked robot"   # 2.5 before saw-v9, when bulges also burned
        behind = (env.end_reason[m] == END_PTS_LOSS).float().mean().item()
        # saw-v9: 0.9 before, when a robot parked over a bulge also burned on it. The bulges are wall now, and a
        # robot shoved into a corner with their blade stalled on it is touched for nothing in about one fight in six.
        assert behind >= 0.7, f"a parked robot lost only {behind:.0%} of its fights with {TYPE_NAMES[t]}"


def test_the_clock_scores_the_fight_on_damage():
    env = _env(n=3)
    _place(env, 12.0, 24.0, 0.0, 36.0, 24.0, 0.0)
    env.tally[:, T_DEALT] = torch.tensor([2.0, 0.5, 0.1], device=DEVICE)
    env.tally[:, T_HURT]  = torch.tensor([0.5, 2.0, 0.0], device=DEVICE)
    env.steps[:] = MAX_STEPS - PHYS_PER_CTRL
    _, _, done, info = env.step(_act(env, blade=-1.0))
    assert done.all() and info.timeout.all(), "the clock must end the fight as a cut, not a terminal state"
    assert info.end_reason.tolist() == [END_PTS_WIN, END_PTS_LOSS, END_PTS_DRAW]
    assert env.won.tolist() == [True, False, False]


def test_training_mixes_fight_lengths_and_evaluation_does_not():
    torch.manual_seed(0)
    env = SawEnv(8192, stage="combat", p=1.0, auto_reset=True)
    for seconds, share in FIGHT_LEN_S:
        got = (env.horizon == round(seconds / DT)).float().mean().item()
        assert abs(got - share) < 0.03, (seconds, got, share)
    assert (SawEnv(64, stage="combat", p=1.0, auto_reset=False).horizon == MAX_STEPS).all()

    # A one minute fight is still going when a 20 s one would have ended
    env = SawEnv(4, stage="pretrain", auto_reset=True)
    env.horizon[:] = 6000
    env.steps[:] = MAX_STEPS - PHYS_PER_CTRL
    _, _, done, _ = env.step(_act(env))
    assert not done.any()
    env.steps[:] = 6000 - PHYS_PER_CTRL
    _, _, done, info = env.step(_act(env))
    assert done.all() and info.timeout.all()


def test_training_fights_do_not_all_end_on_the_same_step():
    """Every fight lasts exactly its clock, so training starts them spread
    over it. Evaluation starts them together."""
    torch.manual_seed(0)
    n = 4096
    env = SawEnv(n, stage="pretrain", auto_reset=True)
    assert env.steps.float().std().item() > MAX_STEPS / 5
    env.reset()                                              # the trainer resets before its first rollout
    assert env.steps.float().std().item() > MAX_STEPS / 5, "reset() put every clock back to zero"
    assert (env.steps < env.horizon).all()
    expected = (50 * PHYS_PER_CTRL / env.horizon.float()).sum().item()   # the share of each clock that goes by
    ended = 0
    for _ in range(50):
        _, _, done, _ = env.step(_act(env))
        assert done.float().mean().item() < 0.01
        ended += int(done.sum())
    assert 0.6 * expected < ended < 1.4 * expected, (ended, expected)
    assert SawEnv(8, stage="pretrain", auto_reset=False).steps.sum().item() == 0


def test_keeping_away_from_an_armed_opponent_is_charged_by_the_second():
    """Out of reach and not driving at them. Coming in is free with either
    end leading; standing off, or driving away, is not."""
    env = _env(n=5, tactic=PURSUER, stage="combat")
    env.r_slow = 0.0                                          # this is about the keeping-away charge alone
    _place(env, 10.0, 24.0, 0.0, 10.0 + PASSIVE_FT + 6.0, 24.0, math.pi)
    env.ex[1] = 10.0 + PASSIVE_FT - 4.0                       # env 1 is close enough to count as fighting
    env.ovx[2] = 2 * PASSIVE_CLOSE_FT_S                       # env 2 is driving at them nose first
    env.oa[3], env.ovx[3] = math.pi, -2 * PASSIVE_CLOSE_FT_S  # env 3 is backing in, as the vert does
    env.oa[4], env.ovx[4] = math.pi, 2 * PASSIVE_CLOSE_FT_S   # env 4 is driving away
    go = _act(env, blade=-1.0)
    go[2, 0], go[3, 0], go[4, 0] = 1.0, -1.0, 1.0
    for _ in range(10):
        env.step(go)
    charged = R_PASSIVE_PER_S * 10 * CTRL_DT
    assert abs(env.rw[0, RW_TIME].item() + charged) < 1e-6, env.rw[:, RW_TIME]
    assert abs(env.rw[4, RW_TIME].item() + charged) < 1e-6, env.rw[:, RW_TIME]
    assert (env.rw[1:4, RW_TIME] == 0.0).all(), env.rw[:, RW_TIME]
    assert abs(env.tally[0, T_PASSIVE_S].item() - 10 * CTRL_DT) < 1e-5 and (env.tally[1:4, T_PASSIVE_S] == 0).all()
    drill = _env(n=1)                                         # an unarmed target: there is nobody to be judged against
    drill.r_slow = 0.0
    _place(drill, 10.0, 24.0, 0.0, 40.0, 24.0, 0.0)
    for _ in range(10):
        drill.step(_act(drill, blade=-1.0))
    assert drill.rw[0, RW_TIME].item() == 0.0


def test_a_fight_spent_out_of_reach_is_lost_on_the_scorecard():
    """Nobody lands anything in either fight. The robot that stood off for
    the whole clock is behind, by more than a robot that fights has ever
    lost by; the one that stayed in reach is level."""
    env = _env(n=2, tactic=PURSUER, stage="combat")
    env.r_slow = 0.0
    _place(env, 10.0, 24.0, 0.0, 10.0 + PASSIVE_FT + 6.0, 24.0, math.pi)
    env.ex[1] = 10.0 + PASSIVE_FT - 4.0
    done = torch.zeros(2, dtype=torch.bool, device=DEVICE)
    while not done.all():
        _, _, done, info = env.step(_act(env, blade=-1.0))
    assert (env.tally[:, T_DEALT] == 0).all() and (env.tally[:, T_HURT] == 0).all()
    assert abs(env.tally[0, T_PASSIVE_S].item() - MAX_STEPS * DT) < 0.05
    assert info.end_reason[0].item() == END_PTS_LOSS and info.end_reason[1].item() == END_PTS_DRAW
    assert PASSIVE_PTS_PER_S * MAX_STEPS * DT > 4.0 > PTS_MARGIN
    assert abs(env.rw[0, RW_TIME].item() + R_DAMAGE * PASSIVE_PTS_PER_S * env.tally[0, T_PASSIVE_S].item()) < 1e-4


def test_backing_off_after_a_swing_is_paid_and_standing_still_is_charged():
    """saw-v9: for RETREAT_S after a swing lands, distance gained from them is paid (out to
    RETREAT_FT) and staying on their body is charged. All fight long, being slow is charged."""
    env = _env(n=3)
    _place(env, 20.0, 24.0, math.pi, 26.0, 24.0, 0.0)            # tail to them, 6 ft apart
    env.r_slow = 0.0
    env.retreat[0], env.retreat[1] = RETREAT_S, RETREAT_S
    env.last_dist[:] = 6.0
    env.ovx[0] = 10.0                                            # env 0 drives away, env 1 sits, env 2 drives away with no swing behind it
    env.ovx[2] = 10.0
    go = _act(env, blade=-1.0)
    go[0, 0], go[2, 0] = 1.0, 1.0
    env.step(go)
    assert env.rw[0, RW_TIME].item() > 0.5 * R_RETREAT_PER_FT * 10.0 * CTRL_DT
    assert abs(env.rw[2, RW_TIME].item()) < 1e-6
    assert env.retreat[0].item() < RETREAT_S and RETREAT_FT < PASSIVE_FT
    slow = _env(n=2)
    _place(slow, 20.0, 24.0, 0.0, 30.0, 24.0, 0.0)
    slow.ovx[1] = V_MAX_FT
    slow.step(_act(slow, blade=-1.0, throttle=torch.tensor([0.0, 1.0], device=DEVICE)))
    assert slow.rw[0, RW_TIME].item() < -0.9 * R_SLOW_PER_S * CTRL_DT
    assert slow.rw[1, RW_TIME].item() > -0.35 * R_SLOW_PER_S * CTRL_DT


# ─── Hazards and walls ────────────────────────────────────────────────────

def test_an_attack_is_in_when_the_blade_did_real_damage():
    """What the curriculum and the audit count: thumping them with the arm,
    or tapping with a slow blade, is not an attack."""
    tally = torch.zeros(5, len(TALLY_NAMES), device=DEVICE)
    tally[0, T_DEALT] = ATTACK_IN_DMG                                      # one fair hit
    tally[1, T_DEALT] = ATTACK_IN_DMG - 0.01                               # taps that add up to less
    tally[2, T_DEALT], tally[2, T_BLOW] = 4 * DMG_HAMMER_BLOW, 4.0         # four blows of the arm, no blade
    tally[3, T_DEALT], tally[3, T_BLOW] = ATTACK_IN_DMG + DMG_HAMMER_BLOW, 1.0
    assert attack_in(tally).tolist() == [True, False, False, True, False]
    assert attack_in(tally.cpu().numpy()).tolist() == [True, False, False, True, False]
    assert ATTACK_IN_DMG < DMG_HAMMER_SAW * DR_BLADE_TOP[0] ** 2, "a full hit with the blade capped would not count"


def test_walls_are_solid_for_every_part_of_a_body_and_smooth():
    """saw-v9: the three bulges are wall. No corner of a body gets into a wall, whichever way the
    robot points and however hard it drives, and nothing is charged as hazard damage any more."""
    n = 6
    env = _env(n=n)
    ox = torch.tensor([3.0, 45.0, 27.0, 24.0, 2.5, 46.0], device=DEVICE)      # by each bulge, a plain wall, two corners
    oy = torch.tensor([24.5, 24.5, 7.0, 46.0, 45.5, 6.0], device=DEVICE)
    oa = torch.tensor([math.pi * 0.8, 0.3, -math.pi / 2 + 0.4, math.pi / 2 - 0.5, math.pi * 0.75, -0.7], device=DEVICE)
    _place(env, ox, oy, oa, 24.0, 24.0, 0.0)
    env.ovx[:] = 15.0
    for _ in range(40):
        _, _, done, _ = env.step(_act(env, blade=-1.0, throttle=1.0))
        c, s = torch.cos(env.oa).unsqueeze(1), torch.sin(env.oa).unsqueeze(1)
        u, w = env.o_len.unsqueeze(1) * env._rim_u, env.o_wid.unsqueeze(1) * env._rim_v
        px, py = env.ox.unsqueeze(1) + c * u - s * w, env.oy.unsqueeze(1) + s * u + c * w
        deep = torch.stack(saw_walls.penetration(px, py)).max().item()
        assert deep < 0.35, f"a body is {deep:.2f} ft inside a wall"      # one physics step of travel at most
    assert not done.any()
    assert env.o_on_wall.any(), "nobody reached a wall"
    assert (env.tally[:, T_HAZARD_S] == 0).all()
    # The bulges stand where the team drew them and the fourth wall is plain
    y = torch.tensor([24.5, 5.0], device=DEVICE)
    assert saw_walls.inset_x0(y)[0] > 2.0 and saw_walls.inset_x0(y)[1] == 0.0
    assert saw_walls.inset_x1(y)[0] > 2.5 and saw_walls.inset_y0(torch.tensor([27.0]))[0] > 5.5
    wall, dist = saw_walls.raster()
    assert wall[:4].all() and not wall[30:, 5:40].any() and dist[24, 24] > 15
    assert torch.equal(env.hazards, wall)


def test_a_robot_can_scrape_along_a_wall_under_the_saw_rules():
    """Driven into a wall at an angle, a robot keeps most of its motion along
    the wall under the saw rules. Under the classic rules it stops dead."""
    hi = ARENA_FT - WALL_CLAMP_IN
    par = RobotParams(1)
    _nominal(par)

    def run(scrape):
        x, y, a = _full(1, hi - 0.02), _full(1, 24.0), _full(1, math.pi / 4)   # heading up and into the east wall
        v, omega, drift = _full(1, 12.0), _full(1, 0.0), _full(1, 0.0)
        drifting = torch.zeros(1, dtype=torch.bool, device=DEVICE)
        for _ in range(10):
            x, y, a, v, omega, drifting, drift, _ = integrate_robot(
                x, y, a, v, omega, drifting, drift, _full(1, 0.6), _full(1, 0.0), par, _full(1, 1.0),
                scrape_walls=scrape)
        along = (torch.sin(a) * v + torch.cos(a) * drift).item()               # speed along the wall
        return x.item(), y.item() - 24.0, along

    x_c, moved_c, along_c = run(False)
    x_s, moved_s, along_s = run(True)
    assert abs(x_c - hi) < 1e-3 and abs(x_s - hi) < 1e-3, "a robot went through the wall"
    assert along_c < 1.5, along_c
    assert along_s > 4.0 and moved_s > 2.0 * moved_c, (along_s, moved_s, moved_c)


def test_jammed_against_a_wall_a_robot_can_barely_turn():
    env = _env(n=2)
    hi = ARENA_FT - WALL_CLAMP_IN
    _place(env, 24.0, 24.0, 0.0, 24.0, 12.0, math.pi / 2)
    env.oy[0], env.ey[0] = hi, hi - 3.9                       # env 0: on the top wall with the other robot on it
    env.squeeze_yaw[:] = 0.4
    env.manual_cmd[:, 0] = 0.4                                # which keeps leaning in
    spin = _act(env, turn=1.0, blade=-1.0)
    for t in range(30):
        env.step(spin)
        if t < 12:                                            # until our turning body levers the other robot off
            assert env.o_squeezed[0].item() and not env.o_squeezed[1].item(), (t, env.o_squeezed)
            assert env.o_omega[0].item() <= 0.4 * YAW_RATE_MAX + 0.05, (t, env.o_omega)
        if t == 11:
            assert env.o_omega[1].item() > 0.4 * YAW_RATE_MAX + 1.0, env.o_omega
    assert abs(env.o_omega[1].item() - YAW_RATE_MAX) < 0.05, env.o_omega
    assert env.tally[0, T_PINNED_S].item() > 0.35 and env.tally[1, T_PINNED_S].item() == 0, env.tally[:, T_PINNED_S]
    assert env.rw[0, RW_WALL].item() < 0 and env.rw[1, RW_WALL].item() == 0, env.rw[:, RW_WALL]


# ─── Starts ───────────────────────────────────────────────────────────────

def test_bad_spot_starts_are_legal_and_about_as_common_as_configured():
    torch.manual_seed(0)
    n = 8192
    lo, hi = WALL_CLAMP_IN, ARENA_FT - WALL_CLAMP_IN
    env = SawEnv(n, stage="combat", p=1.0, auto_reset=False, scenario=Scenario(from_bell=False))
    for t in (env.ox, env.oy, env.ex, env.ey):
        assert (t >= lo - 1e-4).all() and (t <= hi + 1e-4).all()
    # Some starts are dropped because the place was taken or over a hazard,
    # so each kind comes out a little under its configured share
    for spot, share in ((PINNED, SPAWN_PINNED), (CORNERED, SPAWN_CORNERED),
                        (CONTACT, SPAWN_CONTACT), (PINNING, SPAWN_PINNING)):
        got = (env.spot == spot).float().mean().item()
        assert 0.6 * share < got < 1.1 * share, (spot, got, share)
    on_wall = env.o_on_wall.float().mean().item()
    assert 0.8 * (env.spot == PINNED).float().mean().item() < on_wall < 1.5 * SPAWN_PINNED, on_wall
    assert env.o_on_hazard.any(), "no fight opened pinned on a hazard wall"
    close = (env.spot == PINNED) | (env.spot == CONTACT) | (env.spot == PINNING)
    gap = torch.sqrt((env.ex - env.ox) ** 2 + (env.ey - env.oy) ** 2) - (env.o_len + env.e_len)
    assert gap[close].max().item() < 4.0, "an in-contact start left the robots far apart"

    drills = SawEnv(n, stage="pretrain", p=0.0, auto_reset=False)
    d = torch.sqrt((drills.ex - drills.ox) ** 2 + (drills.ey - drills.oy) ** 2)
    assert (d < 9.0).float().mean().item() < 0.005 and not drills.o_on_wall.any()
    assert not drills.o_on_hazard.any()


def test_the_robot_is_set_up_as_a_hammer_saw_only():
    """The team runs it as a hammer saw: no fight is set up as a vert, and no
    arm is dealt dead (the driver takes over if it dies). Parked, the blade
    touches nothing. A vert setup can still be dealt: it locks the arm and
    makes the parked blade the weapon."""
    torch.manual_seed(0)
    assert VERT_SETUP_P == 0.0 and ARM_JAM_P == 0.0 and ARM_JAM_P_DRILLS == 0.0
    for stage in ("combat", "pretrain"):
        env = SawEnv(4096, stage=stage, p=1.0, auto_reset=False, arm_gate=False)
        for w in (env.saw_o, env.saw_e):
            assert not w.vert.any() and not w.jammed.any()
            assert not w.rear_low.any(), "a parked hammer saw is not a vert"
    for _ in range(20):
        env.step(_act(env, arm=1.0))
    assert (env.saw_o.arm > 0.9).all()

    env = _env()
    env.saw_o.vert[:] = False
    _place(env, 24.0, 24.0, 0.0, 24.0 - 4.3, 24.0, 0.0)       # the target sits on our tail
    env.saw_o.blade[:] = 1.0
    for _ in range(10):
        _, _, _, info = env.step(_act(env, arm=-1.0, blade=1.0))
    for col in (T_VERT, T_DEALT, T_CLASH):
        assert (info.tally[:, col] == 0).all(), "the parked blade of a hammer saw did something"
    assert (env.saw_o.blade > 0.95).all() and not env.saw_o.bound.any() and (env.e_stun == 0).all()

    n = 4096
    w = Weapon(n)
    w.reset(torch.arange(n, device=DEVICE), torch.full((n,), W_SAW, dtype=torch.long, device=DEVICE),
            _full(n, 2.0), _full(n, 2.0), jam_p=0.3, vert_p=0.25)
    assert abs(w.vert.float().mean().item() - 0.25) < 0.03
    assert w.jammed[w.vert].all(), "a robot set up as a vert has its arm locked"
    assert abs(w.jammed[~w.vert].float().mean().item() - 0.3) < 0.04
    assert (w.rear_low == w.vert).all()
    _run(w, arm=1.0, seconds=0.5)
    assert (w.arm[w.jammed] == 0).all() and (w.arm[~w.jammed] > 0.9).all()


def test_only_the_end_that_carries_the_weapon_is_worth_pointing_at_them():
    env = _env(n=4)
    _place(env, 20.0, 24.0, 0.0, 26.0, 24.0, 0.0)             # the target is dead ahead, 6 ft off
    env.oa[1], env.oa[3] = math.pi, math.pi                   # envs 1 and 3 have their tail on it instead
    env.saw_o.vert[:2], env.saw_o.vert[2:] = False, True      # two hammer saws, two set up as verts
    phi = env._potential()
    assert phi[0].item() > phi[1].item() + 1e-3               # a hammer saw: only the nose
    assert phi[3].item() > phi[2].item() + 1e-3               # a vert: only the tail
    assert abs(phi[0].item() - phi[3].item()) < 1e-6


def test_with_the_upper_hand_a_fight_opens_with_our_nose_on_them():
    torch.manual_seed(0)
    env = SawEnv(8192, stage="combat", p=1.0, auto_reset=False)
    k = env.spot == PINNING
    toward = torch.cos(env.oa) * (env.ex - env.ox) + torch.sin(env.oa) * (env.ey - env.oy)
    assert k.sum().item() > 200 and (toward[k] > 0).all(), "a hammer saw opened a pin with its back to them"


def test_the_best_drivers_are_a_little_worse_than_the_scale_allows():
    """The team asked for it. The top level draws its drivers at SHARP_TOP of
    the sharp scale: fewer guard, they react later and none is as fast as
    1.5x us. The first level is what it was, and so is the classic field."""
    torch.manual_seed(0)
    n = 8192
    top = SawEnv(n, stage="combat", p=1.0, auto_reset=False).opponents
    assert 0.5 < SHARP_TOP < 0.9
    assert abs(top.guards.float().mean().item() - GUARD_SHARE[1] * SHARP_TOP) < 0.03
    slow, fast = SHARP["SPEED_RATIO"]
    most = slow[1] + (fast[1] - slow[1]) * SHARP_TOP
    assert top.speed_ratio.max().item() <= most + 1e-5 < fast[1]
    late, soon = SHARP["REACT_S"]
    assert top.react.min().item() >= late[0] + (soon[0] - late[0]) * SHARP_TOP - 1e-5 > soon[0]
    first = SawEnv(n, stage="combat", p=0.0, auto_reset=False).opponents
    assert first.react.min().item() >= late[0] - 1e-5 and not first.guards.any()
    classic = make_env("classic", n, stage="combat", p=1.0, auto_reset=False).opponents
    assert classic.react.min().item() < 0.30, "the classic experts got slower"


# ─── What goes wrong with our own robot ───────────────────────────────────

def test_saw_rules_randomize_wider_and_leave_the_classic_ranges_alone():
    saw, classic = make_env("saw", 8, auto_reset=False), make_env("classic", 8, auto_reset=False)
    assert classic.ranges is RANGES and RANGES["grip"] == DR_GRIP
    for key, rng in SAW_DR.items():
        assert key in RANGES and saw.ranges[key] == rng, key
    assert saw.ranges["grip"][0] < RANGES["grip"][0] and saw.ranges["tau"][1] > RANGES["tau"][1]
    assert saw.ranges["hdg_bias"][1] > 2 * RANGES["hdg_bias"][1]
    assert saw.observer.ranges is saw.ranges and classic.observer.ranges is RANGES
    assert saw.opponents.sharp and not classic.opponents.sharp
    assert classic.observer.lost is None and classic.observer.lost_own is None


def test_reverse_can_be_a_slower_gear():
    env = _env(n=2)
    _place(env, 34.0, 24.0, 0.0, 10.0, 40.0, 0.0)
    env.rev[0], env.rev[1] = 0.75, 1.0
    back = _act(env, throttle=-1.0, blade=-1.0)
    for _ in range(25):
        env.step(back)
    assert abs(env.ovx[0].item() + 15.0) < 0.5 and abs(env.ovx[1].item() + 20.0) < 0.5, env.ovx
    env.ovx[:] = 0.0
    for _ in range(12):
        env.step(_act(env, throttle=1.0, blade=-1.0))
    assert abs(env.ovx[0].item() - env.ovx[1].item()) < 1e-4, "forward gear must not be touched"


def test_a_lost_command_leaves_the_last_one_standing():
    env = _env(n=2)
    _place(env, 14.0, 24.0, 0.0, 40.0, 40.0, 0.0)
    env.cmd_drop[0] = 1.0                                     # env 0's radio delivers nothing
    go = _act(env, throttle=1.0, blade=-1.0)
    for _ in range(10):
        env.step(go)
    assert (env.sent[0] == 0).all() and torch.equal(env.sent[1], go[1])
    assert abs(env.ovx[0].item()) < 1e-6 and env.ovx[1].item() > 5.0, env.ovx


def test_a_lost_track_freezes_that_robot_where_it_was_last_seen():
    torch.manual_seed(0)                                      # the camera's delay is drawn: a long one shows little motion
    env = _env(n=4)
    _place(env, 14.0, 24.0, 0.0, 34.0, 30.0, math.pi)
    env.observer.reset(env._ar, env._truth())
    env.lost_t[:] = 5.0
    env.lost_t[3] = 0.0                                       # env 3: the camera has both
    env.lost_who[:] = torch.tensor([0, 1, 2, 0], device=DEVICE)   # them, us, both
    env.manual_cmd[:, 0] = 0.5                                # they drive, and so do we
    go = _act(env, throttle=0.5, blade=-1.0)
    first, _, _, _ = env.step(go)
    for _ in range(9):
        obs, _, _, _ = env.step(go)
    us   = (obs[:, 0:2] - first[:, 0:2]).abs().amax(dim=1).tolist()
    them = (obs[:, 14:16] - first[:, 14:16]).abs().amax(dim=1).tolist()
    assert them[0] == 0 and us[0] > 0.02, (us, them)
    assert us[1] == 0 and them[1] > 0.02, (us, them)
    assert us[2] == 0 and them[2] == 0, (us, them)
    assert us[3] > 0.02 and them[3] > 0.02, (us, them)


def test_a_hit_shakes_the_heading_estimate():
    n = 128
    env = _env(n=n, tactic=PURSUER, stage="combat")
    _place(env, 24.0, 24.0, 0.0, 24.0, 24.0 - 4.3, math.pi / 2)   # its nose is in our right side
    env.saw_e.blade[:], env.saw_e.power[:] = 1.0, 1.0
    env.observer.bias[:, 4], env.observer.bias_sigma[:, 4] = 0.0, 0.0
    env.hdg_kick[: n // 2], env.hdg_kick[n // 2:] = 0.3, 0.0
    _, _, _, info = env.step(_act(env, blade=-1.0))
    assert (info.tally[:, T_TAKEN] == 1).all()
    off = env.observer.bias[:, 4]
    assert (off[n // 2:] == 0).all()
    assert 0.2 < off[: n // 2].std().item() < 0.4, off[: n // 2].std()


def test_the_arm_and_blade_readings_go_missing_separately():
    torch.manual_seed(0)
    env = SawEnv(8192, stage="pretrain", auto_reset=False)
    no_arm, no_blade = env.no_arm_reading, env.no_blade_reading
    assert abs(no_arm.float().mean().item() - NO_ARM_READING_P) < 0.03
    assert abs(no_blade.float().mean().item() - NO_BLADE_READING_P) < 0.03
    both = (no_arm & no_blade).float().mean().item()
    assert abs(both - NO_ARM_READING_P * NO_BLADE_READING_P) < 0.03, both   # one does not decide the other
    obs = env._observe(env._ar)
    assert (obs[no_arm, 26] == -1).all() and (obs[~no_arm, 26] > -0.5).all()
    assert (obs[no_blade, 27] == -1).all() and (obs[~no_blade, 27] > -0.5).all()


def test_throttle_and_turn_together_are_scaled_back_to_what_the_treads_have():
    env = _env(n=5)
    _place(env, 14.0, 24.0, 0.0, 40.0, 40.0, 0.0)
    env.tread_k[:] = torch.tensor([0.5, 0.0, 0.5, 0.0, 0.0], device=DEVICE)
    env.saw_o.blade[:], env.saw_o.sag[:] = 0.0, 0.4           # the blade spins up slower the harder the drive works
    a = _act(env, blade=1.0)
    a[0, 0], a[0, 1] = 0.9, 0.8                               # asks one tread for 1.3 of its top speed
    a[1, 0], a[1, 1] = 0.9 / 1.3, 0.8 / 1.3                   # the same, already scaled back, on treads without a limit
    a[2, 0], a[2, 1] = 0.5, 0.6                               # 0.8: within what the treads have
    a[3, 0], a[3, 1] = 0.5, 0.6
    a[4, 0], a[4, 1] = 0.9, 0.8                               # what env 0 asked for, with no limit
    for _ in range(20):
        env.step(a)
    pose = torch.stack([env.ox, env.oy, env.oa, env.ovx, env.o_omega, env.saw_o.blade], dim=1)
    assert torch.allclose(pose[0], pose[1], atol=1e-4), (pose[0], pose[1])
    assert torch.allclose(pose[2], pose[3], atol=1e-6), (pose[2], pose[3])
    assert env.o_omega[4].item() > env.o_omega[0].item() + 0.5, env.o_omega    # without the limit it turns harder
    assert env.saw_o.blade[4].item() < env.saw_o.blade[0].item() - 0.01, env.saw_o.blade   # and loads the blade more

    torch.manual_seed(0)
    env = SawEnv(4096, stage="combat", p=1.0, auto_reset=False)
    scale = YAW_RATE_MAX / V_MAX_FT
    lo, hi = DR_TREAD_SHARE[0] * DR_OUR_HALF_WID[0] * scale, DR_TREAD_SHARE[1] * DR_OUR_HALF_WID[1] * scale
    assert env.tread_k.min().item() >= lo - 1e-6 and env.tread_k.max().item() <= hi + 1e-6
    assert 0.40 < env.tread_k.mean().item() < 0.50            # a full turn takes about half of a tread's top speed
    assert (env.e_tread_k == 0).all(), "a scripted opponent's driver model keeps its own limits"


def test_the_speed_input_is_our_own_track_differenced_not_the_wheels():
    assert not make_env("classic", 2, auto_reset=False).observer.speed_from_track
    env = _env(n=2)
    assert env.observer.speed_from_track and env.en_observer.speed_from_track
    _place(env, 10.0, 24.0, 0.0, 40.0, 40.0, 0.0)
    o = env.observer
    o.reset(env._ar, env._truth())
    for t in (o.pos_sigma, o.other_sigma, o.hdg_sigma, o.drop_p, o.outlier_p, o.bias, o.bias_sigma):
        t[:] = 0.0
    o.lag[:] = 1
    o.fix = env._truth()[:, [0, 1, 5, 6]].clone()
    o.own_hist[:] = o.fix[:, 0:2].unsqueeze(1)
    env.lost_t[1], env.lost_who[1] = 60.0, 1                  # env 1: the camera has lost us for the whole test
    go = _act(env, throttle=0.5, blade=-1.0)
    for _ in range(12 + ENEMY_VEL_WINDOW):
        obs, _, _, _ = env.step(go)
    assert abs(env.ovx[0].item() - 10.0) < 0.3 and abs(env.ovx[1].item() - 10.0) < 0.3, env.ovx
    # Seen, the input is the track's speed (the sensing delay jitters by a
    # physics step, which is worth 0.1 at this speed). Lost, the track stands
    # still and so does the input, whatever the wheels are doing.
    assert abs(obs[0, 4].item() - 10.0 / V_MAX_FT) < 0.12, obs[:, 4]
    assert obs[1, 4].item() == 0.0, obs[:, 4]


# ─── What the policy sees and does ────────────────────────────────────────

def test_each_ruleset_has_its_own_observation_and_action():
    saw, classic = make_env("saw", 4, auto_reset=False), make_env("classic", 4, auto_reset=False)
    assert isinstance(saw, SawEnv) and type(classic) is BattleEnv
    assert (saw.obs_dim, saw.act_dim) == (SAW_OBS_DIM, SAW_ACT_DIM) == (44, 4)
    assert (classic.obs_dim, classic.act_dim) == (OBS_DIM, ACT_DIM)
    assert saw.physics_version != classic.physics_version
    obs, _, _, info = saw.step(torch.zeros(4, SAW_ACT_DIM, device=DEVICE))
    assert obs.shape == (4, SAW_OBS_DIM) and info.final_obs.shape == (4, SAW_OBS_DIM)
    assert info.tally.shape == (4, len(TALLY_NAMES)) and saw.tally_names == tuple(TALLY_NAMES) and TALLY_N == 20
    assert torch.equal(info.fight_s, saw.horizon.float() * DT) and info.hdg_mode.shape == (4,)
    _, _, _, info = classic.step(torch.zeros(4, ACT_DIM, device=DEVICE))
    assert info.tally is None and classic.fight_stats() == {} and info.fight_s is None and info.hdg_mode is None
    try:
        make_env("sumo", 4)
        raise AssertionError("an unknown ruleset was accepted")
    except ValueError:
        pass


def test_the_policy_sees_its_own_weapon_and_only_the_card_of_theirs():
    env = _env(n=64, tactic=PURSUER, stage="combat")
    _place(env, 20.0, 24.0, 0.0, 30.0, 30.0, 0.0)
    env.saw_o.arm[:], env.saw_o.blade[:] = 0.5, 0.7
    env.prev_action[:, 2], env.prev_action[:, 3] = 1.0, -0.4
    env.observer.reset(env._ar, env._truth())

    def first_obs():
        torch.manual_seed(3)                                  # pins the sensor noise
        return env._observe(env._ar)

    obs = first_obs()
    assert obs.shape == (64, SAW_OBS_DIM)
    assert (obs[:, 24] == 1.0).all() and torch.allclose(obs[:, 25], _full(64, -0.4))
    assert (obs[:, 26] - 0.5).abs().max().item() < 5 * ARM_NOISE
    assert (obs[:, 27] - 0.7).abs().max().item() < 5 * BLADE_NOISE
    assert (obs[:, 28] == 0).all() and (obs[:, 29] == 0).all()   # a 4 ft square robot
    se = env.saw_e
    se.blade[:], se.power[:], se.rebite[:] = 0.1, 0.3, 0.2
    se.kind[:], se.fore[:], se.rad[:] = W_HORIZ, 1.0, 2.4
    env.e_len[:], env.e_wid[:] = 2.5, 1.3
    assert (env.hdg_mode == 0).all()
    assert torch.equal(first_obs(), obs), "the enemy's true weapon or size leaked into the observation"
    env.field.card_kind[:], env.field.card_len[:] = W_HORIZ, 2.5              # what the team wrote down does show
    new = first_obs()
    assert (new[:, 35] == 1).all() and (new[:, 33] == 0).all() and torch.allclose(new[:, 36], _full(64, 0.5))
    assert torch.equal(new[:, :30], obs[:, :30])


def test_self_play_is_refused_under_saw_v8():
    """A copy of our robot would need a gate, a card and a camera of its own."""
    from opponent_selfplay import SelfPlayController
    env = SawEnv(8, stage="combat", p=1.0, auto_reset=False)
    assert env.selfplay is None and env.selfplay_frac == 0.0
    env.selfplay, env.selfplay_frac = None, 0.0               # what BattleEnv.__init__ does
    for set_it in (lambda: setattr(env, "selfplay", SelfPlayController(8, env.obs_dim, env.act_dim)),
                   lambda: setattr(env, "selfplay_frac", 0.5)):
        try:
            set_it()
            raise AssertionError("self-play was accepted under the saw rules")
        except ValueError:
            pass


def test_shaping_telescopes_under_the_saw_rules():
    """Same guarantee as under the classic rules: discounted shaping over any
    stretch sums to gamma^T * phi(end) - phi(start), so it cannot be farmed.
    Here no fight ends inside the stretch, whatever happens in it. The bell
    is the end of the fight, so the potential after it is 0."""
    torch.manual_seed(0)
    n = 256
    env = SawEnv(n, stage="combat", p=0.5, auto_reset=False)
    env.reset()
    phi0 = env.phi.clone()
    disc = torch.zeros(n, device=DEVICE)
    was = torch.zeros(n, device=DEVICE)
    T = 60
    for t in range(T):
        env.step(torch.rand(n, SAW_ACT_DIM, device=DEVICE) * 2 - 1)
        now = env.rw[:, RW_SHAPE]
        disc += (env.gamma ** t) * (now - was)
        was = now.clone()
    assert not env.done.any()
    expected = (env.gamma ** T) * env.phi - phi0
    assert env.gamma == GAMMA_SAW != GAMMA
    assert torch.allclose(disc, expected, atol=3e-4), (disc - expected).abs().max()
    env.steps[:] = env.horizon - PHYS_PER_CTRL
    before = env.phi.clone()
    _, _, done, info = env.step(torch.zeros(n, SAW_ACT_DIM, device=DEVICE))
    assert done.all() and info.timeout.all()
    assert torch.allclose(env.rw[:, RW_SHAPE] - was, -before, atol=1e-5), "the bell must be worth nothing"


def test_the_saw_bell_is_terminal_and_classic_is_not():
    saw, classic = make_env("saw", 4, auto_reset=False), make_env("classic", 4, auto_reset=False)
    assert saw.timeout_bootstrap is False and classic.timeout_bootstrap is True
    assert saw.gamma == GAMMA_SAW and classic.gamma == GAMMA


def test_observation_columns_30_to_43():
    torch.manual_seed(0)
    n = 4096
    env = SawEnv(n, stage="combat", p=0.7, auto_reset=True)
    obs = env.reset()
    for _ in range(5):
        obs, _, _, _ = env.step(torch.rand(n, SAW_ACT_DIM, device=DEVICE) * 2 - 1)
    g, F = env.gate, env.field
    assert torch.allclose(obs[:, 30], ((g.budget - g.fires).clamp(min=0).float() / 10).clamp(0, 3))
    assert torch.allclose(obs[:, 31], (env.horizon - env.steps).clamp(min=0).float() * DT / 180.0)
    assert torch.equal(obs[:, 32], env.gate_open.float())
    for col, kind in zip((33, 34, 35), SPINNERS):
        assert torch.equal(obs[:, col], (env.saw_e.kind == kind).float())
    for col, card, true in ((36, F.card_len, env.e_len), (37, F.card_wid, env.e_wid)):
        assert torch.allclose(obs[:, col], card - 2.0)
        assert ((card / true - 1.0).abs() <= 0.1 + 1e-5).all() and (card / true).std().item() > 0.03
    # Which way they point: the camera says in the share of fights config gives each mode
    for mode, share in enumerate(HDG_SENSE_P):
        assert abs((env.hdg_mode == mode).float().mean().item() - share) < 0.03
    assert HDG_SENSE_P[0] < 1.0 or (obs[:, 38:44] == 0).all()

    for name, mode in (("axis", 1), ("axis_nose", 2)):
        env = SawEnv(n, stage="combat", p=0.7, auto_reset=False, scenario=Scenario(heading_mode=name, camera_period=4))
        assert (env.hdg_mode == mode).all()
        obs, drops, errs = env.reset(), [], []
        env.track_loss_hz[:] = 0.0                            # a robot the camera has lost keeps its old fix, the other does not
        for _ in range(12):
            before = obs
            obs, _, _, _ = env.step(torch.rand(n, SAW_ACT_DIM, device=DEVICE) * 2 - 1)
            h, fresh = obs[:, 38:44], env.observer.e_fresh
            assert torch.equal(h[~fresh], before[~fresh, 38:44]), "the reading changed between camera frames"
            axis, nose = h[:, 0] > 0.5, h[:, 3] > 0.5
            assert (h[~axis, 0:3] == 0).all() and (h[~nose, 3:6] == 0).all()
            assert torch.allclose((h[axis, 1] ** 2 + h[axis, 2] ** 2), torch.ones(int(axis.sum()), device=DEVICE), atol=1e-4)
            assert nose.any() == (mode == 2) and (axis | ~nose).all()
            drops.append((~axis[fresh]).float().mean().item())
            # The reading is the angle between the line from them to us and their heading in the same camera frame
            fix = env.observer.fix
            true = angle_wrap(torch.atan2(fix[:, 1] - fix[:, 3], fix[:, 0] - fix[:, 2]) - env.observer.other_hdg)
            got = 0.5 * torch.atan2(h[:, 2], h[:, 1])
            err = angle_wrap(2 * (got - true)) / 2
            errs.append((err / env.hdg_sigma)[axis & fresh])
        assert abs(sum(drops) / len(drops) - HDG_SENSE_DROP_P) < 0.02, drops
        assert abs(torch.cat(errs).std().item() - 1.0) < 0.15, torch.cat(errs).std()


def test_a_foul_landing_costs_r_foul():
    env = _env(tactic=PURSUER, stage="combat")
    _place(env, 24.0, 24.0, 0.0, 24.0, 40.0, 0.0)             # nobody near our nose: the hammer finds the floor
    env.saw_o.blade[:] = 1.0
    for _ in range(10):
        env.step(_act(env, arm=1.0, blade=1.0))
    assert torch.allclose(env.tally[:, T_MISS], torch.ones(env.n, device=DEVICE), atol=0.01)
    assert torch.allclose(env.rw[:, RW_OUTCOME], -R_FOUL * env.tally[:, T_MISS], atol=1e-6), env.rw[:, RW_OUTCOME]


def test_stalled_landings_are_not_misses():
    """A swing lands in exactly one place: their body off the weapon side,
    their weapon side, or the floor."""
    env = _env(tactic=PURSUER, stage="combat")
    _place(env, 20.0, 24.0, 0.0, 24.3, 24.0, math.pi)         # nose to nose: the hammer comes down on their front
    env.saw_o.blade[:], env.saw_e.blade[:] = 1.0, 0.0
    for _ in range(10):
        env.step(_act(env, arm=1.0, blade=1.0))
    t = env.tally
    assert torch.allclose(t[:, T_FOUL], torch.ones(env.n, device=DEVICE), atol=0.01), t[:, T_FOUL]
    assert (t[:, T_MISS] == 0).all() and (t[:, T_BLOW] == 0).all() and (t[:, T_SOLID] == 0).all()
    assert (env.rw[:, RW_OUTCOME] <= -R_FOUL * 0.99).all()

    env = _env(tactic=PURSUER, stage="combat")
    _place(env, 20.0, 24.0, 0.0, 24.3, 24.0, 0.0)             # the same on their tail
    env.saw_o.blade[:], env.saw_e.blade[:] = 1.0, 0.0
    for _ in range(10):
        env.step(_act(env, arm=1.0, blade=1.0))
    t = env.tally
    assert torch.allclose(t[:, T_BLOW], torch.ones(env.n, device=DEVICE), atol=0.01)
    assert (t[:, T_MISS] == 0).all() and (t[:, T_FOUL] == 0).all() and (t[:, T_SOLID] == 1).all()
    assert torch.allclose(t[:, T_BLOW] + t[:, T_FOUL] + t[:, T_MISS], torch.ones(env.n, device=DEVICE), atol=0.01)


def test_reach_and_from_bell():
    torch.manual_seed(0)
    env = SawEnv(8192, stage="combat", p=1.0, auto_reset=True)
    env.reset_at(env._ar)                                     # fights as they open, before the clocks are staggered
    lo, hi = DR_HAMMER_REACH
    assert env.reach.min().item() >= lo - 1e-5 and env.reach.max().item() <= hi + 1e-5 and env.reach.std().item() > 0.1
    assert torch.allclose(env.saw_o.fore, env.o_len + env.reach)
    bell, long = env.from_bell, env.horizon == round(180.0 / DT)
    assert long.any() and bell[long].all(), "a three minute fight starts from the bell"
    assert abs(bell[~long].float().mean().item() - BELL_SHORT_P) < 0.03
    spinner = env.saw_e.spinner
    assert env.saw_o.blade[bell].max().item() <= 0.15 + 1e-5 and env.saw_o.blade[~bell].min().item() >= 0.3 - 1e-5
    assert env.saw_e.blade[bell & spinner].max().item() <= 0.15 + 1e-5
    assert env.saw_e.blade[~bell & spinner].min().item() >= 0.3 - 1e-5
    assert (env.ovx[bell] == 0).all() and (env.evx[bell] == 0).all() and (env.spot[bell] == 4).all()
    to_them = torch.atan2(env.ey - env.oy, env.ex - env.ox)
    assert angle_wrap(env.oa - to_them)[bell].abs().max().item() <= BELL_FACE_RAD + 1e-4
    assert angle_wrap(env.ea - to_them - math.pi)[bell].abs().max().item() <= BELL_FACE_RAD + 1e-4
    assert (env.spot[~bell] != 4).any()


def test_scenario_overrides_reach_the_env():
    torch.manual_seed(0)
    sc = Scenario(fight_s=30.0, swings=7, reach_ft=0.8, heading_mode="axis", arm_reading=True, blade_reading=False,
                  camera_period=3, from_bell=True, weapon_shares=(1, 1, 1), traves_share=0.0)
    env = SawEnv(4096, stage="combat", p=1.0, auto_reset=True, scenario=sc)
    assert (env.horizon == 3000).all() and (env.saw_o.swings == 7).all() and (env.gate.budget == 7).all()
    assert (env.tally[:, T_BUDGET] == 7).all()
    assert torch.allclose(env.reach, _full(4096, 0.8)) and (env.hdg_mode == 1).all()
    assert not env.no_arm_reading.any() and env.no_blade_reading.all()
    assert (env.observer.period == 3).all() and env.from_bell.all()
    armed = env.opponents.armed
    for kind in SPINNERS:
        assert abs((env.saw_e.kind[armed] == kind).float().mean().item() - 1 / 3) < 0.03
    assert (SawEnv(64, stage="combat", scenario=Scenario(from_bell=False), auto_reset=True).from_bell == False).all()  # noqa: E712
    for bad in (dict(heading_mode="north"), dict(camera_period=9), dict(weapon_shares=(0, 0, 0)), dict(p_range=(0.8, 0.2))):
        try:
            Scenario(**bad)
            raise AssertionError(f"Scenario accepted {bad}")
        except ValueError:
            pass


def test_a_dead_weapon_stops_and_never_respins():
    env = _env(tactic=PURSUER, stage="combat")
    _place(env, 10.0, 24.0, 0.0, 34.0, 30.0, math.pi)
    env.saw_e.blade[:] = 1.0
    env.field.dead_at[:] = 4
    env.step(_act(env))
    assert not env.saw_e.dead.any() and (env.saw_e.blade > 0.9).all()
    for _ in range(40):
        env.step(_act(env))
        assert (env.steps < 5).any() or (env.saw_e.blade == 0).all()
    assert env.saw_e.dead.all() and (env.saw_e.blade == 0).all()
    drills = SawEnv(1024, stage="pretrain", auto_reset=True)
    assert (drills.field.dead_at == _NEVER).all(), "a drill target has no weapon to fail"


def test_staggered_first_fight_gets_a_pro_rata_budget():
    """Training joins every first fight part-way through its clock. It has
    that part of its swings left, and a weapon that was going to fail later
    still fails inside what remains."""
    torch.manual_seed(0)
    env = SawEnv(8192, stage="combat", p=1.0, auto_reset=True)
    left = (env.horizon - env.steps).float() / env.horizon.float()
    assert (env.steps > 0).any() and (left > 0).all()
    budget = env.gate.budget
    assert torch.equal(budget.float(), env.saw_o.swings) and torch.equal(env.tally[:, T_BUDGET], budget.float())
    assert (budget >= 1).all()
    hi = ARM_SWINGS_3MIN[-1][1][1]
    assert (budget.float() <= (hi * env.horizon.float() * DT / 180.0 + 1.0) * left + 1.0 + 1e-4).all()
    long = env.horizon == round(180.0 / DT)
    mean = sum(share * 0.5 * (a + b) for share, (a, b) in ARM_SWINGS_3MIN)
    rate = (budget.float() / left)[long & (left > 0.5)].mean().item()
    assert abs(rate - mean) < 1.0, rate
    at = env.field.dead_at
    timed = (at > 0) & (at < _NEVER)
    assert timed.any() and (at[timed] >= env.steps[timed]).all() and (at[timed] <= env.horizon[timed]).all()
    obs = env._observe(env._ar)
    assert torch.allclose(obs[:, 31], (env.horizon - env.steps).float() * DT / 180.0)


def test_action_rate_uses_the_gated_arm():
    """With the gate on, asking is free: the smoothness charge is on the arm
    the robot moves. A request that wags while the gate is shut costs what a
    steady one does."""
    totals = []
    for wag in (True, False):
        torch.manual_seed(0)
        env = SawEnv(8, stage="pretrain", auto_reset=False, arm_gate=True, scenario=Scenario(from_bell=True))
        env.ex[:], env.ey[:] = (env.ox + 20.0).clamp(max=44.0), env.oy       # far off: the gate stays shut
        env.ex[:] = torch.where((env.ex - env.ox).abs() < 10.0, env.ox - 20.0, env.ex)
        env.observer.reset(env._ar, env._truth())
        for k in range(20):
            arm = (1.0 if k % 2 else -1.0) if wag else -1.0
            env.step(_act(env, arm=arm, blade=1.0))
            assert not env.gate_open.any() and (env._arm_out == -1).all()
        assert (env.tally[:, T_FIRE] == 0).all() and (env.tally[:, T_REQ] == (10 if wag else 0)).all()
        totals.append(env.rw[:, RW_SMOOTH].clone())
    assert torch.allclose(totals[0], totals[1], atol=1e-7), totals


if __name__ == "__main__":
    _runner.run(globals())
