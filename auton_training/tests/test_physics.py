"""
Physics invariants for the shared robot model.

Run with pytest if you have it, or directly:
    python tests/test_physics.py
"""
import math

import _runner  # noqa: F401  (puts the repo root on sys.path)
import torch

from config import (
    DEVICE, DT, YAW_RATE_MAX, ALPHA_MAX_RPS2, WALL_CLAMP_IN, ARENA_FT, MAX_STEPS,
)
from environment import BattleEnv, RobotParams, integrate_robot, END_NONE, RW_WALL


def _env(n=4):
    """Manual-drive env with both robots on identical, nominal physics."""
    env = BattleEnv(n, stage="combat", p=1.0, auto_reset=False)
    env.manual_enemy = True
    for par in (env.our, env.enemy):
        _nominal(par)
    env.cmd_lag[:] = 0          # our commands reach the motors as fast as the enemy's
    env.wear_at[:] = MAX_STEPS + 1
    return env


def _nominal(par, v_max=20.0):
    par.v_max[:], par.yaw_cap[:] = v_max, YAW_RATE_MAX
    par.dalpha[:] = ALPHA_MAX_RPS2 * DT
    par.ctrl_alpha[:] = 0.05 / (0.05 + DT)
    par.bleed[:] = 0.35
    par.dv_max[:] = 77.0 * DT
    par.grip_s[:], par.grip_k[:] = 64.0, 19.0
    par.a_lat_cap[:] = 98.4
    par.yaw_gain[:], par.pull[:], par.deadband[:] = 1.0, 0.0, 0.0


def _place(env, ox, oy, oa, ex, ey, ea):
    """Put both robots at known poses, at rest."""
    env.ox[:], env.oy[:], env.oa[:] = ox, oy, oa
    env.ex[:], env.ey[:], env.ea[:] = ex, ey, ea
    for t in (env.ovx, env.evx, env.o_omega, env.e_omega, env.o_drift, env.e_drift):
        t[:] = 0.0
    env.done[:] = False
    env.steps[:] = 0
    env.prev_dist[:] = 99.0


def _drive(env, our_throttle, our_turn, en_throttle, en_turn, n_ctrl=1):
    act = torch.zeros(env.n, 2, device=DEVICE)
    act[:, 0], act[:, 1] = our_throttle, our_turn
    env.manual_cmd[:, 0], env.manual_cmd[:, 1] = en_throttle, en_turn
    for _ in range(n_ctrl):
        env.step(act)


# ─── Tests ────────────────────────────────────────────────────────────────

def test_both_robots_share_the_integrator():
    """Same command, same limits, mirrored start: our robot and the enemy
    must trace identical speed, yaw rate and (point-reflected) displacement."""
    env = _env(n=2)
    _place(env, ox=12.0, oy=24.0, oa=0.0, ex=36.0, ey=24.0, ea=math.pi)
    x0_o, y0_o, x0_e, y0_e = env.ox.clone(), env.oy.clone(), env.ex.clone(), env.ey.clone()
    for _ in range(40):
        _drive(env, 0.8, 0.6, 0.8, 0.6)
        assert torch.allclose(env.ovx, env.evx, atol=1e-3), (env.ovx, env.evx)
        assert torch.allclose(env.o_omega, env.e_omega, atol=1e-3)
    d_o = torch.stack([env.ox - x0_o, env.oy - y0_o])
    d_e = torch.stack([env.ex - x0_e, env.ey - y0_e])
    assert torch.allclose(d_o, -d_e, atol=1e-2), (d_o, d_e)
    assert env.done.sum() == 0


def test_integrator_is_deterministic_and_pure():
    par = RobotParams(1)
    _nominal(par)
    t = lambda v: torch.tensor([v], device=DEVICE)
    args = dict(x=t(10.0), y=t(10.0), a=t(0.3), v=t(5.0), omega=t(1.0),
                drifting=torch.tensor([False], device=DEVICE), drift_vel=t(0.0),
                throttle=t(1.0), target_omega=t(3.0), par=par, alive=t(1.0))
    before = {k: v.clone() for k, v in args.items() if torch.is_tensor(v)}
    out1 = integrate_robot(**args)
    out2 = integrate_robot(**args)
    for k, v in before.items():
        assert torch.equal(v, args[k]), f"integrate_robot mutated input {k}"
    for a, b in zip(out1, out2):
        assert torch.equal(a, b)


def test_turn_bleed_does_not_compound():
    """Full throttle through a hard turn settles near
    v_max * (1 - bleed * turn_frac) instead of decaying toward zero."""
    env = _env(n=1)
    env.our.grip_s[:] = 1e6                     # isolate the bleed from the slip model
    _place(env, 24.0, 24.0, 0.0, 40.0, 40.0, 0.0)
    _drive(env, 1.0, 1.0, 0.0, 0.0, n_ctrl=40)
    turn_frac = env.o_omega.abs() / YAW_RATE_MAX
    expected = 20.0 * (1.0 - 0.35 * turn_frac)
    assert env.ovx.item() > 5.0, f"robot stalled while turning: v={env.ovx.item():.2f}"
    assert torch.allclose(env.ovx, expected, rtol=0.05), (env.ovx, expected)


def test_enemy_can_reverse_and_is_capped_like_us():
    env = _env(n=1)
    _place(env, 10.0, 10.0, 0.0, 36.0, 30.0, 0.0)
    _drive(env, 0.0, 0.0, -1.0, 0.0, n_ctrl=25)
    assert not env.done.any(), f"episode ended early: reason={env.end_reason.tolist()}"
    assert env.evx.item() < -15.0, env.evx
    assert env.evx.abs().item() <= 20.0 + 1e-3
    _drive(env, 0.0, 0.0, 0.0, 1.0, n_ctrl=40)
    assert 0.5 * YAW_RATE_MAX < env.e_omega.abs().item() <= YAW_RATE_MAX + 1e-4


def test_slide_goes_wide_and_never_adds_speed():
    """Past the grip limit the robot slides to the outside of the turn (to the
    right in a left turn) and the slide takes speed away, never adds it."""
    par = RobotParams(1)
    _nominal(par, v_max=22.0)
    par.grip_s[:], par.grip_k[:] = 30.0, 10.0
    par.dv_max[:] = 0.0                         # no drive, so only the slide can change the speed
    f = lambda v: torch.tensor([v], device=DEVICE)
    x, y, a, v, om = f(24.0), f(24.0), f(0.0), f(20.0), f(0.0)
    drifting, drift = torch.tensor([False], device=DEVICE), f(0.0)
    slid = False
    for _ in range(60):
        speed_before = math.hypot(v.item(), drift.item())
        x, y, a, v, om, drifting, drift, _ = integrate_robot(
            x, y, a, v, om, drifting, drift, f(1.0), f(6.0), par, f(1.0))   # hard left at speed
        if drifting.item():
            slid = True
            assert drift.item() < 0.0, f"slid to the inside of the turn: {drift.item():.2f}"
            assert math.hypot(v.item(), drift.item()) <= speed_before + 5e-3
    assert slid, "the turn never broke traction"
    assert math.hypot(v.item(), drift.item()) < 19.0, "the slide did not scrub any speed"


def test_walls_are_solid_for_both_robots():
    """Driving into a wall stops the robot inside the arena. Nobody dies of
    it, and our robot is charged for the impact."""
    env = _env(n=1)
    _place(env, 8.0, 12.0, math.pi, 40.0, 36.0, 0.0)    # we face the west wall, the enemy the east wall
    _drive(env, 1.0, 0.0, 1.0, 0.0, n_ctrl=60)
    lo, hi = WALL_CLAMP_IN, ARENA_FT - WALL_CLAMP_IN
    assert not env.done.any(), f"wall contact ended the episode: {env.end_reason.tolist()}"
    assert abs(env.ox.item() - lo) < 1e-3 and abs(env.ex.item() - hi) < 1e-3
    assert env.ovx.item() < 6.0 and env.evx.item() < 6.0, "robots kept their speed against the wall"
    assert env.rw[0, RW_WALL].item() < 0.0


def test_drive_imperfections_show_up_in_the_motion():
    """A pull veers the robot under power with no turn command, a deadband
    swallows a small throttle, and a low yaw gain turns less than asked."""
    env = _env(n=3)
    _place(env, 24.0, 24.0, 0.0, 40.0, 40.0, 0.0)
    env.our.pull[0] = 0.06
    env.our.deadband[1] = 0.10
    env.our.yaw_gain[2] = 0.5
    act = torch.tensor([[1.0, 0.0], [0.05, 0.0], [0.0, 0.4]], device=DEVICE)
    for _ in range(20):
        env.step(act)
    assert env.o_omega[0].item() > 0.2, "pull did not veer the robot"
    assert abs(env.ovx[1].item()) < 1e-3, "throttle inside the deadband moved the robot"
    assert abs(env.o_omega[2].item() - 0.5 * 0.4 * YAW_RATE_MAX) < 0.05, env.o_omega[2]


def test_wear_strikes_once_mid_fight():
    env = _env(n=2)
    _place(env, 12.0, 24.0, 0.0, 36.0, 24.0, 0.0)
    env.wear_at[0], env.wear_speed[:], env.wear_pull[:] = 30, 0.8, 0.05
    _drive(env, 0.0, 0.0, 0.0, 0.0, n_ctrl=30)
    assert abs(env.our.v_max[0].item() - 16.0) < 1e-4 and abs(env.our.pull[0].item() - 0.05) < 1e-6
    assert env.our.v_max[1].item() == 20.0 and env.our.pull[1].item() == 0.0, "wear hit an env it was not due in"


def test_reset_clears_motion_state():
    env = _env(n=3)
    env.evx[:], env.e_omega[:], env.e_drift[:], env.o_omega[:] = 9.0, 2.0, 1.0, 3.0
    env.e_drifting[:] = True
    env.manual_cmd[:] = 0.5
    env.done[:] = True
    env.reset_at(torch.tensor([1], device=DEVICE))
    assert env.e_omega[1] == 0 and env.e_drift[1] == 0 and env.o_omega[1] == 0
    assert not env.e_drifting[1] and not env.done[1]
    assert (env.manual_cmd[1] == 0).all() and env.end_reason[1] == END_NONE
    assert env.e_omega[0] == 2.0 and env.e_omega[2] == 2.0, "reset touched other slots"


if __name__ == "__main__":
    _runner.run(globals())
