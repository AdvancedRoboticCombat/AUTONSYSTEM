"""
Sensors, rewards, opponents and matchups.

Run with pytest if you have it, or directly:
    python tests/test_env.py
"""
import _runner  # noqa: F401  (puts the repo root on sys.path)
import torch

import environment
from config import (
    DEVICE, GAMMA, MAX_STEPS, PHYS_PER_CTRL, V_MAX_FT, OBS_DIM, DR_V_MAX, DR_POS_BIAS_FT, SAW_OBS_RING,
    CAMERA_PERIOD_STEPS,
)
from environment import (
    BattleEnv, Observer, make_env, load_arena, _OBS_RING, RW_SHAPE, END_ENEMY_HIT, END_TIP_CLASH, END_MUTUAL,
    END_TIMEOUT,
)
from opponent_library import (
    COMBAT_TYPES, DRILL_TYPES, TYPE_NAMES, WEAKNESS_NAMES, MAX_EDGE, REACT_FLOOR_S, OpponentMixture,
)
from saw_field import Scenario

_STILL = lambda n: torch.zeros(n, 2, device=DEVICE)


def _first_obs(env, seed):
    """The first observation of an episode from the env's current true state,
    with the sensor noise pinned by ``seed``."""
    torch.manual_seed(seed)
    env.observer.reset(env._ar, env._truth())
    return env.observer.observe(env.prev_action, idx=env._ar)


# ─── Sensors ──────────────────────────────────────────────────────────────

def test_observation_does_not_leak_the_enemy_heading_or_its_limits():
    """The real robot cannot see which way the enemy points or how fast it
    can go, so changing either must leave the observation untouched."""
    env = BattleEnv(64, stage="combat", p=1.0, auto_reset=False)
    env.evx[:] = 0.0
    base = _first_obs(env, 7)
    assert base.shape == (64, OBS_DIM)

    env.ea += 1.3
    env.enemy.v_max *= 0.5
    env.enemy.yaw_cap *= 0.5
    assert torch.equal(_first_obs(env, 7), base), "enemy heading or limits leaked into the observation"

    env.ex += 2.0                                   # sanity: its position does show up
    assert not torch.equal(_first_obs(env, 7), base)


def test_enemy_velocity_is_estimated_from_its_position_track():
    env = BattleEnv(16, stage="combat", p=1.0, auto_reset=False)
    env.manual_enemy = True
    env.ox[:], env.oy[:], env.ex[:], env.ey[:], env.ea[:] = 8.0, 40.0, 10.0, 12.0, 0.0
    env.evx[:], env.ovx[:] = 0.0, 0.0
    env.observer.reset(env._ar, env._truth())
    for t in (env.observer.outlier_p, env.observer.drop_p, env.observer.other_sigma, env.observer.bias_sigma,
              env.observer.bias):
        t.zero_()                                   # a clean track: this checks the estimator, not the noise
    env.manual_cmd[:, 0] = 0.6                      # enemy drives east at a steady speed
    for _ in range(25):
        obs, _, done, _ = env.step(_STILL(16))
    assert not done.any()
    est = torch.sqrt(obs[:, 12] ** 2 + obs[:, 13] ** 2) * V_MAX_FT
    true = env.evx.abs()
    assert ((est - true).abs() < 0.35 * true + 2.5).all(), (est, true)


def test_position_error_persists_between_frames():
    """Part of the position error is an offset that wanders slowly, so the
    error two frames apart is correlated. Averaging frames cannot remove it."""
    torch.manual_seed(0)
    n = 4096
    env = BattleEnv(n, stage="pretrain", p=0.0, auto_reset=False)
    env.reset()
    o = env.observer
    o.pos_sigma[:], o.drop_p[:] = 0.0, 0.0          # leave only the wandering offset
    o.bias_sigma[:, :2] = 0.3
    o.bias[:, :2] = torch.randn(n, 2, device=DEVICE) * 0.3
    env.ovx[:] = 0.0
    errs = []
    for _ in range(12):
        env.step(_STILL(n))
        errs.append(o.fix[:, 0] - env.ox)
    early, late = errs[0], errs[-1]
    corr = torch.corrcoef(torch.stack([early, late]))[0, 1].item()
    assert corr > 0.7, f"position error decorrelated in 0.33 s (corr={corr:.2f})"
    assert 0.2 < early.std().item() < 0.4


def test_stress_factor_widens_the_randomization():
    """evaluate.py --stress pushes physics and sensors past the training ranges."""
    try:
        environment.DR_STRETCH = 1.5
        env = BattleEnv(4096, stage="combat", p=1.0, auto_reset=False)
        scale = env.our.v_max / V_MAX_FT
        assert scale.min() < DR_V_MAX[0] - 0.02 and scale.max() > DR_V_MAX[1] + 0.02
        assert env.observer.bias_sigma[:, 0].max() > DR_POS_BIAS_FT[1] * 1.1
        assert (env.observer.bias_sigma >= 0).all() and (env.observer.pos_sigma >= 0).all()
    finally:
        environment.DR_STRETCH = 1.0
    env = BattleEnv(4096, stage="combat", p=1.0, auto_reset=False)
    scale = env.our.v_max / V_MAX_FT
    assert scale.min() >= DR_V_MAX[0] - 1e-4 and scale.max() <= DR_V_MAX[1] + 1e-4


# ─── Rewards ──────────────────────────────────────────────────────────────

def test_shaping_is_potential_based():
    """Discounted shaping over any stretch of an episode must telescope to
    gamma^T * phi(end) - phi(start), whatever the robot did in between. That
    is what makes it impossible to farm."""
    torch.manual_seed(0)
    n = 256
    env = BattleEnv(n, stage="combat", p=0.5, auto_reset=False)
    env.reset()
    phi0 = env.phi.clone()
    disc = torch.zeros(n, device=DEVICE)
    was = torch.zeros(n, device=DEVICE)
    T = 40
    for t in range(T):
        env.step(torch.rand(n, 2, device=DEVICE) * 2 - 1)
        now = env.rw[:, RW_SHAPE]
        disc += (GAMMA ** t) * (now - was)
        was = now.clone()
    alive = ~env.done
    assert alive.sum() > 20
    expected = (GAMMA ** T) * env.phi - phi0
    assert torch.allclose(disc[alive], expected[alive], atol=2e-4), (disc[alive] - expected[alive]).abs().max()


def test_timeout_is_a_draw_not_a_terminal_state():
    env = BattleEnv(8, stage="pretrain", p=0.0, auto_reset=False)
    env.reset()
    env.steps[:] = MAX_STEPS - PHYS_PER_CTRL
    _, _, done, info = env.step(_STILL(8))
    assert done.all() and info.timeout.all()
    assert (info.end_reason == END_TIMEOUT).all() and not env.won.any()


# ─── Opponents ────────────────────────────────────────────────────────────

def test_every_matchup_is_winnable_on_paper():
    """No opponent is both faster and more agile than us, none reacts faster
    than a person can, and each carries exactly one weakness."""
    for p in (0.0, 0.5, 1.0):
        env = BattleEnv(4096, stage="combat", p=p, auto_reset=False)
        opp = env.opponents
        assert (opp.speed_ratio * opp.agility_ratio <= MAX_EDGE + 1e-4).all()
        assert (opp.react >= REACT_FLOOR_S - 1e-6).all()
        assert ((opp.weakness >= 0) & (opp.weakness < len(WEAKNESS_NAMES))).all()
        assert torch.allclose(env.enemy.v_max, env.our.v_max * opp.speed_ratio)
        assert set(opp.types.unique().tolist()) <= set(COMBAT_TYPES)


def test_every_tactic_can_hit_a_parked_robot():
    """Regression guard for the opponents: against a robot that never moves,
    every tactic must land its weapon well inside the 20 s."""
    torch.manual_seed(0)
    eps = 100
    pinned = torch.tensor(COMBAT_TYPES, device=DEVICE).repeat_interleave(eps)
    env = BattleEnv(pinned.numel(), stage="combat", p=1.0, auto_reset=False)
    env.opponents.forced_types = pinned
    env.reset()
    for _ in range(MAX_STEPS // PHYS_PER_CTRL + 2):
        _, _, done, _ = env.step(_STILL(env.n))
        if done.all():
            break
    landed = (env.end_reason == END_ENEMY_HIT) | (env.end_reason == END_TIP_CLASH)
    for t in COMBAT_TYPES:
        rate = landed[pinned == t].float().mean().item()
        assert rate >= 0.9, f"{TYPE_NAMES[t]} only reached a parked robot {rate:.0%} of the time"


def test_drill_targets_cannot_hurt_us():
    torch.manual_seed(0)
    env = BattleEnv(512, stage="pretrain", p=0.0, auto_reset=False)
    env.reset()
    assert set(env.opponents.types.unique().tolist()) <= set(DRILL_TYPES)
    assert not env.opponents.armed.any()
    for _ in range(300):
        env.step(_STILL(512))
    hurt = ((env.end_reason == END_ENEMY_HIT) | (env.end_reason == END_TIP_CLASH)
            | (env.end_reason == END_MUTUAL))
    assert not hurt.any()


def test_forced_types_pin_tactics_through_resets():
    env = BattleEnv(6, stage="combat", p=1.0, auto_reset=False)
    pinned = torch.tensor(COMBAT_TYPES[:6], device=DEVICE)
    env.opponents.forced_types = pinned
    env.reset()
    assert torch.equal(env.opponents.types, pinned)
    env.reset_at(torch.tensor([2, 3], device=DEVICE))
    assert torch.equal(env.opponents.types, pinned)


def test_self_play_slot_is_driven_by_the_snapshot():
    """A frozen copy of the policy drives SELFPLAY slots through its own
    sensors and the shared physics."""
    from opponent_library import SELFPLAY
    from opponent_selfplay import SelfPlayController
    torch.manual_seed(0)
    n = 32
    env = BattleEnv(n, stage="combat", p=1.0, auto_reset=False)
    env.selfplay, env.selfplay_frac = SelfPlayController(n), 1.0
    with torch.no_grad():
        env.selfplay.agent.actor_mean.bias.copy_(torch.tensor([0.5, 0.25]))   # recognisable command
    env.reset()
    assert (env.opponents.types == SELFPLAY).all()
    env.step(_STILL(n))
    assert torch.allclose(env.en_cmd[:, 0], torch.full((n,), 0.5, device=DEVICE), atol=0.05), env.en_cmd[:4]
    assert (env.en_cmd[:, 1] > 0).all()


def test_tensor_p_matches_scalar_distribution():
    """A per-fight p (the saw rules) deals the same field as the same p for
    every fight; a type override wins over the draw and forced_types over both."""
    _, dist = load_arena()
    n = 65536
    idx = torch.arange(n, device=DEVICE)
    ox = oy = torch.full((n,), 24.0, device=DEVICE)
    for sharp in (False, True):
        for p in (0.2, 0.68):
            a, b = OpponentMixture(n, dist), OpponentMixture(n, dist)
            a.sharp = b.sharp = sharp
            torch.manual_seed(0)
            a.reset(idx, p, False, ox, oy)
            torch.manual_seed(1)
            b.reset(idx, torch.full((n,), p, device=DEVICE), False, ox, oy)
            for name in ("react", "speed_ratio", "agility_ratio", "throttle", "lead"):
                ma, mb = getattr(a, name).mean().item(), getattr(b, name).mean().item()
                assert abs(mb / ma - 1.0) < 0.02, (sharp, p, name, ma, mb)
            for t in range(len(TYPE_NAMES)):
                sa, sb = (a.types == t).float().mean().item(), (b.types == t).float().mean().item()
                assert abs(sa - sb) < 0.01, (sharp, p, TYPE_NAMES[t], sa, sb)

    m = OpponentMixture(8, dist)
    m.sharp = True
    small = torch.arange(8, device=DEVICE)
    override = torch.tensor([3, -1, 3, -1, 5, -1, 2, -1], device=DEVICE)
    m.reset(small, torch.full((8,), 0.5, device=DEVICE), False, ox[:8], oy[:8], types_override=override)
    assert torch.equal(m.types[override >= 0], override[override >= 0])
    m.forced_types = torch.full((8,), COMBAT_TYPES[-1], dtype=torch.long, device=DEVICE)
    m.reset(small, 0.5, False, ox[:8], oy[:8], types_override=override)
    assert (m.types == COMBAT_TYPES[-1]).all()


# ─── Camera frames (saw rules) ────────────────────────────────────────────

def _clean_sensors(o):
    """No noise, offsets, outliers or dropped fixes: a fix is then exactly the truth it was taken from."""
    for t in (o.pos_sigma, o.other_sigma, o.drop_p, o.outlier_p, o.bias_sigma, o.bias):
        t.zero_()


def _saw_camera_env(n, period):
    env = make_env("saw", n, stage="combat", p=0.5, auto_reset=False, arm_gate=False,
                   scenario=Scenario(camera_period=period))
    env.track_loss_hz[:] = 0.0          # the tracker loses nobody: this checks the frames, not the tracker
    return env


def _capture(o):
    """The truth at the capture of each row's current fixes, from the observer's own history."""
    return o.ring[o._ar, (o.ptr - o.last_delay) % o.ring_n]


def test_camera_frames_are_held():
    torch.manual_seed(0)
    n = 512
    env = _saw_camera_env(n, 4)
    o = env.observer
    assert o.camera == 4 and (o.period == 4).all() and o.ring_n == SAW_OBS_RING and o.ring.shape[2] == 8
    _clean_sensors(o)
    still = torch.zeros(n, env.act_dim, device=DEVICE)          # we sit; the scripted enemy drives
    shares = []
    for t in range(80):
        fix0, seen0, hdg0 = o.fix.clone(), o.frame_seen.clone(), o.other_hdg.clone()
        env.step(still)
        new = o.frame_seen != seen0
        assert torch.equal(o.fix[~new], fix0[~new]), "the fix moved between camera frames"
        assert not ((o.other_hdg != hdg0) & ~o.e_fresh).any(), "their heading moved without a fresh fix"
        if t >= 10:
            shares.append(new.float().mean().item())
            # Latency on top of the frame's age: a fresh fix is the truth at the frame's capture
            cap, f = _capture(o), o.e_fresh
            assert torch.allclose(o.fix[f], cap[f][:, [0, 1, 5, 6]])
            assert torch.equal(o.other_hdg[f], cap[f, 7])
    share = sum(shares) / len(shares)
    # min(1, 3 / period). The +-1 step delay jitter now and then carries a control step across two
    # frames, which counts once, so the exact share is 0.722
    assert abs(share - 0.75) < 0.04, share

    # One physics step per frame: a new fix, noise and all, every control step
    env = _saw_camera_env(256, 1)
    o = env.observer
    o.drop_p.zero_()
    for _ in range(20):
        fix0 = o.fix.clone()
        env.step(torch.zeros(256, env.act_dim, device=DEVICE))
        assert o.e_fresh.all() and (o.fix != fix0).any(1).all()

    # Drawn per fight, with a random phase
    env = make_env("saw", 4096, stage="combat", p=0.5, auto_reset=False, arm_gate=False)
    per, lo, hi = env.observer.period, *CAMERA_PERIOD_STEPS
    assert set(per.unique().tolist()) == set(range(lo, hi + 1))
    assert abs(per.float().mean().item() - (lo + hi) / 2) < 0.15
    assert (env.observer.phase >= 0).all() and (env.observer.phase < per).all()


def test_classic_observer_has_no_camera():
    env = BattleEnv(8, stage="combat", p=0.5, auto_reset=False)
    for o in (env.observer, env.en_observer):
        assert o.camera is None and o.ring_n == _OBS_RING and o.ring.shape == (8, _OBS_RING, 7)
    env.step(_STILL(8))
    assert env.observer.e_fresh.dtype == torch.bool
    assert env.field is None and env.timeout_bootstrap and env.gamma == GAMMA


def test_camera_none_on_a_live_saw_observer_is_v7():
    torch.manual_seed(0)
    n = 256
    env = _saw_camera_env(n, 4)
    o = env.observer
    o.camera = None
    o.period.fill_(7)
    torch.manual_seed(1)
    o.reset(env._ar, env._truth())
    after = torch.get_rng_state()
    ref = Observer(n, env.hazards, env.ranges, ring=o.ring_n, cols=o.ring.shape[2], camera=None)
    torch.manual_seed(1)
    ref.reset(env._ar, env._truth())
    assert torch.equal(torch.get_rng_state(), after), "the reset drew camera numbers"
    assert (o.period == 7).all() and torch.equal(o.fix, ref.fix) and torch.equal(o.lag, ref.lag)
    _clean_sensors(o)
    for _ in range(10):
        env.step(torch.zeros(n, env.act_dim, device=DEVICE))
        assert o.e_fresh.all()
        assert (o.last_delay <= o.lag + 1).all(), "a frame age on top of the delay"
        assert torch.allclose(o.fix, _capture(o)[:, [0, 1, 5, 6]]), "the fix is not the delayed truth"


def test_make_env_refuses_saw_only_arguments_for_classic():
    for kw in (dict(scenario=Scenario()), dict(p_range=(0.5, 0.8)), dict(arm_gate=True)):
        try:
            make_env("classic", 4, **kw)
        except ValueError:
            continue
        raise AssertionError(f"the classic rules accepted {kw}")
    env = make_env("classic", 4, arm_gate=False)
    assert env.rules == "classic" and env.observer.camera is None


if __name__ == "__main__":
    _runner.run(globals())
