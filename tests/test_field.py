"""
The saw-v8 opponent field (saw_field.py): physique apart from ours, p as
driver skill only, the Traves-like horizontals, the other horizontals,
weapon failures, the card, and the scenario pins.

Run with pytest if you have it, or directly:
    uv run python tests/test_field.py
"""
import _runner  # noqa: F401  (puts the repo root on sys.path)
import torch

from config import DEVICE, DT, MPH2FT, EN_BLADE_FT, DR_EN_HALF_LEN
from environment import make_env
from opponent_library import TURRET, PURSUER, FEINTER, W_SLOW, W_POOR_TURN
from saw_field import (
    Scenario, W_NONE, W_VERT, W_DRUM, W_HORIZ, PROFILE_FIELD, PROFILE_TRAVES, PROFILE_LEGACY, NEVER,
    TRAVES_SHARE, TRAVES_HUB, TRAVES_BAR_FT, TRAVES_HALF_LEN, TRAVES_HALF_WID, TRAVES_TAU_S, TRAVES_MPH,
    TRAVES_YAW_CAP, TRAVES_RECOIL, TRAVES_POWER, EN_MPH_MIX, EN_YAW_CAP, EN_YAW_ACCEL, HORIZ_HUB, HORIZ_BAR_FT,
    HORIZ_TAU_S, EN_HALF_WID, WEAPON_FAIL_P, FAIL_AT_START, CARD_ERR,
)


def _saw(n, stage="combat", p=0.68, **kw):
    return make_env("saw", n, stage=stage, p=p, auto_reset=False, **kw)


def _within(t, lo, hi, what, eps=1e-4):
    assert t.numel() > 0, f"no rows for {what}"
    assert t.min().item() >= lo - eps and t.max().item() <= hi + eps, (what, t.min().item(), t.max().item(), lo, hi)


def _corr(a, b):
    return torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1].item()


def test_opponent_physique_is_not_tied_to_ours():
    torch.manual_seed(0)
    env = _saw(8192)
    armed, trav = env.opponents.armed, env.field.profile == PROFILE_TRAVES
    assert abs(_corr(env.our.v_max[armed], env.enemy.v_max[armed])) < 0.05
    field = armed & ~trav
    mph = env.field.top_mph[field]
    assert torch.allclose(env.field.top_mph, env.enemy.v_max / MPH2FT)
    for share, (lo, hi) in EN_MPH_MIX:
        got = ((mph >= lo - 1e-3) & (mph <= hi + 1e-3)).float().mean().item()
        assert abs(got - share) < 0.02, (lo, hi, got, share)
    _within(env.enemy.yaw_cap[field], *EN_YAW_CAP, "yaw cap, field rows")
    _within(env.enemy.dalpha[armed] / DT / env.enemy.yaw_cap[armed], *EN_YAW_ACCEL, "yaw accel / cap")

    # The saw-v7 field keeps the old tie to our robot
    torch.manual_seed(1)
    leg = _saw(8192, scenario=Scenario(legacy_field=True))
    assert _corr(leg.our.v_max, leg.enemy.v_max) > 0.6
    assert torch.allclose(leg.enemy.v_max, leg.our.v_max * leg.opponents.speed_ratio)

    # Drill targets keep the v7 expression
    dr = _saw(2048, stage="pretrain", p=0.0)
    opp = dr.opponents
    assert not opp.armed.any()
    assert torch.allclose(dr.enemy.v_max, dr.our.v_max * opp.speed_ratio)
    assert torch.allclose(dr.enemy.yaw_cap, dr.our.yaw_cap * opp.agility_ratio)


def test_p_sets_skill_only():
    torch.manual_seed(0)
    env = _saw(8192, p_range=(0.0, 1.0))
    p, opp, armed = env.field.p, env.opponents, env.opponents.armed
    _within(p, 0.0, 1.0, "p")
    assert p.std().item() > 0.2
    low, high = armed & (p < 0.3), armed & (p > 0.7)
    assert opp.react[high].mean().item() < opp.react[low].mean().item() - 0.1, "experts do not react faster"
    assert abs(_corr(p[armed], env.field.top_mph[armed])) < 0.05, "top speed follows p"
    w = opp.weakness[armed]
    assert not ((w == W_SLOW) | (w == W_POOR_TURN)).any(), "a physique weakness outside the legacy field"

    scalar = _saw(1024, p=0.3)
    assert torch.allclose(scalar.field.p, torch.full((1024,), 0.3, device=DEVICE))

    torch.manual_seed(1)
    leg = _saw(4096, scenario=Scenario(legacy_field=True))
    lw = leg.opponents.weakness
    assert ((lw == W_SLOW) | (lw == W_POOR_TURN)).any(), "the legacy field lost its physique weaknesses"


def test_traves_like_ranges():
    torch.manual_seed(0)
    env = _saw(8192)
    se, armed = env.saw_e, env.opponents.armed
    trav = env.field.profile == PROFILE_TRAVES
    share = trav[armed].float().mean().item()
    assert abs(share - TRAVES_SHARE) < 0.02, share
    assert (se.kind[trav] == W_HORIZ).all()
    _within(se.fore[trav] / env.e_len[trav], *TRAVES_HUB, "hub / half length")
    _within(se.rad[trav], *TRAVES_BAR_FT, "bar")
    _within(env.e_len[trav], *TRAVES_HALF_LEN, "half length")
    _within(env.e_wid[trav], *TRAVES_HALF_WID, "half width")
    _within(se.tau[trav], *TRAVES_TAU_S, "spin-up")
    _within(env.field.top_mph[trav], *TRAVES_MPH, "top speed", eps=1e-3)
    _within(env.enemy.yaw_cap[trav], *TRAVES_YAW_CAP, "yaw cap")
    _within(se.recoil[trav], *TRAVES_RECOIL, "recoil")
    _within(se.power[trav], *TRAVES_POWER, "power")
    types = env.opponents.types[trav]
    for t, want in ((TURRET, 0.50), (PURSUER, 0.30), (FEINTER, 0.20)):
        got = (types == t).float().mean().item()
        assert abs(got - want) < 0.04, (t, got, want)


def test_traves_only_on_horizontals():
    torch.manual_seed(0)
    env = _saw(2048, scenario=Scenario(traves_share=1.0))
    env.forced_weapon = W_VERT
    env.reset()
    assert not (env.field.profile == PROFILE_TRAVES).any()
    assert (env.saw_e.rad == EN_BLADE_FT).all()


def test_other_horizontals():
    torch.manual_seed(0)
    env = _saw(8192)
    se = env.saw_e
    rows = (se.kind == W_HORIZ) & (env.field.profile != PROFILE_TRAVES)
    _within(se.fore[rows] / env.e_len[rows], *HORIZ_HUB, "hub / half length")
    _within(se.rad[rows], *HORIZ_BAR_FT, "bar")
    _within(se.tau[rows], *HORIZ_TAU_S, "spin-up")
    assert abs(_corr(se.rad[rows], env.e_len[rows])) < 0.1, "bar tied to body length"
    _within(env.e_wid[rows], *EN_HALF_WID, "half width")
    _within(env.e_len[rows], *DR_EN_HALF_LEN, "half length")


def test_weapon_failure():
    torch.manual_seed(0)
    env = _saw(8192)
    armed = env.saw_e.spinner
    dead_at, horizon = env.field.dead_at, env.horizon
    fail = armed & (dead_at < NEVER)
    assert abs(fail[armed].float().mean().item() - WEAPON_FAIL_P) < 0.02
    at0 = fail & (dead_at == 0)
    assert abs(at0.sum().item() / fail.sum().item() - FAIL_AT_START) < 0.05
    later = fail & ~at0
    assert (dead_at[later] > 0).all() and (dead_at[later] < horizon[later]).all()
    assert (dead_at[~armed] == NEVER).all()

    dr = _saw(2048, stage="pretrain", p=0.0)
    assert (dr.field.dead_at == NEVER).all(), "a drill target's weapon failed"


def test_card_is_within_ten_percent():
    torch.manual_seed(0)
    env = _saw(8192)
    F, se = env.field, env.saw_e
    assert torch.equal(F.card_kind, se.kind)
    _within(F.card_len / env.e_len, *CARD_ERR, "card length")
    _within(F.card_wid / env.e_wid, *CARD_ERR, "card width")
    sp = se.spinner
    _within(F.card_fore[sp] / se.fore[sp], *CARD_ERR, "card fore")
    _within(F.card_rad[sp] / se.rad[sp], *CARD_ERR, "card rad")
    wide = sp & (se.wide > 0)
    _within(F.card_wide[wide] / se.wide[wide], *CARD_ERR, "card wide")
    assert (F.card_wide[se.kind == W_HORIZ] == 0).all()
    # Each factor is drawn on its own
    assert abs(_corr(F.card_len / env.e_len, F.card_wid / env.e_wid)) < 0.05
    # It is the truth with an error, not the truth
    assert (F.card_len != env.e_len).float().mean().item() > 0.99

    dr = _saw(1024, stage="pretrain", p=0.0)
    assert (dr.field.card_kind == W_NONE).all()
    for t in (dr.field.card_fore, dr.field.card_rad, dr.field.card_wide):
        assert (t == 0).all(), "an unarmed target's card has weapon geometry"


def test_scenario_field_pins():
    torch.manual_seed(0)
    env = _saw(2048, p_range=(0.0, 1.0), scenario=Scenario(p=0.4))
    assert torch.allclose(env.field.p, torch.full((2048,), 0.4, device=DEVICE)), "Scenario.p does not beat p_range"
    env = _saw(2048, p_range=(0.0, 1.0), scenario=Scenario(p_range=(0.3, 0.4)))
    _within(env.field.p, 0.3, 0.4, "Scenario.p_range over the env's")

    env = _saw(4096, scenario=Scenario(traves_share=0.0))
    assert not (env.field.profile == PROFILE_TRAVES).any()
    env = _saw(2048, scenario=Scenario(traves_share=1.0, weapon_shares=(0.0, 0.0, 1.0)))
    assert (env.field.profile == PROFILE_TRAVES).all()

    env = _saw(4096, scenario=Scenario(weapon_fail_share=0.0))
    assert (env.field.dead_at == NEVER).all()
    env = _saw(2048, scenario=Scenario(weapon_fail_share=1.0))
    assert (env.field.dead_at[env.saw_e.spinner] < NEVER).all()

    env = _saw(4096, scenario=Scenario(legacy_field=True))
    assert (env.field.profile == PROFILE_LEGACY).all()
    assert (env.field.dead_at == NEVER).all()
    # the v7 horizontal: hub anywhere from 0.3 of the half length
    horiz = env.saw_e.kind == W_HORIZ
    assert (env.saw_e.fore[horiz] / env.e_len[horiz]).min().item() < HORIZ_HUB[0] - 0.05

    env = _saw(6000, scenario=Scenario(weapon_shares=(1, 1, 1), traves_share=0.0))
    armed = env.opponents.armed
    for kind in (W_VERT, W_DRUM, W_HORIZ):
        got = (env.saw_e.kind[armed] == kind).float().mean().item()
        assert abs(got - 1 / 3) < 0.03, (kind, got)

    for bad in (dict(heading_mode="nose"), dict(camera_period=0), dict(camera_period=9), dict(camera_period=2.5),
                dict(traves_share=1.5), dict(weapon_fail_share=-0.1), dict(weapon_shares=(1.0, 1.0)),
                dict(weapon_shares=(0.0, 0.0, 0.0)), dict(weapon_shares=(-1.0, 1.0, 1.0)), dict(p_range=(0.8, 0.5))):
        try:
            Scenario(**bad)
        except ValueError:
            continue
        raise AssertionError(f"Scenario accepted {bad}")
    Scenario(heading_mode="axis_nose", camera_period=1, weapon_shares=(0.0, 0.0, 1.0), p_range=(0.5, 0.5))


if __name__ == "__main__":
    _runner.run(globals())
