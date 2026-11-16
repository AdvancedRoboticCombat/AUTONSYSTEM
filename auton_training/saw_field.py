"""
The saw-v8 opponent field, and the scenarios the tools pin it with.

Every saw-only draw about the opponent and the fight it opens lives here, so
the classic rules never reach it: BattleEnv calls these hooks only when its
``field`` is set (saw_env.SawEnv sets it), and runs its old code otherwise.

    Scenario    frozen settings that pin parts of a fight (fight length,
                swings, reach, p, sensing, ...). None everywhere: the training
                randomization.
    SawField    per-fight state: the opponent's profile, its p, when its
                weapon dies, its top speed, and the card the policy is told

Profiles:
    PROFILE_FIELD   the heavyweight field: absolute top speed and turn rate,
                    not tied to ours; p is driver skill only
    PROFILE_TRAVES  a Traves-like horizontal: long thin frame, a bar at the
                    very front, slow to spin up, hits harder than we do
    PROFILE_LEGACY  the saw-v7 field (Scenario.legacy_field): speed and
                    agility as a share of ours, v7 weapons, no failures

The card (card_kind, card_len, card_wid, card_fore, card_rad, card_wide) is
what the team writes down about the opponent before the fight: its weapon
kind exactly, and its size and weapon geometry within ten percent.
"""
from dataclasses import dataclass

import torch

from config import DEVICE, DT, MPH2FT, A_LAT_MAX_FT
from opponent_library import TURRET, PURSUER, FEINTER, W_SLOW, W_POOR_TURN, W_NO_WEAKNESS

# Weapon ids, as in saw_env (which asserts they agree)
W_NONE, W_VERT, W_DRUM, W_HORIZ = 0, 1, 2, 3

PROFILE_FIELD, PROFILE_TRAVES, PROFILE_LEGACY = 0, 1, 2
NEVER = 10 ** 12               # a physics step no fight reaches (saw_env._NEVER)

# Traves-like horizontals (the first robot we meet)
TRAVES_SHARE     = 0.20               # of the armed combat fights that carry a horizontal
TRAVES_HUB       = (0.95, 1.10)       # hub, x their half length from the centre
TRAVES_BAR_FT    = (1.4, 1.6)         # bar tip radius
TRAVES_HALF_LEN  = (2.0, 3.0)
TRAVES_HALF_WID  = (0.75, 1.25)
TRAVES_TAU_S     = (2.5, 5.0)         # spin-up time constant
TRAVES_MPH       = (8.0, 15.0)
TRAVES_YAW_CAP   = (2.6, 6.0)         # rad/s: 150 to 300 deg/s on video for big bar spinners
TRAVES_MIX       = ((TURRET, 0.50), (PURSUER, 0.30), (FEINTER, 0.20))
TRAVES_RECOIL    = (0.6, 0.8)
TRAVES_POWER     = (1.1, 1.55)        # 55 to 75 kJ over our 49

# The rest of the field: absolute physique, drawn apart from ours
EN_MPH_MIX       = ((0.30, (9.0, 13.0)), (0.55, (13.0, 20.0)), (0.15, (20.0, 24.0)))   # (share, mph range)
# Witch Doctor's fastest on camera was 12 mph (17.6 ft/s). The slow group used to stop at 12, leaving no enemy between 12 and 13.
EN_YAW_CAP       = (3.0, 8.0)         # rad/s
EN_YAW_ACCEL     = (1.5, 3.0)         # x the yaw cap, per second
HORIZ_HUB        = (0.6, 1.1)         # a horizontal's hub, x their half length
HORIZ_BAR_FT     = (1.2, 2.0)
HORIZ_TAU_S      = (2.5, 5.0)
EN_HALF_WID      = (0.75, 2.5)        # non-Traves, non-legacy bodies
EN_ACT_LAG_S     = (0.02, 0.10)       # scripted drivers' command delay (saw_env applies it)

WEAPON_FAIL_P    = 0.15               # share of armed fights in which their weapon dies
FAIL_AT_START    = 1.0 / 3.0          # of those, the share dead from the bell; the rest die somewhere inside the clock
CARD_ERR         = (0.9, 1.1)         # the card's sizes, x the truth
BLADE_START_BELL = (0.0, 0.15)        # blade speed at the start of a fight from the bell ...
BLADE_START_MID  = (0.3, 1.0)         # ... and of one that opens mid-fight

HEADING_MODES = ("none", "axis", "axis_nose")


def _U(k, lo, hi):
    return torch.empty(k, device=DEVICE).uniform_(lo, hi)


def _mixture(k, table):
    """k draws from a ((share, (lo, hi)), ...) table: one rand picks the part, one places the draw in it."""
    shares = torch.tensor([s for s, _ in table], device=DEVICE).cumsum(0)
    lo = torch.tensor([r[0] for _, r in table], device=DEVICE)
    hi = torch.tensor([r[1] for _, r in table], device=DEVICE)
    part = (torch.rand(k, device=DEVICE).unsqueeze(1) >= shares).sum(1).clamp(max=len(table) - 1)
    return lo[part] + (hi[part] - lo[part]) * torch.rand(k, device=DEVICE)


@dataclass(frozen=True)
class Scenario:
    """Settings that pin parts of a saw fight. None: the training randomization."""
    fight_s: float = None             # fight length (s), whatever auto_reset is
    swings: int = None                # the swing budget, exactly (no pro-rata)
    reach_ft: float = None            # hammer reach beyond our front edge
    traves_share: float = None        # None: TRAVES_SHARE; 0: field only; 1: every armed horizontal-capable row
    p: float = None                   # every fight's p (beats p_range)
    p_range: tuple = None             # (lo, hi): p ~ U(lo, hi) per fight (beats the env's p_range)
    heading_mode: str = None          # "none", "axis" or "axis_nose"
    arm_reading: bool = None          # True: present in every fight; False: absent in every fight
    blade_reading: bool = None
    camera_period: int = None         # physics steps per camera frame, 1..8 (1: a new frame every step, as saw-v7)
    from_bell: bool = None
    weapon_fail_share: float = None
    weapon_shares: tuple = None       # (vert, drum, horiz) for non-Traves fights, in place of EN_WEAPON_SHARE
    legacy_field: bool = False        # the saw-v7 field: physique from ours and p, v7 weapons and weaknesses,
                                      # no failures, no Traves, no enemy command delay, v7 pressing speed

    def __post_init__(self):
        if self.heading_mode is not None and self.heading_mode not in HEADING_MODES:
            raise ValueError(f"heading_mode must be one of {HEADING_MODES}, not {self.heading_mode!r}")
        if self.camera_period is not None and (int(self.camera_period) != self.camera_period
                                               or not 1 <= self.camera_period <= 8):
            raise ValueError(f"camera_period must be a whole number of physics steps in 1..8, not {self.camera_period!r}")
        for name in ("traves_share", "weapon_fail_share"):
            v = getattr(self, name)
            if v is not None and not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], not {v!r}")
        if self.weapon_shares is not None:
            w = tuple(self.weapon_shares)
            if len(w) != 3 or any(x < 0 for x in w) or sum(w) <= 0:
                raise ValueError(f"weapon_shares must be three non-negative numbers with a positive sum, not {w!r}")
        if self.p_range is not None:
            if len(self.p_range) != 2 or self.p_range[0] > self.p_range[1]:
                raise ValueError(f"p_range must be (lo, hi) with lo <= hi, not {self.p_range!r}")


class SawField:
    """Per-fight opponent and field state of the saw rules, reached through
    hooks in BattleEnv.reset_at and SawEnv._adjust_start. All tensors (n,)."""

    def __init__(self, n, scenario):
        self.sc = scenario
        z = lambda dtype=torch.float32: torch.zeros(n, dtype=dtype, device=DEVICE)
        self.p = z()
        self.profile = z(torch.long)
        self.dead_at = torch.full((n,), NEVER, dtype=torch.long, device=DEVICE)   # physics step their weapon dies
        self.top_mph = z()
        self.card_kind = z(torch.long)
        self.card_len, self.card_wid = z(), z()
        self.card_fore, self.card_rad, self.card_wide = z(), z(), z()

    def before_opponents(self, idx, env):
        """Before the opponents are dealt: the profile, p and, for Traves-like
        rows, the tactic. Returns (p_k (k,) float, types_override (k,) long,
        -1 where the drawn type stands). Draws: profile, p, tactic."""
        k, sc = idx.numel(), self.sc
        drills = env.stage == "pretrain"
        r_prof = torch.rand(k, device=DEVICE)
        if drills or sc.legacy_field:
            share = 0.0
        else:
            share = TRAVES_SHARE if sc.traves_share is None else sc.traves_share
        prof = torch.where(r_prof < share, PROFILE_TRAVES, PROFILE_FIELD)
        if sc.legacy_field:
            prof = torch.full_like(prof, PROFILE_LEGACY)

        rng = sc.p_range if sc.p_range is not None else env.p_range
        if sc.p is not None:
            p_k = torch.full((k,), float(sc.p), device=DEVICE)
        elif rng is not None:
            p_k = _U(k, float(rng[0]), float(rng[1]))
        else:
            p_k = torch.full((k,), float(env.p), device=DEVICE)

        r_tac = torch.rand(k, device=DEVICE)
        edges = torch.tensor([s for _, s in TRAVES_MIX], device=DEVICE).cumsum(0)
        tactics = torch.tensor([t for t, _ in TRAVES_MIX], dtype=torch.long, device=DEVICE)
        tactic = tactics[(r_tac.unsqueeze(1) >= edges).sum(1).clamp(max=len(TRAVES_MIX) - 1)]
        override = torch.where(prof == PROFILE_TRAVES, tactic, -1)
        self.profile[idx], self.p[idx] = prof, p_k
        return p_k, override

    def _v7_rows(self, env, armed):
        """Rows that keep the saw-v7 physique: the legacy field, the drills and unarmed targets."""
        if self.sc.legacy_field or env.stage == "pretrain":
            return torch.ones_like(armed)
        return ~armed

    def enemy_physique(self, idx, env, grip):
        """The enemy robot's limits, in place of the classic expression (which
        v7 rows keep exactly). Field rows: absolute top speed from EN_MPH_MIX,
        turn rate from EN_YAW_CAP, neither tied to ours or to p. Draws for all
        rows: speed part, speed, Traves speed, yaw cap, Traves yaw cap, yaw
        acceleration, then the grip and lateral-limit draws of the classic code."""
        k, opp = idx.numel(), env.opponents
        v7 = self._v7_rows(env, opp.armed[idx])
        trav = self.profile[idx] == PROFILE_TRAVES
        mph = _mixture(k, EN_MPH_MIX)
        mph = torch.where(trav, _U(k, *TRAVES_MPH), mph)
        cap = _U(k, *EN_YAW_CAP)
        cap = torch.where(trav, _U(k, *TRAVES_YAW_CAP), cap)
        alpha = cap * _U(k, *EN_YAW_ACCEL)
        v_l = env.our.v_max[idx] * opp.speed_ratio[idx]
        y_l = env.our.yaw_cap[idx] * opp.agility_ratio[idx]
        a_l = env.our.dalpha[idx] / DT * opp.agility_ratio[idx]
        env.enemy.randomize(idx,
                            v_max=torch.where(v7, v_l, mph * MPH2FT),
                            yaw_cap=torch.where(v7, y_l, cap),
                            alpha=torch.where(v7, a_l, alpha),
                            grip=grip * _U(k, 0.85, 1.15), ranges=env.ranges)
        env.enemy.a_lat_cap[idx] = A_LAT_MAX_FT * _U(k, 0.8, 1.1)
        self.top_mph[idx] = env.enemy.v_max[idx] / MPH2FT
        # A slow or poorly turning robot is a physique, and the physique no
        # longer comes from the driver: those two weaknesses would do nothing
        w = opp.weakness[idx]
        opp.weakness[idx] = torch.where(~v7 & ((w == W_SLOW) | (w == W_POOR_TURN)), W_NO_WEAKNESS, w)

    def bodies(self, idx, e_len, e_wid, armed):
        """In place, after the v7 body draw. Field rows get a width from
        EN_HALF_WID; armed Traves rows a Traves body. Legacy, drill and
        unarmed rows keep the v7 body. Draws for all rows: field width,
        Traves length, Traves width."""
        k = idx.numel()
        wid_f = _U(k, *EN_HALF_WID)
        len_t = _U(k, *TRAVES_HALF_LEN)
        wid_t = _U(k, *TRAVES_HALF_WID)
        prof = self.profile[idx]
        field = (prof == PROFILE_FIELD) & armed
        trav = (prof == PROFILE_TRAVES) & armed
        e_len[idx] = torch.where(trav, len_t, e_len[idx])
        e_wid[idx] = torch.where(trav, wid_t, torch.where(field, wid_f, e_wid[idx]))

    def force_kind(self, idx, carried):
        """Traves rows carry a horizontal. No draws."""
        return torch.where(self.profile[idx] == PROFILE_TRAVES, W_HORIZ, carried)

    def weapon(self, idx, saw_e, e_len, e_wid, horizon):
        """After Weapon.reset: horizontal and Traves geometry, weapon failure
        and the card. ``horizon`` is (k,) physics steps of these fights.
        Draws for all rows: horizontal hub, bar, tau; Traves hub, bar, tau,
        recoil, power; failure, at the start, when; card factors for length,
        width, fore, rad, wide."""
        k, sc = idx.numel(), self.sc
        leg = sc.legacy_field
        hub_h, bar_h, tau_h = _U(k, *HORIZ_HUB), _U(k, *HORIZ_BAR_FT), _U(k, *HORIZ_TAU_S)
        hub_t, bar_t, tau_t = _U(k, *TRAVES_HUB), _U(k, *TRAVES_BAR_FT), _U(k, *TRAVES_TAU_S)
        rec_t, pow_t = _U(k, *TRAVES_RECOIL), _U(k, *TRAVES_POWER)
        if sc.weapon_fail_share is not None:
            share = sc.weapon_fail_share
        else:
            share = 0.0 if leg else WEAPON_FAIL_P
        fail = torch.rand(k, device=DEVICE) < share
        at_start = torch.rand(k, device=DEVICE) < FAIL_AT_START
        when = torch.rand(k, device=DEVICE)
        f_len, f_wid, f_fore, f_rad, f_wide = (_U(k, *CARD_ERR) for _ in range(5))

        # A Traves profile on a row that does not carry a horizontal (a pinned
        # weapon, an unarmed target) is an ordinary field row
        horiz = saw_e.kind[idx] == W_HORIZ
        trav = (self.profile[idx] == PROFILE_TRAVES) & horiz
        self.profile[idx] = torch.where((self.profile[idx] == PROFILE_TRAVES) & ~trav, PROFILE_FIELD, self.profile[idx])
        if not leg:
            half = e_len[idx]
            fore = torch.where(trav, half * hub_t, torch.where(horiz, half * hub_h, saw_e.fore[idx]))
            rad  = torch.where(trav, bar_t, torch.where(horiz, bar_h, saw_e.rad[idx]))
            tau  = torch.where(trav, tau_t, torch.where(horiz, tau_h, saw_e.tau[idx]))
            saw_e.recoil[idx] = torch.where(trav, rec_t, saw_e.recoil[idx])
            saw_e.power[idx] = torch.where(trav, pow_t, saw_e.power[idx])
            saw_e.fore[idx], saw_e.rad[idx], saw_e.tau[idx] = fore, rad, tau

        armed = saw_e.spinner[idx]
        self.dead_at[idx] = torch.where(armed & fail, torch.where(at_start, 0, (when * horizon).long()), NEVER)

        on = armed.float()                     # an unarmed target's card has no weapon geometry
        self.card_kind[idx] = saw_e.kind[idx]
        self.card_len[idx] = e_len[idx] * f_len
        self.card_wid[idx] = e_wid[idx] * f_wid
        self.card_fore[idx] = saw_e.fore[idx] * f_fore * on
        self.card_rad[idx] = saw_e.rad[idx] * f_rad * on
        self.card_wide[idx] = saw_e.wide[idx] * f_wide * on

    def blade_starts(self, idx, from_bell, saw_o, saw_e):
        """Blade speeds as the fight opens: barely turning from the bell, part
        or fully spun up in a fight that opens mid-way. ``from_bell`` is (k,)
        bool. Draws for all rows: ours bell, ours mid, theirs bell, theirs mid."""
        k = idx.numel()
        o_bell, o_mid = _U(k, *BLADE_START_BELL), _U(k, *BLADE_START_MID)
        e_bell, e_mid = _U(k, *BLADE_START_BELL), _U(k, *BLADE_START_MID)
        saw_o.blade[idx] = torch.where(from_bell, o_bell, o_mid)
        saw_e.blade[idx] = torch.where(saw_e.spinner[idx], torch.where(from_bell, e_bell, e_mid), 0.0)
