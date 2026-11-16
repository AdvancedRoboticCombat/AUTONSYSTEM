#!/usr/bin/env python3
"""
The test drives (README: "The test drives and what each one measures").

A maneuver is a plain dict:

    name       e.g. "step_fwd_0.6"
    kind       still, step, spin, arc, straight or doublet
    throttle   level while the command is on, -1..1 (+ = forward)
    turn       level while the command is on, -1..1 (+ = counter-clockwise)
    lead_s     zero command before it
    hold_s     how long the level is held (a doublet: each half, forward then back)
    tail_s     zero command after it, while the robot comes to rest
    and optionally, for fit():
    min_hold_s   the shortest hold worth playing (default MIN_HOLD_S)
    min_share    how far its level may be lowered, whatever the settings say

    command(m, t)    the (throttle, turn) to send t seconds after the start
    duration(m)      its length, s
    mix(...)         the README's mixer: throttle and turn to left and right treads
    reach(m, k)      how far it can take the robot at the simulator's top speed and yaw rate
    fit(m, ...)      the maneuver shortened (and, if allowed, slowed) to the free floor
    session(spec)    "standard", "quick" or "name,name,..." as a list of maneuvers

    python SIM2REAL/maneuvers.py      prints the catalogue and the sessions

No ROS in here. test_pilot.py plays these.
"""
import math
import os
import sys
from pathlib import Path

# The simulator's numbers come from the simulator. It is a folder of scripts, and
# nothing here needs the GPU (a training job may own it).
os.environ.setdefault("DODGE_DEVICE", "cpu")
_SIM_DIR = str(Path(__file__).resolve().parent.parent / "auton_training")
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
from config import CTRL_DT, SAW_DR, V_MAX_FT, YAW_RATE_MAX        # noqa: E402

# The fastest the simulator lets our robot be: the top of its training ranges.
V_TOP = V_MAX_FT * SAW_DR["v_max"][1]                                         # 24.9 ft/s
YAW_TOP = YAW_RATE_MAX * max(SAW_DR["yaw"][1], SAW_DR["yaw_gain"][1])         # 8.6 rad/s

COAST_S = 0.5         # stopping allowance: the robot is taken to keep its speed this long after the command ends
MIN_HOLD_S = 0.51     # a level held for less than this measures nothing: fit() does not go below it
SPIN_ROOM_FT = 1.0    # floor a turn on the spot is given all round


# ─── The mixer ────────────────────────────────────────────────────────────

def tread_k(half_track_ft):
    """The mixer's turn share (README "Commands"; deploy.SawPilot.tread_k)."""
    return half_track_ft * YAW_RATE_MAX / V_MAX_FT


def mix(throttle, turn, k):
    """Throttle and turn (-1..1) to (left, right) treads, exactly as the fight code does."""
    scale = max(1.0, abs(throttle) + k * abs(turn))
    return (throttle - k * turn) / scale, (throttle + k * turn) / scale


def unmix(left, right, k):
    """(left, right) treads back to (throttle, turn)."""
    return (left + right) / 2.0, (right - left) / (2.0 * k)


# ─── The catalogue ────────────────────────────────────────────────────────

def _m(name, kind, throttle=0.0, turn=0.0, hold_s=1.5, lead_s=0.51, tail_s=1.5, **more):
    """Times are whole control steps (0.03 s), so what is sent is exactly what is written here."""
    return dict(name=name, kind=kind, throttle=throttle, turn=turn, lead_s=lead_s, hold_s=hold_s, tail_s=tail_s, **more)


LEVELS = (0.3, 0.6, 1.0)
MANEUVERS = {m["name"]: m for m in [
    _m("still", "still", hold_s=2.01, lead_s=0.0, tail_s=0.0),
    *[_m(f"step_{way}_{level}", "step", throttle=sign * level)
      for level in LEVELS for way, sign in (("fwd", 1), ("rev", -1))],
    *[_m(f"spin_{way}_{level}", "spin", turn=sign * level)
      for level in LEVELS for way, sign in (("ccw", 1), ("cw", -1))],
    _m("arc_left", "arc", throttle=0.5, turn=0.3, hold_s=2.01),
    _m("arc_right", "arc", throttle=0.5, turn=-0.3, hold_s=2.01),
    _m("straight_fwd", "straight", throttle=0.4, hold_s=3.0, min_hold_s=1.5),     # drift only shows over a distance
    _m("straight_rev", "straight", throttle=-0.4, hold_s=3.0, min_hold_s=1.5),
    _m("doublet", "doublet", throttle=0.6, hold_s=0.42),
    # Not a measurement: a short hop the pilot adds by itself in --no-imu mode when it does not
    # know which way the robot faces. Always allowed to run slower than asked.
    _m("heading_probe", "step", throttle=0.3, hold_s=0.81, tail_s=1.02, min_share=0.5),
]}

# Straight drives first, forward and back in turn so the robot stays near where it started;
# then the ones that leave it facing somewhere else.
SESSIONS = {
    "standard": ["still",
                 "step_fwd_0.3", "step_rev_0.3", "step_fwd_0.6", "step_rev_0.6", "step_fwd_1.0", "step_rev_1.0",
                 "straight_fwd", "straight_rev", "doublet",
                 "arc_left", "arc_right",
                 "spin_ccw_0.3", "spin_cw_0.3", "spin_ccw_0.6", "spin_cw_0.6", "spin_ccw_1.0", "spin_cw_1.0"],
    "quick": ["still", "step_fwd_0.6", "step_rev_0.6", "spin_ccw_0.6", "spin_cw_0.6"],
}


def session(spec="standard"):
    """A session name, or maneuver names separated by commas, as a list of maneuvers (copies)."""
    names = SESSIONS.get(spec) or [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in MANEUVERS]
    if unknown:
        raise ValueError(f"unknown maneuver {unknown}; known: {', '.join(MANEUVERS)}; sessions: {', '.join(SESSIONS)}")
    return [dict(MANEUVERS[n]) for n in names]


# ─── Playing one ──────────────────────────────────────────────────────────

def _on_s(m):
    """Seconds the command is not zero."""
    return m["hold_s"] * (2 if m["kind"] == "doublet" else 1)


def duration(m):
    """Length of the maneuver, lead and tail included, s."""
    return m["lead_s"] + _on_s(m) + m["tail_s"]


def command(m, t):
    """(throttle, turn) to send t seconds after the maneuver started."""
    t -= m["lead_s"]
    if t < -1e-9 or t >= _on_s(m) - 1e-9:
        return 0.0, 0.0
    sign = -1.0 if m["kind"] == "doublet" and t >= m["hold_s"] - 1e-9 else 1.0
    return sign * m["throttle"], sign * m["turn"]


# ─── How much floor one needs ─────────────────────────────────────────────

def _sent(m, k):
    """The maneuver's (throttle, turn) after the mixer's scale-back: what the treads really get."""
    over = max(1.0, abs(m["throttle"]) + k * abs(m["turn"]))
    return m["throttle"] / over, m["turn"] / over


def reach(m, k):
    """The most maneuver m can move and turn the robot, were it as fast as the simulator allows
    (V_TOP, YAW_TOP) and at that speed from the first instant until COAST_S after the command
    ends: (ahead_ft, behind_ft, turn_rad).

    A straight drive needs its distance ahead of the robot (behind, in reverse; both for a
    doublet). One that turns can end up anywhere within that distance, so it reports the same
    distance both ways and the pilot gives it that much room all round."""
    throttle, turn = _sent(m, k)
    far = abs(throttle) * V_TOP * (m["hold_s"] + COAST_S)
    turned = abs(turn) * YAW_TOP * (_on_s(m) + COAST_S)
    if turn:
        far = max(far, SPIN_ROOM_FT)
        return far, far, turned
    if m["kind"] == "doublet":
        return far, far, 0.0
    return (far, 0.0, 0.0) if throttle > 0 else (0.0, far, 0.0)


def mildest(m, k, min_share=1.0):
    """m at the shortest hold and the lowest level fit() will go down to."""
    throttle, turn = _sent(m, k)
    share = min(1.0, m.get("min_share", min_share))
    return dict(m, hold_s=min(m["hold_s"], m.get("min_hold_s", MIN_HOLD_S)), throttle=throttle * share, turn=turn * share)


def fit(m, ahead_ft, behind_ft, k, min_share=1.0):
    """Maneuver m as it can be played with this much free floor ahead of and behind the robot.

    First the hold is shortened, down to MIN_HOLD_S (or the maneuver's own "min_hold_s"). If
    that is not enough, throttle and turn are lowered together, down to ``min_share`` of what
    was asked (1.0: never lowered; a maneuver's own "min_share" wins). Returns the maneuver to
    play, or None when even its mildest form does not fit."""
    m = dict(m)
    ahead, behind, _ = reach(m, k)
    if ahead <= ahead_ft and behind <= behind_ft:
        return m
    low = mildest(m, k, min_share)
    low_ahead, low_behind, _ = reach(low, k)
    if low_ahead > ahead_ft or low_behind > behind_ft or not m["throttle"]:
        return None
    room = min(free for need, free in ((ahead, ahead_ft), (behind, behind_ft)) if need)
    throttle, turn = _sent(m, k)
    hold = room / (abs(throttle) * V_TOP) - COAST_S
    if hold < low["hold_s"]:                    # too short to measure anything: slow it down instead
        share = room / (abs(throttle) * V_TOP * (low["hold_s"] + COAST_S))
        hold = low["hold_s"]
        m["throttle"] = math.copysign(math.floor(abs(throttle) * share * 1000) / 1000, throttle)
        m["turn"] = turn * share
    m["hold_s"] = round(CTRL_DT * math.floor(hold / CTRL_DT + 1e-9), 4)
    return m


def describe(m):
    """One line: what the maneuver sends."""
    what = {"still": "nothing", "doublet": f"throttle {m['throttle']:+.2f} then {-m['throttle']:+.2f}"}.get(
        m["kind"], " ".join(f"{name} {m[name]:+.2f}" for name in ("throttle", "turn") if m[name]))
    return f"{what} for {m['hold_s']:.2f} s" + (" each" if m["kind"] == "doublet" else "")


if __name__ == "__main__":
    k = tread_k(1.48)
    print(f"Simulator top speed {V_TOP:.1f} ft/s, top yaw rate {YAW_TOP:.1f} rad/s; mixer tread_k {k:.3f} (half track 1.48 ft)\n")
    print(f"{'maneuver':15s} {'kind':9s} {'sends':40s} {'s':>5s}  needs at most (ft ahead, behind; rad)")
    for m in MANEUVERS.values():
        ahead, behind, turned = reach(m, k)
        room = f"{ahead:5.1f} all round" if turned else f"{ahead:5.1f} {behind:5.1f}"
        print(f"{m['name']:15s} {m['kind']:9s} {describe(m):40s} {duration(m):5.2f}  {room}   {turned:5.1f}")
    for name, names in SESSIONS.items():
        print(f"\n{name} ({sum(duration(MANEUVERS[n]) for n in names):.0f} s of driving): {', '.join(names)}")
