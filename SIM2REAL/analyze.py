"""
Turn one recorded run into the robot's numbers.

    python SIM2REAL/analyze.py SIM2REAL/runs/<run>          # writes fit.json, replay.csv, replay.png, report.md
    python SIM2REAL/analyze.py RUN --no-imu                 # ignore imu.csv
    python SIM2REAL/analyze.py RUN --numbers OTHER/fit.json # start from an earlier fit (a policy flight measures nothing itself)
    python SIM2REAL/analyze.py --selftest 1 2 3             # made-up runs with known numbers: truth against fitted

How it works, in the order it runs:

    1. Cut the run into maneuvers (events.csv) and sort each by what was
       commanded, not by its name: nothing, throttle only, turn only, both.
    2. Read the simple things straight off the data: noise while standing
       still; each throttle step as a delayed first-order lag; the steady yaw
       rate of each spin; a circle through each arc. These are the "direct"
       values in fit.json. They start the fit and are a check on it that a
       person can redo with a ruler on a plot.
    3. Fit the simulator's own drive model (plant.py) to the same data by
       least squares, each number from the maneuvers that decide it:
           spins in place (gyro)     yaw_gain, yaw, alpha, delay to the gyro
           throttle and arcs (path)  v_max, reverse, deadband, tau, grip, pull, delay to the camera,
                                     bleed, and the turn rate while driving
       The camera gives no heading, so a measured path is compared with the
       replayed one after turning the replay about the start point to fit.
       Each number's uncertainty is the larger of what the least squares says
       and how far the answer moves when one maneuver is left out.
    4. Replay every maneuver's commands through the simulator with the fitted
       numbers and with the simulator's nominal ones, from the measured start,
       and measure how far each is from the camera. Free driving (a flight of
       the policy) is replayed once and scored in 3 s stretches, each laid
       back on the camera at its start.

Without an IMU nothing that needs a gyro is estimated (spins in place, yaw
cap, yaw acceleration, IMU noise): those are "not measured". Turn rate, bleed
and pull still come from the shape of the path and are marked path-based.

fit.json: `numbers` (one entry per number: value in the simulator's terms,
sigma, quality = measured / weak / bound / not measured, the same in real
units, the direct value, how it was obtained), `plant` and `delays` (what the
replay used), `simulator` (each number against config.SAW_DR: nominal, range,
inside or not, suggested range), `replay` (errors per maneuver), `warnings`.
"""
import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plant                                                     # noqa: E402
from plant import (                                              # noqa: E402
    NOMINAL, NOMINAL_LAG, TRAINS_OVER, V_MAX_FT, YAW_RATE_MAX, A_LONG_MAX_FT, ALPHA_MAX_RPS2, SILENCE_S,
)

PRE_S, TAIL_S = 1.0, 1.5     # how much of the standing start and of the roll-out is read with each maneuver
HOLD_MIN_S = 0.25            # a command held at least this long is a "hold"
ZERO = 0.02                  # a command smaller than this is no command
MOVED_FT = 2.0               # a maneuver that travels less than this has no direction of travel
FREE_S, WARM_S = 3.0, 1.5    # free driving (a policy flight) is scored in stretches this long; its replay starts this early
NOMINAL_DELAY = NOMINAL_LAG["act_lag"] + NOMINAL_LAG["obs_lag"]
PLANT_KEYS = [k for k in NOMINAL if k != "half_track_ft"]

# Least squares: lowest and highest value allowed, and the step used to feel out the slope
LIMITS = dict(
    v_max=(0.05, 2.0, 0.01), reverse=(0.2, 1.5, 0.01), deadband=(0.0, 0.4, 0.005), tau=(0.005, 1.5, 0.004),
    grip=(0.05, 3.0, 0.02), pull=(-0.5, 0.5, 0.004), yaw=(0.1, 3.0, 0.01), yaw_gain=(0.1, 3.0, 0.01),
    alpha=(0.05, 8.0, 0.02), bleed=(0.0, 0.95, 0.01), d_cam=(0.0, 0.8, 0.004), d_imu=(0.0, 0.8, 0.002),
)
# What one unit is in real units, and the margin a suggested training range keeps around a measurement
REAL = dict(
    v_max=(V_MAX_FT, "ft/s"), grip=(A_LONG_MAX_FT, "ft/s^2"), yaw=(YAW_RATE_MAX, "rad/s"),
    yaw_gain=(YAW_RATE_MAX, "rad/s per unit turn"), alpha=(ALPHA_MAX_RPS2, "rad/s^2"), tau=(1.0, "s"),
)
MARGIN = dict(v_max=0.10, reverse=0.08, grip=0.20, tau=0.05, deadband=0.02, yaw=0.10, yaw_gain=0.10, alpha=0.30,
              bleed=0.10, pull=0.03, act_lag=0.02, obs_lag=0.02, lag=0.03, pos_noise=0.05, dropout=0.03, hdg_noise=0.01)

wrap = lambda a: (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi
mad = lambda r: 1.4826 * float(np.median(np.abs(r - np.median(r)))) if len(r) else float("nan")


# ─── Reading a run folder ─────────────────────────────────────────────────

def read_table(path):
    """A CSV as {column: array}: floats where every value is a number,
    strings otherwise. A missing or empty file gives {}."""
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for key in (rows[0] if rows else {}):
        col = [(r[key] or "").strip() for r in rows]
        try:
            out[key.strip()] = np.array([{"": "nan", "True": "1", "False": "0"}.get(v, v) for v in col], dtype=float)
        except ValueError:
            out[key.strip()] = np.array(col, dtype=object)
    return out


def load_run(run_dir, use_imu=True):
    """Everything in a run folder, on a clock that starts at the first row."""
    d = Path(run_dir)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    cmd, pos, ev = (read_table(d / f) for f in ("commands.csv", "position.csv", "events.csv"))
    imu = read_table(d / "imu.csv") if use_imu else {}
    if "t" not in pos or len(pos["t"]) < 10:
        raise SystemExit(f"{d}: position.csv is missing or nearly empty; nothing to analyse")
    run = SimpleNamespace(dir=d, name=d.name, meta=meta, source=meta.get("source", "?"))
    run.t0 = float(np.nanmin(pos["t"]))
    if len(cmd.get("t", [])):
        run.t0 = min(run.t0, float(np.nanmin(cmd["t"])))

    mixer = meta.get("mixer", {})
    run.half_track = float(mixer.get("half_track_ft") or
                           (mixer["tread_k"] * V_MAX_FT / YAW_RATE_MAX if mixer.get("tread_k") else NOMINAL["half_track_ft"]))
    run.k = plant.tread_k(run.half_track)

    order = np.argsort(cmd["t"]) if len(cmd.get("t", [])) else np.array([], dtype=int)
    run.tc = cmd["t"][order] - run.t0 if len(order) else np.array([])
    run.left = cmd["left"][order] if len(order) else np.array([])
    run.right = cmd["right"][order] if len(order) else np.array([])

    coasting = pos.get("coasting", np.zeros(len(pos["t"]))) > 0.5
    good = np.isfinite(pos["t"] + pos["x_ft"] + pos["y_ft"])
    run.frame_t = np.sort(pos["t"][good] - run.t0)                 # every frame, coasting or not (for the frame rate)
    keep = good & ~coasting
    order = np.argsort(pos["t"][keep])
    run.tp = (pos["t"][keep] - run.t0)[order]
    run.pxy = np.array([pos["x_ft"][keep][order], pos["y_ft"][keep][order]])
    stamp = pos.get("t_stamp", pos["t"])
    run.pos_stamp_age = float(np.nanmedian(pos["t"] - stamp))

    run.has_imu = False
    if len(imu.get("t", [])) > 50:
        gyro = imu.get("gyro_z", np.full(len(imu["t"]), np.nan))
        yaw = imu.get("yaw_rad", np.full(len(imu["t"]), np.nan))
        ok = np.isfinite(imu["t"]) & (np.isfinite(gyro) | np.isfinite(yaw))
        order = np.argsort(imu["t"][ok])
        run.ti = (imu["t"][ok] - run.t0)[order]
        gyro, yaw = gyro[ok][order], yaw[ok][order]
        if np.isfinite(yaw).all():
            yaw = np.unwrap(yaw)
        if not np.isfinite(gyro).all() or np.ptp(gyro) == 0:       # no rate gyro in the log: differentiate the yaw
            gyro = np.gradient(yaw, run.ti)
        if not np.isfinite(yaw).all():                             # no yaw in the log: add the gyro up
            yaw = np.concatenate([[0.0], np.cumsum(0.5 * (gyro[1:] + gyro[:-1]) * np.diff(run.ti))])
        run.gyro, run.yaw = gyro, yaw
        run.acc = np.array([imu.get(a, np.full(len(imu["t"]), np.nan))[ok][order] for a in ("accel_x", "accel_y")])
        run.imu_stamp_age = float(np.nanmedian(imu["t"] - imu.get("t_stamp", imu["t"])))
        run.has_imu = len(run.ti) > 50

    run.events = [(float(t) - run.t0, str(e), str(m), str(det)) for t, e, m, det in
                  zip(ev.get("t", []), ev.get("event", []), ev.get("maneuver", []), ev.get("detail", []))]
    run.events.sort(key=lambda e: e[0])
    return run


def command_at(run, tq):
    """Throttle and turn in force at times tq: the last command stands, and
    the treads are centred after 0.5 s of silence, as on the PPM output."""
    if not len(run.tc):
        return np.zeros_like(tq), np.zeros_like(tq)
    i = np.clip(np.searchsorted(run.tc, tq, side="right") - 1, 0, None)
    live = (tq >= run.tc[0]) & (tq - run.tc[i] <= SILENCE_S)
    return plant.unmix(run.left[i] * live, run.right[i] * live, run.k)


# ─── Maneuvers ────────────────────────────────────────────────────────────

def find_holds(g, thr, trn):
    """Stretches where a non-zero command stands still: (t_on, t_off, throttle, turn)."""
    key = np.round(np.c_[thr, trn] / ZERO).astype(int)
    edges = np.r_[0, np.nonzero(np.any(np.diff(key, axis=0) != 0, axis=1))[0] + 1, len(g)]
    holds = []
    for a, b in zip(edges[:-1], edges[1:]):
        if g[b - 1] - g[a] >= HOLD_MIN_S and (abs(thr[a]) > ZERO or abs(trn[a]) > ZERO):
            holds.append((g[a], g[b - 1] + (g[1] - g[0]), float(np.mean(thr[a:b])), float(np.mean(trn[a:b]))))
    return holds


def cut(run, n, name, lo, hi, whole=None):
    """One stretch of the run as a segment: its commands, camera fixes and
    IMU samples, and what kind of drive it is. A test maneuver starts from
    rest. A stretch of free driving does not: it is a piece of a longer drive
    (`whole`, the segment it was cut from), which is replayed once from its
    start, and the piece of that replay is laid on the measured position."""
    free = whole is not None
    s = SimpleNamespace(n=n, name=name, lo=lo, hi=hi, warm=free, th0=0.0)
    g = np.arange(lo, hi, 0.01)
    thr, trn = command_at(run, g)
    active = (np.abs(thr) > ZERO) | (np.abs(trn) > ZERO)
    s.t_go = float(g[np.argmax(active)]) if active.any() and not free else float(lo)
    s.holds = find_holds(g, thr, trn)
    held = sum(np.sum((g >= a) & (g < b)) for a, b, _, _ in s.holds) / max(1, active.sum())
    only_thr = any(abs(trn_) < ZERO for _, _, _, trn_ in s.holds)
    only_trn = any(abs(thr_) < ZERO for _, _, thr_, _ in s.holds)
    both = any(abs(thr_) > ZERO and abs(trn_) > ZERO for _, _, thr_, trn_ in s.holds)
    s.kind = ("free" if free else "still" if not active.any() else "free" if held < 0.8 or (only_thr and only_trn)
              else "arc" if both else "throttle" if only_thr else "spin")

    # Commands to replay: from just before the robot is first asked to move (free driving: from WARM_S before it
    # begins, the driver's sticks included, so the simulated robot is already moving as the real one was)
    if free:
        s.tc, s.left, s.right, s.sg = whole.tc, whole.left, whole.right, whole.sg
    else:
        first = lo - WARM_S if s.kind == "free" else s.t_go - 0.15
        m = (run.tc >= first) & (run.tc <= hi) if len(run.tc) else np.zeros(0, bool)
        s.tc, s.left, s.right = run.tc[m], run.left[m], run.right[m]
        s.sg = np.arange(s.tc[0], hi + 0.011, 0.01) if m.any() else np.array([lo, hi])

    m = (run.tp >= lo) & (run.tp <= hi)
    tp, pxy = run.tp[m], run.pxy[:, m]
    if len(tp) >= 7:                                               # drop fixes that jump away from their neighbours
        med = np.array([np.median(np.lib.stride_tricks.sliding_window_view(np.pad(c, 2, mode="edge"), 5), axis=1)
                        for c in pxy])
        near = np.hypot(*(pxy - med)) < 1.5
        tp, pxy = tp[near], pxy[:, near]
    s.tp, s.pxy = tp, pxy
    if len(tp) < 5:
        return None
    if free:                                                       # where the camera had the robot as the stretch began
        m = tp <= lo + 0.3
        s.p0 = (np.array([np.polyval(np.polyfit(tp[m], c[m], 1), lo) for c in pxy]) if m.sum() >= 4 else pxy[:, :3].mean(axis=1))
    else:
        rest = tp <= s.t_go + 0.02
        s.p0 = np.median(pxy[:, rest] if rest.sum() >= 3 else pxy[:, :3], axis=1)
    s.travel = float(np.hypot(*(pxy - s.p0[:, None])).max())
    if run.has_imu:
        m = (run.ti >= lo) & (run.ti <= hi)
        s.ti, s.gyro, s.yaw = run.ti[m], run.gyro[m], run.yaw[m]
    return s


def find_segments(run):
    """The run cut into segments: one per maneuver in events.csv. Where there
    are none (a flight of the policy), or a maneuver is not a test drive, the
    driving is cut into stretches of FREE_S seconds, each replayed by itself."""
    ev = run.events
    starts = [(t, m, det) for t, e, m, det in ev if e == "maneuver_start"]
    taken = [t for t, e, _, _ in ev if e == "override_on"]         # the driver has the robot from here ...
    given = [t for t, e, _, _ in ev if e == "override_off"]        # ... to here
    last = run.tp[-1]
    windows = []
    for i, (t, name, det) in enumerate(starts):
        nxt = starts[i + 1][0] if i + 1 < len(starts) else float("inf")
        ends = [te for te, e, _, _ in ev if e in ("maneuver_end", "abort") and t < te <= nxt]
        end = ends[0] if ends else min(nxt, (run.tc[run.tc < nxt][-1] if (run.tc < nxt).any() else t))
        lo = max([t - PRE_S] + [x for x in given if x <= t][-1:] + [w[1] for w in windows[-1:]])
        hi = min([end + TAIL_S, nxt - 0.02, last] + [x for x in taken if x > t][:1])
        windows.append((lo, hi, name))
    if not starts:
        flown = run.meta.get("maneuvers") or [{}]
        name = str(flown[0].get("name", "drive"))
        if given:                                                  # each stretch the driver left the robot to itself
            windows = [(t, min([x for x in taken if x > t][:1] + [last]), name) for t in given]
        elif len(run.tc):                                          # no events at all: each burst of commands
            gaps = np.nonzero(np.diff(run.tc) > 1.0)[0]
            windows = [(run.tc[a] - PRE_S, min(run.tc[b] + TAIL_S, last), name)
                       for a, b in zip(np.r_[0, gaps + 1], np.r_[gaps, len(run.tc) - 1])]
        else:
            windows = [(run.tp[0], last, "whole run")]

    segs = []
    for lo, hi, name in windows:
        if hi - lo < 0.2:
            continue
        s = cut(run, len(segs), name, lo, hi)
        if s is not None and s.kind == "free":
            pieces = [cut(run, 0, f"{name} {a - lo:.0f}-{min(a + FREE_S, hi) - lo:.0f} s", a, min(a + FREE_S, hi), whole=s)
                      for a in np.arange(lo, hi - 0.5, FREE_S)]
        else:
            pieces = [s]
        for piece in pieces:
            if piece is not None:
                piece.n = len(segs)
                segs.append(piece)
    return segs


# ─── Noise and timing of the sensors ──────────────────────────────────────

def measure_noise(run, segs):
    """Scatter of the camera and the IMU while the robot stands still: the
    `still` maneuvers and the standing start of every other one."""
    out = dict(pos=float("nan"), n_pos=0, gyro=float("nan"), gyro_bias=0.0, yaw=float("nan"), yaw_drift=float("nan"),
               acc=float("nan"))
    quiet = [(s.lo, s.hi if s.kind == "still" else s.t_go) for s in segs]
    quiet = [(a, b) for a, b in quiet if b - a >= 0.3]
    dev = []
    for a, b in quiet:
        m = (run.tp >= a) & (run.tp <= b)
        if m.sum() >= 5:
            dev.append(run.pxy[:, m] - np.median(run.pxy[:, m], axis=1, keepdims=True))
    if dev:
        dev = np.concatenate(dev, axis=1).ravel()
        out.update(pos=1.4826 * float(np.median(np.abs(dev))), n_pos=dev.size // 2)

    period = float(np.median(np.diff(run.frame_t)))
    spans = [(s.lo, s.hi) for s in segs]
    want = sum(b - a for a, b in spans) / period
    got = sum(np.sum((run.tp >= a) & (run.tp <= b)) for a, b in spans)
    out.update(period=period, frames=max(want, 1.0), dropout=float(np.clip(1.0 - got / max(want, 1.0), 0.0, 1.0)))

    if run.has_imu:
        gy, yw, drift, ac = [], [], [], []
        for a, b in quiet:
            m = (run.ti >= a) & (run.ti <= b)
            if m.sum() >= 20:
                gy.append(run.gyro[m])
                line = np.polyfit(run.ti[m], run.yaw[m], 1)
                yw.append(run.yaw[m] - np.polyval(line, run.ti[m]))
                drift.append(line[0])
                ac.append(run.acc[:, m] - np.nanmedian(run.acc[:, m], axis=1, keepdims=True))
        if gy:
            gy = np.concatenate(gy)
            out.update(gyro_bias=float(np.median(gy)), gyro=mad(gy), yaw=mad(np.concatenate(yw)),
                       yaw_drift=float(np.median(drift)))
            ac = np.concatenate(ac, axis=1).ravel()
            if np.isfinite(ac).any():
                out["acc"] = mad(ac[np.isfinite(ac)])
    return out


# ─── Direct estimators: read off the data, no model ───────────────────────

def first_order(run, s, sigma):
    """One throttle-only drive as a delayed first-order lag, nothing else:
    distance along the drive = the throttle command, late by `delay`, smoothed
    with time constant `tau`, times `speed` (ft/s per unit of throttle), added
    up. Returns (speed, tau, delay)."""
    from scipy.signal import lfilter
    u = command_at(run, s.sg)[0]
    d = s.pxy - s.p0[:, None]
    far = d[:, np.hypot(*d) > 0.5 * s.travel]
    line = np.linalg.svd(far @ far.T)[0][:, 0]                     # the line it drove along
    dist = line @ d

    def travelled(speed, tau, delay):
        a = math.exp(-0.01 / tau)
        return np.interp(s.tp - delay, s.sg, np.cumsum(lfilter([1 - a], [1, -a], u)) * 0.01 * speed)
    if np.dot(dist, travelled(10.0, 0.15, 0.15)) < 0:
        dist = -dist
    sol = least_squares(lambda p: travelled(*p) - dist, [10.0, 0.15, 0.15], bounds=([0.5, 0.01, 0.0], [45.0, 1.5, 0.8]),
                        loss="soft_l1", f_scale=3 * sigma, x_scale=[5.0, 0.1, 0.1])
    return sol.x


def direct_throttle(run, L, sigma):
    """Each throttle-only drive read by itself as a delayed first-order lag
    (first_order), then top speed, dead zone and reverse from the line through
    the steady speeds of the throttle levels."""
    fwd, rev, tau, delay = {}, {}, [], []
    for s in L:
        if len(s.holds) != 1:                                      # a doublet has two gears in it; the fit takes care of it
            continue
        speed, t, d = first_order(run, s, sigma)
        level = s.holds[0][2]
        (fwd if level > 0 else rev).setdefault(round(abs(level), 2), []).append(speed * abs(level))
        tau.append(t)
        delay.append(d)

    def line(levels):
        """Speed at full command and the command that just gives none."""
        lv = sorted(levels)
        v = [float(np.median(levels[x])) for x in lv]
        if len(lv) >= 2:
            a, b = np.polyfit(lv, v, 1)
            return a + b, float(np.clip(-b / a, 0.0, 0.4))
        return v[0] / lv[0], None
    out = dict(n_fwd=len(fwd), n_rev=len(rev), top_level=max(list(fwd) + list(rev) + [0.0]), speeds={"fwd": fwd, "rev": rev},
               tau=float(np.median(tau)) if tau else None, delay=float(np.median(delay)) if delay else None)
    if fwd:
        top, db = line(fwd)
        out.update(v_max=top / V_MAX_FT, deadband=db)
        if rev:
            top_rev, _ = line(rev)
            out["reverse"] = (top_rev / top) * (1 - (db or 0.0)) + (db or 0.0)
    return out


def direct_spin(S, bias):
    """Steady yaw rate of each turn hold (median gyro over its second half),
    the slope of the gyro on the way up, and when the way up began."""
    ratio, rate, slope, onset = {}, [], [], []
    for s in S:
        for t_on, t_off, _, trn in s.holds:
            w = (s.ti > t_on + 0.5 * (t_off - t_on) + 0.05) & (s.ti < t_off)
            if w.sum() < 5:
                continue
            om = float(np.median(s.gyro[w])) - bias
            ratio.setdefault(round(trn, 2), []).append(om / (trn * YAW_RATE_MAX))
            rate.append(abs(om))
            up = (s.gyro - bias) * np.sign(om)
            i = np.nonzero((s.ti > t_on) & (up > 0.7 * abs(om)))[0]
            j = np.nonzero((s.ti > t_on) & (up < 0.2 * abs(om)) & (s.ti < (s.ti[i[0]] if len(i) else t_off)))[0]
            if len(i) and len(j) and i[0] - j[-1] >= 3:
                a, b = np.polyfit(s.ti[j[-1]:i[0] + 1], up[j[-1]:i[0] + 1], 1)
                slope.append(a)
                onset.append(-b / a - t_on)
    if not ratio:
        return None
    by_level = {}
    for lvl, r in ratio.items():
        by_level.setdefault(abs(lvl), []).append(float(np.median(r)))
    gain = max(float(np.mean(r)) for r in by_level.values())
    left = [np.median(r) for lvl, r in ratio.items() if lvl > 0]
    right = [np.median(r) for lvl, r in ratio.items() if lvl < 0]
    return dict(yaw_gain=gain, top=max(rate) / YAW_RATE_MAX,
                alpha=float(np.median(slope)) / ALPHA_MAX_RPS2 if slope else None,
                d_imu=float(np.clip(np.median(onset), 0.0, 0.5)) if onset else None,
                left_right=float(max(left) / max(right) - 1.0) if left and right else None,
                ratios={str(k): float(np.median(v)) for k, v in sorted(ratio.items())})


def direct_arc(A, num, sigma):
    """Each steady arc as a circle: turn rate is how fast the robot goes round
    its centre, speed is radius x turn rate. From those, the turn rate per unit
    of turn command and the share of speed the turn costs. Only arcs long
    enough to show a clean piece of circle are read this way.

    Also which way every arc bends, from how the line from its start to the
    robot swings round: `mirrored` is True when most bend against their turn
    command."""
    gain, bleed, bends = [], [], []
    for s in A:
        for t_on, t_off, thr, trn in s.holds:
            if abs(thr) < ZERO or abs(trn) < ZERO:
                continue
            d = s.pxy - s.p0[:, None]
            far = (np.hypot(*d) > max(1.0, 5 * sigma)) & (s.tp < t_off + num["d_cam"] + 0.3)
            if far.sum() >= 6:
                bearing = np.unwrap(np.arctan2(d[1, far], d[0, far]))
                swing = np.median(bearing[-3:]) - np.median(bearing[:3])
                if abs(swing) > 0.15:
                    bends.append(np.sign(swing) == np.sign(trn))
            w = (s.tp > t_on + num["d_cam"] + 0.4 * (t_off - t_on)) & (s.tp < t_off + 0.3 * num["d_cam"])
            if w.sum() < 12 or np.ptp(s.tp[w]) < 0.6:
                continue
            x, y = s.pxy[:, w]
            a, b, c = np.linalg.lstsq(np.c_[x, y, np.ones_like(x)], -(x * x + y * y), rcond=None)[0]
            cx, cy = -a / 2, -b / 2
            radius = math.sqrt(max(cx * cx + cy * cy - c, 1e-9))
            off_circle = np.hypot(x - cx, y - cy) - radius
            phi = np.unwrap(np.arctan2(y - cy, x - cx))
            if np.ptp(phi) < 0.5 or radius < 0.5 or np.sqrt(np.mean(off_circle ** 2)) > 3 * sigma:
                continue                                           # not a clean piece of a circle
            omega = np.polyfit(s.tp[w], phi, 1)[0]
            eff = thr * (num["reverse"] if thr < 0 else 1.0)
            eff = np.sign(eff) * max(abs(eff) - num["deadband"], 0.0) / (1 - num["deadband"])
            cap = YAW_RATE_MAX * num["yaw"]
            gain.append((omega - num["pull"] * eff * cap) / (trn * YAW_RATE_MAX))
            straight = abs(eff) * num["v_max"] * V_MAX_FT
            if straight > 0.5:
                bleed.append((1 - radius * abs(omega) / straight) * cap / abs(omega))
    return dict(yaw_gain=float(np.median(gain)) if gain else None, bleed=float(np.median(bleed)) if bleed else None,
                mirrored=len(bends) > 0 and sum(bends) < len(bends) / 2)


# ─── Fitting the simulator's drive model ──────────────────────────────────

def replay_cols(seg, num, names=(), X=None):
    """Replay a segment's commands from rest at the origin, once per row of X
    (the numbers `names` take that row's values; the rest come from `num`).
    Returns the simulator's output on seg.sg, one column per row, and each
    column's delays."""
    k = 1 if X is None else len(X)
    cols = {key: np.full(k, float(num[key])) for key in list(NOMINAL) + ["d_cam", "d_imu"]}
    for i, name in enumerate(names):
        cols[name] = X[:, i]
    out = plant.simulate(seg.tc, seg.left, seg.right, (0.0, 0.0, 0.0), t_out=seg.sg, **{key: cols[key] for key in NOMINAL})
    if k == 1:
        out = {key: (v if key == "t" else v[:, None]) for key, v in out.items()}
    return out, cols


def path_error(seg, sx, sy, delay, until=None):
    """Measured minus replayed position at each camera fix, (2, N), with the
    replay turned about the start point by the angle that fits best. That
    angle is the heading the robot started the maneuver with (for a stretch
    of free driving: the turn from the replay's frame to the floor's)."""
    m = slice(None) if until is None else seg.tp <= until
    at = lambda t: np.array([np.interp(t - delay, seg.sg, sx), np.interp(t - delay, seg.sg, sy)])
    a = at(seg.tp[m]) - (at(seg.lo)[:, None] if seg.warm else 0.0)
    b = (seg.pxy - seg.p0[:, None])[:, m]
    th = math.atan2(np.sum(a[0] * b[1] - a[1] * b[0]), np.sum(a[0] * b[0] + a[1] * b[1]))
    c, s = math.cos(th), math.sin(th)
    return b - np.array([c * a[0] - s * a[1], s * a[0] + c * a[1]]), th


def solve(names, x0, residual_cols):
    """Least squares over the numbers `names`. residual_cols(X) returns one
    column of residuals (in units of the noise) per row of X. Large residuals
    count less (soft-L1), so a few bad camera fixes do not steer the answer.

    Returns the numbers, their covariance, the rms residual, and the pieces
    the covariance was made from (residuals, slopes, weights). The covariance
    is widened when the residuals are larger than the noise or hang together
    in time (the model does not have the shape of the data). A number the
    data does not react to at all gets an infinite variance."""
    lo, hi, h = (np.array([LIMITS[n][i] for n in names]) for i in range(3))
    x0 = np.clip(np.asarray(x0, dtype=float), lo + 1e-6, hi - 1e-6)
    memo = {}

    def both(x):
        """Residuals and their slopes at x, from one batched replay."""
        key = x.tobytes()
        if key not in memo:
            memo.clear()
            step = np.where(x + h < hi, h, -h)
            R = residual_cols(np.vstack([x, x + np.diag(step)]))
            memo[key] = (R[:, 0], (R[:, 1:] - R[:, :1]) / step)
        return memo[key]

    sol = least_squares(lambda x: both(x)[0], x0, jac=lambda x: both(x)[1], bounds=(lo, hi), x_scale=10 * h,
                        loss="soft_l1", f_scale=3.0, max_nfev=40, xtol=1e-6, ftol=1e-8)
    r, J = both(sol.x)
    w = 1.0 / np.sqrt(1.0 + (r / 3.0) ** 2)
    rho = float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], 0.0, 0.95))
    widen = max(1.0, 1.4826 * float(np.median(np.abs(r)))) * math.sqrt((1 + rho) / (1 - rho))
    A = (J * w[:, None]).T @ J
    live = np.diag(A) * h * h > 1e-3                               # one step of this number changes something
    cov = np.full((len(names), len(names)), np.inf)
    if live.any():
        cov[np.ix_(live, live)] = widen ** 2 * np.linalg.pinv(A[np.ix_(live, live)])
    return sol.x, cov, float(np.sqrt(np.mean(r * r))), (r, J, w, live, widen)


def fit_group(segs, num, names, signal, sigma, bias=0.0):
    """Fit the numbers `names` to the camera path ("path") or the gyro
    ("gyro") of a group of segments, and update `num`.

    The uncertainty of each number is the larger of two: the one the least
    squares gives (solve), and how much the answer moves when each maneuver in
    turn is left out (a jackknife, taken as one least-squares step from the
    answer so it costs no more replays). The second is the honest one when
    the maneuvers disagree with each other more than the noise explains.

    Returns sigma by name, the rms residual in units of the noise, and two
    functions: the sigma of a sum of two of the numbers, and the range of
    values of one number the data allows (see `allowed`)."""
    rows = []                                                      # residuals per segment, to take one out again

    def residual_cols(X):
        parts = []
        for s in segs:
            out, cols = replay_cols(s, num, names, X)
            if signal == "path":
                parts.append(np.stack([path_error(s, out["x_ft"][:, j], out["y_ft"][:, j], cols["d_cam"][j])[0].ravel()
                                       for j in range(len(X))], axis=1) / sigma)
            else:
                parts.append(np.stack([s.gyro - bias - np.interp(s.ti - cols["d_imu"][j], s.sg, out["omega_rps"][:, j])
                                       for j in range(len(X))], axis=1) / sigma)
        rows[:] = [len(p) for p in parts]
        return np.vstack(parts)
    x, cov, rms, (r, J, w, live, widen) = solve(names, [num[n] for n in names], residual_cols)
    num.update(zip(names, map(float, x)))

    n, ends = len(segs), np.cumsum([0] + rows)
    left_out = np.tile(x, (n, 1))
    if n > 1 and live.any():
        Jl = J[:, live]
        A, g = (Jl * w[:, None]).T @ Jl, Jl.T @ (w * r)
        for i in range(n):
            m = slice(ends[i], ends[i + 1])
            Ai, gi = (Jl[m] * w[m, None]).T @ Jl[m], Jl[m].T @ (w[m] * r[m])
            left_out[i, live] = x[live] - np.linalg.pinv(A - Ai) @ (g - gi)
    spread = lambda v: math.sqrt((n - 1) / n * float(np.sum((v - v.mean()) ** 2)))
    sig = {name: float(max(math.sqrt(cov[i, i]), spread(left_out[:, i]))) for i, name in enumerate(names)}

    def sigma_of_sum(a, b):
        i, j = names.index(a), names.index(b)
        var = cov[i, i] + cov[j, j] + 2 * cov[i, j]
        return float(max(math.sqrt(max(var, 0.0)) if np.isfinite(var) else float("inf"), spread(left_out[:, i] + left_out[:, j])))

    def allowed(name, grid):
        """The lowest and highest value in `grid` that `name` can be held at
        without the fit getting clearly worse (2 sigma), the other numbers
        re-adjusting to first order, and the value that fits best. For a
        number that acts through a limit, which the data may only bound from
        one side."""
        i = names.index(name)
        X = np.tile(x, (len(grid), 1))
        X[:, i] = grid
        R = residual_cols(X)
        others = live & (np.arange(len(names)) != i)
        Jo, cost = J[:, others], []
        for c in range(len(grid)):
            wc = 1.0 / np.sqrt(1.0 + (R[:, c] / 3.0) ** 2)
            g = Jo.T @ (wc * R[:, c])
            cost.append(18.0 * np.sum(np.sqrt(1.0 + (R[:, c] / 3.0) ** 2) - 1.0) - g @ np.linalg.pinv((Jo * wc[:, None]).T @ Jo) @ g)
        ok = (np.array(cost) - min(cost)) / widen ** 2 < 4.0
        return float(grid[ok].min()), float(grid[ok].max()), float(grid[int(np.argmin(cost))])
    return sig, rms, sigma_of_sum, allowed


def entry(value, sigma=None, quality="measured", how="", note="", direct=None, unit="", real=None, real_sigma=None):
    """One number of fit.json. `value` is in the simulator's terms, `real`
    in real units; quality is measured / weak / bound / not measured."""
    f = lambda v: None if v is None or not np.isfinite(v) else float(v)
    return dict(value=f(value), sigma=f(sigma), quality=quality, real=f(real if real is not None else value),
                real_sigma=f(real_sigma if real_sigma is not None else sigma), unit=unit, direct=f(direct),
                how=how, note=note)


def judged(name, value, sigma, **kw):
    """An entry for a fitted simulator number, with its quality judged from
    its uncertainty against the width of the range the simulator trains over."""
    scale, unit = REAL.get(name, (1.0, kw.pop("unit", "")))
    lo, hi = TRAINS_OVER.get(name, (0.0, abs(value) or 1.0))
    quality = ("not measured" if not np.isfinite(sigma) else "weak" if sigma > 0.15 * (hi - lo) else "measured")
    if quality == "not measured":
        kw["note"] = (kw.get("note", "") + " The data does not react to this number.").strip()
    return entry(value, sigma, kw.pop("quality", quality), unit=unit, real=kw.pop("real", value * scale),
                 real_sigma=sigma * scale if np.isfinite(sigma) else None, **kw)


def fit_run(run, segs, prior, say=print):
    """The whole fit. Returns the numbers for fit.json, the plant numbers
    and delays to replay with, the noise measurements, and warnings about
    the data (a sign or a frame the wrong way round)."""
    num = dict(prior, half_track_ft=run.half_track)
    num.setdefault("d_cam", NOMINAL_DELAY)
    num.setdefault("d_imu", NOMINAL_LAG["act_lag"])
    res, warnings = {}, []
    noise = measure_noise(run, segs)
    sig_p = noise["pos"] if np.isfinite(noise["pos"]) and noise["n_pos"] >= 20 else 0.2
    L = [s for s in segs if s.kind == "throttle" and s.travel >= MOVED_FT and len(s.tc)]
    S = [s for s in segs if s.kind == "spin" and run.has_imu and len(s.tc) and len(s.ti) > 20]
    A = [s for s in segs if s.kind == "arc" and s.travel >= MOVED_FT and len(s.tc)]
    path_based = "" if run.has_imu else " Path-based: no IMU in this run."

    # ── Turn only: the gyro against the replayed yaw rate ──
    if S:
        d = direct_spin(S, noise["gyro_bias"]) or {}
        if d.get("yaw_gain", 1.0) < 0:
            # The robot turns against the command on every spin: the IMU counts the other way round from the
            # simulator (counter-clockwise positive, seen from above). Turn it over rather than fit nonsense.
            warnings.append("The gyro reads NEGATIVE for a turn command to the left. The simulator's yaw is positive "
                            "counter-clockwise seen from above; the IMU's sign was flipped for this analysis. Check the IMU "
                            "mounting or the tread wiring before trusting the heading in a fight.")
            noise["gyro_bias"] *= -1
            for obj in [run] + [s for s in segs if hasattr(s, "ti")]:
                obj.gyro, obj.yaw = -obj.gyro, -obj.yaw
            d = direct_spin(S, noise["gyro_bias"]) or {}
        num.update({k: d[k] for k in ("yaw_gain", "alpha", "d_imu") if d.get(k)})
        num["yaw"] = 1.05 * d["top"] if d.get("top") else num["yaw"]
        sig_g = max(noise["gyro"], 0.01) if np.isfinite(noise["gyro"]) else 0.05
        sig, rms, _, _ = fit_group(S, num, ["yaw_gain", "yaw", "alpha", "d_imu"], "gyro", sig_g, noise["gyro_bias"])
        say(f"  gyro: {len(S)} spins in place, residual {rms:.1f} x gyro noise")
        # Is the yaw cap ever what holds the replay back? If not, the spins only say it is at least what they reached.
        cap = YAW_RATE_MAX * num["yaw"]
        om = np.concatenate([np.abs(replay_cols(s, num)[0]["omega_rps"][:, 0]) for s in S])
        capped, top = np.mean(om > 0.99 * cap) * len(om) * 0.01 > 0.15, float(om.max())
        how = f"simulator fitted to the gyro of {len(S)} spins in place"
        res["yaw_gain"] = judged("yaw_gain", num["yaw_gain"], sig["yaw_gain"], direct=d.get("yaw_gain"), how=how)
        res["alpha"] = judged("alpha", num["alpha"], sig["alpha"], direct=d.get("alpha"), how=how + " (the ramp of the yaw rate)")
        if capped and np.isfinite(sig["yaw"]):
            res["yaw"] = judged("yaw", num["yaw"], sig["yaw"], direct=d.get("top"), how=how + " (where the yaw rate stops rising)")
        else:
            res["yaw"] = judged("yaw", top / YAW_RATE_MAX, 0.0, quality="bound", how=how,
                                note="At least this: the spins never ran into a cap, so it may be higher.")
            num["yaw"] = max(prior["yaw"], 1.02 * top / YAW_RATE_MAX)
        res["delay_imu"] = entry(num["d_imu"], sig["d_imu"], "measured" if sig["d_imu"] < 0.03 else "weak", unit="s",
                                 direct=d.get("d_imu"), how="command sent to the gyro moving (command delay + IMU link)")
        if d.get("left_right") is not None:
            res["spin_left_right"] = entry(d["left_right"], None, "measured", unit="share",
                                           how="steady yaw rate turning left over turning right, minus 1",
                                           note="The simulator has no such number; a large one argues for a wider yaw_gain.")
    else:
        why = ("no spin-in-place maneuvers in this run" if run.has_imu else
               "needs a gyro: no IMU in this run, and the camera gives no heading while the robot turns on the spot")
        for name in ("yaw", "alpha"):
            res[name] = entry(None, None, "not measured", note=why, unit=REAL[name][1])
        res["delay_imu"] = entry(None, None, "not measured", note=why, unit="s")

    # ── Throttle only, then throttle and arcs together: the camera path against the replayed path ──
    names, d, da, spin_gain = [], {}, {}, num["yaw_gain"]
    if L:
        d = direct_throttle(run, L, sig_p)
        names = ["v_max", "tau", "grip", "pull", "d_cam"]
        if "v_max" not in d:                                       # only reverse drives: reverse cannot be told from top speed
            names[0] = "reverse"
        else:
            num["v_max"] = d["v_max"]
            if d["n_fwd"] >= 2 or d["n_rev"] >= 2:
                names.append("deadband")
                num["deadband"] = d["deadband"] if d["deadband"] is not None else num["deadband"]
            if d["n_rev"]:
                names.append("reverse")
                num["reverse"] = d.get("reverse", num["reverse"])
        if d["delay"] is not None:
            num.update(d_cam=float(np.clip(d["delay"], 0.02, 0.5)), tau=float(np.clip(d["tau"], 0.02, 1.0)))
        sig, rms, sigma_of_sum, allowed = fit_group(L, num, names, "path", sig_p)
    group, arcs_apart = L, ""
    if A:
        # The arcs lean on the throttle numbers (speed in the arc against speed in a straight line), so they are
        # fitted together with them: the uncertainty of one then carries into the other.
        da = direct_arc(A, num, sig_p)
        if da["mirrored"]:
            warnings.append("The arcs curve the OPPOSITE way to the turn command. Either the camera's arena frame is a mirror "
                            "image of the simulator's (x and y along two walls, counter-clockwise positive seen from above) or "
                            "the treads are swapped. The turn numbers below are not to be trusted until that is settled.")
        alone = (dict(num), sig, rms, sigma_of_sum, allowed) if L else None
        num.update({k: float(np.clip(da[k], *LIMITS[k][:2])) for k in ("yaw_gain", "bleed") if da[k] is not None})
        # With no spins to say how fast the robot swings into a turn, the arcs have to be allowed their own answer,
        # or a wrong one bends the delay and tau to make up for it. It is not reported as a measurement.
        arc_names = ["bleed", "yaw_gain"] + ([] if S else ["alpha"])
        group, names = L + A, names + arc_names
        sig, rms, sigma_of_sum, allowed = fit_group(group, num, names, "path", sig_p)
        if L and rms > 1.5 * max(alone[2], 1.0):
            # The arcs do not move the way the model does (a robot that slides, say). Take the throttle numbers
            # from the straight drives alone, and fit the arcs by themselves so they cannot bend them.
            arcs_apart = (f" The arcs do not fit: with them the path fit is {rms:.1f} x the camera noise off, without them "
                          f"{alone[2]:.1f} x. A robot that slides in its turns does this, and so does a way of turning the "
                          f"simulator's model does not have. They were fitted apart from the straight drives.")
            keep = {k: num[k] for k in arc_names}
            num.update(alone[0], **keep)
            sig_arc = fit_group(A, num, arc_names, "path", sig_p)[0]
            group, names = L, names[:-len(arc_names)]
            sig, rms, sigma_of_sum, allowed = dict(alone[1], **sig_arc), alone[2], alone[3], alone[4]
    if L:
        # Grip acts through a limit (the hardest acceleration), and a slow drive response hides it. The search can
        # stop where grip no longer matters, short of the best fit: look along grip itself and start again from there.
        grips = np.linspace(0.1, 1.5 * TRAINS_OVER["grip"][1], 33)
        g_lo, g_hi, g_best = allowed("grip", grips)
        if not g_lo <= num["grip"] <= g_hi:
            num["grip"] = g_best
            again = fit_group(group, num, names, "path", sig_p)
            sig, rms, sigma_of_sum, allowed = dict(sig, **again[0]), again[1], again[2], again[3]
            g_lo, g_hi, g_best = allowed("grip", grips)
    if A and not S:
        res["alpha"]["note"] += (f" (Inside the path fit the arcs were given their own value, {num['alpha']:.2f} +- "
                                 f"{sig['alpha']:.2f}: too rough to report, and not used in the replay.)")
        num["alpha"] = prior["alpha"]
    if L or A:
        say(f"  camera path: {len(L)} throttle-only maneuvers and {len(A)} arcs, residual {rms:.1f} x camera noise")
    how = f"simulator fitted to the camera path of {len(L)} throttle-only maneuvers" + (f" and {len(A)} arcs" if A else "")

    if L:
        ext = "" if d["top_level"] >= 0.95 else f"Extrapolated: the hardest throttle driven was {d['top_level']:.2f}."
        if "v_max" in sig:
            res["v_max"] = judged("v_max", num["v_max"], sig["v_max"], direct=d.get("v_max"), how=how, note=ext)
        if "reverse" in sig:
            back = num["v_max"] * V_MAX_FT * (num["reverse"] - num["deadband"]) / (1 - num["deadband"])
            res["reverse"] = judged("reverse", num["reverse"], sig["reverse"], direct=d.get("reverse"), how=how, unit="ft/s backwards",
                                    real=back, note="" if "v_max" in sig else "No forward drive in this run: this is reverse "
                                    "speed over the top speed assumed beforehand.")
            res["reverse"]["real_sigma"] = None
        if "deadband" in sig:
            res["deadband"] = judged("deadband", num["deadband"], sig["deadband"], direct=d.get("deadband"), how=how,
                                     unit="share of throttle")
        res["tau"] = judged("tau", num["tau"], sig["tau"], direct=d["tau"], how=how + " (the shape of each start and stop)")
        if g_hi >= grips[-1] - 1e-9:
            res["grip"] = judged("grip", g_lo, 0.0, quality="bound", how=how + " (the hardest acceleration)",
                                 note="At least this: the drive response (tau), not traction, set the hardest acceleration, so "
                                      "the data cannot tell this value from any higher one.")
            num["grip"] = max(prior["grip"], g_lo)
        else:
            wide = max(sig["grip"] if np.isfinite(sig["grip"]) else 0.0, (g_hi - g_lo) / 4)
            res["grip"] = judged("grip", num["grip"], wide, how=how + " (the hardest acceleration)",
                                 note="Acceleration limit only. The simulator uses the same number for sideways slip.")
        res["pull"] = judged("pull", num["pull"], sig["pull"], how=how + " (how the path curves)",
                             real=num["pull"] * num["yaw"] * YAW_RATE_MAX, note=(
                                 f"Real value is the yaw rate picked up at full throttle. As a share it assumes a yaw cap of "
                                 f"{num['yaw']:.2f}; what the path measures is pull x yaw cap." + path_based))
        res["pull"]["unit"] = "rad/s at full throttle"
        res["pull"]["real_sigma"] = sig["pull"] * num["yaw"] * YAW_RATE_MAX if np.isfinite(sig["pull"]) else None
        res["delay_total"] = entry(num["d_cam"], sig["d_cam"], "measured" if sig["d_cam"] < 0.03 else "weak", unit="s", direct=d["delay"],
                                   how="command sent to the camera position moving (act_lag + obs_lag), from the same fit")
        both = sigma_of_sum("d_cam", "tau")
        res["lag_plus_tau"] = entry(num["d_cam"] + num["tau"], both, "measured" if both < 0.03 else "weak", unit="s",
                                    direct=None if d["delay"] is None else d["delay"] + d["tau"],
                                    how="sum of delay_total and tau: how far behind its commands the robot runs",
                                    note="Better determined than either part: a later start and a slower response look alike.")
        res["speeds_direct"] = {k: {str(l): float(np.median(v)) for l, v in sorted(lv.items())} for k, lv in d["speeds"].items()}
    else:
        for name in ("v_max", "reverse", "deadband", "tau", "grip", "pull"):
            res.setdefault(name, entry(None, None, "not measured", note="no throttle-only maneuvers in this run",
                                       unit=REAL.get(name, (1, ""))[1]))
        res["delay_total"] = entry(None, None, "not measured", note="no throttle-only maneuvers in this run", unit="s")
    for name in ("reverse", "deadband"):
        res.setdefault(name, entry(None, None, "not measured", unit="share",
                                   note="needs reverse drives" if name == "reverse" else "needs two or more throttle levels"))

    if A:
        res["bleed"] = judged("bleed", num["bleed"], sig["bleed"], direct=da["bleed"],
                              how=how + " (speed in the arc against speed in a straight line)", unit="share of speed at the yaw cap",
                              note=f"Assumes a yaw cap of {num['yaw']:.2f}; what the arcs measure is bleed / yaw cap = "
                                   f"{num['bleed'] / num['yaw']:.3f}." + path_based + arcs_apart)
        arc = judged("yaw_gain", num["yaw_gain"], sig["yaw_gain"], direct=da["yaw_gain"], how=how + " (how tightly the arcs curve)",
                     note=("Turn rate while driving, from the path. Assumes the arcs stayed under the yaw cap." + path_based
                           + arcs_apart).strip())
        if arcs_apart:
            res["bleed"]["quality"] = "weak"
            warnings.append(arcs_apart.strip())
        if S:
            res["yaw_gain_arc"] = arc
            if abs(num["yaw_gain"] - spin_gain) > 0.1 * spin_gain + 3 * (arc["sigma"] or 0):
                res["yaw_gain_arc"]["note"] += " Differs from the spins in place: the simulator has one number for both."
            num["yaw_gain"] = spin_gain
        else:
            res["yaw_gain"] = arc
    else:
        res["bleed"] = entry(None, None, "not measured", note="no arc maneuvers in this run", unit="share")
        res.setdefault("yaw_gain", entry(None, None, "not measured", unit=REAL["yaw_gain"][1],
                                         note="no arcs in this run" + ("" if run.has_imu else ", and no IMU for the spins")))

    # ── Splitting the delay, as far as the data allows ──
    total, imu = res["delay_total"], res["delay_imu"]
    stamped = run.pos_stamp_age > 0.005
    if total["value"] is not None and stamped:
        res["obs_lag"] = entry(run.pos_stamp_age, None, "weak", unit="s", how="arrival time minus the position message's own stamp",
                               note="Right only if that stamp is when the frame was taken; it misses the camera's own delay.")
        res["act_lag"] = entry(total["value"] - run.pos_stamp_age, total["sigma"], "weak", unit="s", how="delay_total minus obs_lag")
    elif total["value"] is not None and imu["value"] is not None:
        res["act_lag"] = entry(imu["value"], imu["sigma"], "bound", unit="s", how="delay to the gyro",
                               note="At most this: it still contains the IMU link's own delay.")
        res["obs_lag"] = entry(total["value"] - imu["value"], math.hypot(total["sigma"] or 0, imu["sigma"] or 0), "bound", unit="s",
                               how="delay_total minus the delay to the gyro", note="At least this, for the same reason.")
    else:
        why = ("Only their sum is measured (delay_total): the position messages are stamped when they are sent, not when "
               "the frame was taken, and there is no gyro to time the command alone.")
        res["act_lag"] = entry(None, None, "not measured", unit="s", note=why)
        res["obs_lag"] = entry(None, None, "not measured", unit="s", note=why)

    # ── Sensors ──
    res["pos_noise"] = entry(noise["pos"], 1.17 * noise["pos"] / math.sqrt(2 * noise["n_pos"]) if noise["n_pos"] else None,
                             "measured" if noise["n_pos"] >= 20 else "not measured", unit="ft",
                             how=f"scatter of {noise['n_pos']} camera fixes while standing still (per axis)")
    res["dropout"] = entry(noise["dropout"], math.sqrt(max(noise["dropout"], 0.005) / noise["frames"]), "measured", unit="share of frames",
                           how=f"camera frames missing or coasting, at one frame per {noise['period'] * 1000:.0f} ms")
    res["camera_period"] = entry(noise["period"], None, "measured", unit="s", how="median time between position messages")
    if run.has_imu and np.isfinite(noise["gyro"]):
        res["gyro_noise"] = entry(noise["gyro"], None, "measured", unit="rad/s", how="gyro scatter while standing still",
                                  note=f"Bias {noise['gyro_bias']:+.4f} rad/s.")
        res["hdg_noise"] = entry(noise["yaw"], None, "measured", unit="rad", how="IMU yaw scatter while standing still",
                                 note=f"Yaw drifts {noise['yaw_drift']:+.4f} rad/s while standing still.")
        res["accel_noise"] = entry(noise["acc"], None, "measured" if np.isfinite(noise["acc"]) else "not measured",
                                   unit="the log's own units", how="accelerometer scatter while standing still")
    else:
        why = "no IMU in this run" if not run.has_imu else "no standing-still stretch with IMU samples"
        for name, unit in (("gyro_noise", "rad/s"), ("hdg_noise", "rad"), ("accel_noise", "")):
            res[name] = entry(None, None, "not measured", unit=unit, note=why)
    res["side_slip"] = entry(None, None, "not measured", unit="rad",
                             note="Needs a heading that does not come from the path itself." if not run.has_imu else "")
    return res, num, noise, warnings


# ─── Replay: the same commands through the simulator ──────────────────────

def both_replays(run, segs, num):
    """Each segment's commands through the simulator from rest at the origin,
    column 0 with the fitted numbers and column 1 with the nominal ones.
    Stretches cut from one free drive share one replay."""
    nominal, done = dict(NOMINAL, half_track_ft=run.half_track), {}
    for s in segs:
        if len(s.tc) and id(s.tc) not in done:
            done[id(s.tc)] = plant.simulate(s.tc, s.left, s.right, (0.0, 0.0, 0.0), t_out=s.sg,
                                            **{k: np.array([num[k], nominal[k]]) for k in NOMINAL})
        s.out = done.get(id(s.tc)) if len(s.tc) else None


def start_headings(run, segs, num):
    """The heading each maneuver started with: the turn that lines the fitted
    replay up with the measured path (free driving: with the first second of
    each stretch). Where the robot does not travel, the IMU's yaw if there is
    one. Returns the IMU yaw's offset from the arena frame over time, as
    (times, offsets) from the throttle-only maneuvers, and over all of them
    its mean and how much it wanders."""
    offs, straight = [], []
    for s in segs:
        if s.travel >= MOVED_FT and s.out is not None:
            s.th0 = path_error(s, s.out["x_ft"][:, 0], s.out["y_ft"][:, 0], num["d_cam"], s.lo + 1.0 if s.warm else None)[1]
            if s.warm:      # free driving has no standing start to share: the nominal replay gets its own best turn
                s.th0_nominal = path_error(s, s.out["x_ft"][:, 1], s.out["y_ft"][:, 1], NOMINAL_DELAY, s.lo + 1.0)[1]
            if run.has_imu and len(s.ti) and not s.warm:
                offs.append(float(wrap(s.th0 - np.interp(s.t_go, s.ti, s.yaw))))
                if s.kind == "throttle":
                    straight.append((s.t_go, offs[-1]))
    off = float(np.arctan2(np.mean(np.sin(offs)), np.mean(np.cos(offs)))) if offs else 0.0
    for s in segs:
        if run.has_imu and len(s.ti) and not (s.travel >= MOVED_FT and s.out is not None):
            s.th0 = float(np.interp(s.t_go, s.ti, s.yaw)) + off
    track = (np.array([t for t, _ in straight]), off + wrap(np.array([o for _, o in straight]) - off)) if straight else None
    return track, off, (mad(wrap(np.array(offs) - off)) if len(offs) > 1 else None)


def placed(seg, out, col, delay, t):
    """Column `col` of a replay put on the floor at times t: turned by the
    start heading and moved to the measured start. Returns x, y, heading."""
    at = lambda key, when: np.interp(np.asarray(when) - delay, seg.sg, out[key][:, col])
    x, y = at("x_ft", t), at("y_ft", t)
    if seg.warm:
        x, y = x - at("x_ft", seg.lo), y - at("y_ft", seg.lo)
    th0 = getattr(seg, "th0_nominal", seg.th0) if col == 1 else seg.th0
    c, s = math.cos(th0), math.sin(th0)
    return seg.p0[0] + c * x - s * y, seg.p0[1] + s * x + c * y, th0 + at("heading_rad", t)


def replay(run, segs, num):
    """Every segment's commands through the simulator with the fitted numbers
    and with the nominal ones, from the measured start. Returns the rows of
    replay.csv and one summary per maneuver."""
    rows, summary = [], []
    for s in segs:
        fine, out = s.sg[(s.sg >= s.lo) & (s.sg <= s.hi)], s.out
        if out is not None:
            fit, nom = placed(s, out, 0, num["d_cam"], s.tp), placed(s, out, 1, NOMINAL_DELAY, s.tp)
            s.plot = SimpleNamespace(fit=placed(s, out, 0, num["d_cam"], fine)[:2], nom=placed(s, out, 1, NOMINAL_DELAY, fine)[:2])
        else:
            fit = nom = [np.full(len(s.tp), v) for v in (s.p0[0], s.p0[1], s.th0)]
            s.plot = SimpleNamespace(fit=fit[:2], nom=nom[:2])
        s.plot.e_fit = e_fit = np.hypot(s.pxy[0] - fit[0], s.pxy[1] - fit[1])
        s.plot.e_nom = e_nom = np.hypot(s.pxy[0] - nom[0], s.pxy[1] - nom[1])
        hdg = np.full(len(s.tp), np.nan)
        if run.has_imu and len(s.ti) > 2:
            # the IMU's change of heading since the start, read at the moment each camera fix shows, on the fitted start
            shift = num["d_cam"] - num["d_imu"]
            hdg = (placed(s, out, 0, num["d_cam"], [s.t_go])[2] if out is not None else s.th0) \
                + np.interp(s.tp - shift, s.ti, s.yaw) - np.interp(s.t_go - shift, s.ti, s.yaw)
        for i in range(len(s.tp)):
            rows.append([s.tp[i] + run.t0, s.name, s.kind, s.pxy[0, i], s.pxy[1, i], hdg[i], fit[0][i], fit[1][i], fit[2][i],
                         e_fit[i], nom[0][i], nom[1][i], nom[2][i], e_nom[i]])
        seen = np.isfinite(hdg).any()
        summary.append(dict(maneuver=s.name, kind=s.kind, seconds=round(float(s.hi - s.lo), 2), travel_ft=round(s.travel, 1),
                            fit_typical_ft=float(np.median(e_fit)), fit_worst_ft=float(e_fit.max()),
                            nominal_typical_ft=float(np.median(e_nom)), nominal_worst_ft=float(e_nom.max()),
                            fit_heading_worst_rad=float(np.nanmax(np.abs(wrap(hdg - fit[2])))) if seen else None,
                            nominal_heading_worst_rad=float(np.nanmax(np.abs(wrap(hdg - nom[2])))) if seen else None))
    return rows, summary


def side_slip(run, segs, num, track, sigma):
    """Sideways over forward travel in the steady part of each arc, from the
    camera's travel against the IMU's heading (set on the arena frame by the
    straight drives before and after). Returns, for the arc where it stands
    out most from the camera noise: the slip angle, its one-sigma noise, the
    sideways acceleration (speed x yaw rate) and the arc's name."""
    worst = None
    shift = num["d_cam"] - num["d_imu"]
    for s in segs:
        if s.kind != "arc" or len(s.ti) < 20 or track is None:
            continue
        for t_on, t_off, _, _ in s.holds:
            w = np.nonzero((s.tp > t_on + num["d_cam"] + 0.4 * (t_off - t_on)) & (s.tp < t_off))[0]
            if len(w) < 8:
                continue
            mid = 0.5 * (s.tp[w][1:] + s.tp[w][:-1])
            hdg = np.interp(mid - shift, s.ti, s.yaw) + np.interp(s.t_go, *track)
            step = np.diff(s.pxy[:, w], axis=1)
            fwd = float(np.sum(step[0] * np.cos(hdg) + step[1] * np.sin(hdg)))
            side = float(np.sum(-step[0] * np.sin(hdg) + step[1] * np.cos(hdg)))
            om = float(np.median(s.gyro[(s.ti > s.tp[w][0]) & (s.ti < s.tp[w][-1])]))
            slip = math.atan2(side, abs(fwd))
            noise = math.hypot(2.0 * sigma / max(abs(fwd), 0.1), 0.03)      # camera scatter over the travel, and the heading offset
            if worst is None or abs(slip) / noise > abs(worst[0]) / worst[1]:
                worst = (slip, noise, abs(fwd / (s.tp[w][-1] - s.tp[w][0]) * om), s.name)
    return worst


# ─── Pictures and words ───────────────────────────────────────────────────

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
C_FIT, C_NOM, C_MEAS = "#2a78d6", "#eb6834", "#8b8a85"     # fitted, nominal, measured (ink grey: it is the reference)


def draw(run, segs, noise, path):
    """replay.png: the paths on the floor, and the error of each replay over time."""
    import warnings as py_warnings
    import matplotlib
    matplotlib.use("Agg")
    with py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore")                         # a second matplotlib on this PC makes it complain about 3D axes
        import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw=dict(width_ratios=[1, 1.35]), facecolor=SURFACE)
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
    drive = [s for s in segs if s.kind != "still"] or segs
    for n, s in enumerate(drive):
        lab = lambda text: text if n == 0 else None
        ax.plot(*s.pxy, ".", color=C_MEAS, markersize=3, label=lab("camera (measured)"), zorder=2)
        ax.plot(*s.plot.nom, color=C_NOM, linewidth=1.6, label=lab("simulator, nominal numbers"), zorder=3)
        ax.plot(*s.plot.fit, color=C_FIT, linewidth=1.6, label=lab("simulator, fitted numbers"), zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x, ft")
    ax.set_ylabel("y, ft")
    ax.set_title("Where the robot went, and where the simulator puts it", loc="left", fontsize=11, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, markerscale=3, borderaxespad=0.2)

    # Error over time, maneuvers set side by side. A running median over 9 fixes takes the camera's own
    # scatter out of the lines, so what is left is the replay being in the wrong place.
    smooth = lambda e: np.median(np.lib.stride_tricks.sliding_window_view(np.pad(e, 4, mode="edge"), 9), axis=1)
    at, ticks = 0.0, []
    for n, s in enumerate(drive):
        t = s.tp - s.lo + at
        bx.plot(t, np.maximum(smooth(s.plot.e_nom), 0.03), color=C_NOM, linewidth=1.6, label="nominal numbers" if n == 0 else None)
        bx.plot(t, np.maximum(smooth(s.plot.e_fit), 0.03), color=C_FIT, linewidth=1.6, label="fitted numbers" if n == 0 else None)
        ticks.append((at + 0.5 * (s.hi - s.lo), s.name))
        at += s.hi - s.lo + 0.4
        bx.axvline(at - 0.2, color=GRID, linewidth=0.8)
    if np.isfinite(noise["pos"]):
        bx.axhline(noise["pos"] * 1.18, color=MUTED, linewidth=0.8)
        bx.text(at, noise["pos"] * 1.18, " camera\n noise", color=MUTED, va="center", fontsize=8)
    bx.set_yscale("log")
    bx.set_xlim(0, max(at, 1.0))
    every = max(1, len(ticks) // 40)
    bx.set_xticks([x for x, _ in ticks[::every]])
    bx.set_xticklabels([name for _, name in ticks[::every]], rotation=90, fontsize=7)
    bx.grid(False, axis="x")
    bx.set_ylabel("distance between camera and replay, ft (log scale, 9-fix running median)")
    bx.set_title("How far each replay is from the camera, maneuver by maneuver", loc="left", fontsize=11, pad=24)
    bx.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, borderaxespad=0.2)
    fig.suptitle(f"{run.name}: the logged commands replayed in the simulator", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.98, 0.96))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def simulator_table(res):
    """Each simulator number against its nominal value and the range it is
    trained over, with a suggested range: the measurement with a margin."""
    total = res["delay_total"]
    rows = {k: res[k] for k in PLANT_KEYS + ["act_lag", "obs_lag", "pos_noise", "hdg_noise", "dropout"] if k in res}
    table = {}
    for name, e in list(rows.items()) + [("act_lag + obs_lag", total)]:
        if name == "act_lag + obs_lag":
            lo, hi = (TRAINS_OVER["act_lag"][0] + TRAINS_OVER["obs_lag"][0], TRAINS_OVER["act_lag"][1] + TRAINS_OVER["obs_lag"][1])
            margin = MARGIN["lag"]
        else:
            lo, hi = TRAINS_OVER[name]
            margin = MARGIN[name]
        row = dict(measured=e["value"], sigma=e["sigma"], quality=e["quality"], nominal=(lo + hi) / 2, trains_over=[lo, hi],
                   inside=None, suggest=[lo, hi], verdict="not measured: keep the range")
        v = e["value"]
        if v is not None and e["quality"] != "not measured":
            m = max(margin, 3 * (e["sigma"] or 0.0))
            if e["quality"] == "bound" and name in ("yaw", "grip", "obs_lag"):        # "at least"
                row.update(inside=bool(v <= hi), suggest=[round(max(0.0, min(lo, v) if v < lo else max(lo, v - m)), 3), round(max(hi, v + m), 3)],
                           verdict="at least the measured value" + ("" if v <= hi else ": ABOVE the range"))
            elif e["quality"] == "bound":                                             # act_lag: "at most"
                row.update(inside=bool(v >= lo), suggest=[round(min(lo, max(0.0, v - m)), 3), round(min(hi, v + m) if v >= lo else v + m, 3)],
                           verdict="at most the measured value" + ("" if v >= lo else ": BELOW the range"))
            else:
                s_lo, s_hi = (-(abs(v) + m), abs(v) + m) if name == "pull" else (v - m, v + m)
                if name not in ("pull",):
                    s_lo = max(0.0, s_lo)
                row.update(inside=bool(lo <= v <= hi), suggest=[round(s_lo, 3), round(s_hi, 3)],
                           verdict="inside" if lo <= v <= hi else ("BELOW the range" if v < lo else "ABOVE the range"))
                if e["quality"] == "weak":
                    row["verdict"] += " (weakly determined)"
            if name in ("pull", "bleed") and res["yaw"]["quality"] != "measured":
                row["verdict"] += "; as a share of the yaw cap it assumes the cap, which was not measured"
        table[name] = row
    return table


def write_report(run, res, table, summary, overall, noise, extras, warnings, path):
    """report.md: what was measured, against what the simulator assumes."""
    f = lambda v, n=3: "n/a" if v is None else f"{v:.{n}f}"
    pm = lambda e, key="value", sk="sigma": ("not measured" if e[key] is None else
                                             f"{'>= ' if e['quality'] == 'bound' and e is not res.get('act_lag') else ''}"
                                             f"{'<= ' if e is res.get('act_lag') and e['quality'] == 'bound' else ''}"
                                             f"{e[key]:.3f}" + (f" +- {e[sk]:.3f}" if e[sk] else ""))
    out = [f"# {run.name}: the robot against the simulator", "",
           f"Source: `{run.source}`. IMU: {'yes' if run.has_imu else 'NO (everything that needs a gyro is marked not measured)'}. "
           f"{len(summary)} maneuvers. Written by `SIM2REAL/analyze.py`; every number is in `fit.json` with how it was obtained.", ""]
    out += [f"**WARNING. {w}**\n" for w in warnings]
    out += ["## The robot's numbers against the simulator", "",
           "Values are in the simulator's own terms (a share of its scale, or seconds), so they map straight onto "
           "`config.SAW_DR`. The simulator's nominal value is the middle of the range it trains over. "
           "The +- is one sigma from the fit; it does not cover ways the real robot differs from the simulator's model.", "",
           "| Number | Measured | In real units | Simulator nominal | Trains over | Verdict | Suggested range |",
           "|---|---|---|---|---|---|---|"]
    for name, row in table.items():
        e = res.get(name) or res["delay_total"]
        real = "" if e["real"] is None or e["unit"].startswith("share") or not e["unit"] else f"{e['real']:.3g} {e['unit']}"
        out.append(f"| `{name}` | {pm(e)} | {real} | {row['nominal']:.3f} | {row['trains_over'][0]:g} to {row['trains_over'][1]:g} | "
                   f"{row['verdict']} | {row['suggest'][0]:g} to {row['suggest'][1]:g} |")
    outside = [n for n, r in table.items() if r["inside"] is False]
    out += ["", ("**Outside the range the simulator trains over: " + ", ".join(f"`{n}`" for n in outside) + ".**") if outside
            else "Nothing measured falls outside the range the simulator trains over.", "",
            "Suggested range = measured value +- the larger of three sigma and a fixed margin per number (kept broad on purpose). "
            "A number that was not measured keeps its current range.", "", "## Other things measured", ""]
    for name in ("lag_plus_tau", "delay_imu", "yaw_gain_arc", "spin_left_right", "side_slip", "camera_period", "gyro_noise", "accel_noise"):
        e = res.get(name)
        if e and e["value"] is not None:
            out.append(f"- `{name}`: {pm(e)} {e['unit']}. {e['how'][:1].upper() + e['how'][1:]}. {e['note']}".rstrip())
    out += [f"- {x}" for x in extras]
    notes = [(n, e) for n, e in res.items() if isinstance(e, dict) and "quality" in e and (e["quality"] != "measured") and e["note"]]
    if notes:
        out += ["", "## Not measured, bounded or weak", ""]
        out += [f"- `{n}` ({e['quality']}): {e['note']}" for n, e in notes]
    chk = [(n, e) for n, e in res.items() if isinstance(e, dict) and e.get("direct") is not None and e.get("value") is not None]
    if chk:
        out += ["", "## Read straight off the data, as a check on the fit", "",
                "The same numbers without the simulator's model: steady speed per throttle level, steady yaw rate per turn "
                "level, circles through the arcs. They are rougher (short holds read low), but a large gap means the robot's "
                "response does not have the shape the simulator assumes.", "", "| Number | Fitted | Direct |", "|---|---|---|"]
        out += [f"| `{n}` | {e['value']:.3f} | {e['direct']:.3f} |" for n, e in chk]
    out += ["", "## How well the simulator replays the drive", "",
            f"The logged commands replayed through the simulator, from each maneuver's measured start, against the camera. "
            f"Fitted numbers: typically **{f(overall['fit_typical_ft'], 2)} ft** off, worst **{f(overall['fit_worst_ft'], 2)} ft**. "
            f"Nominal numbers (and the nominal {NOMINAL_DELAY:.3f} s delay): typically **{f(overall['nominal_typical_ft'], 2)} ft**, "
            f"worst **{f(overall['nominal_worst_ft'], 2)} ft**. Camera noise alone gives about {f(noise['pos'] * 1.18 if np.isfinite(noise['pos']) else None, 2)} ft.",
            "The start heading of each maneuver is the one that lines the fitted replay up with the measured path; the nominal "
            "replay starts from the same pose. Free driving (a flight of the policy) is replayed once from its start and scored "
            f"in {FREE_S:.0f} s stretches: each replay is laid on the camera position at the start of the stretch and turned to "
            "fit its first second, so the numbers say how far it strays in the seconds after. See `replay.png` and `replay.csv`.", "",
            "| Maneuver | Kind | Travel ft | Fitted typical / worst ft | Nominal typical / worst ft | Heading worst rad (fitted / nominal) |",
            "|---|---|---|---|---|---|"]
    for m in summary:
        hd = "n/a" if m["fit_heading_worst_rad"] is None else f"{m['fit_heading_worst_rad']:.2f} / {m['nominal_heading_worst_rad']:.2f}"
        out.append(f"| {m['maneuver']} | {m['kind']} | {m['travel_ft']:.1f} | {m['fit_typical_ft']:.2f} / {m['fit_worst_ft']:.2f} | "
                   f"{m['nominal_typical_ft']:.2f} / {m['nominal_worst_ft']:.2f} | {hd} |")
    Path(path).write_text("\n".join(out) + "\n")


def analyze(run_dir, use_imu=True, numbers=None, say=print):
    """Analyse one run folder and write fit.json, replay.csv, replay.png and
    report.md into it. Returns the contents of fit.json."""
    run = load_run(run_dir, use_imu)
    segs = find_segments(run)
    kinds = {k: sum(s.kind == k for s in segs) for k in ("still", "throttle", "spin", "arc", "free")}
    say(f"{run.name}: {len(segs)} maneuvers {kinds}, IMU {'yes' if run.has_imu else 'no'}")
    prior = dict(NOMINAL)
    if numbers:
        earlier = json.loads(Path(numbers).read_text())
        prior.update({k: v for k, v in earlier["plant"].items() if k in NOMINAL})
        prior.update({k: earlier["delays"][k] for k in ("d_cam", "d_imu")})
    res, num, noise, warnings = fit_run(run, segs, prior, say)
    for w in warnings:
        say("  WARNING: " + w)
    both_replays(run, segs, num)
    track, off, wander = start_headings(run, segs, num)
    extras = []
    if run.has_imu:
        extras.append(f"IMU yaw against the arena frame: offset {off:+.3f} rad" +
                      (f", wandering {wander:.3f} rad between maneuvers." if wander is not None else "."))
        slip = side_slip(run, segs, num, track, noise["pos"] if np.isfinite(noise["pos"]) else 0.2)
        if slip:
            angle, sd, accel, where = slip
            seen = abs(angle) > max(0.15, 3 * sd)
            res["side_slip"] = entry(angle, sd, "measured" if seen else "weak", unit="rad",
                                     how=f"sideways over forward travel in the arc '{where}' (camera travel against IMU heading)",
                                     note=(f"At {accel:.0f} ft/s^2 of sideways acceleration ({accel / 32.174:.2f} g). " +
                                           ("The robot slides in its arcs." if seen else "No slip shown: this is within what "
                                            "camera noise, the heading offset and the two delays can fake.")))
    rows, summary = replay(run, segs, num)
    drive = [m for m in summary if m["kind"] != "still"] or summary
    w = np.array([m["seconds"] for m in drive])
    overall = {k: float(np.average([m[k] for m in drive], weights=w)) if "typical" in k else float(max(m[k] for m in drive))
               for k in ("fit_typical_ft", "fit_worst_ft", "nominal_typical_ft", "nominal_worst_ft")}
    table = simulator_table(res)
    fit = dict(run=run.name, source=run.source, imu=run.has_imu, analysed=time.strftime("%Y-%m-%d %H:%M:%S"),
               numbers=res, plant={k: float(num[k]) for k in NOMINAL}, delays=dict(d_cam=num["d_cam"], d_imu=num["d_imu"]),
               simulator=table, replay=dict(overall=overall, per_maneuver=summary), warnings=warnings,
               truth=run.meta.get("truth"))
    d = run.dir
    (d / "fit.json").write_text(json.dumps(fit, indent=2))
    with open(d / "replay.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t", "maneuver", "kind", "x_ft", "y_ft", "heading_imu_rad", "x_fit", "y_fit", "heading_fit", "err_fit_ft",
                     "x_nominal", "y_nominal", "heading_nominal", "err_nominal_ft"])
        wr.writerows([[f"{v:.4f}" if isinstance(v, float) else v for v in r] for r in rows])
    draw(run, segs, noise, d / "replay.png")
    write_report(run, res, table, summary, overall, noise, extras, warnings, d / "report.md")
    say(f"  replay: fitted typically {overall['fit_typical_ft']:.2f} ft off (worst {overall['fit_worst_ft']:.2f}), "
        f"nominal {overall['nominal_typical_ft']:.2f} ft (worst {overall['nominal_worst_ft']:.2f})")
    say(f"  wrote fit.json, replay.csv, replay.png, report.md in {d}")
    return fit


# ─── Self-test: a made-up run with known numbers ──────────────────────────

COASTING_P = 0.02    # share of the made-up camera frames in which the tracker is coasting


def test_drives():
    """A session like the README's, named like maneuvers.py names its own:
    (maneuver, detail, [(seconds, throttle, turn), ...])."""
    lead, out = (0.3, 0.0, 0.0), (1.0, 0.0, 0.0)
    drives = [("still", "nothing for 2 s", [(2.0, 0.0, 0.0)])]
    for lvl in (0.3, 0.6, 1.0):
        hold = min(1.5, 24.0 / (lvl * 22.0))
        drives += [(f"step_{way}_{lvl}", f"throttle {s * lvl:+.1f} for {hold:.2f} s", [lead, (hold, s * lvl, 0.0), out])
                   for way, s in (("fwd", 1), ("rev", -1))]
    for lvl in (0.3, 0.6, 1.0):
        drives += [(f"spin_{way}_{lvl}", f"turn {s * lvl:+.1f} for 1.5 s", [lead, (1.5, 0.0, s * lvl), out])
                   for way, s in (("ccw", 1), ("cw", -1))]
    drives += [(f"arc_{n}", f"throttle {a:+.1f} turn {b:+.2f} for 2 s", [lead, (2.0, a, b), out])
               for n, (a, b) in enumerate(((0.5, 0.4), (0.5, -0.4), (0.8, 0.25), (-0.5, 0.4)), 1)]
    drives += [("straight_fwd", "throttle +0.5 for 2 s", [lead, (2.0, 0.5, 0.0), out]),
               ("doublet", "throttle +0.6 then -0.6 for 0.6 s each", [lead, (0.6, 0.6, 0.0), (0.6, -0.6, 0.0), out])]
    return drives


def make_run(folder, truth, seed):
    """Write a run folder (README format) for a robot with the numbers in
    `truth`, straight from plant.simulate: delayed commands, a noisy, late
    camera that drops frames and sometimes coasts, a noisy IMU whose yaw has
    its own zero."""
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    numbers = {k: truth[k] for k in NOMINAL}
    k = plant.tread_k(truth["half_track_ft"])
    t0 = 1.7e9
    cmd, events, cam, imu = [], [], [], []
    clock, yaw_zero = 2.0, rng.uniform(-math.pi, math.pi)
    drives = test_drives()
    for n, (name, detail, parts) in enumerate(drives):
        # commands every 0.03 s (with a little jitter), then the robot's true motion from the delayed commands
        tc, thr, trn = [], [], []
        t = clock
        for seconds, a, b in parts:
            for _ in range(int(round(seconds / 0.03))):
                tc.append(t + rng.uniform(-0.001, 0.001)); thr.append(a); trn.append(b)
                t += 0.03
        tc = np.array(tc)
        left, right = plant.mix(np.array(thr), np.array(trn), k)
        end = tc[-1]
        heading = rng.uniform(-math.pi, math.pi)
        start = (24.0 + rng.uniform(-6, 6), 24.0 + rng.uniform(-6, 6), heading)
        t_cam = np.arange(clock - 1.9, end + 2.2, 1 / 30.0)
        t_cam = t_cam + rng.uniform(-0.002, 0.002, len(t_cam))
        t_imu = np.arange(clock - 1.9, end + 2.2, 0.01)
        true_cam = plant.simulate(tc + truth["act_lag"], left, right, start, t_out=t_cam, **numbers)
        true_imu = plant.simulate(tc + truth["act_lag"], left, right, start, t_out=t_imu, **numbers)
        cmd += [(t0 + a, l, r, -1.0, 0.0) for a, l, r in zip(tc, left, right)]
        events += [(t0 + tc[0], "maneuver_start", name, detail), (t0 + end, "maneuver_end", name, detail),
                   (t0 + end + 1.8, "override_on", "", ""), (t0 + end + 2.6, "override_off", "", "")]
        last = None
        for i, tt in enumerate(t_cam):
            if rng.random() < truth["dropout"]:
                continue
            arrive = tt + truth["obs_lag"] + rng.uniform(-0.002, 0.002)
            x = true_cam["x_ft"][i] + rng.normal(0, truth["pos_noise"])
            y = true_cam["y_ft"][i] + rng.normal(0, truth["pos_noise"])
            coasting = last is not None and rng.random() < truth["coasting"]
            if coasting:
                x, y = last[0] + rng.normal(0, 1.0), last[1] + rng.normal(0, 1.0)
            last = (x, y)
            cam.append((t0 + arrive, t0 + arrive - 0.001, x, y, int(coasting)))
        accel = np.gradient(true_imu["v_fts"], t_imu) * 0.3048
        for i, tt in enumerate(t_imu):
            imu.append((t0 + tt + truth["imu_lag"], t0 + tt + truth["imu_lag"] - 0.001,
                        true_imu["heading_rad"][i] + yaw_zero + 0.002 * tt + rng.normal(0, truth["hdg_noise"]),
                        true_imu["omega_rps"][i] + truth["gyro_bias"] + rng.normal(0, truth["gyro_noise"]),
                        accel[i] + rng.normal(0, 0.3),
                        true_imu["v_fts"][i] * true_imu["omega_rps"][i] * 0.3048 + rng.normal(0, 0.3)))
        clock = end + 4.5

    def write(name, header, rows):
        with open(folder / name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
    write("commands.csv", ["t", "left", "right", "blade", "arm"], cmd)
    write("events.csv", ["t", "event", "maneuver", "detail"], events)
    write("position.csv", ["t", "t_stamp", "x_ft", "y_ft", "coasting"], sorted(cam))
    write("opponent.csv", ["t", "t_stamp", "x_ft", "y_ft", "coasting"], [])
    write("imu.csv", ["t", "t_stamp", "yaw_rad", "gyro_z", "accel_x", "accel_y"], imu)
    (folder / "meta.json").write_text(json.dumps(dict(
        name=folder.name, started=t0, source="fake", arena_ft=[48, 48],
        maneuvers=[dict(name=a, detail=b) for a, b, _ in drives],
        mixer=dict(half_track_ft=truth["half_track_ft"], tread_k=k), limits={}, truth=truth), indent=2))


def draw_truth(seed):
    """A robot away from nominal: every number drawn from the range the
    simulator trains over, with delays and sensor noise to match."""
    rng = np.random.default_rng(1000 + seed)
    truth = {k: float(rng.uniform(*TRAINS_OVER[k])) for k in PLANT_KEYS + ["act_lag", "obs_lag"]}
    truth.update(half_track_ft=NOMINAL["half_track_ft"], imu_lag=0.010, pos_noise=float(rng.uniform(0.1, 0.3)),
                 dropout=float(rng.uniform(0.02, 0.08)), coasting=COASTING_P, gyro_noise=0.05,
                 gyro_bias=float(rng.uniform(-0.02, 0.02)), hdg_noise=0.01)
    return truth


def compare(truth, fits):
    """Print truth against fitted, one column per analysis. Returns the rows."""
    want = dict(truth)
    want.update(delay_total=truth["act_lag"] + truth["obs_lag"], delay_imu=truth["act_lag"] + truth.get("imu_lag", 0.0),
                lag_plus_tau=truth["act_lag"] + truth["obs_lag"] + truth["tau"],
                dropout=1 - (1 - truth["dropout"]) * (1 - truth.get("coasting", 0.0)))
    # What the path alone decides when the yaw cap is unknown: these two products
    want["pull x yaw"], want["bleed / yaw"] = truth["pull"] * truth["yaw"], truth["bleed"] / truth["yaw"]
    names = [n for n in PLANT_KEYS + ["pull x yaw", "bleed / yaw", "delay_total", "lag_plus_tau", "delay_imu", "act_lag",
                                      "obs_lag", "pos_noise", "dropout", "gyro_noise", "hdg_noise"] if n in want]
    head = f"{'number':<13}{'truth':>8}  " + "  ".join(f"{label:<36}" for label in fits)
    print(head + "\n" + "-" * len(head))
    rows = []
    for name in names:
        line = f"{name:<13}{want[name]:>8.3f}  "
        for label, fit in fits.items():
            e, pl = fit["numbers"].get(name), fit["plant"]
            if " " in name:
                pull_bleed = fit["numbers"]["pull" if "pull" in name else "bleed"]["value"] is not None
                e = dict(value=pl["pull"] * pl["yaw"] if "pull" in name else pl["bleed"] / pl["yaw"], sigma=None,
                         quality="measured") if pull_bleed else None
            if not e or e["value"] is None:
                cell = "not measured"
            else:
                mark = {"bound": ">=" if name != "act_lag" else "<=", "weak": "~ "}.get(e["quality"], "  ")
                pm = f"+-{e['sigma']:6.3f}" if e["sigma"] is not None else " " * 8
                cell = f"{mark}{e['value']:7.3f} {pm}  err {e['value'] - want[name]:+7.3f}"
                rows.append((name, label, want[name], e["value"], e["sigma"], e["quality"]))
            line += f"{cell:<36}  "
        print(line)
    for label, fit in fits.items():
        o = fit["replay"]["overall"]
        print(f"replay, {label}: fitted {o['fit_typical_ft']:.2f} ft typical / {o['fit_worst_ft']:.2f} worst; "
              f"nominal {o['nominal_typical_ft']:.2f} / {o['nominal_worst_ft']:.2f}")
    return rows


def selftest(seeds, keep=None):
    """For each seed: make a run with hidden numbers, fit it with and without
    its IMU, and print truth against fitted."""
    rows = []
    for seed in seeds:
        truth = draw_truth(seed)
        base = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="sim2real_selftest_"))
        folder = base / f"selftest_{seed}"
        make_run(folder, truth, seed)
        bare = base / f"selftest_{seed}_no_imu"
        shutil.copytree(folder, bare, dirs_exist_ok=True)
        (bare / "imu.csv").write_text("t,t_stamp,yaw_rad,gyro_z,accel_x,accel_y\n")
        print(f"\n=== selftest seed {seed}: camera noise {truth['pos_noise']:.2f} ft, {truth['dropout'] * 100:.0f}% of frames dropped ===")
        started = time.time()
        fits = {"with IMU": analyze(folder), "no IMU (path only)": analyze(bare)}
        print(f"  ({time.time() - started:.0f} s)\n")
        rows += [(seed,) + r for r in compare(truth, fits)]
        if not keep:
            shutil.rmtree(base)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", nargs="?", help="a run folder (SIM2REAL/runs/<name>)")
    ap.add_argument("--no-imu", action="store_true", help="ignore imu.csv")
    ap.add_argument("--numbers", help="fit.json of an earlier run: numbers this run does not measure start from it")
    ap.add_argument("--selftest", nargs="*", type=int, metavar="SEED", help="fit made-up runs with known numbers (default seeds 1 2 3)")
    ap.add_argument("--keep", help="with --selftest: keep the made-up run folders in this directory")
    args = ap.parse_args()
    if args.selftest is not None:
        selftest(args.selftest or [1, 2, 3], args.keep)
    elif args.run_dir:
        analyze(args.run_dir, use_imu=not args.no_imu, numbers=args.numbers)
    else:
        ap.error("give a run folder or --selftest")


if __name__ == "__main__":
    main()
