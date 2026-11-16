"""
One robot on the training simulator's own drive model, driven by tread commands.

This does not re-implement the motion model: every physics step is a call to
`auton_training/environment.integrate_robot`, the function both robots go
through in training, wrapped the way `saw_env.SawEnv._physics_step` wraps it for
our robot (tread scale-back, slower reverse, turn command x YAW_RATE_MAX). So a
drive here is a drive in the simulator, with numbers you choose.

    plant = Plant(v_max=0.7, tau=0.2)          # anything not named stays nominal
    plant.reset(10.0, 10.0, 0.0)
    state = plant.step(left, right)            # one control step (0.03 s)

    out = simulate(t, left, right, (x0, y0, heading0), v_max=0.7)   # a whole command log

    python SIM2REAL/plant.py          # sanity: top speed, yaw cap, replay against stepping
    python SIM2REAL/plant.py --env    # the plant against a real SawEnv fight (must agree to rounding)

Not in here, on purpose:
    latency  the fake robot delays commands and observations itself, and the
             analysis shifts the command log
    walls    the test pilot keeps the robot off them
    weapon   the blade is stopped in the test drives, so it does not slow the turn

Units: feet, seconds, radians; heading counter-clockwise from +x, yaw rate
positive the same way (the simulator's and the README's frame).
"""
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# The simulator is a folder of scripts, not a package. One robot needs no GPU
# (and a training job may own it), so the simulator is put on the CPU unless
# the caller already chose.
os.environ.setdefault("DODGE_DEVICE", "cpu")
_SIM_DIR = Path(__file__).resolve().parent.parent / "auton_training"
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

import torch                                                    # noqa: E402
from config import (                                            # noqa: E402
    DT, CTRL_DT, V_MAX_FT, YAW_RATE_MAX, A_LAT_MAX_FT, A_LONG_MAX_FT, ALPHA_MAX_RPS2,
    GRIP_STATIC_FT, GRIP_KINETIC_FT, SAW_DR, DR_REVERSE,
)
from environment import integrate_robot                         # noqa: E402

# The ranges the simulator draws each fight's robot from, by name (read from
# config when this file is imported, so they follow it).
TRAINS_OVER = dict(SAW_DR, reverse=DR_REVERSE)
_mid = lambda name: (TRAINS_OVER[name][0] + TRAINS_OVER[name][1]) / 2.0

# Every number of the drive model at the simulator's nominal value, which is
# the middle of the range it trains over (the simulator draws mid +- half).
# The names are the simulator's (config.SAW_DR, DR_REVERSE).
NOMINAL = dict(
    v_max=_mid("v_max"),        # top speed forward, as a share of V_MAX_FT (22.64 ft/s, the simulator's speed scale)
    reverse=_mid("reverse"),    # top speed in reverse, as a share of forward
    grip=_mid("grip"),          # floor grip: x A_LONG_MAX_FT (77.06 ft/s^2 acceleration limit) and x both sideways
                                # slip limits (lets go at 2.0 g, grips again under 0.6 g)
    tau=_mid("tau"),            # s, how fast the drive follows a throttle command
    deadband=_mid("deadband"),  # throttle travel around zero that does nothing, share of full throttle
    yaw=_mid("yaw"),            # fastest yaw rate the robot can reach, x YAW_RATE_MAX (7.5 rad/s)
    yaw_gain=_mid("yaw_gain"),  # yaw rate delivered per unit of turn command, x YAW_RATE_MAX
    alpha=_mid("alpha"),        # yaw acceleration, x ALPHA_MAX_RPS2 (18.42 rad/s^2)
    bleed=_mid("bleed"),        # share of top speed lost at the full yaw rate
    pull=_mid("pull"),          # yaw picked up at full throttle, as a share of the yaw cap (+ = to the left)
    half_track_ft=1.48,         # not the robot's: the mixer's setting (deploy.PilotConfig.half_track), needed
                                # to turn left/right back into throttle and turn
)
# The same for the two delays. They are not in the plant (see above); the
# fake robot and the analysis use them.
NOMINAL_LAG = dict(act_lag=_mid("act_lag"), obs_lag=_mid("obs_lag"))    # s

SILENCE_S = 0.5      # the PPM output centres the sticks after this long without a command
_NO_WALL = -1.0e6    # "closest the centre comes to a wall": far outside, so there are none


def tread_k(half_track_ft=NOMINAL["half_track_ft"]):
    """The mixer's turn share (README, deploy.SawPilot). V_MAX_FT is the
    simulator's speed scale here, as in deploy.py, not the robot's top speed."""
    return half_track_ft * YAW_RATE_MAX / V_MAX_FT


def mix(throttle, turn, k=None):
    """Throttle and turn (-1..1) to left and right treads, as the fight code does."""
    k = tread_k() if k is None else k
    scale = np.maximum(1.0, np.abs(throttle) + k * np.abs(turn))
    return (throttle - k * turn) / scale, (throttle + k * turn) / scale


def unmix(left, right, k=None):
    """Left and right treads (clipped to -1..1, as the radio sends them) back to
    the throttle and turn the simulator's drive takes."""
    k = tread_k() if k is None else k
    left, right = np.clip(left, -1.0, 1.0), np.clip(right, -1.0, 1.0)
    return (left + right) / 2.0, (right - left) / (2.0 * k)


class Plant:
    """One robot (or a batch: give any number as an array and every state
    becomes an array of that length; the analysis uses this to try many sets of
    numbers in one pass)."""

    def __init__(self, **numbers):
        unknown = set(numbers) - set(NOMINAL)
        if unknown:
            raise TypeError(f"unknown plant numbers {sorted(unknown)}; known: {sorted(NOMINAL)}")
        vals = {k: np.atleast_1d(np.asarray(numbers.get(k, v), dtype=float)) for k, v in NOMINAL.items()}
        self.n = max(v.size for v in vals.values())
        t = {k: torch.as_tensor(np.broadcast_to(v, (self.n,)).copy(), dtype=torch.float64) for k, v in vals.items()}
        self.numbers = {k: (float(v[0]) if self.n == 1 else v.numpy().copy()) for k, v in t.items()}
        # Exactly RobotParams.randomize, with the numbers given instead of drawn
        self.par = SimpleNamespace(
            v_max=V_MAX_FT * t["v_max"],
            yaw_cap=YAW_RATE_MAX * t["yaw"],
            a_lat_cap=torch.full((self.n,), A_LAT_MAX_FT, dtype=torch.float64),
            dv_max=A_LONG_MAX_FT * t["grip"] * DT,
            dalpha=ALPHA_MAX_RPS2 * t["alpha"] * DT,
            ctrl_alpha=t["tau"] / (t["tau"] + DT),
            bleed=t["bleed"],
            grip_s=GRIP_STATIC_FT * t["grip"],
            grip_k=GRIP_KINETIC_FT * t["grip"],
            yaw_gain=t["yaw_gain"], pull=t["pull"], deadband=t["deadband"],
        )
        self.rev = t["reverse"]
        self.k = t["half_track_ft"] * YAW_RATE_MAX / V_MAX_FT
        self._alive = torch.ones(self.n, dtype=torch.float64)
        self._wall = torch.full((self.n,), _NO_WALL, dtype=torch.float64)
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x_ft, y_ft, heading_rad, v_fts=0.0, omega_rps=0.0):
        """Put the robot somewhere, standing still unless told otherwise."""
        full = lambda v: torch.full((self.n,), float(v), dtype=torch.float64)
        self.x, self.y, self.a = full(x_ft), full(y_ft), full(heading_rad)
        self.v, self.omega, self.slide = full(v_fts), full(omega_rps), full(0.0)
        self.slipping = torch.zeros(self.n, dtype=torch.bool)
        self._owed = 0.0                      # time asked for and not yet simulated (dt is rarely a whole step)
        return self.state()

    def state(self):
        out = dict(x_ft=self.x, y_ft=self.y, heading_rad=self.a, v_fts=self.v, omega_rps=self.omega,
                   slide_fts=self.slide)
        return {k: (float(v[0]) if self.n == 1 else v.numpy().copy()) for k, v in out.items()}

    def step(self, left, right, dt=CTRL_DT):
        """One control step from tread commands as sent on /control/channels.
        Returns x_ft, y_ft, heading_rad (wrapped to -pi..pi), v_fts (signed,
        along the body), omega_rps, and slide_fts (sideways slip speed, + = to
        the left). No latency inside."""
        throttle, turn = unmix(np.asarray(left, dtype=float), np.asarray(right, dtype=float), self.k.numpy())
        self._owed += dt
        steps = int(round(self._owed / DT))
        self._owed -= steps * DT
        thr = torch.as_tensor(np.broadcast_to(throttle, (self.n,)).copy(), dtype=torch.float64)
        trn = torch.as_tensor(np.broadcast_to(turn, (self.n,)).copy(), dtype=torch.float64)
        for _ in range(steps):
            self._physics(thr, trn)
        return self.state()

    @torch.no_grad()
    def _physics(self, throttle, turn):
        """One 100 Hz step, as SawEnv._physics_step does it for our robot."""
        over = (throttle.abs() + self.k * turn.abs()).clamp(min=1.0)       # a tread has a top speed
        throttle, turn = throttle / over, turn / over
        throttle = torch.where(throttle < 0, throttle * self.rev, throttle)    # reverse can be a slower gear
        (self.x, self.y, self.a, self.v, self.omega, self.slipping, self.slide, _) = integrate_robot(
            self.x, self.y, self.a, self.v, self.omega, self.slipping, self.slide,
            throttle, turn * YAW_RATE_MAX, self.par, self._alive, scrape_walls=True, wall_in=self._wall)


def _step_average(t, value, grid):
    """A logged tread command as the physics steps see it: its average over
    each step from grid[i] to grid[i+1]. The command stands until the next one
    and is zero after SILENCE_S without one. A command that lands on a step
    boundary (every one in the simulator) takes effect whole on that step, as
    there; one that lands inside a step is shared by time, so a log whose
    clock is not the physics clock is not rounded to it."""
    value = np.clip(value, -1.0, 1.0)
    off = np.minimum(t + SILENCE_S, np.r_[t[1:], np.inf])   # when each command stops standing
    knots = np.r_[np.c_[t, off].ravel(), off[-1] + 1.0]     # on, off, on, off, ..., and a point past the end
    held = np.c_[value * (off - t), np.zeros(len(t))].ravel()             # integral over each stretch between knots
    total = np.r_[0.0, np.cumsum(held)]
    total[-1] = total[-2]
    return np.diff(np.interp(grid, knots, total)) / np.diff(grid)


def simulate(t, left, right, start, t_out=None, **numbers):
    """Replay a command log through the plant.

    t, left, right   the log (Unix seconds or any clock; need not be regular).
                     Each command is held until the next one, and the treads
                     go to zero after 0.5 s of silence, like the PPM output.
    start            (x_ft, y_ft, heading_rad) at t[0], optionally followed by
                     v_fts and omega_rps
    t_out            times to report at (default: t)
    numbers          any NOMINAL keys; arrays run a batch, and every output
                     then has one column per set of numbers

    Returns a dict of arrays at t_out: t, x_ft, y_ft, heading_rad (continuous,
    not wrapped), v_fts, omega_rps, slide_fts.
    """
    t = np.asarray(t, dtype=float)
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    t_out = t if t_out is None else np.asarray(t_out, dtype=float)
    plant = Plant(**numbers)
    plant.reset(*start)

    steps = max(1, int(math.ceil((max(t[-1], t_out[-1]) - t[0]) / DT - 1e-9)))
    grid = t[0] + DT * np.arange(steps + 1)                 # the state is known at these times
    k = plant.k.numpy()[None, :]
    throttle, turn = unmix(_step_average(t, left, grid)[:, None], _step_average(t, right, grid)[:, None], k)
    throttle, turn = torch.as_tensor(throttle), torch.as_tensor(turn)

    names = ("x_ft", "y_ft", "heading_rad", "v_fts", "omega_rps", "slide_fts")
    log = np.empty((steps + 1, len(names), plant.n))
    read = lambda: torch.stack([plant.x, plant.y, plant.a, plant.v, plant.omega, plant.slide]).numpy()
    log[0] = read()
    for s in range(steps):
        plant._physics(throttle[s], turn[s])
        log[s + 1] = read()
    log[:, 2] = np.unwrap(log[:, 2], axis=0)

    out = {"t": t_out}
    for j, name in enumerate(names):
        cols = np.stack([np.interp(t_out, grid, log[:, j, c]) for c in range(plant.n)], axis=1)
        out[name] = cols[:, 0] if plant.n == 1 else cols
    return out


def _sanity():
    """Full throttle reaches the nominal top speed, full turn the yaw cap, and
    the replay of a log matches stepping through it."""
    N = NOMINAL
    print("nominal numbers:", {k: round(v, 4) for k, v in N.items()})
    p = Plant()
    p.reset(10.0, 10.0, 0.0)
    for _ in range(100):
        s = p.step(1.0, 1.0)
    print(f"full throttle 3 s: v = {s['v_fts']:.2f} ft/s (nominal top speed {N['v_max'] * V_MAX_FT:.2f}), "
          f"x = {s['x_ft']:.1f} ft")
    p.reset(10.0, 10.0, 0.0)
    left, right = mix(0.0, 1.0)
    for _ in range(100):
        s = p.step(left, right)
    print(f"full turn 3 s: omega = {s['omega_rps']:.2f} rad/s (nominal yaw cap {N['yaw'] * YAW_RATE_MAX:.2f}), "
          f"moved {math.hypot(s['x_ft'] - 10, s['y_ft'] - 10):.3f} ft")
    p.reset(10.0, 10.0, 0.0)
    left, right = mix(-1.0, 0.0)
    for _ in range(100):
        s = p.step(left, right)
    rev = (N["reverse"] - N["deadband"]) / (1 - N["deadband"]) * N["v_max"] * V_MAX_FT
    print(f"full reverse 3 s: v = {s['v_fts']:.2f} ft/s (nominal {-rev:.2f})")

    t = np.arange(0, 3.0, CTRL_DT)
    left, right = mix(np.where(t < 2.0, 0.8, 0.0), np.where((t > 0.5) & (t < 1.5), 0.6, 0.0))
    out = simulate(t, left, right, (10.0, 10.0, 0.3), v_max=0.7, tau=0.2)
    q = Plant(v_max=0.7, tau=0.2)
    q.reset(10.0, 10.0, 0.3)
    for i in range(len(t) - 1):
        s = q.step(left[i], right[i])
    print(f"simulate vs step, end of a 3 s drive: {out['x_ft'][-1] - s['x_ft']:+.2e} ft, "
          f"{out['y_ft'][-1] - s['y_ft']:+.2e} ft")
    both = simulate(t, left, right, (10.0, 10.0, 0.3), v_max=[0.7, 1.0], tau=[0.2, 0.05])
    print(f"batch of two sets of numbers: column 0 differs from the single run by "
          f"{np.abs(both['x_ft'][:, 0] - out['x_ft']).max():.2e} ft")


def _against_env():
    """The plant against a real SawEnv fight: one env, our robot given known
    numbers and no weapon effects, random throttle and turn for a few seconds.
    The two must agree to rounding, or the simulator's drive has changed and
    this file has to follow. Reaches into SawEnv's fields on purpose."""
    from saw_env import SawEnv
    numbers = dict(v_max=0.8, reverse=0.8, grip=0.5, tau=0.12, deadband=0.05, yaw=0.7, yaw_gain=1.05, alpha=0.6,
                   bleed=0.3, pull=0.04)
    p = Plant(**numbers)
    env = SawEnv(1, auto_reset=False)
    env.reset()
    for name, value in vars(p.par).items():
        getattr(env.our, name)[:] = value.float()
    env.rev[:], env.tread_k[:], env.cmd_lag[:], env.cmd_drop[:] = numbers["reverse"], float(p.k[0]), 1, 0.0
    env.saw_o.gyro[:], env.o_stun[:], env.o_air[:], env.wear_at[:] = 0.0, 0.0, 0.0, 10 ** 9
    env.ox[:], env.oy[:], env.oa[:], env.ex[:], env.ey[:] = 24.0, 30.0, 0.3, 44.0, 44.0     # the opponent well away
    for state in (env.ovx, env.o_omega, env.o_drift):
        state[:] = 0.0
    env.o_drifting[:] = False
    rng = np.random.default_rng(0)
    sent, seen = [], []
    for step in range(120):
        if step % 12 == 0:
            throttle, turn = rng.uniform(-1, 1), rng.uniform(-1, 1) * (step > 24)
        sent.append((throttle, turn))
        env.step(torch.tensor([[throttle, turn, -1.0, -1.0]]))
        seen.append([float(v[0]) for v in (env.ox, env.oy, env.ovx, env.o_omega)])
    sent, seen = np.array(sent), np.array(seen)
    t = CTRL_DT * np.arange(len(sent))
    left, right = mix(sent[:, 0], sent[:, 1], float(p.k[0]))
    out = simulate(t + DT, left, right, (24.0, 30.0, 0.3), t_out=t + CTRL_DT, **numbers)   # the env's shortest command lag: one step
    for j, key in enumerate(("x_ft", "y_ft", "v_fts", "omega_rps")):
        print(f"{key:10s} plant against SawEnv over {t[-1]:.1f} s: largest difference {np.abs(out[key] - seen[:, j]).max():.1e} "
              f"(the env moved it over a range of {np.ptp(seen[:, j]):.1f})")


if __name__ == "__main__":
    _against_env() if "--env" in sys.argv else _sanity()
