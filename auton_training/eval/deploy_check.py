"""
Does the deploy pilot fight as well as the policy does inside the simulator?

Each fight is played twice from the same seed:
  sim     the policy reads the simulator's own observation (how it trained)
  pilot   deploy.SawPilot builds the observation from the camera fixes and
          a drifting gyro, with deploy.HeadingEstimator for our heading,
          started the way the robot will be: facing the opponent

    python eval/deploy_check.py
    python eval/deploy_check.py CKPT --fights 12 --seconds 180 --gyro-bias 0.01

A pilot row far below the sim row means the robot-side code, not the policy,
is losing the fight.
"""
import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from agent import load_agent
from config import CTRL_DT, DEVICE, DT, YAW_RATE_MAX, angle_wrap
from deploy import HeadingEstimator, PilotConfig, SawPilot
from environment import env_for_checkpoint
from saw_env import T_DEALT, T_FIRE, T_HURT, T_SOLID
from saw_field import Scenario
from saw_gate import CAL_KEYS

DEFAULT = ROOT / "ppo_models" / "saw_v8a_traves_hold" / "SAW_upd375.pt"


@torch.no_grad()
def fight(ckpt, seed, seconds, use_pilot, gyro_bias):
    torch.manual_seed(seed)
    agent, meta = load_agent(str(ckpt), DEVICE)
    sc = Scenario(from_bell=True, arm_reading=False, blade_reading=False, weapon_fail_share=0.0,
                  heading_mode="none", fight_s=seconds, swings=10)
    env = env_for_checkpoint(meta, 1, stage="combat", p=0.6, auto_reset=False, scenario=sc)
    obs = env.reset()
    state, fresh = agent.initial_state(1, DEVICE), torch.ones(1, device=DEVICE)
    if use_pilot:
        pilot = SawPilot(ckpt, PilotConfig(half_len=float(env.o_len[0]), half_wid=float(env.o_wid[0]),
                                           swings=int(env.gate.budget[0]), fight_s=float(env.horizon[0]) * DT))
        for key in CAL_KEYS:
            getattr(pilot.gate.cal, key)[0] = getattr(env.gate.cal, key)[0]
        pilot.card = {k: v.clone() for k, v in env._card().items()}
        est = HeadingEstimator()
        x, y, ex, ey = (float(v) for v in env.observer.fix[0])
        est.reset(math.atan2(ey - y, ex - x))                  # set down facing the opponent
    last_fix, throttle, hdg_err = None, 0.0, []
    while not bool(env.done[0]):
        if use_pilot:
            x, y, ex, ey = (float(v) for v in env.observer.fix[0])
            gyro = float(obs[0, 5]) * YAW_RATE_MAX + gyro_bias
            a = est.update(x, y, (x, y) != last_fix, gyro, gyro * CTRL_DT, throttle, math.hypot(ex - x, ey - y))
            last_fix = (x, y)
            hdg_err.append(float(angle_wrap(torch.tensor(a - float(env.oa[0])))))
            cmd = pilot.step(x, y, ex, ey, a, gyro)
            action = torch.tensor([[cmd.throttle, cmd.turn, cmd.arm_request, cmd.blade]], device=DEVICE)
            throttle = cmd.throttle
        else:
            action, state = agent.act_deterministic(obs, state, fresh)
            fresh = torch.zeros(1, device=DEVICE)
        obs, _, _, _ = env.step(action)
    t = env.tally[0]
    rms = math.sqrt(sum(e * e for e in hdg_err) / len(hdg_err)) if hdg_err else 0.0
    return dict(dealt=float(t[T_DEALT]), taken=float(t[T_HURT]), solid=float(t[T_SOLID]), swings=float(t[T_FIRE]),
                hdg_rms=rms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", default=str(DEFAULT))
    ap.add_argument("--fights", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--gyro-bias", type=float, default=0.005, help="rad/s the gyro drifts by (default 0.005)")
    args = ap.parse_args()

    rows = {"sim": [], "pilot": []}
    for k in range(args.fights):
        for name in rows:
            rows[name].append(fight(args.checkpoint, 1000 + k, args.seconds, name == "pilot", args.gyro_bias))
        s, p = rows["sim"][-1], rows["pilot"][-1]
        print(f"fight {k + 1:2d}  sim dealt {s['dealt']:.2f} taken {s['taken']:.2f} solid {s['solid']:.0f} | "
              f"pilot dealt {p['dealt']:.2f} taken {p['taken']:.2f} solid {p['solid']:.0f} "
              f"heading rms {p['hdg_rms']:.2f} rad", flush=True)
    print(f"\n{args.fights} fights of {args.seconds:.0f} s, means per fight")
    print(f"{'':6s} {'dealt':>6s} {'taken':>6s} {'solid':>6s} {'swings':>6s} {'fights with a solid strike':>27s}")
    for name, r in rows.items():
        mean = lambda key: sum(v[key] for v in r) / len(r)
        hit = sum(v["solid"] > 0 for v in r)
        print(f"{name:6s} {mean('dealt'):6.2f} {mean('taken'):6.2f} {mean('solid'):6.2f} {mean('swings'):6.2f} "
              f"{hit:>20d} of {len(r)}")
    print(f"pilot heading error, rms over all fights: "
          f"{math.sqrt(sum(v['hdg_rms'] ** 2 for v in rows['pilot']) / len(rows['pilot'])):.2f} rad")


if __name__ == "__main__":
    main()
