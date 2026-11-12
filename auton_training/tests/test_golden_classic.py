"""
The classic ruleset must not change when the saw ruleset does.

Each scenario below plays the classic rules from a fixed seed with a fixed
action sequence on the CPU and hashes every observation, reward and done flag.
The hashes in tests/golden/classic_cpu.json were written before the saw-v8
work started (2026-11-11). A different hash means the classic rules now draw
random numbers in a different order or compute something differently.

    python tests/test_golden_classic.py            # check (CPU only; set DODGE_DEVICE=cpu)
    python tests/test_golden_classic.py --write    # rewrite the reference. Only when the classic rules are meant to change.
"""
import _runner  # noqa: F401  (puts the repo root on sys.path)
import hashlib
import json
import os
import sys

import torch

from config import DEVICE
from environment import make_env

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "classic_cpu.json")
SCENARIOS = [                      # (name, stage, p, envs, control steps, seed)
    ("combat_p0",   "combat",   0.0, 96, 700, 101),
    ("combat_p05",  "combat",   0.5, 96, 700, 202),
    ("combat_p1",   "combat",   1.0, 96, 700, 303),
    ("pretrain_p0", "pretrain", 0.0, 96, 700, 404),
]
HOLD = 10                          # control steps each random action is held for


def _play(stage, p, n, steps, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = make_env("classic", n, stage=stage, p=p)
    obs = env.reset()
    gen = torch.Generator(device="cpu").manual_seed(seed + 7)
    h = hashlib.sha256()
    h.update(obs.detach().cpu().numpy().tobytes())
    total, ended = 0.0, 0
    action = torch.zeros(n, env.act_dim)
    for t in range(steps):
        if t % HOLD == 0:
            action = torch.rand(n, env.act_dim, generator=gen) * 2.0 - 1.0
        obs, reward, done, info = env.step(action.to(DEVICE))
        for part in (obs, reward, done, info.end_reason, info.timeout):
            h.update(part.detach().cpu().numpy().tobytes())
        total += float(reward.sum())
        ended += int(done.sum())
    return {"sha256": h.hexdigest(), "reward_sum": round(total, 4), "episodes_ended": ended}


def _all():
    return {name: _play(stage, p, n, steps, seed) for name, stage, p, n, steps, seed in SCENARIOS}


def test_classic_rules_are_bit_identical_to_the_reference():
    if DEVICE.type != "cpu":
        print("SKIP  the reference is for the CPU; run with DODGE_DEVICE=cpu")
        return
    with open(GOLDEN) as f:
        want = json.load(f)
    got = _all()
    for name in want:
        assert got[name]["sha256"] == want[name]["sha256"], (
            f"{name}: classic rules changed (reward sum {got[name]['reward_sum']} vs {want[name]['reward_sum']}, "
            f"episodes {got[name]['episodes_ended']} vs {want[name]['episodes_ended']})")


if __name__ == "__main__":
    if "--write" in sys.argv:
        if DEVICE.type != "cpu":
            sys.exit("write the reference on the CPU: set DODGE_DEVICE=cpu")
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        out = _all()
        with open(GOLDEN, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
    else:
        _runner.run(globals())
