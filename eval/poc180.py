"""
PoC-180: the yardstick of the saw-v8 work. Does the policy land at least
one solid hammer strike on a legal side in a full three-minute fight, from
the bell, with the real swing budget and nothing but the overhead camera?

Three fixed panels, each with its own seed, so two checkpoints meet the same
fights:

  T  Traves-like horizontal bar spinners (the first real opponent)
  F  the field: verts, drums and horizontals at the levels a fight is likely to bring
  L  the saw-v7 field, for comparing with the runs before saw-v8

Every weapon stays alive (no failures) and neither of our optional readings
(arm angle, blade speed) is there.

Run:
    uv run python eval/poc180.py ppo_models/saw_v8a_run1/SAW_upd100.pt
    uv run python eval/poc180.py CKPT --lite                 128 fights a panel instead of 512
    uv run python eval/poc180.py CKPT --gate off             the policy drives the arm itself
    uv run python eval/poc180.py CKPT --gate on --gate-rule body
    uv run python eval/poc180.py CKPT --swings 20 --reach 1.0 --heading axis_nose --panels T F

It runs on the CPU unless --gpu is given, so it can run beside a training
run. Results go to eval/poc180/<tag>.json and .txt.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if "--gpu" not in sys.argv:
    os.environ.setdefault("DODGE_DEVICE", "cpu")

import torch

from agent import load_agent, gate_from_meta
from config import DEVICE, CTRL_DT
from environment import make_env
from saw_env import T_BLOW, T_FOUL, T_MISS, T_SOLID, T_TAKEN, T_HURT, T_DEALT, T_FIRE, W_HORIZ
from saw_field import Scenario, PROFILE_TRAVES
from train_ppo import attack_in

PANELS = {
    "T": dict(seed=1000, scenario=dict(traves_share=1.0, p=0.68, weapon_shares=(0.0, 0.0, 1.0))),
    "F": dict(seed=2000, scenario=dict(traves_share=0.0, p_range=(0.5, 0.8))),
    "L": dict(seed=3000, scenario=dict(traves_share=0.0, p=0.68, legacy_field=True)),
}
COMMON = dict(from_bell=True, arm_reading=False, blade_reading=False, weapon_fail_share=0.0)
CLEAN_HITS = 3          # a first strike is "clean" when we had taken fewer hits than this before it


@torch.no_grad()
def run_panel(agent, meta, panel, n=512, swings=10, reach=0.5, gate="ckpt", heading="none", seconds=180.0,
              gate_rule=None):
    """One panel's fights, played to the bell with the policy's mean action."""
    torch.manual_seed(PANELS[panel]["seed"])
    kw = {**COMMON, **PANELS[panel]["scenario"]}
    kw.update(swings=swings, reach_ft=reach, heading_mode=heading, fight_s=seconds)
    sc = Scenario(**kw)
    arm_gate = {"on": True, "off": False, "ckpt": gate_from_meta(meta)}[gate]
    env = make_env("saw", n, "combat", p=0.68, auto_reset=False, scenario=sc, arm_gate=arm_gate)
    env.gate_rule = gate_rule or meta.get("gate_rule") or "heading"
    if meta.get("gate_persist"):
        env.gate_persist = int(meta["gate_persist"])
    obs = env.reset()
    state = agent.initial_state(n, DEVICE)
    fresh = torch.ones(n, device=DEVICE)
    first_t = torch.full((n,), float("nan"), device=DEVICE)
    taken_at_first = torch.zeros(n, device=DEVICE)
    for k in range(math.ceil(seconds / CTRL_DT) + 2):
        action, state = agent.act_deterministic(obs, state, fresh)
        fresh = torch.zeros(n, device=DEVICE)
        obs, _, done, _ = env.step(action)
        new = (env.tally[:, T_SOLID] >= 1) & first_t.isnan()
        first_t[new] = (k + 1) * CTRL_DT
        taken_at_first[new] = env.tally[new, T_TAKEN]
        if done.all():
            break
    assert env.done.all(), "some fights never reached the bell"

    t = env.tally
    solid = t[:, T_SOLID]
    used = t[:, T_BLOW] + t[:, T_FOUL] + t[:, T_MISS]
    total = used.sum().item()
    share = lambda col: t[:, col].sum().item() / total if total > 0 else 0.0
    struck = ~first_t.isnan()
    mean = lambda x: x.float().mean().item()
    return {
        "panel": panel, "seed": PANELS[panel]["seed"], "scenario": asdict(sc), "arm_gate": arm_gate,
        "gate_rule": env.gate_rule if arm_gate else None,
        "n_fights": n,
        "n_traves": int((env.field.profile == PROFILE_TRAVES).sum()),
        "n_horiz": int((env.saw_e.kind == W_HORIZ).sum()),
        "poc_share": mean(solid >= 1),                      # the headline: a solid legal strike in the fight
        "clean_share": mean(struck & (taken_at_first < CLEAN_HITS)),
        "solid_per_fight": mean(solid),
        "ge2_share": mean(solid >= 2),
        "attack_in_share": mean(attack_in(t)),
        "damage_dealt": mean(t[:, T_DEALT]),
        "hits_taken": mean(t[:, T_TAKEN]),
        "damage_taken": mean(t[:, T_HURT]),
        "swings_used": mean(used),
        "fires": mean(t[:, T_FIRE]),
        "land_legal": share(T_BLOW), "land_side": share(T_FOUL), "land_floor": share(T_MISS),
        "first_strike_median_s": first_t[struck].median().item() if struck.any() else None,
    }


COLUMNS = (("panel", "{}"), ("n_fights", "{}"), ("poc_share", "{:.3f}"), ("clean_share", "{:.3f}"),
           ("solid_per_fight", "{:.2f}"), ("ge2_share", "{:.3f}"), ("attack_in_share", "{:.3f}"),
           ("swings_used", "{:.2f}"), ("land_legal", "{:.2f}"), ("land_side", "{:.2f}"), ("land_floor", "{:.2f}"),
           ("hits_taken", "{:.2f}"), ("damage_taken", "{:.2f}"), ("first_strike_median_s", "{:.1f}"))


def table(report):
    rows = [[name for name, _ in COLUMNS]]
    for r in report["panels"].values():
        rows.append(["-" if r[name] is None else fmt.format(r[name]) for name, fmt in COLUMNS])
    widths = [max(len(row[c]) for row in rows) for c in range(len(COLUMNS))]
    head = (f"PoC-180  {report['meta']['checkpoint']}  update {report['meta']['update']}  "
            + "  ".join(f"{k}={v}" for k, v in report["settings"].items()))
    return "\n".join([head] + ["  ".join(cell.rjust(w) for cell, w in zip(row, widths)) for row in rows])


def poc180(checkpoint, swings=10, reach=0.5, gate="ckpt", heading="none", panels=("T", "F", "L"), lite=False,
           n=None, seconds=180.0, out_dir=ROOT / "eval" / "poc180", write=True, gate_rule=None, quiet=False):
    ckpt = Path(checkpoint)
    agent, meta = load_agent(str(ckpt), DEVICE)
    if meta["rules"] != "saw":
        raise SystemExit(f"{ckpt} is a {meta['rules']}-rules checkpoint: PoC-180 is a saw-rules yardstick")
    n = n or (128 if lite else 512)
    out, t0 = {}, time.time()
    for panel in panels:
        out[panel] = run_panel(agent, meta, panel, n, swings, reach, gate, heading, seconds, gate_rule)
        if not quiet:
            print(f"  panel {panel}: poc={out[panel]['poc_share']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    first = next(iter(out.values()))
    report = {
        "meta": {"checkpoint": str(ckpt), "update": meta.get("update", 0), "physics": meta.get("physics"),
                 "env_physics": make_env("saw", 1).physics_version, "arm_gate": first["arm_gate"],
                 "gate_rule": first["gate_rule"], "seconds_taken": round(time.time() - t0, 1)},
        "settings": dict(swings=swings, reach=reach, gate=gate, gate_rule=first["gate_rule"], heading=heading,
                         n=n, seconds=seconds, lite=lite),
        "panels": out,
    }
    if write:
        rule = f"-{first['gate_rule']}" if first["gate_rule"] not in (None, "heading") else ""
        tag = (f"{ckpt.parent.name}_{ckpt.stem}_u{meta.get('update', 0)}_g{gate}{rule}_s{swings}_r{reach:g}_{heading}"
               + ("_lite" if lite else ""))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_dir / f"{tag}.txt").write_text(table(report) + "\n", encoding="utf-8")
        report["written"] = str(out_dir / f"{tag}.json")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--swings", type=int, default=10, help="swing budget for the fight (default 10)")
    ap.add_argument("--reach", type=float, default=0.5, help="hammer reach past our front edge, ft (default 0.5)")
    ap.add_argument("--gate", choices=("ckpt", "on", "off"), default="ckpt",
                    help="the arm gate: as the checkpoint was trained (default), on, or off")
    ap.add_argument("--gate-rule", choices=("heading", "body"), help="default: the checkpoint's, else heading")
    ap.add_argument("--heading", choices=("none", "axis", "axis_nose"), default="none",
                    help="what the camera says of the opponent's heading (default: nothing, as the real tracker)")
    ap.add_argument("--panels", nargs="+", choices=tuple(PANELS), default=list(PANELS))
    ap.add_argument("--lite", action="store_true", help="128 fights a panel")
    ap.add_argument("--n", type=int, help="fights a panel")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    rep = poc180(a.checkpoint, a.swings, a.reach, a.gate, a.heading, tuple(a.panels), a.lite, a.n, a.seconds,
                 gate_rule=a.gate_rule)
    print(table(rep))
    print(f"[SAVE] {rep['written']}")
