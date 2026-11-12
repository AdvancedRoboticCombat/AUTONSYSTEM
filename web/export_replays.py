"""
Play a checkpoint's fights and write them out for the web viewer.

    uv run python web/export_replays.py ppo_models/saw_v8a_body_priv/SAW_upd625.pt
    uv run python web/export_replays.py CKPT --panels T F --fights 4 --seconds 180

The fights come from the PoC-180 panels (eval/poc180.py): T is Traves-like
horizontal spinners, F the field, L the saw-v7 field. Each fight is written to
web/replays/<run>_<checkpoint>_<panel><k>.json: the two robots' positions, the
arm, the blades and every swing with how it landed, one row per control step.

Runs on the CPU unless --gpu is given, so it can run beside a training run.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
if "--gpu" not in sys.argv:
    os.environ.setdefault("DODGE_DEVICE", "cpu")

import numpy as np
import torch

from agent import load_agent, gate_from_meta
from config import DEVICE, CTRL_DT, ARENA_FT, HAZ_PATH, ARM_PARKED, ARM_DOWN, EN_FRONT_DISCS
from environment import make_env
from opponent_library import TYPE_NAMES, WEAKNESS_NAMES
from poc180 import PANELS, COMMON
from saw_env import W_NAMES, T_BLOW, T_FOUL, T_MISS, T_SOLID, T_TAKEN, T_DEALT, T_HURT, T_FIRE
from saw_field import Scenario, PROFILE_TRAVES

OUT_DIR = ROOT / "web" / "replays"
PANEL_NAMES = {"T": "Traves-like", "F": "Field", "L": "Old field"}
COLUMNS = ["ox", "oy", "oa", "ex", "ey", "ea", "arm", "blade", "eblade", "flags", "dealt", "taken"]
EVENTS = ((T_FIRE, "swing"), (T_SOLID, "solid"), (T_BLOW, "legal"), (T_FOUL, "side"), (T_MISS, "floor"), (T_TAKEN, "hit"))


@torch.no_grad()
def export(checkpoint, panels=("T", "F"), fights=4, seconds=180.0, swings=10, reach=0.5, seed_offset=0,
           out_dir=OUT_DIR):
    ckpt = Path(checkpoint)
    agent, meta = load_agent(str(ckpt), DEVICE)
    if meta["rules"] != "saw":
        raise SystemExit(f"{ckpt} is a {meta['rules']}-rules checkpoint: the web viewer shows saw-rules fights")
    hazards = np.load(HAZ_PATH) if HAZ_PATH.exists() else np.zeros((ARENA_FT, ARENA_FT), bool)
    haz_cells = [[int(x), int(y)] for y, x in zip(*np.nonzero(hazards))]
    import saw_walls                                   # saw-v9: the walls are smooth curves, drawn from their outline
    walls = saw_walls.outline()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    n = fights
    for panel in panels:
        torch.manual_seed(PANELS[panel]["seed"] + 7 + seed_offset)
        sc = Scenario(**{**COMMON, **PANELS[panel]["scenario"]}, swings=swings, reach_ft=reach,
                      heading_mode="none", fight_s=seconds)
        gated = gate_from_meta(meta)
        env = make_env("saw", n, "combat", p=0.68, auto_reset=False, scenario=sc, arm_gate=gated)
        env.gate_rule = meta.get("gate_rule") or "heading"
        obs = env.reset()
        state = agent.initial_state(n, DEVICE)
        fresh = torch.ones(n, device=DEVICE)
        frames = [[] for _ in range(n)]
        events = [[] for _ in range(n)]

        def snap():
            rows = torch.stack([
                env.ox, env.oy, env.oa, env.ex, env.ey, env.ea, env.saw_o.arm, env.saw_o.blade, env.saw_e.blade,
                (env.o_stun > 0).float() + 2 * (env.e_stun > 0).float() + 4 * env.gate_open.float()
                + 8 * env.gate_ready.float(),
                env.tally[:, T_DEALT], env.tally[:, T_HURT]], dim=1).cpu().tolist()
            return rows

        static = []
        for k in range(n):
            so, se = env.saw_o, env.saw_e
            static.append({
                "our": {"half_len": env.o_len[k].item(), "half_wid": env.o_wid[k].item(),
                        "aft": so.aft[k].item(), "fore": so.fore[k].item(), "rad": so.rad[k].item()},
                "enemy": {"half_len": env.e_len[k].item(), "half_wid": env.e_wid[k].item(),
                          "weapon": W_NAMES[int(se.kind[k])], "fore": se.fore[k].item(), "wide": se.wide[k].item(),
                          "rad": se.rad[k].item(), "tactic": TYPE_NAMES[int(env.opponents.types[k])],
                          "weakness": WEAKNESS_NAMES[int(env.opponents.weakness[k])],
                          "level": env.field.p[k].item(),
                          "traves": bool(env.field.profile[k] == PROFILE_TRAVES)},
            })
        for k, row in enumerate(snap()):
            frames[k].append(row)
        for step in range(math.ceil(seconds / CTRL_DT) + 2):
            live = ~env.done.clone()
            before = env.tally.clone()
            action, state = agent.act_deterministic(obs, state, fresh)
            fresh = torch.zeros(n, device=DEVICE)
            obs, _, done, _ = env.step(action)
            delta = (env.tally - before).cpu()
            rows = snap()
            for k in range(n):
                if not live[k]:
                    continue
                frames[k].append(rows[k])
                for col, name in EVENTS:
                    if delta[k, col] > 0:
                        events[k].append([len(frames[k]) - 1, name])
            if done.all():
                break
        t = env.tally.cpu()
        for k in range(n):
            f = np.round(np.array(frames[k], dtype=np.float64), 3)
            name = f"{ckpt.parent.name}_{ckpt.stem}_{panel}{k + 1 + seed_offset}"
            doc = {
                "name": name, "run": ckpt.parent.name, "checkpoint": ckpt.name, "update": meta.get("update", 0),
                "panel": panel, "panel_name": PANEL_NAMES[panel], "fight": k + 1 + seed_offset,
                "dt": CTRL_DT, "seconds": seconds, "arena_ft": ARENA_FT, "hazards": [] if walls else haz_cells, "walls": walls,
                "swings": swings, "reach_ft": reach, "arm_gate": bool(gated),
                "gate_rule": env.gate_rule if gated else None,
                "arm_parked": ARM_PARKED, "arm_down": ARM_DOWN, "front_discs": EN_FRONT_DISCS,
                **static[k],
                "summary": {"solid": t[k, T_SOLID].item(), "legal": t[k, T_BLOW].item(), "side": t[k, T_FOUL].item(),
                            "floor": t[k, T_MISS].item(), "swings_used": t[k, T_FIRE].item(),
                            "hits_taken": t[k, T_TAKEN].item(), "dealt": t[k, T_DEALT].item(),
                            "taken": t[k, T_HURT].item()},
                "columns": COLUMNS, "events": events[k], "frames": f.tolist(),
            }
            path = out_dir / f"{name}.json"
            path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
            written.append(path)
            s = doc["summary"]
            print(f"[saved] {path.name}  {doc['enemy']['tactic']} ({doc['enemy']['weapon']})  "
                  f"solid strikes {s['solid']:.0f}  swings {s['swings_used']:.0f}  hits taken {s['hits_taken']:.0f}",
                  flush=True)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--panels", nargs="+", choices=tuple(PANELS), default=["T", "F"])
    ap.add_argument("--fights", type=int, default=4, help="fights per panel (default 4)")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--swings", type=int, default=10)
    ap.add_argument("--reach", type=float, default=0.5)
    ap.add_argument("--seed-offset", type=int, default=0, help="a different set of fights")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    export(a.checkpoint, tuple(a.panels), a.fights, a.seconds, a.swings, a.reach, a.seed_offset)
