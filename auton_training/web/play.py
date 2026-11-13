"""
Fight the policy in the browser: a person drives the opponent (a Traves-like
horizontal bar spinner) with the keyboard, the newest checkpoint drives our
robot. One fight at a time, on the CPU, so it can run beside training.

web/serve.py loads this the first time somebody presses "Fight the policy".
"""
import json
import math
import os
import time
import sys
import threading
from pathlib import Path

os.environ.setdefault("DODGE_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch                                                    # noqa: E402

torch.set_num_threads(2)

import saw_walls                                                # noqa: E402
from agent import load_agent, gate_from_meta                    # noqa: E402
from config import DEVICE, CTRL_DT, ARENA_FT, ARM_PARKED, ARM_DOWN, EN_FRONT_DISCS   # noqa: E402
from environment import make_env                                # noqa: E402
from poc180 import PANELS, COMMON                               # noqa: E402
from saw_env import W_NAMES, T_BLOW, T_FOUL, T_MISS, T_SOLID, T_TAKEN, T_DEALT, T_HURT, T_FIRE   # noqa: E402
from saw_field import Scenario                                  # noqa: E402

EVENTS = ((T_FIRE, "swing"), (T_SOLID, "solid"), (T_BLOW, "legal"), (T_FOUL, "side"), (T_MISS, "floor"), (T_TAKEN, "hit"))
COLUMNS = ["ox", "oy", "oa", "ex", "ey", "ea", "arm", "blade", "eblade", "flags", "dealt", "taken"]
PREFER = ("saw_v9b_traves", "saw_v9b_traves_r05", "saw_v8a_traves_hold")     # newest checkpoint of the first run that has one
MAX_STEPS_PER_CALL = 8
PLAYS = ROOT / "web" / "plays"       # every live fight of 10 s or more is kept here: frames plus the person's inputs
_lock = threading.Lock()
_game = None


def newest_checkpoint():
    # DODGE_PLAY_CHECKPOINT picks the policy to fight: a checkpoint file, or a run folder
    # (its newest SAW_upd*.pt). Relative paths are taken from this folder's parent.
    chosen = os.environ.get("DODGE_PLAY_CHECKPOINT")
    if chosen:
        path = Path(chosen) if os.path.isabs(chosen) else ROOT / chosen
        if path.is_dir():
            found = sorted(path.glob("SAW_upd*.pt"), key=lambda p: p.stat().st_mtime)
            if not found:
                raise FileNotFoundError(f"no SAW_upd*.pt in {path}")
            return found[-1]
        if not path.is_file():
            raise FileNotFoundError(f"DODGE_PLAY_CHECKPOINT: {path} does not exist")
        return path
    for run in PREFER:
        found = sorted((ROOT / "ppo_models" / run).glob("SAW_upd*.pt"), key=lambda p: p.stat().st_mtime)
        if found:
            return found[-1]
    raise FileNotFoundError("no checkpoint to fight: looked in ppo_models/" + ", ppo_models/".join(PREFER))


class Game:
    @torch.no_grad()
    def __init__(self, reach=0.75, seconds=180.0, swings=10):
        self.ckpt = newest_checkpoint()
        self.agent, self.meta = load_agent(str(self.ckpt), DEVICE)
        torch.manual_seed(int.from_bytes(os.urandom(4), "little"))
        sc = Scenario(**{**COMMON, **PANELS["T"]["scenario"]}, swings=swings, reach_ft=reach, heading_mode="none",
                      fight_s=seconds)
        gated = gate_from_meta(self.meta)
        env = make_env("saw", 1, "combat", p=0.68, auto_reset=False, scenario=sc, arm_gate=gated)
        env.gate_rule = self.meta.get("gate_rule") or "heading"
        if self.meta.get("gate_persist"):
            env.gate_persist = int(self.meta["gate_persist"])
        env.manual_enemy = True
        self.env, self.seconds, self.frames = env, seconds, 1
        self.inputs, self.saved, self.began = [], False, time.strftime("%Y%m%d_%H%M%S")
        self.obs = env.reset()
        env.manual_enemy = True
        self.state = self.agent.initial_state(1, DEVICE)
        self.fresh = torch.ones(1, device=DEVICE)
        so, se = env.saw_o, env.saw_e
        self.head = {
            "name": "live", "run": self.ckpt.parent.name, "checkpoint": self.ckpt.name, "update": self.meta.get("update", 0),
            "panel": "P", "panel_name": "You drive the opponent", "fight": 1, "live": True,
            "dt": CTRL_DT, "seconds": seconds, "arena_ft": ARENA_FT, "hazards": [], "walls": saw_walls.outline(),
            "swings": swings, "reach_ft": reach, "arm_gate": bool(gated), "gate_rule": env.gate_rule if gated else None,
            "arm_parked": ARM_PARKED, "arm_down": ARM_DOWN, "front_discs": EN_FRONT_DISCS,
            "our": {"half_len": env.o_len[0].item(), "half_wid": env.o_wid[0].item(),
                    "aft": so.aft[0].item(), "fore": so.fore[0].item(), "rad": so.rad[0].item()},
            "enemy": {"half_len": env.e_len[0].item(), "half_wid": env.e_wid[0].item(),
                      "weapon": W_NAMES[int(se.kind[0])], "fore": se.fore[0].item(), "wide": se.wide[0].item(),
                      "rad": se.rad[0].item(), "tactic": "You", "weakness": "none", "level": 1.0, "traves": True},
            "columns": COLUMNS, "events": [], "frames": [self.snap()],
        }
        self.log_frames, self.log_events = list(self.head["frames"]), []

    def snap(self):
        e = self.env
        row = [e.ox, e.oy, e.oa, e.ex, e.ey, e.ea, e.saw_o.arm, e.saw_o.blade, e.saw_e.blade,
               (e.o_stun > 0).float() + 2 * (e.e_stun > 0).float() + 4 * e.gate_open.float() + 8 * e.gate_ready.float(),
               e.tally[:, T_DEALT], e.tally[:, T_HURT]]
        return [round(float(v[0]), 3) for v in row]

    @torch.no_grad()
    def step(self, throttle, turn, k):
        env, frames, events = self.env, [], []
        for _ in range(max(1, min(int(k), MAX_STEPS_PER_CALL))):
            if bool(env.done[0]):
                break
            env.manual_cmd[0, 0], env.manual_cmd[0, 1] = max(-1.0, min(1.0, throttle)), max(-1.0, min(1.0, turn))
            self.inputs.append([float(env.manual_cmd[0, 0]), float(env.manual_cmd[0, 1])])
            before = env.tally.clone()
            action, self.state = self.agent.act_deterministic(self.obs, self.state, self.fresh)
            self.fresh = torch.zeros(1, device=DEVICE)
            self.obs, _, _, _ = env.step(action)
            delta = env.tally - before
            frames.append(self.snap())
            for col, name in EVENTS:
                if delta[0, col] > 0:
                    events.append([self.frames, name])
            self.frames += 1
        self.log_frames.extend(frames)
        self.log_events.extend(events)
        if bool(env.done[0]):
            self.save()
        t = env.tally[0]
        return {"frames": frames, "events": events, "done": bool(env.done[0]),
                "summary": {"solid": t[T_SOLID].item(), "side": t[T_FOUL].item(), "floor": t[T_MISS].item(),
                            "swings_used": t[T_FIRE].item(), "hits_taken": t[T_TAKEN].item(),
                            "dealt": t[T_DEALT].item(), "taken": t[T_HURT].item()}}

    def save(self):
        """Keep the fight for later: what happened each control step and what the person asked for."""
        if self.saved or len(self.inputs) * CTRL_DT < 10.0:
            return
        PLAYS.mkdir(parents=True, exist_ok=True)
        doc = {**self.head, "frames": self.log_frames, "events": self.log_events, "human": "enemy",
               "inputs": self.inputs, "input_columns": ["throttle", "turn"]}
        (PLAYS / f"play_{self.began}.json").write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        self.saved = True


def start(body):
    global _game
    with _lock:
        if _game is not None:
            _game.save()
        _game = Game(reach=float(body.get("reach", 0.75)))
        return _game.head


def step(body):
    with _lock:
        if _game is None:
            return {"error": "no fight is running"}
        f = lambda key: float(body.get(key, 0.0)) if math.isfinite(float(body.get(key, 0.0))) else 0.0
        return _game.step(f("throttle"), f("turn"), int(body.get("steps", 3)))
