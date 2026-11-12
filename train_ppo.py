"""
Recurrent PPO trainer (CleanRL style). One loop, two stages, two rulesets:

    uv run python train_ppo.py --stage pretrain    # drills against unarmed targets
    uv run python train_ppo.py                     # combat; seeds from the pretrain checkpoint if present
    uv run python train_ppo.py --resume ppo_models/ZACAI_upd120.pt

    uv run python train_ppo.py --rules saw --stage pretrain    # the same two stages under the saw rules
    uv run python train_ppo.py --rules saw

    # saw-v8: a run in its own folder, started from a saw-v7 policy (30 inputs, padded to 44)
    uv run python train_ppo.py --name saw_v8a_run1/SAW --init-pad ppo_models/saw_v7_run7/SAW_upd290.pt         --epochs 2 --minibatch-envs 2048 --target-kl 0.02 --lr 1e-4 --critic-warmup 10 --p-range 0.4 0.85

A run never writes over another: a fresh run whose name already has
checkpoints stops and says so (the default names included). Pick a new
--name, or --resume the old run.

Pretraining is the same environment, robot, sensors, rules and PPO update as
combat. The only difference is who is on the other side: parked and drifting
targets with no weapon, so the policy learns to drive, aim and strike before
anything fights back.

Combat runs a performance-gated curriculum: the level p (opponent skill and
tactic mix) only rises while the policy is winning (classic), or doing about
as well as a competent hand-written driver at that level (saw: see gate()),
and frozen snapshots of the policy join the opponent pool once p is high
enough.

Under the saw rules with --p-range the curriculum is off: every fight draws
its opponent's level from the range. Self-play snapshots are classic only.

The ruleset (config.RULES) decides the fight and the size of the policy's
observation and action. Each keeps its own checkpoints (ZACAI_* for classic,
SAW_* for saw); a checkpoint cannot be continued under the other ruleset.

Sized for a large CUDA GPU. For a quick local check pass --n-envs 4096.
"""
import argparse
import glob
import math
import random
import re
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from agent import Agent, save_checkpoint, load_checkpoint, load_padded, read_meta
from config import DEVICE, HIDDEN, SAVE_DIR, RULES, DEFAULT_NAME, CTRL_DT, PRIV_DIM
from environment import (
    env_class, make_env, RW_N, RW_NAMES, END_NAMES, WIN_ENDS, LOSS_ENDS, DRAW_ENDS,
)
from opponent_library import (
    N_TYPES, TYPE_NAMES, DRILL_TYPES, COMBAT_TYPES, SOFT_TYPES,
)
from opponent_selfplay import SelfPlayController
from saw_field import Scenario
from saw_env import (
    attack_in, T_BLOW, T_MISS, T_DEALT, T_HURT, T_SOLID, T_FOUL, T_REQ, T_OPEN, T_FIRE, T_READY, T_REQ_OPEN, T_BUDGET,
)

say = partial(print, flush=True)     # a log file follows the run line by line

# ── PPO hyperparameters ───────────────────────────────────────────────────
N_BATCH        = 131072
N_STEPS        = 64      # rollout length; each minibatch replays whole rollouts through the LSTM
MINIBATCH_SIZE = 32768   # samples per minibatch = (MINIBATCH_SIZE // N_STEPS) envs x N_STEPS steps
N_EPOCHS       = 5
GAE_LAMBDA     = 0.95
GAE_LAMBDA_SAW = 0.97    # with gamma 0.998: a swing pays off seconds after the driving that set it up
CLIP_EPS       = 0.2
ENT_COEF       = 0.002
VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5
TARGET_KL      = 0.03    # stop an update early once the policy has moved this far
LR_START       = 3e-4
LR_MIN         = 3e-5
LR_MIN_SAW     = 1e-4    # the saw curriculum reaches p = 1 long before the policy is finished (update 205 of the first
                         # run, which then gained 6 points in 290 updates at 3e-5), so its rate stays higher there
STD_FLOOR      = 0.25    # exploration never collapses below this
STD_NEW_STAGE  = 0.60    # action std when combat starts from a pretrain checkpoint
COLD_START_LR  = 0.1     # lr multiplier for the first update on trained weights with a fresh optimizer

SAVE_EVERY     = 10

# ── Pretrain stage ────────────────────────────────────────────────────────
PRETRAIN_MAX_UPDATES = 300
PRETRAIN_MIN_UPDATES = 15     # early updates only see the quick episodes, so their hit rate flatters
PRETRAIN_TARGET_HIT  = 0.90   # done once this share of drills end in a hit ...
PRETRAIN_PATIENCE    = 3      # ... for this many updates in a row

# ── Combat curriculum ─────────────────────────────────────────────────────
P_STEP        = 0.02    # how far p moves when the gate opens
FAST_WIN      = 0.65    # winning this comfortably moves p twice as far
GATE_WIN      = 0.45    # win rate over the hard tactics needed to advance (and wins must outnumber losses)
MIN_TYPE_WIN  = 0.20    # no hard tactic may be below this
MIN_SAMPLES   = 20      # episodes before a tactic's win rate counts
# The saw rules are not an even match. Under saw-v5 a spinner's weapon side
# is off limits and a hit has to be charged, and no hand-written driver that
# sees the true state finishes ahead of an expert at all; the best is ahead
# of beginners in 17% of fights (docs/decisions/0004). What a policy can
# reach is not known beforehand, and it has to be ready for experts whether
# or not it beats them. So the gate asks for little, and it does not wait
# for ever: after SAW_GATE_WAIT updates at one level the opponents get
# better regardless.
# saw-v6 (hammer only): the run before this one was moved up a level every 24
# updates whether it had earned it or not, reached p = 0.80 ahead in 7% of
# fights, and was no better at any level than it had been 200 updates earlier.
# So the gate now asks for the same record at every level and waits for it:
# the level a run sits at is the level it holds its own at.
# Later the same day the team said the robot does not need to win: "we want
# to get some attacks in and get proof of concept". So the saw gate does not
# read the scorecard any more. It reads the share of fights in which the
# blade did real damage (saw_env.attack_in), and it still waits for it.
# saw-v7: half was set when a saw laid on the floor counted. With strikes from
# above only, 86 updates against the easiest opponents left the log at 33% and
# the audits at 20 to 24%, flat, and those opponents drive at 3.5 mph: a policy
# that waits there for ever never meets a robot moving at a real speed. A
# quarter lets it climb for as long as it still lands some.
SAW_GATE_HIT  = (0.25, 0.25)    # share of fights with an attack in needed to advance, p = 0 -> p = 1
# An opponent is drawn at the level its fight starts at, and the shortest
# fight takes 10.4 updates to finish. A gate that looks every update is
# reading fights against easier opponents than the level it is judging: the
# first saw-v5 run went up a level an update on 44% ahead in the log while it
# was ahead in 15 to 20% of fights at the level it had reached (and in 6% by
# p = 0.68). So after a level change the saw gate waits until the 20 s fights
# it reads were fought there. Longer fights still lag, by a level or two.
SAW_GATE_DWELL = 11             # updates at a level before the saw gate looks
SAW_GATE_WAIT  = 0              # updates at one level before the saw curriculum moves on anyway (0: it waits)
SAW_P_STEP     = 0.04           # with a dwell that long, levels half as fine


def gate(rules, p):
    """What the curriculum asks for at level p: (record over the hard
    tactics, the least any one of them may show, whether wins must outnumber
    losses). The record is the win rate under the classic rules and the share
    of fights with an attack in under the saw rules."""
    if rules == "classic":
        return GATE_WIN, MIN_TYPE_WIN, True
    need = SAW_GATE_HIT[0] + (SAW_GATE_HIT[1] - SAW_GATE_HIT[0]) * p
    return need, need / 4, False


def advance(rules, p, win, loss, blocked, held):
    """Does the curriculum move on after this update? "" for no, "gate" when
    the record asks for it, "waited" when the saw rules have spent
    SAW_GATE_WAIT updates at this level. ``blocked``: some tactic is under
    its floor. ``held``: updates since the level last changed."""
    if p >= 1.0 or (rules == "saw" and held < SAW_GATE_DWELL):
        return ""
    need, _, must_lead = gate(rules, p)
    if win >= need and (win > loss or not must_lead) and not blocked:
        return "gate"
    if rules == "saw" and SAW_GATE_WAIT and held >= SAW_GATE_WAIT:
        return "waited"
    return ""


def learning_rate(rules, pretrain, p):
    """Full rate in the drills; in combat it falls as the level rises."""
    if pretrain:
        return LR_START
    return LR_START - p * (LR_START - (LR_MIN if rules == "classic" else LR_MIN_SAW))

# ── Self-play ─────────────────────────────────────────────────────────────
SELFPLAY_FRAC    = 0.25  # share of envs facing a frozen snapshot
SELFPLAY_MIN_P   = 0.50  # only once the curriculum is this far along
SNAPSHOT_REFRESH = 50    # updates between snapshot swaps

N_ENDS = max(END_NAMES) + 1
# Row of the reward log each end reason belongs to: 0 win, 1 loss, 2 draw
END_ROW = torch.ones(N_ENDS, dtype=torch.long, device=DEVICE)
END_ROW[list(WIN_ENDS)] = 0
END_ROW[list(DRAW_ENDS)] = 2


class RolloutStats:
    """Episode outcomes by opponent type, mean reward terms by outcome and
    (saw rules) mean fight tallies, overall and by fight length and by what
    the camera says of the opponent's heading, accumulated on the device and
    read once per update."""

    LENGTHS = (20, 60, 180)     # fight lengths the saw log is split by, s

    def __init__(self, tally_names=()):
        self.ends   = torch.zeros(N_TYPES * N_ENDS, device=DEVICE)
        self.rw_sum = torch.zeros(3, RW_N, device=DEVICE)     # rows: win, loss, draw
        self.rw_cnt = torch.zeros(3, device=DEVICE)
        self.tally_names = tally_names
        self.tally  = torch.zeros(len(tally_names), device=DEVICE)
        self.hits   = torch.zeros(N_TYPES, device=DEVICE)     # fights with an attack in, by opponent type
        # Per group: the summed tallies, then fights, fights with an attack in, control steps fought
        width = len(tally_names) + 3
        self.by_len = torch.zeros(len(self.LENGTHS), width, device=DEVICE)
        self.by_hdg = torch.zeros(3, width, device=DEVICE)

    def add(self, done, info):
        d = done.float()
        self.ends += torch.bincount(info.opp_type * N_ENDS + info.end_reason,
                                    weights=d, minlength=N_TYPES * N_ENDS)
        row = END_ROW[info.end_reason]
        self.rw_sum.index_add_(0, row, info.rw * d.unsqueeze(1))
        self.rw_cnt.index_add_(0, row, d)
        if info.tally is None:
            return
        attack = attack_in(info.tally).float()
        self.tally += (info.tally * d.unsqueeze(1)).sum(0)
        self.hits += torch.bincount(info.opp_type, weights=d * attack, minlength=N_TYPES)
        seconds = getattr(info, "fight_s", None)
        if seconds is None:
            return
        rows = torch.cat([info.tally, torch.stack([torch.ones_like(d), attack, seconds / CTRL_DT], dim=1)], dim=1)
        rows = rows * d.unsqueeze(1)
        lengths = torch.tensor(self.LENGTHS, device=DEVICE)
        bucket = (seconds.round().long().unsqueeze(1) == lengths).long().argmax(1)
        known = (seconds.round().long().unsqueeze(1) == lengths).any(1).float().unsqueeze(1)
        self.by_len.index_add_(0, bucket, rows * known)
        self.by_hdg.index_add_(0, info.hdg_mode.clamp(0, 2), rows)

    def summary(self, types):
        """Rates over the given opponent types, plus each type's win rate.
        A draw is a timeout under the classic rules and a fight level on
        damage under the saw rules."""
        ends = self.ends.view(N_TYPES, N_ENDS).cpu()
        per_type = {t: (ends[t, list(WIN_ENDS)].sum().item() / ends[t].sum().item(), int(ends[t].sum()))
                    for t in range(N_TYPES) if ends[t].sum() > 0}
        sel = ends[list(types)].sum(0)
        total = max(1.0, sel.sum().item())
        hits, fought = self.hits.cpu(), ends.sum(1)
        return {
            "episodes": int(sel.sum()),
            "win":      sel[list(WIN_ENDS)].sum().item() / total,
            "loss":     sel[list(LOSS_ENDS)].sum().item() / total,
            "draw":     sel[list(DRAW_ENDS)].sum().item() / total,
            "per_type": per_type,
            # Saw rules: the share of fights in which our blade did real damage
            "attack":   hits[list(types)].sum().item() / total,
            "attack_per_type": {t: hits[t].item() / fought[t].item() for t in per_type},
        }

    def reward_lines(self, draw_label):
        rows = []
        for name, s, c in zip(("WIN ", "LOSS", draw_label.upper()), self.rw_sum.cpu(), self.rw_cnt.cpu()):
            if c == 0:
                rows.append(f"  rw[{name}]: n=0")
                continue
            mean = (s / c).tolist()
            rows.append(f"  rw[{name}]: n={int(c)} total={sum(mean):+.2f} "
                        + " ".join(f"{k}={v:+.2f}" for k, v in zip(RW_NAMES, mean)))
        return rows

    def tally_line(self):
        """Mean of each fight tally over the fights that finished, or None
        under rules that keep none."""
        fights = self.rw_cnt.sum().item()
        if not self.tally_names or fights == 0:
            return None
        return "  fight: " + " ".join(f"{k}={v:.2f}" for k, v in zip(self.tally_names, (self.tally / fights).tolist()))

    def length_lines(self):
        """Saw rules: one line per fight length that finished fights this
        update, and one per heading-reading mode when more than the plain
        one is in play. Solid strikes, attacks in, damage, swings and where
        they landed (legal / weapon side / floor), and what the gate did:
        the share of control steps with a request, with the gate open and
        ready, the share of open steps that were used, firings, and swings
        left over at the bell."""
        if len(self.tally_names) <= T_BUDGET:
            return []
        k = len(self.tally_names)
        share = lambda a, b: f"{a / b:.2f}" if b > 0 else "-"

        def line(label, row):
            t, f, atk, cs = row[:k], row[k], row[k + 1], row[k + 2]
            used = t[T_BLOW] + t[T_FOUL] + t[T_MISS]
            return (f"  {label}: n={int(f)} solid={t[T_SOLID] / f:.2f} atk={atk / f:.2f} dealt={t[T_DEALT] / f:.2f} "
                    f"taken={t[T_HURT] / f:.2f} swings={used / f:.2f} "
                    f"land L/W/F={share(t[T_BLOW], used)}/{share(t[T_FOUL], used)}/{share(t[T_MISS], used)} "
                    f"req={t[T_REQ] / cs:.3f} open={t[T_OPEN] / cs:.3f} ready={t[T_READY] / cs:.2f} "
                    f"req|open={share(t[T_REQ_OPEN], t[T_OPEN])} fires={t[T_FIRE] / f:.2f} "
                    f"unused={(t[T_BUDGET] - t[T_FIRE]) / f:.2f}")

        by_len, by_hdg = self.by_len.cpu().tolist(), self.by_hdg.cpu().tolist()
        lines = [line(f"len{sec}", row) for sec, row in zip(self.LENGTHS, by_len) if row[k] > 0]
        if any(row[k] > 0 for row in by_hdg[1:]):
            lines += [line(f"hdg{m}", row) for m, row in enumerate(by_hdg) if row[k] > 0]
        return lines


def timeout_bootstrap(env, agent, info, reward, lstm_state, gamma, priv=None):
    """Where the clock running out is not a terminal state (the classic
    rules), the fight would go on, so the value of where it stood is added
    back to the reward. Where the bell is the end (env.timeout_bootstrap is
    False, the saw rules) the reward is returned as it is."""
    if not env.timeout_bootstrap:
        return reward
    cut = info.timeout.nonzero(as_tuple=False).squeeze(1)
    if cut.numel() > 0:
        with torch.no_grad():
            v = agent.get_value(info.final_obs[cut],
                                (lstm_state[0][:, cut].contiguous(), lstm_state[1][:, cut].contiguous()),
                                torch.zeros(cut.numel(), device=DEVICE),
                                priv=None if priv is None else priv[cut])
        reward[cut] += gamma * v.flatten()
    return reward


def ppo_update(agent, optimizer, buf, initial_state, next_value, next_done, n_envs, *,
               gamma, lam, epochs, envs_per_mb, target_kl, critic_only=False, priv=None, mask=None):
    """GAE, then up to ``epochs`` of clipped PPO. Minibatches are whole env
    rollouts so the LSTM is unrolled from the state it actually had at the
    rollout start. ``priv`` (steps, envs, priv_dim) feeds a privileged
    critic; ``mask`` (steps, envs, act_dim) leaves action dimensions that had
    no effect out of the log-probability and the entropy. ``critic_only``
    trains the value head alone: nothing the actor depends on moves.

    Returns dict(kl, epochs_run, ev, v_loss, clipfrac): the last KL, epochs
    done (a fraction when the KL stop cut one short), how much of the
    returns' variance the value before the update explained, the last value
    loss and the share of samples the clip touched."""
    obs, actions, logprobs, rewards, dones, values = buf
    with torch.no_grad():
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(N_STEPS)):
            nonterminal = 1.0 - (next_done if t == N_STEPS - 1 else dones[t + 1])
            nv = next_value if t == N_STEPS - 1 else values[t + 1]
            delta = rewards[t] + gamma * nv * nonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        returns = advantages + values
        var = returns.var().item()
        ev = 1.0 - (returns - values).var().item() / var if var > 0 else 0.0

    b_obs      = obs.reshape(-1, obs.shape[-1])
    b_actions  = actions.reshape(-1, actions.shape[-1])
    b_logprobs = logprobs.reshape(-1)
    b_dones    = dones.reshape(-1)
    b_returns  = returns.reshape(-1)
    b_adv      = (advantages.reshape(-1) - advantages.mean()) / (advantages.std() + 1e-8)
    b_priv     = None if priv is None else priv.reshape(-1, priv.shape[-1])
    b_mask     = None if mask is None else mask.reshape(-1, mask.shape[-1])
    heads      = ("critic.", "critic_priv.")

    envs_per_mb = max(1, min(n_envs, envs_per_mb))
    n_mb = math.ceil(n_envs / envs_per_mb)
    steps = torch.arange(N_STEPS, device=DEVICE).unsqueeze(1) * n_envs
    out = dict(kl=0.0, epochs_run=0.0, ev=ev, v_loss=0.0, clipfrac=0.0)
    for epoch in range(epochs):
        perm = torch.randperm(n_envs, device=DEVICE)
        for k, start in enumerate(range(0, n_envs, envs_per_mb)):
            env_idx = perm[start:start + envs_per_mb]
            mb = (steps + env_idx).flatten()               # time-major: all envs at t=0, then t=1, ...
            _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                b_obs[mb], (initial_state[0][:, env_idx], initial_state[1][:, env_idx]),
                b_dones[mb], b_actions[mb],
                priv=None if b_priv is None else b_priv[mb], dim_mask=None if b_mask is None else b_mask[mb])
            logratio = newlogprob - b_logprobs[mb]
            ratio = logratio.exp()
            pg_loss = torch.max(-b_adv[mb] * ratio,
                                -b_adv[mb] * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)).mean()
            v_loss = F.mse_loss(newvalue.view(-1), b_returns[mb])
            loss = VF_COEF * v_loss if critic_only else pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

            optimizer.zero_grad()
            loss.backward()
            if critic_only:
                for name, par in agent.named_parameters():
                    if not name.startswith(heads):
                        par.grad = None
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            with torch.no_grad():
                agent.actor_logstd.clamp_(min=math.log(STD_FLOOR))
                out["kl"] = ((ratio - 1) - logratio).mean().item()
                out["clipfrac"] = ((ratio - 1).abs() > CLIP_EPS).float().mean().item()
                out["v_loss"] = v_loss.item()
            out["epochs_run"] = epoch + (k + 1) / n_mb
            if out["kl"] > target_kl and not critic_only:
                return out
    return out


def _run_prefix(args, rules, pretrain):
    """Where this run's checkpoints go, as a path prefix. A run never writes
    over another one's."""
    name = getattr(args, "name", None)
    if name and (Path(name).is_absolute() or ".." in Path(name).parts):
        raise SystemExit(f"--name {name!r}: give a name inside {SAVE_DIR}, without '..'")
    prefix = SAVE_DIR / (name or DEFAULT_NAME[rules])
    fresh = True
    if args.resume:
        res = Path(args.resume)
        own = res.parent / re.sub(r"(_pretrain)?(_latest|_upd\d+)$", "", res.stem)
        if not name:
            prefix = own
        fresh = prefix.resolve() != own.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if fresh:
        if pretrain:
            taken = [f"{prefix}_pretrain_latest.pt"] if Path(f"{prefix}_pretrain_latest.pt").exists() else []
        else:
            taken = [f for f in [f"{prefix}_latest.pt"] if Path(f).exists()] + sorted(glob.glob(f"{prefix}_upd*.pt"))
        if taken:
            raise SystemExit(f"a run already wrote {taken[0]}" + (f" and {len(taken) - 1} more" if len(taken) > 1 else "")
                             + ". Pick a new --name, or --resume it.")
    return str(prefix)


def train(args):
    stage, n = args.stage, args.n_envs
    pretrain = stage == "pretrain"
    opt = lambda name: getattr(args, name, None)
    init_pad = opt("init_pad")
    if init_pad and (args.resume or args.init):
        raise SystemExit("--init-pad starts a run from padded weights: it does not go with --resume or --init")
    if init_pad and pretrain:
        raise SystemExit("--init-pad is for the combat stage")
    # The ruleset: the one asked for, else the one the starting checkpoint was trained under
    start = args.resume or init_pad or args.init
    start_meta = read_meta(start) if start else {}
    rules = args.rules or start_meta.get("rules", "classic")
    if start and start_meta.get("rules", "classic") != rules:
        raise SystemExit(f"{start} was trained under the {start_meta.get('rules', 'classic')} rules, not {rules}")
    saw = rules == "saw"
    if not saw and (opt("p_range") or opt("no_gate") or opt("priv_critic") or opt("gate_rule")
                    or opt("traves_share") is not None or opt("weapon_shares")):
        raise SystemExit("--p-range, --no-gate, --gate-rule, --priv-critic, --traves-share and --weapon-shares: "
                         "saw rules only")
    prefix = _run_prefix(args, rules, pretrain)
    pretrain_path = f"{prefix}_pretrain_latest.pt"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Schedule: what the command line gives, else (on a resume) what the run was using, else the defaults
    was = start_meta if args.resume else {}
    save_every  = opt("save_every") or SAVE_EVERY
    epochs      = opt("epochs") or N_EPOCHS
    target_kl   = opt("target_kl") or TARGET_KL
    envs_per_mb = opt("minibatch_envs") or MINIBATCH_SIZE // N_STEPS
    lam         = opt("lam") or was.get("lam") or (GAE_LAMBDA_SAW if saw else GAE_LAMBDA)
    warmup      = opt("critic_warmup") or 0
    p_range     = opt("p_range") or was.get("p_range")
    p_range     = tuple(float(x) for x in p_range) if p_range else None
    no_gate     = opt("no_gate")
    arm_gate    = (not no_gate) if no_gate is not None else bool(was.get("arm_gate", True))
    gate_rule   = opt("gate_rule") or was.get("gate_rule") or "heading"
    priv_dim    = PRIV_DIM if opt("priv_critic") else int(was.get("priv_dim", 0))
    # The opponent mix a specialist is trained on (None: the field's own shares)
    traves_share  = opt("traves_share") if opt("traves_share") is not None else was.get("traves_share")
    weapon_shares = opt("weapon_shares") or was.get("weapon_shares")
    weapon_shares = tuple(float(x) for x in weapon_shares) if weapon_shares else None
    heading_mode  = opt("heading_mode") or was.get("heading_mode")
    reach_ft      = opt("reach_ft") if opt("reach_ft") is not None else was.get("reach_ft")
    gate_persist  = opt("gate_persist") or was.get("gate_persist")
    if not saw and heading_mode:
        raise SystemExit("--heading-mode: saw rules only")
    swings_3min   = opt("swings_3min") if opt("swings_3min") is not None else was.get("swings_3min")
    foul          = opt("foul") if opt("foul") is not None else was.get("foul")
    if not saw and (swings_3min is not None or foul is not None):
        raise SystemExit("--swings-3min and --foul: saw rules only")
    if args.resume and opt("priv_critic") and int(start_meta.get("priv_dim", 0)) != PRIV_DIM:
        raise SystemExit(f"{args.resume} was trained without the privileged critic: --priv-critic cannot be added on a resume")

    cls = env_class(rules)
    agent = Agent(cls.obs_dim, cls.act_dim, priv_dim).to(DEVICE)
    optimizer = Adam(agent.parameters(), lr=LR_START, eps=1e-5)
    update, p = 0, args.p or 0.0
    if p_range:
        p = 0.5 * (p_range[0] + p_range[1])
    init_from = padded_from = None

    def narrower(path):
        if read_meta(path).get("obs_dim", cls.obs_dim) < cls.obs_dim:
            raise SystemExit(f"{path} has {read_meta(path)['obs_dim']} inputs and the {rules} rules now have "
                             f"{cls.obs_dim}: start from it with --init-pad")

    # Trained weights under a fresh Adam take one oversized step, so the first
    # update after such a start that trains the actor runs at a fraction of
    # the learning rate.
    cold_start = False
    if args.resume:
        narrower(args.resume)
        meta = load_checkpoint(args.resume, agent, DEVICE, optimizer)
        cold_start = not meta.get("optimizer", False)
        if meta.get("stage", stage) == stage:
            update = meta.get("update", 0)
            if args.p is None and not opt("p_range"):
                p = meta.get("p", p)
        init_from, padded_from = meta.get("init_from"), meta.get("padded_from")
        say(f"[RESUME] {args.resume}  update={update}  p={p:.2f}")
    elif init_pad:
        padded_from = load_padded(init_pad, agent, DEVICE)["padded_from"]
        cold_start, init_from = True, str(init_pad)
        say(f"[INIT] combat starts from {init_pad}, padded from {padded_from} to {cls.obs_dim} inputs")
        if warmup and not priv_dim:
            say("[WARN] --critic-warmup after --init-pad without --priv-critic: the critic cannot see inputs "
                f"{padded_from}..{cls.obs_dim - 1} during the warm-up (their first-layer weights are zero and frozen)")
    elif not pretrain and Path(args.init or pretrain_path).exists():
        narrower(args.init or pretrain_path)
        load_checkpoint(args.init or pretrain_path, agent, DEVICE)
        cold_start, init_from = True, str(args.init or pretrain_path)
        with torch.no_grad():
            agent.actor_logstd.fill_(math.log(STD_NEW_STAGE))
        say(f"[INIT] combat starts from {args.init or pretrain_path}")
    elif not pretrain:
        say(f"[INIT] no pretrain checkpoint at {pretrain_path}; combat starts from scratch. "
            "Run --stage pretrain first for a much faster start.")
    if opt("init_std") and (init_pad or args.init):
        with torch.no_grad():
            agent.actor_logstd.fill_(math.log(opt("init_std")))

    pinned = saw and (traves_share is not None or weapon_shares is not None or heading_mode is not None or reach_ft is not None)
    scenario = Scenario(traves_share=traves_share, weapon_shares=weapon_shares, heading_mode=heading_mode, reach_ft=reach_ft) if pinned else None
    env = make_env(rules, n, stage=stage, p=p, arm_gate=arm_gate if saw else None, p_range=p_range if saw else None,
                   scenario=scenario)
    gamma = opt("gamma") or was.get("gamma") or env.gamma
    env.gamma = gamma
    if saw:
        env.gate_rule = gate_rule
        if gate_persist:
            env.gate_persist = int(gate_persist)
        if swings_3min is not None:
            env.swings_3min = float(swings_3min)
        if foul is not None:
            env.r_foul = float(foul)
    selfplay = SelfPlayController(n, env.obs_dim, env.act_dim) if not pretrain and not saw else None
    tracked = DRILL_TYPES if pretrain else tuple(t for t in COMBAT_TYPES if t not in SOFT_TYPES)
    draw_label = "tout" if rules == "classic" else "draw"
    gated = saw and env.arm_gate

    obs_buf      = torch.zeros((N_STEPS, n, env.obs_dim), device=DEVICE)
    actions_buf  = torch.zeros((N_STEPS, n, env.act_dim), device=DEVICE)
    logprobs_buf = torch.zeros((N_STEPS, n), device=DEVICE)
    rewards_buf  = torch.zeros((N_STEPS, n), device=DEVICE)
    dones_buf    = torch.zeros((N_STEPS, n), device=DEVICE)
    values_buf   = torch.zeros((N_STEPS, n), device=DEVICE)
    priv_buf     = torch.zeros((N_STEPS, n, priv_dim), device=DEVICE) if priv_dim else None
    mask_buf     = torch.ones((N_STEPS, n, env.act_dim), device=DEVICE) if gated else None

    next_obs   = env.reset()                     # (deals the swing budgets with the settings above)
    next_priv  = env.privileged() if priv_dim else None
    next_done  = torch.zeros(n, device=DEVICE)
    lstm_state = agent.initial_state(n, DEVICE)
    global_step = update * N_STEPS * n
    streak = 0
    held = 0                                        # updates since the curriculum last moved
    done_here = 0                                   # updates this process has run
    t_start = t_last = time.time()
    say(f"[INFO] rules={rules}  stage={stage}  envs={n}  device={DEVICE}  "
        f"obs={env.obs_dim}  act={env.act_dim}  hidden={HIDDEN}  physics={env.physics_version}")
    say(f"[INFO] gamma={gamma}  lam={lam}  epochs={epochs}  envs/minibatch={envs_per_mb}  target_kl={target_kl}"
        + (f"  gate={'on (' + env.gate_rule + ')' if gated else 'off'}  p_range={p_range}  priv_critic={bool(priv_dim)}"
           f"  critic_warmup={warmup}" if saw else "")
        + (f"  traves_share={traves_share}  weapon_shares={weapon_shares}  heading_mode={heading_mode}  reach_ft={reach_ft}" if pinned else "")
        + (f"  swings_3min={swings_3min}" if swings_3min is not None else "")
        + (f"  gate_persist={gate_persist}" if gate_persist else "")
        + (f"  foul={foul}" if foul is not None else ""))

    while args.updates is None or update < args.updates:
        update += 1
        done_here += 1
        warming = done_here <= warmup

        # Self-play: bring in a frozen snapshot once the curriculum is far enough along
        if selfplay is not None and p >= SELFPLAY_MIN_P and (env.selfplay is None or update % SNAPSHOT_REFRESH == 0):
            pool = sorted(glob.glob(f"{prefix}_upd*.pt"))
            if pool:
                selfplay.load(random.choice(pool))
                env.selfplay, env.selfplay_frac = selfplay, SELFPLAY_FRAC
                say(f"[SELFPLAY] snapshot {selfplay.path} on {SELFPLAY_FRAC:.0%} of envs")

        lr = opt("lr") if opt("lr") and not pretrain else learning_rate(rules, pretrain, p)
        if cold_start and not warming:
            lr, cold_start = lr * COLD_START_LR, False
        optimizer.param_groups[0]["lr"] = lr
        initial_state = (lstm_state[0].clone(), lstm_state[1].clone())
        stats = RolloutStats(env.tally_names)
        earned = torch.zeros(n, device=DEVICE)      # reward from the environment this rollout, per env

        # ── Rollout ───────────────────────────────────────────────────────
        for step in range(N_STEPS):
            obs_buf[step], dones_buf[step] = next_obs, next_done
            mask = None
            if gated:
                # A request made while the gate is shut does nothing: it is left out of the policy gradient
                mask = mask_buf[step]
                mask[:, 2] = (next_obs[:, 32] > 0.5).float()
            if priv_dim:
                priv_buf[step] = next_priv
            with torch.no_grad():
                action, logprob, _, value, lstm_state = agent.get_action_and_value(
                    next_obs, lstm_state, next_done, priv=next_priv, dim_mask=mask)
            actions_buf[step], logprobs_buf[step], values_buf[step] = action, logprob, value.flatten()

            next_obs, reward, done, info = env.step(action)
            earned += reward
            reward = timeout_bootstrap(env, agent, info, reward, lstm_state, gamma)
            next_priv = env.privileged() if priv_dim else None
            rewards_buf[step] = reward
            next_done = done.float()
            stats.add(done, info)
        global_step += N_STEPS * n
        t_rolled = time.time()

        with torch.no_grad():
            next_value = agent.get_value(next_obs, lstm_state, next_done, priv=next_priv).flatten()
        ppo = ppo_update(agent, optimizer,
                         (obs_buf, actions_buf, logprobs_buf, rewards_buf, dones_buf, values_buf),
                         initial_state, next_value, next_done, n,
                         gamma=gamma, lam=lam, epochs=epochs, envs_per_mb=envs_per_mb, target_kl=target_kl,
                         critic_only=warming, priv=priv_buf, mask=mask_buf)
        kl = ppo["kl"]

        # ── Progress ──────────────────────────────────────────────────────
        s = stats.summary(tracked)
        finished = False
        if pretrain:
            streak = streak + 1 if s["win"] >= PRETRAIN_TARGET_HIT else 0
            finished = (streak >= PRETRAIN_PATIENCE and update >= PRETRAIN_MIN_UPDATES) or update >= PRETRAIN_MAX_UPDATES
        elif not p_range:
            need, type_floor, must_lead = gate(rules, p)
            # What the gate reads: fights won, or under the saw rules fights with an attack in
            record = s["attack"] if saw else s["win"]
            by_type = s["attack_per_type"] if saw else {t: wr for t, (wr, _) in s["per_type"].items()}
            blocking = [t for t in tracked
                        if t in s["per_type"] and s["per_type"][t][1] >= MIN_SAMPLES
                        and by_type[t] < type_floor]
            why = advance(rules, p, record, s["loss"], bool(blocking), held)
            if why:
                p = min(1.0, p + (SAW_P_STEP if saw else P_STEP) * (2 if record >= FAST_WIN else 1))
                env.p = p
                note = f" (not earned: {held} updates at {record:.0%})" if why == "waited" else ""
                held = 0
                say(f"  >>> CURRICULUM UP: p={p:.2f}{note} <<<")
            else:
                held += 1
                if blocking and record >= need and (s["win"] > s["loss"] or not must_lead) and p < 1.0:
                    say(f"  >>> held at p={p:.2f} by {', '.join(TYPE_NAMES[t] for t in blocking)} "
                        f"(< {type_floor:.0%}) <<<")

        now = time.time()
        sps = int(N_STEPS * n / max(now - t_last, 1e-6))
        roll_s, ppo_s = t_rolled - t_last, now - t_rolled
        t_last = now
        stds = torch.exp(agent.actor_logstd).flatten().tolist()
        std = sum(stds) / len(stds)
        hit = f" hit={s['attack']:.3f}" if saw else ""      # share of fights with an attack in
        say(f"[{global_step:>13,}] upd={update:>4} sps={sps:>7} win={s['win']:.3f} loss={s['loss']:.3f} "
            f"{draw_label}={s['draw']:.3f}{hit} p={p:.2f} lr={lr:.2e} std={std:.3f} kl={kl:.4f} "
            f"ret={earned.mean().item():+.3f}")
        say("  types: " + "  ".join(f"{TYPE_NAMES[t][:5]}:{wr:.0%}" for t, (wr, _) in sorted(s["per_type"].items())))
        for line in stats.reward_lines(draw_label) + [stats.tally_line()] + stats.length_lines():
            if line:
                say(line)
        say(f"  ppo: epochs={ppo['epochs_run']:.2f}/{epochs} kl={kl:.4f} ev={ppo['ev']:.3f} vloss={ppo['v_loss']:.4f} "
            f"clipfrac={ppo['clipfrac']:.3f} std=[{', '.join(f'{x:.3f}' for x in stds)}] "
            f"rollout={roll_s:.1f}s update={ppo_s:.1f}s" + (" crit=warmup" if warming else ""))

        # ── Checkpoints ───────────────────────────────────────────────────
        meta = dict(update=update, p=p, stage=stage, rules=rules, physics=env.physics_version,
                    arm_gate=bool(gated), gate_rule=env.gate_rule if saw else None, gamma=gamma, lam=lam,
                    p_range=list(p_range) if p_range else None, priv_dim=priv_dim,
                    swings_3min=swings_3min, foul=foul, heading_mode=heading_mode, reach_ft=reach_ft, gate_persist=gate_persist, traves_share=traves_share, weapon_shares=list(weapon_shares) if weapon_shares else None,
                    init_from=init_from, padded_from=padded_from)
        if pretrain:
            if update % save_every == 0 or finished:
                save_checkpoint(pretrain_path, agent, optimizer, **meta)
                say(f"[SAVE] {pretrain_path}")
        else:
            save_checkpoint(f"{prefix}_latest.pt", agent, optimizer, **meta)
            if update % save_every == 0:
                save_checkpoint(f"{prefix}_upd{update}.pt", agent, **meta)
                say(f"[SAVE] {prefix}_upd{update}.pt")
        if finished:
            name = opt("name")
            say(f"[DONE] pretrain finished after {update} updates "
                f"({(time.time() - t_start) / 60:.1f} min). Now run: uv run python train_ppo.py"
                + ("" if rules == "classic" else f" --rules {rules}") + (f" --name {name}" if name else ""))
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", choices=RULES,
                    help="ruleset (default: the one the starting checkpoint was trained under, else classic)")
    ap.add_argument("--stage", choices=("pretrain", "combat"), default="combat")
    ap.add_argument("--resume", help="checkpoint to continue from (keeps its update count, its settings, and its p "
                                     "unless --p is given; writes on into its own folder unless --name is given)")
    ap.add_argument("--init", help="combat only: weights to start from instead of the pretrain checkpoint")
    ap.add_argument("--init-pad", help="combat only: start from a checkpoint with fewer inputs (a saw-v7 policy); "
                                       "the new inputs start with zero weights, so it drives as it did")
    ap.add_argument("--init-std", type=float, help="action std to start from after --init or --init-pad")
    ap.add_argument("--n-envs", type=int, default=N_BATCH, help="parallel envs (shrink for a small machine)")
    ap.add_argument("--updates", type=int, help="stop after this many updates (default: run until stopped)")
    ap.add_argument("--p", type=float, help="combat: curriculum level to start at (default 0, or the checkpoint's on --resume)")
    ap.add_argument("--p-range", type=float, nargs=2, metavar=("LO", "HI"),
                    help="saw: every fight draws its opponent's level from this range; the curriculum is off")
    ap.add_argument("--name", help="checkpoint name prefix inside ppo_models/, e.g. saw_v8a_run1/SAW "
                                   "(default: ZACAI for classic, SAW for saw)")
    ap.add_argument("--epochs", type=int, help=f"PPO epochs per update (default {N_EPOCHS})")
    ap.add_argument("--minibatch-envs", type=int, help=f"envs per minibatch (default {MINIBATCH_SIZE // N_STEPS})")
    ap.add_argument("--target-kl", type=float, help=f"stop an update once the policy has moved this far (default {TARGET_KL})")
    ap.add_argument("--lr", type=float, help="combat: a flat learning rate instead of the one that falls with p")
    ap.add_argument("--gamma", type=float, help="discount (default: the ruleset's)")
    ap.add_argument("--lam", type=float, help=f"GAE lambda (default {GAE_LAMBDA} classic, {GAE_LAMBDA_SAW} saw)")
    ap.add_argument("--no-gate", action="store_true", default=None, help="saw: the policy drives the arm itself")
    ap.add_argument("--gate-rule", choices=("heading", "body"),
                    help="saw: what the gate asks for when the camera gives no heading (saw_gate.ArmGate.open)")
    ap.add_argument("--gate-persist", type=int,
                    help="saw: control steps in a row a landing must look legal before the gate opens (default 1)")
    ap.add_argument("--reach-ft", type=float,
                    help="saw: pin the hammer reach past our front edge (default: drawn per fight)")
    ap.add_argument("--heading-mode", choices=("none", "axis", "axis_nose"),
                    help="saw: what the camera says of the opponent's heading in every fight (default: config HDG_SENSE_P)")
    ap.add_argument("--traves-share", type=float,
                    help="saw: share of the horizontal-spinner fights that are Traves-like (a specialist's mix)")
    ap.add_argument("--weapon-shares", type=float, nargs=3, metavar=("VERT", "DRUM", "HORIZ"),
                    help="saw: opponent weapon mix in place of the field's")
    ap.add_argument("--swings-3min", type=float,
                    help="saw: a fixed swing budget per three minutes (shorter fights get their share)")
    ap.add_argument("--foul", type=float,
                    help="saw: cost of a swing that lands on the floor or on their weapon side (default: config R_FOUL)")
    ap.add_argument("--critic-warmup", type=int, help="first N updates of this process train the critic only")
    ap.add_argument("--priv-critic", action="store_true", default=None,
                    help="saw: a critic that also sees the true state (training only)")
    ap.add_argument("--save-every", type=int, help=f"updates between numbered checkpoints (default {SAVE_EVERY})")
    ap.add_argument("--seed", type=int)
    train(ap.parse_args())
