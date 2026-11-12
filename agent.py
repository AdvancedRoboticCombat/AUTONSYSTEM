"""
The policy network and its checkpoint format. Every script (trainer,
evaluator, visualizer, self-play) imports the one Agent defined here.

Architecture: obs -> 3 x Linear(HIDDEN)+Tanh -> LSTM(HIDDEN) -> actor / critic.
The observation and action sizes come from the ruleset (config.RULES): 24 and
2 for the classic rules, 44 and 4 for the saw rules (30 before saw-v8). An
agent reads the first ``obs_dim`` columns of whatever observation it is given,
so a 30-input saw-v7 policy runs on the 44-column saw-v8 observation.

A training run may add a privileged critic (``priv_dim > 0``): a second value
head that also sees true-state features the policy never does. It is used by
the trainer only; evaluation and deployment carry it and never call it.
"""
import math
import os
import time

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn.utils.rnn import PackedSequence

from config import OBS_DIM, ACT_DIM, HIDDEN, SAW_ACT_DIM, BLADE_ON_BIAS

LSTM_CHUNK = 8192      # widest batch handed to the LSTM in one call (see Agent._lstm_step)
LSTM_SEQ_CHUNK = 32768 # most row-steps (columns x steps) in one sequence call (see Agent._seq_fast)


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    # Sequences longer than one step go through the LSTM in a few calls
    # instead of one per step (see _seq_fast). DODGE_SLOW_LSTM=1 restores the
    # step loop.
    fast_seq = os.environ.get("DODGE_SLOW_LSTM", "") != "1"

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, priv_dim=0):
        super().__init__()
        self.obs_dim, self.act_dim, self.priv_dim = obs_dim, act_dim, priv_dim
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, HIDDEN)), nn.Tanh(),
            layer_init(nn.Linear(HIDDEN, HIDDEN)),  nn.Tanh(),
            layer_init(nn.Linear(HIDDEN, HIDDEN)),  nn.Tanh(),
        )
        self.lstm         = nn.LSTM(HIDDEN, HIDDEN, batch_first=False)
        self.critic       = layer_init(nn.Linear(HIDDEN, 1), std=1.0)
        self.actor_mean   = layer_init(nn.Linear(HIDDEN, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        if act_dim == SAW_ACT_DIM:                       # throttle, turn, arm, blade: start with the blade on
            nn.init.constant_(self.actor_mean.bias[3:4], BLADE_ON_BIAS)
        if priv_dim > 0:                                 # built last, so the other layers draw the same init
            self.critic_priv = nn.Sequential(
                layer_init(nn.Linear(HIDDEN + priv_dim, 256)), nn.Tanh(),
                layer_init(nn.Linear(256, 1), std=0.01),
            )

    def initial_state(self, n, device):
        return (torch.zeros(1, n, HIDDEN, device=device),
                torch.zeros(1, n, HIDDEN, device=device))

    def get_states(self, obs, lstm_state, done):
        """Run the trunk and LSTM over a (T*B, >= obs_dim) batch laid out
        time-major. ``done[t]`` marks that obs[t] starts a new episode, so the
        recurrent state is zeroed before it is consumed."""
        obs        = obs[..., :self.obs_dim]
        hidden     = self.network(obs)
        batch_size = lstm_state[0].shape[1]
        hidden     = hidden.reshape((-1, batch_size, HIDDEN))
        done       = done.reshape((-1, batch_size))
        if self.fast_seq and hidden.shape[0] > 1:
            out, lstm_state = self._seq_fast(hidden, lstm_state, done)
            return torch.flatten(out, 0, 1), lstm_state
        out = []
        for h, d in zip(hidden, done):
            keep = (1.0 - d.float()).view(1, -1, 1)
            h, lstm_state = self._lstm_step(h, (keep * lstm_state[0], keep * lstm_state[1]))
            out.append(h)
        return torch.flatten(torch.cat(out), 0, 1), lstm_state

    def _lstm_step(self, x, state):
        """One LSTM step over a (B, HIDDEN) batch; returns (1, B, HIDDEN) and
        the new state. cuDNN's workspace grows with the batch (about 55 KB a
        row on the GPU: 7 GB for a 131,072-fight rollout step, which the
        allocator then keeps), so a batch wider than LSTM_CHUNK goes through
        in pieces. PPO minibatches are narrower and take the single call."""
        if x.shape[0] <= LSTM_CHUNK:
            return self.lstm(x.unsqueeze(0), state)
        outs, hs, cs = [], [], []
        for i in range(0, x.shape[0], LSTM_CHUNK):
            j = i + LSTM_CHUNK
            o, (h, c) = self.lstm(x[i:j].unsqueeze(0), (state[0][:, i:j].contiguous(), state[1][:, i:j].contiguous()))
            outs.append(o)
            hs.append(h)
            cs.append(c)
        return torch.cat(outs, 1), (torch.cat(hs, 1), torch.cat(cs, 1))

    def _seq_fast(self, hidden, lstm_state, done):
        """The step loop of get_states in a few LSTM calls. hidden is
        (T, B, HIDDEN), done (T, B). Returns ((T, B, HIDDEN), final state),
        what the loop returns before flattening.

        Columns with no fight end after their first step ("clean", most PPO
        rows) run as whole sequences. The others ("dirty") are cut at each
        fight end into segments that start from a zero state and go through
        packed calls (_seq_segments). cuDNN's workspace grows with the rows in
        a call, so each call takes at most LSTM_SEQ_CHUNK row-steps (512
        columns at T = 64). Measured on the RTX 5070 Ti, one forward and
        backward of get_action_and_value at T = 64: 24 ms against the loop's
        34 ms at 512 columns, 46 against 53 at 1,024, for about 0.3 GB more
        peak memory than the loop (one call's workspace, reused call after
        call). No loop over rows or time steps."""
        T, B = done.shape
        keep = (1.0 - done[0].float()).view(1, -1, 1)
        h0, c0 = keep * lstm_state[0], keep * lstm_state[1]
        ends = done[1:].bool()
        dirty = ends.any(0)
        if T * B <= LSTM_SEQ_CHUNK and not bool(dirty.any()):
            return self.lstm(hidden, (h0, c0))
        width = max(1, LSTM_SEQ_CHUNK // T)
        out = hidden.new_empty(T, B, HIDDEN)
        h_n, c_n = h0.new_empty(1, B, HIDDEN), c0.new_empty(1, B, HIDDEN)
        for cols, has_ends in (((~dirty).nonzero().squeeze(1), False), (dirty.nonzero().squeeze(1), True)):
            for i in range(0, cols.numel(), width):
                j = cols[i:i + width]
                state = (h0[:, j], c0[:, j])
                if has_ends:
                    o, (h, c) = self._seq_segments(hidden[:, j], state, ends[:, j])
                else:
                    o, (h, c) = self.lstm(hidden[:, j], state)
                out[:, j] = o
                h_n[:, j], c_n[:, j] = h, c
        return out, (h_n, c_n)

    def _seq_segments(self, hidden, state, ends):
        """Columns with fight ends: ends[t - 1] marks that step t starts a new
        fight from a zero state. Every segment of every column goes through
        one packed LSTM call. The packed layout (steps outer, segments by
        falling length inner) holds each element exactly once, so packing is
        one gather and unpacking another. pack_padded_sequence and
        pad_packed_sequence would go through a padded copy whose backward
        costs a full-size copy per time step: on the GPU that made this path
        slower than the loop."""
        T, n = hidden.shape[:2]
        dev, N = hidden.device, n * T
        ar = torch.arange(N, device=dev)
        starts = torch.cat([torch.ones(1, n, dtype=torch.bool, device=dev), ends], 0)
        starts = starts.t().reshape(-1)                                  # column-major: a segment is a contiguous run
        seg = starts.long().cumsum(0) - 1                                # segment of each (column, step)
        first = starts.nonzero().squeeze(1)                              # where each segment starts
        n_seg = first.numel()
        lengths = torch.diff(first, append=first.new_tensor([N]))
        pos = ar - first[seg]                                            # step within its segment
        order = torch.argsort(lengths, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(n_seg, device=dev)
        batch = (lengths.unsqueeze(0) > torch.arange(T, device=dev).unsqueeze(1)).sum(1)   # segments alive at each step
        where = (batch.cumsum(0) - batch)[pos] + rank[seg]              # packed index of each element
        src = torch.empty_like(where)
        src[where] = ar
        batch = batch.cpu()
        x = hidden.transpose(0, 1).reshape(N, HIDDEN)
        packed = PackedSequence(x[src], batch[batch > 0], order, rank)

        col_first = seg[torch.arange(n, device=dev) * T]                 # each column's first segment ...
        col_last = seg[torch.arange(n, device=dev) * T + T - 1]          # ... and its last
        at = (torch.zeros_like(col_first), col_first)
        hs = state[0].new_zeros(1, n_seg, HIDDEN).index_put(at, state[0][0])
        cs = state[1].new_zeros(1, n_seg, HIDDEN).index_put(at, state[1][0])
        o, (h, c) = self.lstm(packed, (hs, cs))
        return o.data[where].reshape(n, T, HIDDEN).transpose(0, 1), (h[:, col_last], c[:, col_last])

    def value(self, hidden, priv=None):
        """The critic's value of LSTM features; an agent with a privileged
        critic needs the true-state features ``priv`` as well."""
        if self.priv_dim > 0:
            if priv is None:
                raise ValueError("this agent has a privileged critic: pass priv")
            return self.critic_priv(torch.cat([hidden, priv], 1))
        return self.critic(hidden)

    def get_value(self, obs, lstm_state, done, priv=None):
        hidden, _ = self.get_states(obs, lstm_state, done)
        return self.value(hidden, priv)

    def get_action_and_value(self, obs, lstm_state, done, action=None, priv=None, dim_mask=None):
        """Sample (or score) an action. ``dim_mask`` ((N, act_dim), 0 or 1)
        drops action dimensions from the log-prob and the entropy: the trainer
        masks the arm on steps where the gate was closed, because the arm
        request then had no effect."""
        hidden, lstm_state = self.get_states(obs, lstm_state, done)
        mean  = self.actor_mean(hidden)
        probs = Normal(mean, torch.exp(self.actor_logstd.expand_as(mean)))
        if action is None:
            action = probs.sample()
        logprob, entropy = probs.log_prob(action), probs.entropy()
        if dim_mask is not None:
            logprob, entropy = logprob * dim_mask, entropy * dim_mask
        return action, logprob.sum(1), entropy.sum(1), self.value(hidden, priv), lstm_state

    @torch.no_grad()
    def act_deterministic(self, obs, lstm_state, done):
        """Actor mean, clamped to the command range. Used for evaluation,
        the visualizer, self-play opponents and deployment. One row per robot.

        A row whose observation holds a NaN or an infinity (a bad sensor
        frame) gets an all-zero command and keeps the memory it had, so one
        bad frame cannot poison the recurrent state for the rest of the fight.
        Only the columns this agent reads are checked.
        """
        obs = obs[..., :self.obs_dim]
        ok = torch.isfinite(obs).all(dim=1)
        obs = torch.where(ok.unsqueeze(1), obs, torch.zeros_like(obs))
        hidden, new_state = self.get_states(obs, lstm_state, done)
        action = self.actor_mean(hidden).clamp(-1.0, 1.0)
        keep = ok.view(1, -1, 1)
        new_state = (torch.where(keep, new_state[0], lstm_state[0]),
                     torch.where(keep, new_state[1], lstm_state[1]))
        return torch.where(ok.unsqueeze(1), action, torch.zeros_like(action)), new_state


# ─── Checkpoints ──────────────────────────────────────────────────────────

def save_checkpoint(path, agent, optimizer=None, **meta):
    """Weights plus whatever the trainer needs to resume (update, p, stage,
    rules). Pass the optimizer for a checkpoint that training will continue
    from.

    The file is written under a temporary name and then swapped into place,
    so a kill or a crash mid-save leaves the previous checkpoint intact."""
    ckpt = {"model": agent.state_dict(),
            "meta": {"obs_dim": agent.obs_dim, "act_dim": agent.act_dim, **meta}}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    path = str(path)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    for _ in range(40):                      # another process may be reading the old file (Windows)
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    print(f"[WARN] could not replace {path} (held open by another program); "
          f"the new checkpoint is in {tmp}", flush=True)


def _read(path, device):
    """(weights, meta, raw checkpoint) of a checkpoint file."""
    ckpt = torch.load(path, map_location=device)
    state, meta = ckpt, {}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state, meta = ckpt["model"], dict(ckpt.get("meta", {}))
    meta["obs_dim"] = state["network.0.weight"].shape[1]
    meta["act_dim"] = state["actor_mean.weight"].shape[0]
    meta["priv_dim"] = state["critic_priv.0.weight"].shape[1] - HIDDEN if "critic_priv.0.weight" in state else 0
    meta.setdefault("rules", "classic")      # checkpoints from before there was a choice
    return state, meta, ckpt


def read_meta(path):
    """A checkpoint's meta dict without building an agent: rules, obs_dim,
    act_dim, priv_dim, and whatever the trainer stored (update, p, stage, ...)."""
    return _read(path, "cpu")[1]


def load_checkpoint(path, agent, device, optimizer=None):
    """Load weights into ``agent`` and return the checkpoint's meta dict
    (``meta["optimizer"]`` says whether the optimizer state was restored too).
    Raises ValueError for a checkpoint whose observation, action or
    privileged-critic layout is not the agent's: one trained under the other
    ruleset, from before the 24-dim sensor-only observation, or with a
    narrower observation (that one starts a run through --init-pad)."""
    state, meta, ckpt = _read(path, device)
    have = (meta["obs_dim"], meta["act_dim"], meta["priv_dim"])
    want = (agent.obs_dim, agent.act_dim, agent.priv_dim)
    if have != want:
        hint = "Retrain from the pretrain stage."
        if meta["act_dim"] == agent.act_dim and meta["obs_dim"] < agent.obs_dim:
            hint = "Start from it with --init-pad, which widens its first layer."
        elif have[:2] == want[:2]:
            hint = "The privileged critic differs (priv_dim)."
        raise ValueError(f"{path} was trained on a {meta['obs_dim']}-dim observation and "
                         f"{meta['act_dim']} actions with priv_dim {meta['priv_dim']} ({meta['rules']} rules); "
                         f"this agent uses {agent.obs_dim}, {agent.act_dim} and {agent.priv_dim}. {hint}")
    meta["optimizer"] = optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt
    if meta["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    agent.load_state_dict(state)
    return meta


def load_agent(path, device):
    """Build the Agent a checkpoint was saved from, load it and switch it to
    evaluation mode. Returns (agent, meta); ``meta["rules"]`` names the
    ruleset to build the environment with."""
    state, meta, _ = _read(path, device)
    agent = Agent(meta["obs_dim"], meta["act_dim"], meta["priv_dim"]).to(device)
    agent.load_state_dict(state)
    agent.eval()
    meta["optimizer"] = False
    return agent, meta


def load_padded(path, agent, device):
    """Load a checkpoint with a narrower observation into ``agent``: the first
    layer's weights for the new columns are zero, so the agent acts and values
    exactly as the old one did until training moves them. A privileged critic
    the checkpoint lacks keeps its fresh init. Returns the checkpoint's meta
    with ``padded_from`` = its obs_dim."""
    state, meta, _ = _read(path, device)
    old = meta["obs_dim"]
    if meta["act_dim"] != agent.act_dim or old > agent.obs_dim:
        raise ValueError(f"{path} has a {old}-dim observation and {meta['act_dim']} actions; "
                         f"padding needs {agent.act_dim} actions and at most {agent.obs_dim} inputs")
    own = agent.state_dict()
    extra = sorted(set(state) - set(own))
    missing = sorted(k for k in set(own) - set(state) if not k.startswith("critic_priv."))
    if extra or missing:
        raise ValueError(f"{path} does not fit this agent: keys it has that the agent lacks {extra}, "
                         f"keys it lacks {missing}")
    new = dict(own)
    for k, v in state.items():
        if k == "network.0.weight":
            v = torch.cat([v, v.new_zeros(v.shape[0], agent.obs_dim - old)], 1)
        if v.shape != own[k].shape:
            raise ValueError(f"{path}: {k} has shape {tuple(v.shape)}, the agent's is {tuple(own[k].shape)}")
        new[k] = v
    agent.load_state_dict(new)
    meta["padded_from"] = old
    return meta


def gate_from_meta(meta):
    """Whether the checkpoint's policy was trained with the arm gate on
    (saw rules only; off for checkpoints from before the gate existed)."""
    return meta.get("rules") == "saw" and bool(meta.get("arm_gate", False))
