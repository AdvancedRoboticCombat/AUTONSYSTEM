"""
Self-play opponent: a frozen snapshot of our own policy drives the enemy
robot in the slots the opponent mixture marks SELFPLAY.

The snapshot gets the same deal we do. It sees the fight through its own
Observer (latency, noise, no knowledge of our heading), outputs the same
command, and its robot is moved by integrate_robot.

Classic rules only: the saw-v8 environment refuses a self-play opponent
(its opponents carry weapons and bodies the policy's own robot does not).
"""
import torch

from agent import Agent, load_checkpoint
from config import DEVICE, OBS_DIM, ACT_DIM


class SelfPlayController:
    def __init__(self, n, obs_dim=OBS_DIM, act_dim=ACT_DIM):
        self.n = n
        self.agent = Agent(obs_dim, act_dim).to(DEVICE).eval()
        for prm in self.agent.parameters():
            prm.requires_grad_(False)
        self.state = self.agent.initial_state(n, DEVICE)
        self.prev_action = torch.zeros(n, act_dim, device=DEVICE)
        self.fresh = torch.ones(n, device=DEVICE)     # 1: this slot starts a new episode
        self.path = None

    def load(self, path):
        """Swap in the weights of another snapshot."""
        load_checkpoint(path, self.agent, DEVICE)
        self.path = path
        self.fresh.fill_(1.0)

    def reset(self, idx):
        self.fresh[idx] = 1.0
        self.prev_action[idx] = 0.0

    @torch.no_grad()
    def act(self, observer, mask, extra=None):
        """One control step. ``observer`` is the enemy-side Observer, ``mask``
        the (n,) bool of SELFPLAY slots and ``extra`` any observation columns
        the ruleset adds after the Observer's. Returns (n, act_dim) actions,
        zero outside the mask. Only the masked slots go through the network."""
        obs = observer.observe(self.prev_action)
        if extra is not None:
            obs = torch.cat([obs, extra], dim=1)
        action = torch.zeros(self.n, self.agent.act_dim, device=DEVICE)
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            h, c = self.state
            act, (h_new, c_new) = self.agent.act_deterministic(
                obs[idx], (h[:, idx].contiguous(), c[:, idx].contiguous()), self.fresh[idx])
            h[:, idx], c[:, idx] = h_new, c_new
            action[idx] = act
            self.fresh[idx] = 0.0
        self.prev_action = action
        return action
