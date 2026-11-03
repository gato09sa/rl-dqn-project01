"""Deep Q-Network (Mnih et al., 2015) and the variants used in the project, in PyTorch.

Components:
    QNetwork      MLP state -> one Q-value per action; optional dueling head
                  (Wang et al., 2016)
    ReplayBuffer  fixed-size FIFO of transitions, sampled uniformly
    DQNAgent      epsilon-greedy acting + TD update toward a bootstrap target

Flags (all in the config dict):
    use_target  True : target computed with a frozen copy, synced every C steps
                False: target computed with the online network itself (ablation)
    use_replay  True : uniform mini-batches from a large buffer
                False: buffer holds only the last `batch_size` transitions and every
                       update uses exactly those, i.e. consecutive, correlated data
                       (ablation; same number of gradient steps as vanilla DQN)
    double      Double DQN target: online net picks a', target net evaluates it
                (van Hasselt et al., 2016)
    dueling     Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)

Interface shared with TabularAgent:
    act(obs, epsilon), greedy(obs), update(), save(path), load(path)
plus maybe_sync_target(step) and buffer.add(...).

Parallel runs: call configure_torch_for_parallel() once at the start of each
worker process. Each process then uses one core, so 7-8 seeds/configs run side by
side on an 8-core laptop instead of fighting over threads. Everything stays on CPU:
with a small MLP and batches of 64-128, copying to a GPU every step costs more than
it saves.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CONFIG = {
    "hidden": [128, 128],     # hidden layer sizes
    "lr": 5e-4,               # Adam learning rate
    "gamma": 0.99,            # discount factor
    "buffer_size": 50_000,    # replay capacity (transitions)
    "batch_size": 64,
    "target_update": 1_000,   # env steps between hard target syncs (C)
    "grad_clip": 10.0,        # max global grad norm
    "use_target": True,
    "use_replay": True,
    "double": False,
    "dueling": False,
}


def configure_torch_for_parallel(seed=None):
    """One intra-op thread per process (call once per worker process)."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass   # PyTorch only allows this once per process; already set
    if seed is not None:
        torch.manual_seed(seed)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(128, 128), dueling=False):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.value = nn.Linear(d, 1)               # V(s)
            self.advantage = nn.Linear(d, n_actions)   # A(s, a)
        else:
            self.head = nn.Linear(d, n_actions)        # Q(s, a) directly

    def forward(self, x):
        z = self.trunk(x)
        if not self.dueling:
            return self.head(z)
        a = self.advantage(z)
        # Subtracting the mean advantage makes V and A identifiable
        # (otherwise any constant could move between them).
        return self.value(z) + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, rng, use_replay=True):
        self.capacity, self.rng, self.use_replay = capacity, rng, use_replay
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, obs, action, reward, next_obs, terminated):
        """Store `terminated`, not `terminated or truncated`: a time-limit cut is not a
        real terminal state, so its target must still bootstrap from next_obs."""
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity       # overwrite the oldest (FIFO)
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        if self.use_replay:
            idx = self.rng.integers(0, self.size, size=batch_size)   # uniform over the buffer
        else:
            idx = np.arange(self.size)                                # the most recent window only
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.actions[idx]),
                torch.from_numpy(self.rewards[idx]), torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.dones[idx]))


class DQNAgent:
    def __init__(self, obs_dim, n_actions, config=None, seed=None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg
        self.n_actions = n_actions
        self.gamma = cfg["gamma"]
        self.batch_size = cfg["batch_size"]
        self.use_target, self.double = cfg["use_target"], cfg["double"]
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)   # network initialization

        self.q = QNetwork(obs_dim, n_actions, cfg["hidden"], cfg["dueling"])
        # Without a target network the "target" is the online network itself.
        self.q_target = copy.deepcopy(self.q) if self.use_target else self.q
        if self.use_target:
            self.q_target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])

        capacity = cfg["buffer_size"] if cfg["use_replay"] else cfg["batch_size"]
        self.buffer = ReplayBuffer(capacity, obs_dim, self.rng, cfg["use_replay"])
        self.n_updates = 0

    # ---------------------------------------------------------------- acting
    @torch.no_grad()
    def greedy(self, obs):
        q = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def act(self, obs, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(obs)

    # -------------------------------------------------------------- learning
    def can_update(self):
        return len(self.buffer) >= self.batch_size

    def update(self):
        """One gradient step on a mini-batch. Returns (loss, mean Q(s,a)) for logging;
        the mean Q is useful to show overestimation/divergence in the ablation."""
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        q_sa = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)       # Q(s, a; theta)
        with torch.no_grad():
            if self.double:
                a_star = self.q(next_obs).argmax(dim=1, keepdim=True)        # online net selects
                q_next = self.q_target(next_obs).gather(1, a_star).squeeze(1)  # target net evaluates
            else:
                q_next = self.q_target(next_obs).max(dim=1).values           # max_a' Q(s', a'; theta-)
            target = rewards + self.gamma * (1.0 - dones) * q_next

        loss = F.smooth_l1_loss(q_sa, target)   # Huber loss: less sensitive to large TD errors
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg["grad_clip"])
        self.optimizer.step()
        self.n_updates += 1
        return float(loss.item()), float(q_sa.mean().item())

    def maybe_sync_target(self, step):
        """Hard copy theta -> theta- every `target_update` env steps (no-op without target)."""
        if self.use_target and step % self.cfg["target_update"] == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    # ------------------------------------------------------------------- I/O
    def save(self, path):
        torch.save({"state_dict": self.q.state_dict(), "config": self.cfg}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.q.load_state_dict(ckpt["state_dict"])
        if self.use_target:
            self.q_target.load_state_dict(ckpt["state_dict"])
