"""Phase 7: TD3 and SAC agents (PyTorch), hyper-parameters from the ch6 table."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AgentConfig:
    actor_hidden: tuple = (1024, 512)     # ch6
    critic_hidden: tuple = (512, 256)     # ch6
    actor_lr: float = 1e-4                # ch6
    critic_lr: float = 1e-3               # ch6
    gamma: float = 0.99                   # ch6
    tau: float = 0.005                    # ch6 lists 0.0005; 0.005 is the TD3/SAC default (switchable)
    buffer_size: int = 50_000             # ch6
    batch_size: int = 128
    warmup: int = 500                     # random actions before learning
    # TD3
    expl_noise: float = 0.1
    target_noise: float = 0.2             # ch6: N(0, 0.2) on target actions
    noise_clip: float = 0.5
    policy_delay: int = 2
    # SAC
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2


def mlp(inp, out, hidden, out_act=None):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class Replay:
    """FIFO ring buffer."""

    def __init__(self, obs_dim, act_dim, size, extra_dim=0):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.x = np.zeros((size, extra_dim), np.float32)       # e.g. the other agent's action (CTDE)
        self.r = np.zeros(size, np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.x2 = np.zeros((size, extra_dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.size, self.ptr, self.n = size, 0, 0

    def add(self, o, a, r, o2, d, x=None, x2=None):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        if x is not None:
            self.x[i], self.x2[i] = x, x2
        self.ptr = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, k, rng):
        idx = rng.integers(0, self.n, k)
        t = lambda z: torch.as_tensor(z[idx])
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d), t(self.x), t(self.x2)


def checkpoint_state(agent):
    """Everything needed to continue training except the replay buffer (kept off disk for size)."""
    st = {}
    for k, v in vars(agent).items():
        if isinstance(v, (nn.Module, torch.optim.Optimizer)):
            st[k] = v.state_dict()
        elif isinstance(v, torch.Tensor):
            st[k] = v.detach().clone()
        elif isinstance(v, np.random.Generator):
            st[k] = v.bit_generator.state
    st["steps"] = agent.steps
    return st


def load_checkpoint_state(agent, st):
    """Restore checkpoint_state(); the buffer refills from the restored policy (no random warmup)."""
    for k, v in st.items():
        cur = getattr(agent, k)
        if isinstance(cur, (nn.Module, torch.optim.Optimizer)):
            cur.load_state_dict(v)
        elif isinstance(cur, torch.Tensor):
            with torch.no_grad():
                cur.copy_(v)
        elif isinstance(cur, np.random.Generator):
            cur.bit_generator.state = v
    agent.steps = st["steps"]
    agent.resumed = True


def soft_update(target, source, tau):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(tau * p)


class TD3:
    """Twin critics, delayed actor, target policy smoothing (Fujimoto et al. 2018).

    extra_dim > 0 makes the critic centralised: Q(o, a, x) with x the other agent's action (MATD3).
    """

    def __init__(self, obs_dim, act_dim, cfg=AgentConfig(), extra_dim=0, seed=0):
        torch.manual_seed(seed)
        self.cfg, self.act_dim, self.extra_dim = cfg, act_dim, extra_dim
        self.rng = np.random.default_rng(seed)
        self.actor = mlp(obs_dim, act_dim, cfg.actor_hidden, nn.Tanh())
        self.actor_t = mlp(obs_dim, act_dim, cfg.actor_hidden, nn.Tanh())
        qin = obs_dim + act_dim + extra_dim
        self.q1, self.q2 = mlp(qin, 1, cfg.critic_hidden), mlp(qin, 1, cfg.critic_hidden)
        self.q1_t, self.q2_t = mlp(qin, 1, cfg.critic_hidden), mlp(qin, 1, cfg.critic_hidden)
        for t, s in ((self.actor_t, self.actor), (self.q1_t, self.q1), (self.q2_t, self.q2)):
            t.load_state_dict(s.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.critic_lr)
        self.buf = Replay(obs_dim, act_dim, cfg.buffer_size, extra_dim)
        self.steps = 0
        self.resumed = False

    def act(self, o, explore=True):
        if explore and not self.resumed and self.buf.n < self.cfg.warmup:
            return self.rng.uniform(-1, 1, self.act_dim).astype(np.float32)
        with torch.no_grad():
            a = self.actor(torch.as_tensor(o, dtype=torch.float32)[None])[0].numpy()
        if explore:
            a = np.clip(a + self.rng.normal(0, self.cfg.expl_noise, self.act_dim), -1, 1)
        return a.astype(np.float32)

    def _q_in(self, o, a, x):
        return torch.cat([o, a, x], 1) if self.extra_dim else torch.cat([o, a], 1)

    def update(self):
        c = self.cfg
        if self.buf.n < max(c.batch_size, 0 if self.resumed else c.warmup):
            return {}
        o, a, r, o2, d, x, x2 = self.buf.sample(c.batch_size, self.rng)
        with torch.no_grad():
            noise = (torch.randn_like(a) * c.target_noise).clamp(-c.noise_clip, c.noise_clip)
            a2 = (self.actor_t(o2) + noise).clamp(-1, 1)
            qi = self._q_in(o2, a2, x2)
            y = r[:, None] + c.gamma * (1 - d[:, None]) * torch.min(self.q1_t(qi), self.q2_t(qi))
        qi = self._q_in(o, a, x)
        lq = F.mse_loss(self.q1(qi), y) + F.mse_loss(self.q2(qi), y)
        self.opt_q.zero_grad()
        lq.backward()
        self.opt_q.step()
        self.steps += 1
        out = {"q_loss": float(lq)}
        if self.steps % c.policy_delay == 0:
            la = -self.q1(self._q_in(o, self.actor(o), x)).mean()
            self.opt_a.zero_grad()
            la.backward()
            self.opt_a.step()
            for t, s in ((self.actor_t, self.actor), (self.q1_t, self.q1), (self.q2_t, self.q2)):
                soft_update(t, s, c.tau)
            out["actor_loss"] = float(la)
        return out

    def state_dict(self):
        return {k: getattr(self, k).state_dict() for k in ("actor", "q1", "q2")}


class SAC:
    """Soft actor-critic with twin critics and automatic entropy tuning (Haarnoja et al. 2018)."""

    def __init__(self, obs_dim, act_dim, cfg=AgentConfig(), extra_dim=0, seed=0):
        torch.manual_seed(seed)
        self.cfg, self.act_dim, self.extra_dim = cfg, act_dim, extra_dim
        self.rng = np.random.default_rng(seed)
        self.actor = mlp(obs_dim, 2 * act_dim, cfg.actor_hidden)
        qin = obs_dim + act_dim + extra_dim
        self.q1, self.q2 = mlp(qin, 1, cfg.critic_hidden), mlp(qin, 1, cfg.critic_hidden)
        self.q1_t, self.q2_t = mlp(qin, 1, cfg.critic_hidden), mlp(qin, 1, cfg.critic_hidden)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.tensor(np.log(cfg.init_alpha), requires_grad=True)
        self.target_entropy = -float(act_dim)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.critic_lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.buf = Replay(obs_dim, act_dim, cfg.buffer_size, extra_dim)
        self.steps = 0
        self.resumed = False

    def _dist(self, o):
        mu, log_std = self.actor(o).chunk(2, -1)
        return mu, log_std.clamp(-20, 2).exp()

    def _sample(self, o):
        mu, std = self._dist(o)
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (torch.distributions.Normal(mu, std).log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        return a, logp

    def act(self, o, explore=True):
        if explore and not self.resumed and self.buf.n < self.cfg.warmup:
            return self.rng.uniform(-1, 1, self.act_dim).astype(np.float32)
        with torch.no_grad():
            o = torch.as_tensor(o, dtype=torch.float32)[None]
            a = self._sample(o)[0] if explore else torch.tanh(self._dist(o)[0])
        return a[0].numpy().astype(np.float32)

    def _q_in(self, o, a, x):
        return torch.cat([o, a, x], 1) if self.extra_dim else torch.cat([o, a], 1)

    def update(self):
        c = self.cfg
        if self.buf.n < max(c.batch_size, 0 if self.resumed else c.warmup):
            return {}
        o, a, r, o2, d, x, x2 = self.buf.sample(c.batch_size, self.rng)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self._sample(o2)
            qi = self._q_in(o2, a2, x2)
            q_t = torch.min(self.q1_t(qi), self.q2_t(qi))[:, 0] - alpha * logp2
            y = (r + c.gamma * (1 - d) * q_t)[:, None]
        qi = self._q_in(o, a, x)
        lq = F.mse_loss(self.q1(qi), y) + F.mse_loss(self.q2(qi), y)
        self.opt_q.zero_grad()
        lq.backward()
        self.opt_q.step()
        an, logp = self._sample(o)
        qi = self._q_in(o, an, x)
        la = (alpha * logp - torch.min(self.q1(qi), self.q2(qi))[:, 0]).mean()
        self.opt_a.zero_grad()
        la.backward()
        self.opt_a.step()
        l_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        l_alpha.backward()
        self.opt_alpha.step()
        soft_update(self.q1_t, self.q1, c.tau)
        soft_update(self.q2_t, self.q2, c.tau)
        self.steps += 1
        return {"q_loss": float(lq), "actor_loss": float(la), "alpha": float(alpha)}

    def state_dict(self):
        return {k: getattr(self, k).state_dict() for k in ("actor", "q1", "q2")}


AGENTS = {"td3": TD3, "sac": SAC}
