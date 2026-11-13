"""Phase 6: two-timescale environment of ch5 for the RL methods.

Time: one env step = one slot (small timescale). Every K = cfg.slots_per_large_step slots
the large-timescale decision (association alpha + CU/DU placement psi) is taken; every slot
the small-timescale decision (PRBs kappa + power omega) is taken.

Actions are continuous in [-1, 1] and are decoded so that every action is structurally
valid (eq.1, 2, 3, a5, 6 hold by construction):
  large  a_L in R^{U*B + 2*B*N}: per-user RU scores -> argmax;  per-RU CU / DU node scores -> argmax
  small  a_S in R^{2U}:           per-user PRB share and power share (softmax within each RU);
                                  PRB counts by largest remainder (>= 1 each), PRBs dealt in
                                  order of channel quality, power split equally over a user's PRBs
Reward (shared by all agents, ch5): r = -(J + penalty * #violated) / J_ref, with J_ref the best
static ch6 scenario at reset, so r ~ -1 for the static baseline and r -> -J*/J_ref at the optimum.
"""
from dataclasses import dataclass

import numpy as np

from .channel import large_scale_gain, small_scale_fading
from .config import Config
from .kpis import Decision
from .network import Network
from .reference_policies import penalised_objective, reference_decision
from .users import build_users


@dataclass
class EnvConfig:
    large_steps_per_episode: int = 10      # episode = 10 large steps = 100 slots by default
    randomize_users: bool = True           # new user drop per training episode (topology fixed)
    penalty: float = 1.0                   # per violated constraint, in units of J_ref
    reward_clip: float = 10.0
    reward_mode: str = "negative"          # "negative" (default) | "inverse" (ch5: r = J_ref / J)
    large_prior: float = 0.0               # residual prior: scores += large_prior * one-hot(best static
                                           # decision), so a zero action reproduces the static baseline
    small_temp: float = 3.0                # PRB / power shares = softmax(small_temp * a); lower = no starving
    action_penalty: float = 0.0            # reward -= action_penalty * mean(a^2) over the RL actions used


class TwoTimescaleEnv:
    def __init__(self, cfg: Config, env_cfg: EnvConfig = EnvConfig()):
        self.cfg, self.ecfg = cfg, env_cfg
        self.K = cfg.slots_per_large_step
        self.net = Network(cfg)
        t = self.net.topo
        self.B, self.N, self.U, self.M = t.B, t.N, cfg.n_ue, cfg.n_prb
        self.large_dim = self.U * self.B + 2 * self.B * self.N
        self.small_dim = 2 * self.U
        self.episode = 0
        self.reset(0)

    # ------------------------------------------------------------------ episode control
    def reset(self, episode=None):
        """Episode 0 is exactly the Phase-5 snapshot Network(cfg); later episodes redraw users."""
        self.episode = self.episode + 1 if episode is None else episode
        self.net = Network(self.cfg)
        if self.ecfg.randomize_users and self.episode > 0:
            rng = np.random.default_rng([self.cfg.seed, 7, self.episode])
            self.net.rng = rng
            self.net.users = build_users(self.cfg, rng)
            self.net.G = large_scale_gain(self.net.topo.ru_pos, self.net.users.pos, self.cfg, rng)
            self.net.H = small_scale_fading((self.B, self.U, self.M), self.cfg, rng)
        self.slot = 0
        refs = [reference_decision(self.net, s) for s in ("all_edge", "cu_regional_du_edge", "all_regional")]
        vals = [penalised_objective(self.net, d) for d in refs]
        self.J_ref = min(vals)
        best = refs[int(np.argmin(vals))]
        self.prior = np.concatenate([np.eye(self.B)[best.assoc].ravel(), np.eye(self.N)[best.cu_node].ravel(),
                                     np.eye(self.N)[best.du_node].ravel()])
        ref = reference_decision(self.net, "all_regional")
        self.assoc, self.cu, self.du = ref.assoc, ref.cu_node, ref.du_node
        self.last = self.net.evaluate(ref)
        return self.observe()

    @property
    def is_large_step(self):
        return self.slot % self.K == 0

    @property
    def done(self):
        return self.slot >= self.K * self.ecfg.large_steps_per_episode

    # ------------------------------------------------------------------ observations
    def observe(self):
        """Global observation; the small agent also sees per-PRB fading of each user at its RU."""
        net, cfg = self.net, self.cfg
        u, t = net.users, net.topo
        logG = (10 * np.log10(net.G) + 120) / 40                                    # ~[-1, 1]
        slice_oh = np.eye(len(cfg.slices))[u.slice_idx]
        assoc_oh = np.eye(self.B)[self.assoc]
        cu_oh, du_oh = np.eye(self.N)[self.cu], np.eye(self.N)[self.du]
        util = self.last["cpu_util"]
        phi = self.last["phi_node"] / (t.cost_factor.max() * cfg.cost_unit_v)
        large = np.concatenate([logG.ravel(), slice_oh.ravel(), assoc_oh.ravel(), cu_oh.ravel(),
                                du_oh.ravel(), util, phi]).astype(np.float32)
        h_serv = net.H[self.assoc, np.arange(self.U)]                               # (U, M)
        small = np.concatenate([large, (10 * np.log10(np.maximum(h_serv, 1e-6)) / 10).ravel()]).astype(np.float32)
        return {"large": large, "small": small}

    @property
    def obs_dims(self):
        o = self.observe()
        return len(o["large"]), len(o["small"])

    # ------------------------------------------------------------------ action decoding
    def decode_large(self, a):
        a = np.asarray(a, float) + self.ecfg.large_prior * self.prior
        sc = a[: self.U * self.B].reshape(self.U, self.B)
        pl = a[self.U * self.B:].reshape(2, self.B, self.N)
        return sc.argmax(axis=1), pl[0].argmax(axis=1), pl[1].argmax(axis=1)

    def decode_small(self, a, assoc):
        """PRB owner (B,M) and power (B,M) from per-user PRB/power shares."""
        a = np.asarray(a, float)
        z, y = a[: self.U], a[self.U:]
        owner = -np.ones((self.B, self.M), int)
        power = np.zeros((self.B, self.M))
        P = self.cfg.p_max_w
        for b in range(self.B):
            mine = np.flatnonzero(assoc == b)
            if not len(mine):
                continue
            n = len(mine)
            share = _softmax(self.ecfg.small_temp * z[mine])
            raw = share * self.M
            cnt = np.floor(raw).astype(int)
            if self.M >= n:
                cnt = np.maximum(cnt, 1)
            while cnt.sum() > self.M:
                cnt[np.argmax(cnt)] -= 1
            for i in np.argsort(-(raw - np.floor(raw)))[: self.M - cnt.sum()]:
                cnt[i] += 1
            gains = self.net.H[b][mine]                                             # (n, M)
            free = np.ones(self.M, bool)
            left = cnt.copy()
            while left.sum() > 0:                                                   # deal best PRBs in turn
                for k in np.argsort(-share):
                    if left[k] == 0:
                        continue
                    m = int(np.argmax(np.where(free, gains[k], -np.inf)))
                    owner[b, m] = mine[k]
                    free[m] = False
                    left[k] -= 1
            pshare = _softmax(self.ecfg.small_temp * y[mine])
            for k, uidx in enumerate(mine):
                ms = owner[b] == uidx
                if ms.any():
                    power[b, ms] = P * pshare[k] / ms.sum()
            if power[b].sum() > 0:                                                  # users without a PRB
                power[b] *= P / power[b].sum()                                      # leave no power unused
        return owner, power

    # ------------------------------------------------------------------ dynamics
    def step(self, a_small=None, a_large=None, small_decision=None):
        """Advance one slot.

        a_large is used only when is_large_step (ignored otherwise). The small-timescale decision
        is either decoded from a_small or given directly as (owner, power) by a non-RL method.
        """
        used = []                                                                   # RL actions applied now
        if self.is_large_step and a_large is not None:
            self.assoc, self.cu, self.du = self.decode_large(a_large)
            used.append(np.asarray(a_large, float))
        if small_decision is not None:
            owner, power = small_decision
        else:
            owner, power = self.decode_small(a_small, self.assoc)
            used.append(np.asarray(a_small, float))
        dec = Decision.from_compact(self.assoc, owner, power, self.cu, self.du, self.B, self.N)
        r = self.net.evaluate(dec)
        self.last = r
        n_viol = sum(not v for v in r["constraints"].values())
        J = r["objective"] if np.isfinite(r["objective"]) else self.ecfg.reward_clip * self.J_ref
        if self.ecfg.reward_mode == "inverse":
            reward = self.J_ref / (J + self.ecfg.penalty * n_viol * self.J_ref)
        else:
            reward = -min((J + self.ecfg.penalty * n_viol * self.J_ref) / self.J_ref, self.ecfg.reward_clip)
        if self.ecfg.action_penalty and used:
            reward -= self.ecfg.action_penalty * float(np.mean(np.concatenate(used) ** 2))
        info = dict(objective=float(r["objective"]), feasible=bool(r["feasible"]), n_viol=n_viol,
                    J_ref=self.J_ref, decision=dec)
        self.slot += 1
        self.net.small_step()                                                       # fading; mobility every K
        return self.observe(), float(reward), self.done, info


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
