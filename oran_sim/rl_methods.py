"""Phase 7-8: the three learning methods and their training / evaluation loop.

  marl_td3    two TD3 agents (ch5 two-timescale MATD3): large agent -> association + placement every
              K slots; small agent -> PRB + power every slot, critic centralised on the large action
  hybrid_td3  large-timescale TD3 agent + convex small timescale (PRB local search + exact KKT power)
  hybrid_sac  as hybrid_td3 with a SAC large agent
  single_sac  one SAC agent outputs [a_large, a_small] every slot (a_large applied every K slots)

Large-agent transitions are semi-MDP: reward = mean slot reward over its K slots. Episodes end by
time limit only, so transitions are stored as non-terminal (no bias from truncation).
"""
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .agents import AGENTS, AgentConfig, checkpoint_state, load_checkpoint_state
from .config import Config
from .marl_env import EnvConfig, TwoTimescaleEnv
from .optimal import RadioRU, objective_weights

METHODS = ("marl_td3", "hybrid_td3", "hybrid_sac", "single_sac")
LARGE_CFG = dict(warmup=50, batch_size=64)      # the large agent sees only ~10 transitions per episode
LARGE_UPDATES = 10                              # gradient steps per large transition
CONVEX_ITERS = 40                               # power iterations per slot in the hybrid methods


def convex_small_decision(env):
    """Small timescale by optimisation: PRB local search + exact convex power, per RU."""
    net = env.net
    w, _ = objective_weights(net)
    owner = -np.ones((env.B, env.M), int)
    power = np.zeros((env.B, env.M))
    for b in range(env.B):
        users = np.flatnonzero(env.assoc == b)
        if not len(users):
            continue
        ru = RadioRU(net, b, users, w)
        split = int(env.cu[b] != env.du[b])
        local = ru.assign_local_search(split)
        sol = ru.optimal_power(local, split, max_iter=CONVEX_ITERS, tol=1e-9, golden_iters=30)
        if sol is None:
            sol = ru._solution(local, ru.equal_power(), split)
        owner[b], power[b] = sol.owner, sol.power
    return owner, power


class Method:
    def __init__(self, name, env: TwoTimescaleEnv, agent_cfg=AgentConfig(), seed=0, large_kw=None,
                 large_updates=LARGE_UPDATES):
        if name not in METHODS:
            raise ValueError(name)
        self.name, self.env = name, env
        dl, ds = env.obs_dims
        large_cfg = replace(agent_cfg, **{**LARGE_CFG, **(large_kw or {})})
        self.large_updates = large_updates
        self.large = self.small = None
        if name == "marl_td3":
            self.large = AGENTS["td3"](dl, env.large_dim, large_cfg, seed=seed)
            self.small = AGENTS["td3"](ds, env.small_dim, agent_cfg, extra_dim=env.large_dim, seed=seed + 1)
        elif name in ("hybrid_td3", "hybrid_sac"):
            self.large = AGENTS[name.split("_")[1]](dl, env.large_dim, large_cfg, seed=seed)
        else:
            self.single = AGENTS["sac"](ds, env.large_dim + env.small_dim, agent_cfg, seed=seed)

    # ---------------------------------------------------------------- one episode
    def run_episode(self, episode, explore=True, learn=True):
        env = self.env
        o = env.reset(episode)
        L = env.large_dim
        log = dict(reward=0.0, objective=[], feasible=[], ratio=[], first_objective=None)
        pend = None                       # pending large transition (o_L, a_L, reward sum)
        a_large = None
        while not env.done:
            if env.is_large_step and self.name != "single_sac":
                if pend is not None and learn:
                    self._store_large(pend, o["large"])
                a_large = self.large.act(o["large"], explore)
                pend = [o["large"], a_large, 0.0]
            if self.name == "marl_td3":
                a_s = self.small.act(o["small"], explore)
                o2, r, done, info = env.step(a_small=a_s, a_large=a_large)
                if learn:
                    self.small.buf.add(o["small"], a_s, r, o2["small"], 0.0, x=a_large, x2=a_large)
                    self.small.update()
            elif self.name.startswith("hybrid"):
                big = env.is_large_step
                if big:                           # the convex step needs the new association first
                    env.assoc, env.cu, env.du = env.decode_large(a_large)
                o2, r, done, info = env.step(small_decision=convex_small_decision(env),
                                             a_large=a_large if big else None)
            else:
                a = self.single.act(o["small"], explore)
                o2, r, done, info = env.step(a_small=a[L:], a_large=a[:L])
                if learn:
                    self.single.buf.add(o["small"], a, r, o2["small"], 0.0)
                    self.single.update()
            if pend is not None:
                pend[2] += r
            log["reward"] += r
            log["objective"].append(info["objective"])
            log["feasible"].append(info["feasible"])
            log["ratio"].append(info["objective"] / info["J_ref"])
            if log["first_objective"] is None:
                log["first_objective"] = info["objective"]
                log["first_feasible"] = info["feasible"]
            o = o2
        if pend is not None and learn:
            self._store_large(pend, o["large"])
        return log

    def _store_large(self, pend, o_next):
        o_l, a_l, rsum = pend
        self.large.buf.add(o_l, a_l, rsum / self.env.K, o_next, 0.0)
        for _ in range(self.large_updates):
            self.large.update()

    def agents(self):
        return {k: getattr(self, k) for k in ("large", "small", "single") if getattr(self, k, None) is not None}

    def save(self, path):
        sd = {}
        for k in ("large", "small", "single"):
            ag = getattr(self, k, None)
            if ag is not None:
                sd[k] = ag.state_dict()
        torch.save(sd, path)


def train(method, n_ue, seed, episodes, out_dir, eval_every=50, agent_cfg=AgentConfig(),
          env_cfg=EnvConfig(), cfg_kw=None, ckpt_every=10, large_kw=None, large_updates=LARGE_UPDATES):
    """Train one method on the Phase-5 topology (seed, n_ue); evaluate on the Phase-5 snapshot.

    Every ckpt_every episodes the training state (networks, optimisers, RNGs, logs; not the replay
    buffer) goes to <out>/ckpt.pt; a later call with the same arguments continues from there.
    """
    torch.set_num_threads(1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(seed=seed, n_ue=n_ue, **(cfg_kw or {}))
    env = TwoTimescaleEnv(cfg, env_cfg)
    m = Method(method, env, agent_cfg, seed=seed, large_kw=large_kw, large_updates=large_updates)
    rows, evals, best_eval, start, t_prev = [], [], np.inf, 1, 0.0
    ck = out / "ckpt.pt"
    if ck.exists():
        st = torch.load(ck, weights_only=False)
        for k, ag in m.agents().items():
            load_checkpoint_state(ag, st["agents"][k])
        rows, evals, best_eval, start = st["rows"], st["evals"], st["best_eval"], st["episode"] + 1
        t_prev = rows[-1]["time"] if rows else 0.0
    t0 = time.time() - t_prev

    def write_logs():
        with open(out / "log.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        (out / "eval.json").write_text(json.dumps(dict(
            method=method, n_ue=n_ue, seed=seed, episodes=episodes, evals=evals, best_first_objective=best_eval,
            agent_cfg=asdict(agent_cfg), env_cfg=asdict(env_cfg)), indent=1))

    for ep in range(start, episodes + 1):
        lg = m.run_episode(ep, explore=True, learn=True)
        rows.append(dict(episode=ep, reward=lg["reward"], mean_objective=float(np.mean(lg["objective"])),
                         mean_ratio=float(np.mean(lg["ratio"])), feasible=float(np.mean(lg["feasible"])),
                         time=time.time() - t0))
        if ep % eval_every == 0 or ep == episodes:
            ev = m.run_episode(0, explore=False, learn=False)       # Phase-5 snapshot, deterministic
            e = dict(episode=ep, first_objective=ev["first_objective"], first_feasible=ev["first_feasible"],
                     mean_objective=float(np.mean(ev["objective"])), feasible=float(np.mean(ev["feasible"])),
                     mean_ratio=float(np.mean(ev["ratio"])), time=time.time() - t0)
            evals.append(e)
            if e["first_feasible"] and e["first_objective"] < best_eval:
                best_eval = e["first_objective"]
                m.save(out / "best.pt")
            write_logs()
        if ep % ckpt_every == 0 and ep < episodes:
            tmp = out / "ckpt.tmp"
            torch.save(dict(episode=ep, rows=rows, evals=evals, best_eval=best_eval,
                            agents={k: checkpoint_state(ag) for k, ag in m.agents().items()}), tmp)
            tmp.replace(ck)                                          # atomic: a kill never leaves half a file
    m.save(out / "final.pt")
    ck.unlink(missing_ok=True)
    return evals
