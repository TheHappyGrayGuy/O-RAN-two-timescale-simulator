"""Diagnose why the large agent does not learn: action decoding, prior, and reward landscape."""
import sys, time
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from oran_sim.config import Config
from oran_sim.kpis import Decision
from oran_sim.marl_env import EnvConfig, TwoTimescaleEnv
from oran_sim.rl_methods import convex_small_decision

U = int(sys.argv[1])
cfg = Config(n_ue=U, seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
env = TwoTimescaleEnv(cfg, EnvConfig(large_prior=0.0))
env.reset(0)
B, N = env.B, env.N

best = env.prior
sa = best[:U * B].reshape(U, B).argmax(1); sc = best[U * B:].reshape(2, B, N)
s_cu, s_du = sc[0].argmax(1), sc[1].argmax(1)


def J(assoc, cu, du):
    env.assoc, env.cu, env.du = assoc, cu, du
    owner, power = convex_small_decision(env)
    r = env.net.evaluate(Decision.from_compact(assoc, owner, power, cu, du, B, N))
    nv = sum(not v for v in r["constraints"].values())
    return (r["objective"] if np.isfinite(r["objective"]) else 10 * env.J_ref) + nv * env.J_ref, nv


J0, _ = J(sa, s_cu, s_du)
print(f"U={U}  J_static(+convex)={J0:.5f}  J_ref={env.J_ref:.5f}  large_dim={env.large_dim}")

rng = np.random.default_rng(1)
# A. can the action change the decision at all, for each prior?
for beta in (0, 0.5, 1, 2):
    env.ecfg.large_prior = beta
    ch = []
    for _ in range(2000):
        a = np.tanh(rng.normal(0, 1, env.large_dim))
        x, y, z = env.decode_large(a)
        ch.append((x != sa).sum() + (y != s_cu).sum() + (z != s_du).sum())
    ch = np.array(ch)
    print(f"A beta={beta}: P(decision != static)={np.mean(ch > 0):.3f}  mean #changed={ch.mean():.1f}")

# B. random full decisions (beta=0): how often better than static?
env.ecfg.large_prior = 0.0
vals = []
t = time.time()
for _ in range(100):
    x, y, z = env.decode_large(np.tanh(rng.normal(0, 1, env.large_dim)))
    vals.append(J(x, y, z)[0])
vals = np.array(vals) / J0
print(f"B random full decision: P(better than static)={np.mean(vals < 1):.2f}  median J/Jstatic={np.median(vals):.2f}  "
      f"best={vals.min():.3f}  ({(time.time()-t)/100:.2f}s per eval)")

# C. beta=1 with small exploration noise around 0
env.ecfg.large_prior = 1.0
vals, nch = [], []
for _ in range(100):
    x, y, z = env.decode_large(np.clip(rng.normal(0, 0.3, env.large_dim), -1, 1))
    nch.append((x != sa).sum() + (y != s_cu).sum() + (z != s_du).sum())
    vals.append(J(x, y, z)[0])
vals = np.array(vals) / J0
print(f"C beta=1, noise 0.3: mean #changed={np.mean(nch):.1f}  P(better)={np.mean(vals < 1):.2f}  "
      f"median J/Jstatic={np.median(vals):.2f}  best={vals.min():.3f}")

# D. single local moves from the static decision
moves = []
for u in range(U):
    for b in range(B):
        if b != sa[u]:
            x = sa.copy(); x[u] = b; moves.append(("ue", J(x, s_cu, s_du)[0]))
for which in range(2):
    for b in range(B):
        for n in range(N):
            cu, du = s_cu.copy(), s_du.copy()
            (cu if which == 0 else du)[b] = n
            if (cu != s_cu).any() or (du != s_du).any():
                moves.append(("cu" if which == 0 else "du", J(sa, cu, du)[0]))
d = np.array([m[1] for m in moves]) / J0
print(f"D single local moves: n={len(d)}  P(improves)={np.mean(d < 0.999):.2f}  best={d.min():.3f}  "
      f"median={np.median(d):.3f}")

# E. greedy chain of local moves (what a perfect local-move agent could reach)
a_, c_, d_ = sa.copy(), s_cu.copy(), s_du.copy()
cur = J0
for step in range(30):
    cand = []
    for u in range(U):
        for b in range(B):
            if b != a_[u]:
                x = a_.copy(); x[u] = b; cand.append((J(x, c_, d_)[0], x, c_, d_))
    for b in range(B):
        for n in range(N):
            if n != c_[b]:
                y = c_.copy(); y[b] = n; cand.append((J(a_, y, d_)[0], a_, y, d_))
            if n != d_[b]:
                z = d_.copy(); z[b] = n; cand.append((J(a_, c_, z)[0], a_, c_, z))
    v, x, y, z = min(cand, key=lambda c: c[0])
    if v >= cur * 0.9995:
        break
    cur, a_, c_, d_ = v, x, y, z
print(f"E greedy local moves: {step} moves -> J/Jstatic={cur / J0:.3f}  J={cur:.5f}")
