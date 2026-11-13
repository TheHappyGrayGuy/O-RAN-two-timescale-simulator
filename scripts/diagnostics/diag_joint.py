import sys, json
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from oran_sim.config import Config
from oran_sim.kpis import Decision
from oran_sim.marl_env import EnvConfig, TwoTimescaleEnv
from oran_sim.rl_methods import convex_small_decision
full = {(c['U'], c['seed']): c['optimum'] for c in json.load(open(ROOT / "results" / "phase5" / "full.json"))}
for U, s in [(5,1),(10,1),(15,1),(20,1),(10,0),(20,0)]:
    env = TwoTimescaleEnv(Config(n_ue=U, seed=s), EnvConfig()); env.reset(0)
    B, N = env.B, env.N
    def J(a, c, d):
        env.assoc, env.cu, env.du = a, c, d
        ow, pw = convex_small_decision(env)
        r = env.net.evaluate(Decision.from_compact(a, ow, pw, c, d, B, N))
        nv = sum(not v for v in r["constraints"].values())
        return (r["objective"] if np.isfinite(r["objective"]) else 10 * env.J_ref) + nv * env.J_ref
    p = env.prior
    a = p[:U*B].reshape(U, B).argmax(1); sc = p[U*B:].reshape(2, B, N); c, d = sc[0].argmax(1), sc[1].argmax(1)
    top3 = np.argsort(-env.net.G.T, axis=1)[:, :3]                     # 3 best RUs per user
    cur = J(a, c, d); jstar = full[(U, s)]['objective']; n = 0
    while True:
        cand = []
        for u in range(U):
            for b in top3[u]:
                if b != a[u]:
                    x = a.copy(); x[u] = b; cand.append((J(x, c, d), x, c, d))
        for b in range(B):
            for n1 in range(N):
                for n2 in range(N):
                    if (n1, n2) != (c[b], d[b]):
                        y, z = c.copy(), d.copy(); y[b], z[b] = n1, n2
                        cand.append((J(a, y, z), a, y, z))
        v, x, y, z = min(cand, key=lambda t: t[0])
        if v >= cur * 0.9995 or n > 40: break
        cur, a, c, d, n = v, x, y, z, n + 1
    print(f"U={U:2d} s={s}: {len(cand)} moves/step, {n} greedy steps -> gap {100*(cur/jstar-1):5.1f}%")
