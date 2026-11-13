import sys, json, time
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
    B, N = env.B, env.N; o = full[(U, s)]; jstar = o['objective']
    def J(a, c, d):
        env.assoc, env.cu, env.du = a, c, d
        ow, pw = convex_small_decision(env)
        r = env.net.evaluate(Decision.from_compact(a, ow, pw, c, d, B, N))
        nv = sum(not v for v in r["constraints"].values())
        return (r["objective"] if np.isfinite(r["objective"]) else 10 * env.J_ref) + nv * env.J_ref
    c, d = np.array(o['cu']), np.array(o['du'])
    maxg = env.net.G.argmax(axis=0)                                    # strongest RU per user
    top3 = np.argsort(-env.net.G.T, axis=1)[:, :3]
    t = time.time(); a = maxg.copy(); cur = J(a, c, d); sweeps = 0
    while sweeps < 5:                                                  # user best-response sweeps
        imp = False
        for u in range(U):
            for b in top3[u]:
                if b != a[u]:
                    x = a.copy(); x[u] = b; v = J(x, c, d)
                    if v < cur * 0.9995: a, cur, imp = x, v, True
        sweeps += 1
        if not imp: break
    print(f"U={U:2d} s={s}: optimal placement + max-gain assoc {100*(J(maxg,c,d)/jstar-1):6.1f}% | + user sweeps {100*(cur/jstar-1):5.1f}% ({sweeps} sweeps, {time.time()-t:.1f}s)")
