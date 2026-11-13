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
for U in (5, 10, 15, 20):
    for s in (0, 1):
        env = TwoTimescaleEnv(Config(n_ue=U, seed=s), EnvConfig()); env.reset(0)
        o = full[(U, s)]
        def J(a, c, d):
            env.assoc, env.cu, env.du = a, c, d
            ow, pw = convex_small_decision(env)
            return env.net.evaluate(Decision.from_compact(a, ow, pw, c, d, env.B, env.N))["objective"]
        p = env.prior; B, N = env.B, env.N
        sa = p[:U*B].reshape(U, B).argmax(1); sc = p[U*B:].reshape(2, B, N)
        js = J(sa, sc[0].argmax(1), sc[1].argmax(1))
        jo = J(np.array(o['assoc']), np.array(o['cu']), np.array(o['du']))
        jstar = o['objective']
        print(f"U={U:2d} s={s}: optimum {jstar:.5f} | static assoc+fast PRB {100*(js/jstar-1):5.1f}% | optimal assoc+fast PRB {100*(jo/jstar-1):5.1f}% "
              f"| assoc differs {np.sum(sa != np.array(o['assoc']))}/{U}, cu {np.sum(sc[0].argmax(1)!=np.array(o['cu']))}/4, du {np.sum(sc[1].argmax(1)!=np.array(o['du']))}/4")
