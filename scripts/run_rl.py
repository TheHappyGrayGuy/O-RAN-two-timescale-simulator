"""Phase 8-9: train every (method, U, seed) in parallel, then compare against the Phase-5 optimum.

Usage:
  python scripts/run_rl.py                         # all 4 methods x U {5,10,15,20} x seeds {0,1,2}
  python scripts/run_rl.py --wait-phase5           # start automatically once Phase 5 has finished
  python scripts/run_rl.py --episodes 200 --users 10 --seeds 0   # quick pass
Finished runs (eval.json with all episodes) are skipped, so the script can be re-run to resume.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # one core per run
    os.environ.setdefault(k, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COST = {"single_sac": 1.5, "marl_td3": 1.0, "hybrid_td3": 0.8, "hybrid_sac": 0.8}   # relative s/episode


def job(method, U, seed, episodes, out_root, eval_every, prior=0.0, temp=3.0, act_pen=0.0, large=None):
    import warnings
    warnings.filterwarnings("ignore")
    from oran_sim.marl_env import EnvConfig
    from oran_sim.rl_methods import train
    out = Path(out_root) / f"{method}_U{U}_s{seed}"
    t0 = time.time()
    evals = train(method, U, seed, episodes, out, eval_every=eval_every, env_cfg=EnvConfig(large_prior=prior, small_temp=temp, action_penalty=act_pen),
                  **(large or {}))
    return method, U, seed, evals[-1]["first_objective"], time.time() - t0


def done(out_root, method, U, seed, episodes):
    f = Path(out_root) / f"{method}_U{U}_s{seed}" / "eval.json"
    if not f.exists():
        return False
    d = json.loads(f.read_text())
    return d["episodes"] == episodes and d["evals"] and d["evals"][-1]["episode"] == episodes


def phase5_finished(users, seeds):
    """Every (U, seed) of the grid is in full.json and report.md was written after it."""
    p5 = ROOT / "results" / "phase5"
    rep, full = p5 / "report.md", p5 / "full.json"
    if not (rep.exists() and full.exists()) or rep.stat().st_mtime < full.stat().st_mtime:
        return False
    try:
        have = {(c["U"], c["seed"]) for c in json.loads(full.read_text())}
    except (ValueError, KeyError):           # file being written
        return False
    return all((U, s) in have for U in users for s in seeds)


def main():
    from oran_sim.rl_methods import METHODS
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--users", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=str(ROOT / "results" / "rl"))
    ap.add_argument("--wait-phase5", action="store_true")
    ap.add_argument("--prior", type=float, default=0.0, help="residual prior on the best static decision")
    ap.add_argument("--temp", type=float, default=3.0, help="softmax temperature of PRB / power shares")
    ap.add_argument("--act-pen", type=float, default=0.0, help="reward penalty on mean squared action")
    ap.add_argument("--large-updates", type=int, default=10, help="gradient steps per large transition")
    ap.add_argument("--large-hidden", type=int, nargs="+", default=None, help="large agent layer sizes")
    ap.add_argument("--large-actor-lr", type=float, default=None)
    a = ap.parse_args()
    if a.wait_phase5:
        print("waiting for Phase 5 to finish ...", flush=True)
        while not phase5_finished(a.users, a.seeds):
            time.sleep(60)
    jobs = [(m, U, s) for m in a.methods for U in a.users for s in a.seeds
            if not done(a.out, m, U, s, a.episodes)]
    jobs.sort(key=lambda j: -COST[j[0]] * (1 + j[1] / 10))                 # longest first
    print(f"{len(jobs)} runs on {a.workers} workers, {a.episodes} episodes each", flush=True)
    kw = {}
    if a.large_hidden:
        kw.update(actor_hidden=tuple(a.large_hidden), critic_hidden=tuple(a.large_hidden))
    if a.large_actor_lr:
        kw["actor_lr"] = a.large_actor_lr
    large = dict(large_kw=kw, large_updates=a.large_updates)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, m, U, s, a.episodes, a.out, a.eval_every, a.prior, a.temp, a.act_pen, large) for m, U, s in jobs]
        for f in as_completed(futs):
            try:
                m, U, s, obj, dt = f.result()
                print(f"done {m} U={U} seed={s} final objective={obj:.6f} ({dt / 60:.0f} min)", flush=True)
            except Exception as e:                                         # keep the other runs going
                print(f"FAILED: {e!r}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_rl.py"), "--rl", a.out])


if __name__ == "__main__":
    main()
