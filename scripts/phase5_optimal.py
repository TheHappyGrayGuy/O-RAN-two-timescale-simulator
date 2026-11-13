"""Phase 5 study: certified optimum on small instances, optimum vs baselines at ch6 scale.

Usage: python scripts/phase5_optimal.py [--seeds 5] [--users 5 10 15 20] [--workers 10]
Writes results/phase5/{small.json, full.json, report.md}.
"""
import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oran_sim import Config, Network  # noqa: E402
from oran_sim.baselines import BASELINES  # noqa: E402
from oran_sim.optimal import Solver  # noqa: E402
from oran_sim.reference_policies import penalised_objective  # noqa: E402

OUT = ROOT / "results" / "phase5"
SMALL = dict(n_ru=2, n_edge=1, n_regional=1, n_ue=3, n_prb=3)


def small_case(seed):
    warnings.filterwarnings("ignore")
    net = Network(Config(seed=seed, **SMALL))
    t0 = time.time()
    ex = Solver(net).solve_exact()
    t1 = time.time()
    sol = Solver(net, exhaustive_limit=0, refine_in_search=True).solve(restarts=10)
    t2 = time.time()
    return dict(seed=seed, exact=ex.value, exact_lb=ex.lower_bound, certified=ex.certified,
                solver=sol.value, t_exact=t1 - t0, t_solver=t2 - t1)


def full_case(args):
    U, seed = args
    warnings.filterwarnings("ignore")
    net = Network(Config(seed=seed, n_ue=U))
    rng = np.random.default_rng(seed)
    out = dict(U=U, seed=seed)
    for name, fn in BASELINES.items():
        if name == "joint_optimum":
            continue
        d = fn(net, rng)
        r = net.evaluate(d)
        out[name] = dict(objective=float(r["objective"]), penalised=float(penalised_objective(net, d)),
                         feasible=bool(r["feasible"]), sat=float(r["satisfaction_ratio"]),
                         phi=float(r["phi_bar"]), d_ms=(r["d_slice"] * 1e3).tolist())
    t0 = time.time()
    s = Solver(net, refine_in_search=True)
    res = s.solve(restarts=6)
    t1 = time.time()
    cond = s.conditional_lower_bound(res.assoc)
    glob = s.lower_bound(res)
    t2 = time.time()
    r = res.info["kpis"]
    out["optimum"] = dict(objective=float(res.value), feasible=bool(r["feasible"]), sat=float(r["satisfaction_ratio"]),
                          phi=float(r["phi_bar"]), d_ms=(r["d_slice"] * 1e3).tolist(),
                          cond_lb=float(cond), global_lb=float(glob),
                          restarts=[float(v) for v in res.info["restart_values"]],
                          edge_share=float(r["edge_share"]),
                          cu=res.cu.tolist(), du=res.du.tolist(), assoc=res.assoc.tolist(),
                          t_solve=t1 - t0, t_bounds=t2 - t1)
    return out


def report(small, full):
    L = ["# Phase 5: optimal solutions", ""]
    L += ["## Small instances (B=2, N=2, U=3, M=3): certified global optimum by full enumeration", "",
          "| seed | exact optimum | certified | scalable solver | difference |", "|---|---|---|---|---|"]
    for c in small:
        L.append(f"| {c['seed']} | {c['exact']:.6f} | {'yes' if c['certified'] else 'no*'} | {c['solver']:.6f} | "
                 f"{(c['solver'] / c['exact'] - 1) * 100:+.4f}% |")
    hits = sum(abs(c["solver"] / c["exact"] - 1) < 1e-6 for c in small)
    L += ["", f"Scalable solver reaches the exact optimum in **{hits}/{len(small)}** instances.",
          "*no = the relaxation bound is not attained, because the fronthaul budget binds once fronthaul transmission time is included; "
          "the reported optimum is then exact only for V-optimal radio solutions.", ""]
    names = ["random", "all_edge", "cu_regional_du_edge", "all_regional", "static_best", "placement_search"]
    L += ["## ch6 scale (B=4, N=6, M=25): optimum vs baselines (mean over seeds)", "",
          "| U | " + " | ".join(names) + " | **optimum** | gain vs static_best | cond. gap | global gap | solve time |",
          "|---|" + "---|" * (len(names) + 5)]
    for U in sorted({c["U"] for c in full}):
        cs = [c for c in full if c["U"] == U]
        row = [f"{np.mean([c[n]['penalised'] for c in cs]):.4f}" + ("" if all(c[n]["feasible"] for c in cs) else "†")
               for n in names]
        opt = np.mean([c["optimum"]["objective"] for c in cs])
        gain = np.mean([1 - c["optimum"]["objective"] / c["static_best"]["penalised"] for c in cs]) * 100
        cg = np.mean([(c["optimum"]["objective"] - c["optimum"]["cond_lb"]) / c["optimum"]["objective"] for c in cs]) * 100
        gg = np.mean([(c["optimum"]["objective"] - c["optimum"]["global_lb"]) / c["optimum"]["objective"] for c in cs]) * 100
        ts = np.mean([c["optimum"]["t_solve"] for c in cs])
        L.append(f"| {U} | " + " | ".join(row) + f" | **{opt:.4f}** | {gain:.1f}% | {cg:.3f}% | {gg:.1f}% | {ts:.0f}s |")
    L += ["", "† at least one seed infeasible (penalised value includes 1e3 per violated constraint).",
          "cond. gap: certified bound over all PRB / power / placement choices for the association found.",
          "global gap: certified bound with association relaxed as well.", ""]
    L += ["## Per-slice delay at the optimum [ms] and edge share", "", "| U | eMBB | URLLC | mMTC | edge share | QoS sat. |",
          "|---|---|---|---|---|---|"]
    for U in sorted({c["U"] for c in full}):
        cs = [c for c in full if c["U"] == U]
        d = np.mean([c["optimum"]["d_ms"] for c in cs], axis=0)
        L.append(f"| {U} | {d[0]:.3f} | {d[1]:.3f} | {d[2]:.3f} | "
                 f"{np.mean([c['optimum']['edge_share'] for c in cs]):.2f} | {np.mean([c['optimum']['sat'] for c in cs]):.2f} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--small-seeds", type=int, default=12)
    ap.add_argument("--users", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # resume: cases already in small.json / full.json are kept, only the missing ones are solved
    sf, ff = OUT / "small.json", OUT / "full.json"
    small = json.loads(sf.read_text()) if sf.exists() else []
    full = json.loads(ff.read_text()) if ff.exists() else []
    have = {(c["U"], c["seed"]) for c in full}
    with ProcessPoolExecutor(a.workers) as ex:
        if len(small) < a.small_seeds:
            small = list(ex.map(small_case, range(a.small_seeds)))
            sf.write_text(json.dumps(small, indent=1))
        print("small done", flush=True)
        jobs = [(U, s) for U in a.users for s in range(a.seeds) if (U, s) not in have]
        print(f"{len(have)} cases kept, {len(jobs)} to solve", flush=True)
        for res in ex.map(full_case, jobs):
            full.append(res)
            print(f"U={res['U']} seed={res['seed']} opt={res['optimum']['objective']:.6f} "
                  f"static_best={res['static_best']['penalised']:.6f} t={res['optimum']['t_solve']:.0f}s", flush=True)
            ff.write_text(json.dumps(full, indent=1))
    text = report(small, full)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
