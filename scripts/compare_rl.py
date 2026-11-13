"""Phase 9-10: compare the learned policies with the Phase-5 optimum and the static baselines.

Every policy is evaluated on the Phase-5 snapshot of (U, seed), so the comparison is like for like.
Writes <rl>/report.md, <rl>/convergence.png, <rl>/optimality.png.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"marl_td3": "MARL (2 x TD3)", "hybrid_td3": "TD3 + convex", "hybrid_sac": "SAC + convex",
         "single_sac": "single SAC"}
COLOR = {"marl_td3": "#2a7ab9", "hybrid_td3": "#e8a33d", "hybrid_sac": "#c0392b", "single_sac": "#5a9e4b"}


def load(rl_dir):
    runs = {}
    for f in Path(rl_dir).glob("*/eval.json"):
        d = json.loads(f.read_text())
        rows = list(csv.DictReader(open(f.parent / "log.csv"))) if (f.parent / "log.csv").exists() else []
        runs[(d["method"], d["n_ue"], d["seed"])] = dict(eval=d, log=rows)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", default=str(ROOT / "results" / "rl"))
    ap.add_argument("--phase5", default=str(ROOT / "results" / "phase5" / "full.json"))
    a = ap.parse_args()
    runs = load(a.rl)
    if not runs:
        print("no runs found")
        return
    p5 = {(c["U"], c["seed"]): c for c in json.loads(Path(a.phase5).read_text())} if Path(a.phase5).exists() else {}
    methods = [m for m in LABEL if any(k[0] == m for k in runs)]
    users = sorted({k[1] for k in runs})

    def final(r):                         # policy at the end of training, on the Phase-5 snapshot
        e = r["eval"]["evals"][-1]
        return e["first_objective"] if e["first_feasible"] else np.inf

    def best(r):
        v = r["eval"]["best_first_objective"]
        return v if v is not None and np.isfinite(v) else np.inf

    L = ["# Learned policies vs the Phase-5 optimum", "",
         "Objective on the Phase-5 snapshot (lower is better); gap = objective / optimum - 1, mean over seeds.", "",
         "| U | optimum | static best | " + " | ".join(f"{LABEL[m]} (gap)" for m in methods) + " |",
         "|---|---|---|" + "---|" * len(methods)]
    gaps = {m: [] for m in methods}
    for U in users:
        seeds = sorted({k[2] for k in runs if k[1] == U})
        opt = [p5[(U, s)]["optimum"]["objective"] for s in seeds if (U, s) in p5]
        stat = [p5[(U, s)]["static_best"]["penalised"] for s in seeds if (U, s) in p5]
        cells = []
        for m in methods:
            vals, gs = [], []
            for s in seeds:
                if (m, U, s) in runs:
                    v = best(runs[(m, U, s)])
                    vals.append(v)
                    if (U, s) in p5:
                        gs.append(v / p5[(U, s)]["optimum"]["objective"] - 1)
            gaps[m].append(np.mean(gs) * 100 if gs else np.nan)
            cells.append(f"{np.mean(vals):.4f} ({np.mean(gs) * 100:+.1f}%)" if gs else
                         (f"{np.mean(vals):.4f}" if vals else "-"))
        L.append(f"| {U} | {np.mean(opt):.4f} | {np.mean(stat):.4f} | " + " | ".join(cells) + " |"
                 if opt else f"| {U} | - | - | " + " | ".join(cells) + " |")
    L += ["", "Infeasible final policies count as inf. 'best' = best feasible evaluation during training.", ""]
    (Path(a.rl) / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    fig, axes = plt.subplots(1, len(users), figsize=(4.2 * len(users), 3.6), squeeze=False)
    for ax, U in zip(axes[0], users):
        for m in methods:
            curves = [r["log"] for k, r in runs.items() if k[0] == m and k[1] == U and r["log"]]
            if not curves:
                continue
            n = min(len(c) for c in curves)
            y = np.array([[float(row["mean_ratio"]) for row in c[:n]] for c in curves])
            k = max(1, n // 50)
            ys = np.convolve(y.mean(0), np.ones(k) / k, mode="valid")
            ax.plot(np.arange(len(ys)) + k, ys, color=COLOR[m], label=LABEL[m], lw=1.4)
        opt_ratio = [p5[(U, s)]["optimum"]["objective"] / p5[(U, s)]["static_best"]["penalised"]
                     for s in {k[2] for k in runs if k[1] == U} if (U, s) in p5]
        if opt_ratio:
            ax.axhline(np.mean(opt_ratio), color="k", ls="--", lw=1, label="optimum (Phase 5)")
        ax.axhline(1.0, color="#999", ls=":", lw=1, label="static best")
        ax.set_title(f"U = {U}")
        ax.set_xlabel("episode")
        ax.set_ylabel("objective / static best")
        ax.set_ylim(0, 3)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(a.rl) / "convergence.png", dpi=150)
    plt.close(fig)

    if p5:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(users))
        wd = 0.8 / max(1, len(methods))
        for i, m in enumerate(methods):
            ax.bar(x + i * wd, gaps[m], wd, color=COLOR[m], label=LABEL[m])
        ax.set_xticks(x + wd * (len(methods) - 1) / 2, [f"U={u}" for u in users])
        ax.set_ylabel("gap to optimum [%]")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(a.rl) / "optimality.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
