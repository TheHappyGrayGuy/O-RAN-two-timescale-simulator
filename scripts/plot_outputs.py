"""Figures for the thesis: RL convergence and the system-model entities at the Phase-5 optimum.

  python scripts/plot_outputs.py                 # U=20 seed 0, RL runs from results/rl
  python scripts/plot_outputs.py --U 10 --seed 1 --rl results/rl_pilot_prior

Writes results/figures/*.png. The RL figures use whatever episodes are logged so far, so the
script can be re-run while training is still going.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oran_sim import Config, Network                                      # noqa: E402
from oran_sim.optimal import Solver                                        # noqa: E402
from oran_sim.reference_policies import penalised_objective, reference_decision   # noqa: E402

# categorical slots of the reference palette, in fixed order (validated set, light surface)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
METHODS = {"marl_td3": ("MARL (2 × TD3)", C1), "hybrid_td3": ("TD3 + convex", C2),
           "hybrid_sac": ("SAC + convex", C3), "single_sac": ("single SAC", C4)}
SLICE_COLOR, SLICE_MARK = [C1, C2, C3], ["o", "s", "^"]                      # eMBB, URLLC, mMTC
STATIC = ("all_edge", "cu_regional_du_edge", "all_regional")

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.titlesize": 10, "axes.titlecolor": INK,
})


def save(fig, out, name, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect)
    for attempt in range(5):                       # a viewer / virus scan can lock the file briefly
        try:
            fig.savefig(out / f"{name}.png", dpi=200)
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)
    plt.close(fig)
    print("wrote", out / f"{name}.png")


# ---------------------------------------------------------------------------- RL figures
def load_runs(rl_dir):
    runs = {}
    for f in Path(rl_dir).glob("*/log.csv"):
        rows = list(csv.DictReader(open(f)))
        ev = f.parent / "eval.json"
        if rows:
            runs[f.parent.name] = dict(rows=rows, eval=json.loads(ev.read_text()) if ev.exists() else None)
    return runs


def split_name(name):
    m, rest = name.rsplit("_U", 1)
    U, s = rest.split("_s")
    return m, int(U), int(s)


def smooth(y, k):
    if len(y) < k or k < 2:
        return np.asarray(y, float)
    c = np.convolve(y, np.ones(k) / k, mode="valid")
    return np.concatenate([np.cumsum(y[:k - 1]) / np.arange(1, k), c])     # no shrinking at the start


def fig_reward(runs, out, p5):
    """Reward of the greedy policy on the fixed Phase-5 instance: -J / J_static (best static = -1)."""
    users = sorted({split_name(n)[1] for n in runs})
    fig, axes = plt.subplots(1, len(users), figsize=(max(7.5, 3.3 * len(users)), 3.1), squeeze=False)
    for ax, U in zip(axes[0], users):
        seeds = sorted({split_name(n)[2] for n in runs if split_name(n)[1] == U and (U, split_name(n)[2]) in p5})
        if not seeds:
            continue
        ref = {s: p5[(U, s)]["static_best"]["penalised"] for s in seeds}
        opt = np.mean([-p5[(U, s)]["optimum"]["objective"] / ref[s] for s in seeds])
        for m, (label, col) in METHODS.items():
            pts = {}
            for n, v in runs.items():
                mm, UU, s = split_name(n)
                if (mm, UU) != (m, U) or s not in ref or not v["eval"]:
                    continue
                for e in v["eval"]["evals"]:
                    if e["first_feasible"]:
                        pts.setdefault(e["episode"], []).append(-e["first_objective"] / ref[s])
            if pts:
                ep = sorted(pts)
                ax.plot(ep, [np.mean(pts[e]) for e in ep], color=col, label=label)
        ax.axhline(opt, color=INK, lw=1.2)
        ax.text(0.99, opt, "optimum (Phase 5) ", transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=7, color=INK)
        ax.axhline(-1, color=INK2, ls="--", lw=1)
        ax.text(0.99, -1, "best static ", transform=ax.get_yaxis_transform(), ha="right", va="top",
                fontsize=7, color=INK2)
        ax.set_title(f"U = {U}")
        ax.set_xlabel("episode")
        ax.set_ylim(min(-1.6, ax.get_ylim()[0]), max(opt + 0.08, -0.6))
    axes[0][0].set_ylabel("normalised reward  −J / J_static")
    h, l = [], []
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=max(1, len(l)), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Reward convergence of the greedy policy on the Phase-5 instance (mean over seeds)",
                 fontsize=10, color=INK)
    save(fig, out, "rl_reward_convergence", rect=(0, 0.07, 1, 1))


def fig_train_reward(runs, out, slots):
    users = sorted({split_name(n)[1] for n in runs})
    fig, axes = plt.subplots(1, len(users), figsize=(3.3 * len(users), 3.0), squeeze=False, sharey=True)
    for ax, U in zip(axes[0], users):
        for m, (label, col) in METHODS.items():
            curves = [np.array([float(r["reward"]) / slots for r in v["rows"]])
                      for n, v in runs.items() if split_name(n)[:2] == (m, U)]
            if not curves:
                continue
            n = min(len(c) for c in curves)
            Y = np.stack([smooth(c[:n], max(5, n // 25)) for c in curves])
            x = np.arange(1, n + 1)
            if len(curves) > 1:
                ax.fill_between(x, Y.min(0), Y.max(0), color=col, alpha=0.15, lw=0)
            ax.plot(x, Y.mean(0), color=col, label=label)
        ax.axhline(-1, color=INK2, ls="--", lw=1)
        ax.text(1, -1, " best static", color=INK2, va="bottom", fontsize=7)
        ax.set_title(f"U = {U}")
        ax.set_xlabel("episode")
    axes[0][0].set_ylabel("mean reward per slot")
    axes[0][0].set_ylim(-10.3, 0.2)                                           # reward is clipped at -10
    h, l = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Training reward per slot (random user drops, with exploration; shaded: range over seeds)",
                 fontsize=10, color=INK)
    save(fig, out, "rl_training_reward", rect=(0, 0.07, 1, 1))


def fig_eval(runs, out, p5):
    users = sorted({split_name(n)[1] for n in runs})
    fig, axes = plt.subplots(1, len(users), figsize=(3.3 * len(users), 3.0), squeeze=False, sharey=True)
    for ax, U in zip(axes[0], users):
        for m, (label, col) in METHODS.items():
            pts = {}
            for n, v in runs.items():
                mm, UU, s = split_name(n)
                if (mm, UU) != (m, U) or not v["eval"] or (U, s) not in p5:
                    continue
                opt = p5[(U, s)]["optimum"]["objective"]
                for e in v["eval"]["evals"]:
                    if e["first_feasible"]:
                        pts.setdefault(e["episode"], []).append(100 * (e["first_objective"] / opt - 1))
            if pts:
                ep = sorted(pts)
                ax.plot(ep, [np.mean(pts[e]) for e in ep], color=col, marker="o", ms=4, label=label)
        stat = [100 * (p5[(U, s)]["static_best"]["penalised"] / p5[(U, s)]["optimum"]["objective"] - 1)
                for s in range(5) if (U, s) in p5 and p5[(U, s)]["static_best"]["feasible"]]
        if stat:
            ax.axhline(np.mean(stat), color=INK2, ls="--", lw=1)
            ax.text(ax.get_xlim()[0], np.mean(stat), " best static", color=INK2, va="bottom", fontsize=7)
        ax.axhline(0, color=INK, lw=1)
        ax.set_title(f"U = {U}")
        ax.set_xlabel("episode")
    axes[0][0].set_ylabel("gap to Phase-5 optimum [%]")
    axes[0][0].set_ylim(bottom=-2)
    axes[0][-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Greedy policy on the Phase-5 instance (mean over seeds)", fontsize=10, color=INK)
    save(fig, out, "rl_gap_to_optimum")


def fig_phase5(cases, out):
    """Objective of the optimum, the baselines and the certified lower bound vs number of users."""
    users = sorted({c["U"] for c in cases})
    C5 = "#e87ba4"
    series = [("all_edge", "all edge", C4, "s", "--"), ("cu_regional_du_edge", "CU regional / DU edge", C3, "^", "--"),
              ("static_best", "best static", C2, "D", "--"), ("placement_search", "placement search", C5, "v", "-"),
              ("optimum", "optimum (Phase 5)", C1, "o", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4), gridspec_kw=dict(width_ratios=[1, 0.8], wspace=0.75))
    ax = axes[0]
    for key, label, col, mk, ls in series:
        y = [np.mean([c[key]["objective"] if key == "optimum" else c[key]["penalised"]
                      for c in cases if c["U"] == U]) for U in users]
        ax.plot(users, y, color=col, marker=mk, ms=6, ls=ls, label=label)
    lb = [np.mean([c["optimum"]["global_lb"] for c in cases if c["U"] == U]) for U in users]
    ax.plot(users, lb, color=INK2, ls=":", lw=1.5, label="certified lower bound")
    ax.set_yscale("log")
    ax.set_xticks(users)
    ax.set_xlabel("users U")
    ax.set_ylabel("objective J (log)")
    ax.set_title("Objective vs number of users")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax = axes[1]
    for key, label, col, mk, ls in series[2:4]:
        g = [np.mean([100 * (c[key]["penalised"] / c["optimum"]["objective"] - 1)
                      for c in cases if c["U"] == U and c[key]["feasible"]]) for U in users]
        ax.plot(users, g, color=col, marker=mk, ms=6, ls=ls, label=label)
    ax.axhline(0, color=C1, lw=2, label="optimum")
    ax.set_xticks(users)
    ax.set_xlabel("users U")
    ax.set_ylabel("excess over the optimum [%]")
    ax.set_title("How far the baselines are from the optimum")
    ax.legend(fontsize=7)
    fig.suptitle("Phase 5: optimum vs baselines (mean over 5 seeds, feasible seeds)", fontsize=10, color=INK)
    save(fig, out, "phase5_optimum_vs_baselines", rect=(0, 0, 1, 0.94))


# ---------------------------------------------------------------------------- system model
def system_state(U, seed, case):
    """Network, optimum decision (PRB + power re-solved for the stored association / placement)."""
    net = Network(Config(seed=seed, n_ue=U))
    o = case["optimum"]
    assoc, cu, du = (np.array(o[k]) for k in ("assoc", "cu", "du"))
    sol = Solver(net, refine_in_search=True)
    sol.polish(assoc)
    dec = sol.decision(assoc, cu, du)
    r = net.evaluate(dec)
    refs = {s: reference_decision(net, s) for s in STATIC}
    best = min(refs, key=lambda s: penalised_objective(net, refs[s]))
    return net, dec, r, best, refs[best], net.evaluate(refs[best])


def node_names(t):
    e = r_ = 0
    names = []
    for is_edge in t.node_is_edge:
        if is_edge:
            e += 1
            names.append(f"E{e}")
        else:
            r_ += 1
            names.append(f"R{r_}")
    return names


def fig_association(net, dec, out, tag):
    t, u, cfg = net.topo, net.users, net.cfg
    names = node_names(t)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for b in range(t.B):                                                   # user -> serving RU
        for i in np.flatnonzero(dec.assoc == b):
            ax.plot(*zip(t.ru_pos[b], u.pos[i]), color=MUTED, lw=0.8, zorder=1)
    for s, sl in enumerate(cfg.slices):
        i = u.slice_idx == s
        ax.scatter(*u.pos[i].T, s=36, marker=SLICE_MARK[s], color=SLICE_COLOR[s], edgecolor="white",
                   lw=1, zorder=3, label=f"{sl.name} user")
    ax.scatter(*t.ru_pos.T, s=150, marker="^", color=INK, edgecolor="white", lw=1.5, zorder=4, label="RU")
    for b in range(t.B):
        split = dec.cu_node[b] != dec.du_node[b]
        txt = (f"RU{b + 1}\nCU {names[dec.cu_node[b]]} / DU {names[dec.du_node[b]]}" if split
               else f"RU{b + 1}\nCU+DU {names[dec.cu_node[b]]}")
        ax.annotate(txt, t.ru_pos[b], xytext=(8, 8), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax.set_title(f"User association and CU/DU placement at the optimum ({tag})")
    save(fig, out, "system_association")


def fig_power(net, dec, r, out, tag):
    t, u, cfg = net.topo, net.users, net.cfg
    owner = np.where(dec.kappa.any(axis=1), dec.kappa.argmax(axis=1), -1)          # (B, M)
    pw = dec.omega.sum(axis=1)                                                       # (B, M) W
    p_dbm = np.where(owner >= 0, 10 * np.log10(np.maximum(pw, 1e-12)) + 30, np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), gridspec_kw=dict(height_ratios=[1, 1.15]))
    ax = axes[0]
    im = ax.imshow(p_dbm, aspect="auto", cmap="Blues", origin="lower")
    for b in range(t.B):
        for m in range(cfg.n_prb):
            if owner[b, m] >= 0:
                v = p_dbm[b, m]
                ax.text(m, b, str(owner[b, m] + 1), ha="center", va="center", fontsize=6,
                        color="white" if v > np.nanmean(p_dbm) else INK)
    ax.set_yticks(range(t.B), [f"RU{b + 1}" for b in range(t.B)])
    ax.set_xlabel("PRB index")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("PRB power [dBm]")
    ax.set_title(f"Power per PRB; number = serving user ({tag})")
    ax = axes[1]
    P = (dec.kappa * dec.omega).sum(axis=(0, 2))
    n_prb = dec.kappa.sum(axis=(0, 2))
    order = np.lexsort((np.arange(u.U), u.slice_idx))
    x = np.arange(u.U)
    for s, sl in enumerate(cfg.slices):
        k = np.flatnonzero(u.slice_idx[order] == s)
        ax.bar(x[k], P[order][k], width=0.8, color=SLICE_COLOR[s], label=sl.name)
    for xi, i in zip(x, order):
        ax.text(xi, P[i], f"{int(n_prb[i])}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(x, [str(i + 1) for i in order], fontsize=7)
    ax.set_xlabel("user (grouped by slice); number above bar = PRBs")
    ax.set_yscale("log")
    ax.set_ylabel("transmit power [W] (log)")
    ax.set_ylim(bottom=0.5 * P[P > 0].min())
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "system_power")


def fig_delay(net, dec, r, out, tag):
    u, cfg = net.users, net.cfg
    a = dec.assoc
    parts = [("radio (M/M/1)", r["d_u"], C1), ("processing", r["d_proc_u"], C2),
             ("transmission FH+MH", (r["d_tr_mid"] + r["d_tr_front"])[a], C3),
             ("propagation", r["d_prop_b"][a], C4)]
    fig, axes = plt.subplots(1, len(cfg.slices), figsize=(9.5, 3.2),
                             gridspec_kw=dict(width_ratios=[max(1, (u.slice_idx == s).sum()) for s in
                                                            range(len(cfg.slices))]))
    for s, (ax, sl) in enumerate(zip(axes, cfg.slices)):
        idx = np.flatnonzero(u.slice_idx == s)
        x = np.arange(len(idx))
        bottom = np.zeros(len(idx))
        for name, v, col in parts:
            h = 1e3 * v[idx]
            ax.bar(x, h, bottom=bottom, width=0.75, color=col, edgecolor="white", lw=1, label=name)
            bottom += h
        ax.set_xticks(x, [str(i + 1) for i in idx], fontsize=7)
        ax.set_title(f"{sl.name}  (QoS {1e3 * sl.d_qos_s:g} ms)")
        ax.set_xlabel("user")
    axes[0].set_ylabel("end-to-end delay [ms]")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"End-to-end delay per user and its components at the optimum ({tag})", fontsize=10, color=INK)
    save(fig, out, "system_delay")


def fig_processing(net, r_opt, r_stat, best, out, tag):
    t = net.topo
    names = node_names(t)
    x = np.arange(t.N)
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    ax = axes[0]
    ax.bar(x - w / 2 - 0.01, 100 * r_opt["cpu_util"], w, color=C1, label="optimum")
    ax.bar(x + w / 2 + 0.01, 100 * r_stat["cpu_util"], w, color=MUTED, label=f"best static ({best})")
    ax.set_xticks(x, [f"{n}\n{'edge' if e else 'regional'}" for n, e in zip(names, t.node_is_edge)])
    ax.set_ylabel("CPU utilisation [%]")
    ax.set_title("Processing load per O-Cloud node")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.bar(x - w / 2 - 0.01, 1e-9 * r_opt["node_cpu"], w, color=C1, label="used, optimum")
    ax.bar(x + w / 2 + 0.01, 1e-9 * r_stat["node_cpu"], w, color=MUTED, label="used, best static")
    ax.scatter(x, 1e-9 * t.cpu_capacity, marker="_", s=300, color=INK, lw=2, label="capacity", zorder=3)
    ax.set_xticks(x, names)
    ax.set_yscale("log")
    ax.set_ylabel("processing [G cycles/s]")
    ax.set_title("Processing used vs node capacity")
    ax.legend(fontsize=7)
    fig.suptitle(f"Processing power ({tag})", fontsize=10, color=INK)
    save(fig, out, "system_processing")


def fig_slices(cases, net, r_opt, r_stat, out, tag):
    """Each slice's reaction: delay vs number of users, and achieved vs offered rate."""
    cfg = net.cfg
    users = sorted({c["U"] for c in cases})
    fig, axes = plt.subplots(1, len(cfg.slices) + 1, figsize=(12, 3.2))
    for s, (ax, sl) in enumerate(zip(axes, cfg.slices)):
        for key, label, col, ls in (("optimum", "optimum", SLICE_COLOR[s], "-"),
                                    ("static_best", "best static", MUTED, "--")):
            y = [np.mean([c[key]["d_ms"][s] for c in cases if c["U"] == U]) for U in users]
            ax.plot(users, y, color=col, ls=ls, marker=SLICE_MARK[s], ms=6, label=label)
        ax.set_title(f"{sl.name} (QoS {1e3 * sl.d_qos_s:g} ms)")
        ax.set_xlabel("users U")
        ax.set_xticks(users)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("mean slice delay [ms]")
    ax = axes[-1]
    u = net.users
    x = np.arange(len(cfg.slices))
    w = 0.38
    offered = [1e-6 * u.offered_bps[u.slice_idx == s].sum() for s in x]
    got = [1e-6 * r_opt["rate_u"][u.slice_idx == s].sum() for s in x]
    ax.bar(x - w / 2 - 0.01, got, w, color=SLICE_COLOR, label="achieved rate")
    ax.bar(x + w / 2 + 0.01, offered, w, color="white", edgecolor=INK2, hatch="////", lw=1, label="offered load")
    ax.set_xticks(x, [sl.name for sl in cfg.slices])
    ax.set_yscale("log")
    ax.set_ylabel("Mb/s (sum over slice users)")
    ax.set_title(f"Rate vs load ({tag})")
    ax.legend(fontsize=7)
    fig.suptitle("Slice reaction: delay as the network loads up (mean over 5 seeds), and rate headroom",
                 fontsize=10, color=INK)
    save(fig, out, "system_slices")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rl", default=str(ROOT / "results" / "rl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--skip-system", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / "results" / "phase5" / "full.json").read_text())
    p5 = {(c["U"], c["seed"]): c for c in cases}

    fig_phase5(cases, out)
    runs = load_runs(a.rl)
    if runs:
        slots = Config().slots_per_large_step * 10
        fig_reward(runs, out, p5)
        fig_train_reward(runs, out, slots)
        fig_eval(runs, out, p5)
    if a.skip_system:
        return
    tag = f"U={a.U}, seed {a.seed}"
    net, dec, r, best, _, r_stat = system_state(a.U, a.seed, p5[(a.U, a.seed)])
    print(f"re-solved objective {r['objective']:.6f} (Phase 5: {p5[(a.U, a.seed)]['optimum']['objective']:.6f})")
    fig_association(net, dec, out, tag)
    fig_power(net, dec, r, out, tag)
    fig_delay(net, dec, r, out, tag)
    fig_processing(net, r, r_stat, best, out, tag)
    fig_slices(cases, net, r, r_stat, out, tag)


if __name__ == "__main__":
    main()
