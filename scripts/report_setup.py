"""Phase 1-4 sanity report: topology map, channel statistics, KPI table per scenario.

Usage: python scripts/report_setup.py [out_dir]
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_sim import SLICE_NAMES, Config, Network, reference_decision  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "setup")
OUT.mkdir(parents=True, exist_ok=True)
SCENARIOS = ["all_edge", "cu_regional_du_edge", "all_regional"]


def plot_topology(net):
    t, u = net.topo, net.users
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, lim in [(a1, t.cfg.L_regional_max), (a2, t.cfg.L_edge_max)]:
        for (p, q), tier in zip(t.links, t.link_tier):
            x = [t.graph.nodes[p]["pos"], t.graph.nodes[q]["pos"]]
            ax.plot(*np.array(x).T, color="#bbb" if tier == "fronthaul" else "#e8a33d", lw=0.6, zorder=1)
        ax.add_patch(plt.Rectangle((0, 0), t.cfg.L_max, t.cfg.L_max, fill=False, ls="--", color="k"))
        ax.scatter(*t.ru_pos.T, marker="^", s=120, c="k", label="RU", zorder=3)
        ax.scatter(*t.node_pos[t.node_is_edge].T, marker="s", s=90, c="#2a7ab9", label="edge O-Cloud", zorder=3)
        ax.scatter(*t.node_pos[~t.node_is_edge].T, marker="D", s=90, c="#c0392b", label="regional O-Cloud", zorder=3)
        for s, mk in zip(range(3), "o*x"):
            idx = u.users_of_slice(s)
            ax.scatter(*u.pos[idx].T, marker=mk, s=25, label=SLICE_NAMES[s], zorder=2)
        ax.set_xlim(-0.03 * lim, lim * 1.03)
        ax.set_ylim(-0.03 * lim, lim * 1.03)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    a1.set_title("Full topology (regional nodes)")
    a2.set_title("Zoom: RAN area and edge nodes")
    a2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "topology.png", dpi=150)
    plt.close(fig)


def plot_channel(cfg):
    snr = {m: [] for m in ["log_distance", "3gpp_umi"]}
    for model in snr:
        net = Network(cfg.with_(pathloss_model=model, n_ue=200))
        for _ in range(20):
            p = cfg.p_max_w / cfg.n_prb
            best = net.G.argmax(0)
            h = net.H[best, np.arange(net.users.U)]
            snr[model].append(10 * np.log10(net.G[best, np.arange(net.users.U)][:, None] * h * p / cfg.noise_per_prb_w).ravel())
            net.large_step()
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, v in snr.items():
        v = np.sort(np.concatenate(v))
        ax.plot(v, np.linspace(0, 1, len(v)), label=model)
    ax.set_xlabel("per-PRB SNR at serving RU [dB]")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "snr_cdf.png", dpi=150)
    plt.close(fig)
    return {m: np.percentile(np.concatenate(v), [5, 50, 95]) for m, v in snr.items()}


def kpi_table(cfg, n_seeds=10, n_slots=50):
    rows = []
    for unit in ["byte", "kbit"]:
        for U in [5, 10, 15, 20]:
            for sc in SCENARIOS:
                acc = []
                for seed in range(n_seeds):
                    net = Network(cfg.with_(seed=seed, n_ue=U, packet_size_unit=unit))
                    for _ in range(n_slots):
                        r = net.evaluate(reference_decision(net, sc))
                        acc.append((r["d_slice"] * 1e3, r["phi_bar"], r["satisfaction_ratio"], r["feasible"],
                                    r["cpu_util"].max(), r["throughput_bps"] / 1e6))
                        net.small_step()
                d = np.array([a[0] for a in acc])
                fin = np.where(np.isfinite(d), d, np.nan)
                rows.append(dict(unit=unit, U=U, scenario=sc,
                                 d_ms=np.nanmedian(fin, axis=0), phi=np.mean([a[1] for a in acc]),
                                 sat=np.mean([a[2] for a in acc]), feas=np.mean([a[3] for a in acc]),
                                 cpu=np.mean([a[4] for a in acc]), thr=np.mean([a[5] for a in acc])))
    return rows


def main():
    warnings.filterwarnings("ignore", "All-NaN slice")  # unstable slices are reported as inf
    cfg = Config()
    net = Network(cfg)
    plot_topology(net)
    pct = plot_channel(cfg)
    rows = kpi_table(cfg)

    lines = ["# Setup sanity report", "",
             f"PRB bandwidth {cfg.prb_bandwidth_hz/1e6:.2f} MHz, noise/PRB {10*np.log10(cfg.noise_per_prb_w)+30:.1f} dBm, "
             f"P/PRB {10*np.log10(cfg.p_max_w/cfg.n_prb)+30:.1f} dBm", "",
             "## Per-PRB SNR percentiles (5/50/95) [dB]", ""]
    lines += [f"- {m}: " + " / ".join(f"{x:.1f}" for x in v) for m, v in pct.items()]
    lines += ["", "## KPIs with reference policy (max-gain association, round-robin PRB, equal power)", "",
              "| unit | U | scenario | D eMBB / URLLC / mMTC [ms] (median) | phi_bar | QoS sat. | feasible | max CPU util | thr [Mb/s] |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = " / ".join("inf" if np.isnan(x) else f"{x:.2f}" for x in r["d_ms"])
        lines.append(f"| {r['unit']} | {r['U']} | {r['scenario']} | {d} | {r['phi']:.3f} | {r['sat']:.2f} | "
                     f"{r['feas']:.2f} | {r['cpu']:.2f} | {r['thr']:.1f} |")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nfigures and report written to {OUT}")


if __name__ == "__main__":
    main()
