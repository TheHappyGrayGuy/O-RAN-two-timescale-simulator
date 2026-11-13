# Two-timescale O-RAN resource allocation: simulator, optimum and RL baselines

Simulation code for the O-RAN system model of the thesis. The model is a sliced O-RAN network (eMBB, URLLC, mMTC) in which:

- the **large timescale** (every K = 10 slots) chooses user-to-RU association α and CU/DU placement ψ on edge or regional O-Cloud nodes;
- the **small timescale** (every 1 ms slot) chooses PRB allocation κ and transmit power ω.

The objective is a QoS-normalised weighted sum of slice delay and processing cost:
J = Σ_s δ_s · mean(D/D^QoS) + (1 − δ_s) · φ̄/φ_max.

The repository contains:

1. the network simulator (topology, users and traffic, channel, delay and cost KPIs, all constraints), with tests;
2. a near-optimal solver with certified lower bounds (the benchmark);
3. four RL methods: MARL (2 × TD3), TD3 + convex, SAC + convex, and single SAC;
4. results, figures, diagnostics, and a standalone results chapter in English and Persian (LaTeX).

## Layout

| Path | Contents |
|---|---|
| `oran_sim/` | the simulator package (see the table below) |
| `scripts/` | experiment runners and plotting |
| `scripts/diagnostics/` | tests explaining why the association/placement agent did not learn |
| `tests/` | pytest suite (`test_1` to `test_7`, one file per phase) |
| `results/phase5/` | optimum vs static baselines (`report.md`, `small.json`, `full.json`) |
| `results/rl/` | RL grid: 4 methods × U ∈ {5, 10, 15, 20} × seeds {0, 1} × 500 episodes (`report.md`, per-run `eval.json` and `log.csv`) |
| `results/diagnostics/` | diagnostic outputs and their summary (`README.md`) |
| `results/figures*/` | figures: Phase 5, system model, RL convergence |
| `results/pilots/` | earlier RL pilot runs (tuning history) |
| `results/setup/` | topology and SNR sanity report |
| `chapter6/` | standalone results chapter: `chapter6_en.tex` / `chapter6_fa.tex`, PDFs and figures |
| `ROADMAP.md` | development phases, tests per phase, and modelling decisions D1–D3 |

| Module | Role |
|---|---|
| `config.py` | all parameters (ch6 table plus documented assumptions) |
| `topology.py` | RU sites, O-Cloud graph, routes, fibre delay |
| `users.py` | user drops, slices, Poisson traffic, mobility |
| `channel.py` | path loss, shadowing, fading, SNR/SINR, PRB rate |
| `kpis.py`, `network.py` | delay terms, cost, constraints and objective for a decision |
| `reference_policies.py`, `baselines.py` | static scenarios (all edge, CU regional + DU edge, all regional) and other baselines |
| `optimal.py` | power by KKT water-filling, PRB local search, exhaustive placement, association multi-start, Lagrangian lower bounds |
| `marl_env.py` | two-timescale environment: state, action decoding and reward |
| `agents.py`, `rl_methods.py` | TD3 and SAC agents, the four methods, training with checkpoints |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

This was developed with Python 3.14, numpy 2.4, scipy 1.17, torch 2.15 (CPU), networkx 3.7, cvxpy 1.9 and matplotlib 3.10.

## Reproduce

```bash
python scripts/report_setup.py                       # topology and channel sanity report -> results/setup/
python scripts/phase5_optimal.py --seeds 5           # optimum and bounds -> results/phase5/   (U = 20 takes ~30 h per seed)
python scripts/run_rl.py --episodes 500 --seeds 0 1 --eval-every 25 \
    --prior 2 --temp 1 --act-pen 0.1 --large-hidden 256 256 --large-updates 2   # RL grid -> results/rl/ (~10 h, 11 cores)
python scripts/compare_rl.py --rl results/rl         # RL vs optimum table -> results/rl/report.md
python scripts/plot_outputs.py --rl results/rl --out results/figures   # figures
python scripts/diagnostics/diag_opt.py               # one of the diagnostics (see results/diagnostics/README.md)
```

- `run_rl.py` and `phase5_optimal.py` skip finished runs and resume from checkpoints, so they can be stopped and restarted.
- On Windows, `scripts/keep_awake.py <pid>` keeps the PC awake while a run is alive, and `scripts/stop_runs.ps1` stops all runs.
- Trained models (`*.pt`) are not included because of their size; `run_rl.py` regenerates them.

## Results so far

**Optimum vs best static scenario** (Phase 5, mean of 5 seeds; lower J is better):

| U | Optimum J | Best static J | Gain | Gap to conditional bound | Gap to global bound |
|---|---|---|---|---|---|
| 5 | 0.0310 | 0.0561 | 33.2% | 0.001% | 4.5% |
| 10 | 0.0344 | 0.0432 | 19.5% | 0.35% | 7.1% |
| 15 | 0.0384 | 0.0487 | 20.8% | 1.51% | 10.0% |
| 20 | 0.0426 | 0.0525 | 18.5% | 0.26% | 11.3% |

- The solver matches brute-force enumeration on 12 of 12 small instances.
- The conditional bound fixes the association.
- The global bound also relaxes the association; it is weaker, so the result is a near-optimal benchmark with a certified bound.

**RL grid** (`results/rl/report.md`, gap to the optimum, mean of seeds 0 and 1):

| U | MARL (2 × TD3) | TD3 + convex | SAC + convex | Single SAC |
|---|---|---|---|---|
| 5 | +20.2% | +20.0% | +20.0% | +20.3% |
| 10 | +17.0% | +15.3% | +15.3% | +17.7% |
| 15 | +13.1% | +10.9% | +10.9% | +17.3% |
| 20 | +13.3% | +8.8% | +8.8% | +15.8% |

**Status:** these RL results are not final. With the prior β = 2, the association/placement agent could not change the
static decision, so every method kept the best static association and placement. The diagnostics
(`results/diagnostics/README.md`) show the following:

- the PRB/power layer is within 0.6% of the optimum when given the right association and placement;
- the remaining gap comes entirely from association and placement;
- placement is the hard part.

A redesign is proposed there: RL chooses the placement, and the association is computed by local search.

## Thesis text

`chapter6/chapter6_en.tex` and `chapter6/chapter6_fa.tex` are standalone (report class). Build them with XeLaTeX:

```bash
cd chapter6
xelatex chapter6_en.tex && xelatex chapter6_en.tex
xelatex chapter6_fa.tex && xelatex chapter6_fa.tex    # needs IRNazanin*.ttf in chapter6/fonts/ (not included) and the XePersian package
```

The RL result sections still contain `% TODO` markers until the redesigned grid is run.
