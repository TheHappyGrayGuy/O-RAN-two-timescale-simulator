# O-RAN Chapter 4 simulation: roadmap to results

Model source: `Thesis/chapter4/chapter4.tex` (system model), `chapter5.tex` (two-timescale MARL),
`chapter6.tex` (parameter table and result figures).

Run the tests: `python -m pytest -q` from this folder.
Run the setup report: `python scripts/report_setup.py`, which writes `results/setup/`.

| # | Phase | Code | Tests (gate to next phase) | Status |
|---|---|---|---|---|
| 0 | Parameter audit and spec freeze | `config.py`, this file | `test_parameters_from_chapter6`, `test_derived_units` | done; open issues below need decisions |
| 1 | Telecom environment and topology: RU sites, O-Cloud graph G=(N,E), routes p^b, fiber delay | `topology.py` | `tests/test_1_topology.py` (regions, graph size, route indicators, delay = length/v, reproducibility) | done |
| 2 | Users and RU distributions, slices, Poisson traffic, mobility | `users.py`, `topology.ru_positions` | `tests/test_2_users.py` (KS uniformity, slice counts sum to U, Poisson mean = var, bounded mobility) | done |
| 3 | Channel models: log-distance n=3 (ch6), 3GPP UMi, shadowing, Rayleigh/Rician fading, SNR/SINR, Shannon PRB rate | `channel.py` | `tests/test_3_channel.py` (FSPL and TR 38.901 values, 30 dB/decade, E\|h\|²=1, KS exponential, SINR hand calc, interference) | done |
| 4 | KPIs and constraints: eq.1–30, objective, QoS satisfaction, utilisation, fairness | `kpis.py`, `network.py`, `reference_policies.py` | `tests/test_4_kpis.py` (independent hand calculation of every delay term, one test that violates each constraint), `tests/test_5_network_integration.py` | done |
| 5 | Baselines (random, ch6 static scenarios, static best, placement search) and the optimal solution: exact power (KKT water-filling), PRB search with exact-move refinement, exhaustive placement, association multi-start; certified global optimum on small instances; certified dual lower bounds at full scale | `baselines.py`, `optimal.py`, `scripts/phase5_optimal.py` | `tests/test_6_optimal.py` (decomposition equals kpis to 1e-10; power never worse than cvxpy and passes KKT; dual bounds valid; exact = independent brute force; scalable solver = certified optimum on tiny instances; baseline ordering) | done |
| 6 | Two-timescale environment (ch5): large step every K=10 slots (association + CU/DU placement), small step every slot (PRB + power); continuous actions decoded so that eq.1/2/3/a5/6 always hold; shared reward r = -(J + violations)/J_static | `marl_env.py` | `tests/test_7_rl.py` (dimensions; episode 0 = Phase-5 snapshot; random actions structurally valid; large action held for K slots; reward formula; determinism; convex small step beats random) | done |
| 7 | Agents and methods: TD3 and SAC (ch6 sizes 1024/512 and 512/256, lr 1e-4/1e-3, gamma 0.99, buffer 5e4); methods **MARL** (2 x TD3, small critic centralised on the large action), **TD3 + convex** and **SAC + convex** (RL large timescale, Phase-5 radio solver for the small one), **single SAC** | `agents.py`, `rl_methods.py` | FIFO replay, soft update, target smoothing N(0,0.2) clipped; TD3 and SAC both learn a bandit; all 4 methods train, log and checkpoint end to end | done |
| 8 | Parallel training: every (method, U, seed) in its own process (1 thread each), longest first, resumable; auto-start after Phase 5 | `scripts/run_rl.py` | pilot run (U=5, 150 episodes) must improve on its random start | done (pilots in `results/pilots/`) |
| 9 | Experiments: U in {5,10,15,20} x 3 seeds x 1000 episodes; each policy evaluated on the Phase-5 snapshot against the certified optimum and the static baselines | `scripts/compare_rl.py` | report + convergence and optimality-gap figures | grid run (500 episodes, seeds 0-1, β = 2): the association/placement agent did not learn; see `results/diagnostics/README.md` |
| 10 | Figures and tables for ch6, reproducibility package | `scripts/plot_outputs.py` | figures regenerate from saved results only | done for Phase 5 and system figures; RL figures pending the redesign |

## Phase 5: how the optimum is obtained and certified

With `interference=False` (the ch6 setting) the objective splits exactly into per-RU radio terms
V_b(PRBs, power | users, CU/DU split) and a rate-independent infrastructure term I(placement | association).
The solver's evaluator matches `kpis.evaluate` to 1e-10.

| Block | Method | Optimality |
|---|---|---|
| Power ω given PRBs | linearize delays, then exact weighted water-filling, then exact line search | global (convex): KKT verified, never worse than cvxpy/Clarabel, which was itself sometimes inaccurate |
| PRBs κ given users | greedy, vectorized local search, then exact-move refinement (moves and swaps re-scored with exact power) | certified within 0.01–0.8% per RU by the Lagrangian dual of the fractional relaxation |
| Placement ψ given association | exhaustive vectorized enumeration of all N^(2B) = 1.68M placements | exact |
| Association α | multi-start local search; full enumeration on small instances | exact on small instances (12/12 = certified optimum); certified bound at full scale |

Lower bounds use weak duality, so every multiplier gives a valid bound and solver accuracy cannot invalidate one.
The dual has a closed form: water-filling per (RU, PRB, user), plus a valid cut that every user must hold at least one PRB.
- *Conditional bound*: fixed association, everything else relaxed. It certifies the PRB/power/placement blocks.
- *Global bound*: association relaxed as well. It is weaker because users choose their RU independently in the radio part and the infrastructure part.

Dead ends, kept for the record: cvxpy exponential-cone relaxations were inaccurate where fractional PRB shares
reach zero, and Outer Approximation MILPs (HiGHS) did not finish in reasonable time.

## Phase 6-8 modelling choices
- tau = 0.005 instead of the ch6 value 0.0005, which learns about 10x slower within a 1000-episode budget (`AgentConfig.tau`).
- Residual action prior on the large timescale (`EnvConfig.large_prior = 1.0`, `run_rl.py --prior 1.0`): the best static decision is added to the agent's association and placement scores, so a zero action reproduces the static baseline. In the U=5 pilot (150 episodes) the gaps to the optimum fell from +12/+26/+54/+150% to +2.2/+5.9/+7.2/+12.0% (TD3+convex / SAC+convex / MARL / single SAC). The thesis text must say that the learned policies start from the static baseline.
- The thesis uses 2000 episodes; the default here is 1000 per run, because 48 runs share 12 cores (`--episodes`).
- Training episodes redraw users on the fixed Phase-5 topology. Evaluation always uses episode 0, which is exactly the
  Phase-5 snapshot, so every learned policy is compared with the certified optimum of the same instance.
- The small agent's critic uses the large action held over the K slots as the other agent's action (CTDE).

## Decisions D1–D3 (made 2026-11-11, evidence-based)

How they were tested: for each candidate model, a coordinate-descent search over every RU's (CU node, DU node)
was compared with the fixed ch6 scenarios (all-edge, CU-regional + DU-edge, all-regional), over 8 seeds and U ∈ {5, 10, 15, 20}.
A model "has a trade-off" if three things hold: the optimum beats the best fixed scenario by a clear margin,
the optimal placement changes with load, and the edge is used preferentially for delay-critical slices.
`test_tradeoff_exists_under_default_model` guards these results.

| Model | Optimum gain vs best fixed, U = 5 / 10 / 15 / 20 | CU on edge, URLLC vs other users (U=5) |
|---|---|---|
| Thesis literal (eq.20 with CPU = demand, raw objective, no fronthaul budget) | 1.8 / 0.6 / 1.1 / 2.1 % | 0.38 vs 0.12 |
| **Chosen** (node M/M/c + QoS-normalised objective + fronthaul 100 µs) | **23.0 / 8.5 / 7.3 / 7.6 %** | **0.69 vs 0.12** |
| Chosen without fronthaul budget | 19.6 / 3.6 / 1.5 / 1.5 % | 0.69 vs 0.12 |
| Chosen with raw objective | 19.8 / 9.4 / 9.8 / 10.2 % | 0.38 vs 0.12 |

**D1: packet sizes are bytes** (`packet_size_unit="byte"`). The values 1500 / 200 / 20 are the Ethernet MTU,
a typical URLLC payload (3GPP TR 38.913 / TS 22.261 use 32–200 B) and a typical mMTC report.
Read as kilobits, the ch6 CPU figures (from Murti et al.) are 3–43× over capacity in every scenario
and the eMBB radio queue is unstable, so no result in ch6 could have come from that reading.
`"kbit"` is kept for reproduction, and `test_literal_kbit_reading_is_infeasible` documents the problem.

**D2: a real delay–cost trade-off.** Three changes, each needed (see the ablations):
- *Processing = shared M/M/c queue per O-Cloud node* (`processing_model="node_queue"`). "10 × 3.4 GHz" is read as
  10 cores × 3.4 GHz (edge: 1 core). A packet of q bits needs c̃^l·q cycles on one core. Every CU/DU hosted on a node
  shares its queue (Erlang-C waiting time). This removes the eq.20 artifact: with CPU = demand, delay was
  1/(λ_b(c̃γ−1)), the same on every node and shrinking with load. Stability (eq.22) now coincides with CPU capacity (eq.8),
  and γ is no longer needed. `"thesis"` and `"proportional"` remain available.
- *Fronthaul latency budget 100 µs one-way for RU↔DU* (`fronthaul_budget_s=100e-6`): the O-RAN WG4 split 7.2x
  transport class (eCPRI high-performance, ≈ 20 km of fibre). This is the standard reason DUs stay near the RUs.
  It adds a `fronthaul_latency` constraint.
- *QoS-normalised weighted sum* (`objective_norm="qos"`): Σ_s δ_s·mean(D/D^QoS) + (1−δ_s)·φ̄/φ_max, with
  φ_max = 2·φ^e_max (both functions of an RU on a saturated edge node). Both terms are dimensionless, so δ_s is
  a real priority and the result no longer depends on whether delay is in s or ms.
  `"raw"` (ch4 as written, delay in ms) remains available.

**D3: scheduling constraint.** θ = 1 ms (one NR slot at 15 kHz SCS, equal to the small-timescale step).
The constraint is D^prop(CU→DU) + D^proc_CU + D^tr_mid ≤ θ, checked only when CU and DU are on different nodes. This is the CU-side work
that must reach the DU within one slot. Under D2, CU processing is µs-scale, so the constraint holds normally
and binds only when a node or the midhaul is congested. It used to fail in every split case only because of the eq.20 artifact.

## Remaining modelling choices (switchable in `config.py`, default in bold)

4. Eq.24 has `+ λ` in the fronthaul denominator. It is implemented as `− λ` (M/M/1), consistent with eq.23 and eq.25.
5. Eq.10/11 are checked per link, with midhaul and fronthaul load summed on shared links.
6. Eq.a5 uses strict `<`. The code uses `≤ P_max`, so an equal split of the full power is allowed.
7. ch6 "2, 4" nodes is read as **N^r=2, N^e=4**. Link capacities "100 GHz, 10 GHz" are read as **fronthaul 100 Gb/s, midhaul 10 Gb/s**.
8. Not in the thesis, so assumed: f_c **3.5 GHz**, M **25 PRB**, L'max **2 km**, L''max **20 km**, slice weights δ = **0.5 / 0.9 / 0.1**,
   noise figure **0 dB**, cost unit v **1**, **10** slots per large step.
9. ch6 uses SNR without interference (**`interference=False`**). `True` adds co-channel interference between RUs.
10. Eq.30 charges each function the full cost of its node (Σ ψ·φ_{c,n}), which mildly rewards spreading load. Kept as written.

## Thesis text to update (ch4) once D1–D3 are accepted
- Eq.20–22: replace with the M/M/c node model (sojourn = W_q(n) + c̃^l q̄_b / f_core, stability ρ_n < 1).
- Add the fronthaul latency constraint D^prop_{b,DU} + D^tr_front ≤ 100 µs to the problem.
- Objective: normalise the delay by D^QoS and the cost by φ_max. Fix the `+λ` in eq.24.
- ch6 table: packet sizes in bytes. State θ = 1 ms.
