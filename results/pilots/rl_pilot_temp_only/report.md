# Learned policies vs the Phase-5 optimum

Objective on the Phase-5 snapshot (lower is better); gap = objective / optimum - 1, mean over seeds.

| U | optimum | static best | MARL (2 x TD3) (gap) | single SAC (gap) |
|---|---|---|---|---|
| 20 | 0.0399 | 0.0470 | 0.0499 (+25.3%) | 0.0448 (+12.4%) |

Infeasible final policies count as inf. 'best' = best feasible evaluation during training.
