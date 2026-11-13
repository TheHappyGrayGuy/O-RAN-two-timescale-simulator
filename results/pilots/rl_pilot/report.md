# Learned policies vs the Phase-5 optimum

Objective on the Phase-5 snapshot (lower is better); gap = objective / optimum - 1, mean over seeds.

| U | optimum | static best | MARL (2 x TD3) (gap) | TD3 + convex (gap) | SAC + convex (gap) | single SAC (gap) |
|---|---|---|---|---|---|---|
| 5 | 0.0301 | 0.0341 | 0.0465 (+54.2%) | 0.0339 (+12.4%) | 0.0380 (+26.2%) | 0.0754 (+150.2%) |

Infeasible final policies count as inf. 'best' = best feasible evaluation during training.
