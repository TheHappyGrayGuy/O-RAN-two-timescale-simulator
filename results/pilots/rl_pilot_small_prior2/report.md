# Learned policies vs the Phase-5 optimum

Objective on the Phase-5 snapshot (lower is better); gap = objective / optimum - 1, mean over seeds.

| U | optimum | static best | MARL (2 x TD3) (gap) | TD3 + convex (gap) | SAC + convex (gap) | single SAC (gap) |
|---|---|---|---|---|---|---|
| 20 | 0.0399 | 0.0470 | 0.0461 (+15.6%) | 0.0418 (+4.8%) | 0.0418 (+4.8%) | 0.0443 (+11.2%) |

Infeasible final policies count as inf. 'best' = best feasible evaluation during training.
