# Learned policies vs the Phase-5 optimum

Objective on the Phase-5 snapshot (lower is better); gap = objective / optimum - 1, mean over seeds.

| U | optimum | static best | MARL (2 x TD3) (gap) | TD3 + convex (gap) | SAC + convex (gap) | single SAC (gap) |
|---|---|---|---|---|---|---|
| 10 | 0.0335 | 0.0383 | 0.0375 (+11.9%) | 0.0351 (+4.9%) | 0.0348 (+4.0%) | 0.0372 (+11.2%) |
| 20 | 0.0399 | 0.0470 | 0.0478 (+19.9%) | 0.0425 (+6.6%) | 0.0411 (+3.0%) | 0.0441 (+10.8%) |

Infeasible final policies count as inf. 'best' = best feasible evaluation during training.
